# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

import requests
from celery import Celery

from gfmstudio.amo.services import invoke_deploy_models_with_caikit
from gfmstudio.amo.utils import invoke_model_offboarding_handler
from gfmstudio.config import settings
from gfmstudio.fine_tuning.core.kubernetes import (
    check_k8s_job_status,
    deploy_hpo_tuning_job,
    deploy_tuning_job,
)
from gfmstudio.fine_tuning.utils.webhook_event_handlers import (
    handle_dataset_factory_webhooks,
    handle_fine_tuning_webhooks,
    update_tune_status,
)
from gfmstudio.inference.services import (
    invoke_cancel_inference_handler,
    invoke_tune_upload_handler,
)
from gfmstudio.inference.v2.services import invoke_inference_v2_pipelines_handler
from gfmstudio.log import logger
INF_SERVICE_NAME = "inference_gateway"
FT_SERVICE_NAME = "geoft"
celery_app = Celery(
    "gfmstudio",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.task_queues = {
    INF_SERVICE_NAME: {
        "exchange": "default",
        "routing_key": INF_SERVICE_NAME,
    },
    FT_SERVICE_NAME: {
        "exchange": "default",
        "routing_key": FT_SERVICE_NAME,
    },
}
celery_app.conf.task_default_queue = INF_SERVICE_NAME
celery_app.conf.task_default_routing_key = INF_SERVICE_NAME


@celery_app.task(
    name="invoke_fine_tuning_kjob_task",
    queue=FT_SERVICE_NAME,
)
def deploy_tuning_job_celery_task(**kwargs):
    # Inject the monitoring task into kwargs to avoid circular import
    kwargs["_monitor_task"] = monitor_k8_job_completion_task
    return asyncio.run(deploy_tuning_job(**kwargs))


@celery_app.task(
    name="monitor_k8_job_completion_task",
    queue=FT_SERVICE_NAME,
    bind=True,  # Bind to get access to self for retry
    max_retries=30,  # Allow many retries
    default_retry_delay=30,  # Start with 30 seconds
)
def monitor_k8_job_completion_task(self, ftune_id: str):
    """Celery task to monitor Kubernetes job completion with automatic retry."""
    # Get max wait time from settings, fallback to 7200 seconds (2 hours) if not set
    max_wait = settings.KJOB_MAX_WAIT_SECONDS or 7200

    try:
        k8s_job_status, _ = asyncio.run(
            check_k8s_job_status(ftune_id)
        )
    except Exception as exc:
        if "not found" in str(exc):
            # Job not found, consider it done (likely already completed and deleted)
            logger.debug(
                f"{ftune_id}: Job not found, assuming completed and cleaned up"
            )
            return "Completed"
        # Unexpected error, retry with exponential backoff
        logger.warning(f"{ftune_id}: Error checking job status, will retry: {exc}")
        raise self.retry(exc=exc, countdown=min(2**self.request.retries * 30, max_wait))

    # Handle None status (job not found after retries)
    if k8s_job_status is None:
        # Job doesn't exist - either completed and deleted, or never created
        logger.debug(
            f"{ftune_id}: Job status is None, assuming completed and cleaned up"
        )
        return "Completed"
    
    # Handle Unknown status (job/pod not found - likely deleted after completion)
    if k8s_job_status == "Unknown":
        # Job and pods not found - resources were cleaned up after completion
        logger.info(
            f"{ftune_id}: Job status is Unknown (resources deleted), assuming completed and cleaned up"
        )
        return "Completed"

    if k8s_job_status in ["Complete", "Failed"]:
        # Job is done
        logger.info(f"{ftune_id}: Job finished with status: {k8s_job_status}")
        return k8s_job_status

    # Update database status based on pod phase
    if k8s_job_status == "Running":
        # Pod is actually running - update database to In_progress
        try:
            asyncio.run(update_tune_status(ftune_id, "In_progress"))
        except Exception as e:
            logger.warning(f"{ftune_id}: Failed to update status to In_progress: {e}")

    # Job still running, retry with exponential backoff
    # countdown: 30s, 60s, 120s, 240s, 480s, 960s (max with default 600s setting)
    countdown = min(2**self.request.retries * 30, max_wait)
    logger.info(
        f"{ftune_id}: Job status={k8s_job_status}, will check again in {countdown}s"
    )
    raise self.retry(countdown=countdown)


@celery_app.task(
    name="invoke_hpo_fine_tuning_task",
    queue=FT_SERVICE_NAME,
)
def deploy_hpo_tuning_celery_task(**kwargs):
    kwargs["_monitor_task"] = monitor_k8_job_completion_task
    return asyncio.run(deploy_hpo_tuning_job(**kwargs))


@celery_app.task(
    name="invoke_fine_tuning_webhook_task",
    queue=FT_SERVICE_NAME,
)
def invoke_fine_tuning_webhook_handlers(**kwargs):
    return asyncio.run(handle_fine_tuning_webhooks(**kwargs))


@celery_app.task(
    name="invoke_dataset_webhooks_task",
    queue=FT_SERVICE_NAME,
)
def invoke_dataset_webhooks_handlers(**kwargs):
    return asyncio.run(handle_dataset_factory_webhooks(**kwargs))


@celery_app.task(name="invoke_v2_inference_pipelines", queue=INF_SERVICE_NAME)
def invoke_v2_inference_pipelines_task(**kwargs):
    return asyncio.run(
        invoke_inference_v2_pipelines_handler(**kwargs, pipeline_version="v2")
    )


@celery_app.task(
    name="invoke_tune_download",
    queue=INF_SERVICE_NAME,
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def invoke_tune_upload(self, **kwargs):
    try:
        return asyncio.run(invoke_tune_upload_handler(**kwargs))
    except requests.exceptions.ConnectionError as e:
        logger.warning("retrying Celery task after download failure")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="invoke_cancel_inference", queue=INF_SERVICE_NAME)
def invoke_cancel_inference(inference_id, user):
    return asyncio.run(invoke_cancel_inference_handler(inference_id, user))


@celery_app.task(name="invoke_model_onboarding", queue=INF_SERVICE_NAME)
def invoke_model_onboarding(**kwargs):
    return asyncio.run(invoke_deploy_models_with_caikit(**kwargs))


@celery_app.task(name="invoke_model_offboarding", queue=INF_SERVICE_NAME)
def invoke_model_offboarding(**kwargs):
    return asyncio.run(invoke_model_offboarding_handler(**kwargs))
