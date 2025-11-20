# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio

import requests
from celery import Celery

from gfmstudio.amo.services import invoke_deploy_models_with_caikit
from gfmstudio.amo.utils import invoke_model_offboarding_handler
from gfmstudio.config import settings
from gfmstudio.fine_tuning.core.kubernetes import (
    deploy_hpo_tuning_job,
    deploy_tuning_job,
)
from gfmstudio.fine_tuning.utils.webhook_event_handlers import (
    handle_dataset_factory_webhooks,
    handle_fine_tuning_webhooks,
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
    return asyncio.run(deploy_tuning_job(**kwargs))


@celery_app.task(
    name="invoke_hpo_fine_tuning_task",
    queue=FT_SERVICE_NAME,
)
def deploy_hpo_tuning_celery_task(**kwargs):
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
