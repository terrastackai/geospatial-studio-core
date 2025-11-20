# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from http.client import HTTPException

from gfmstudio.amo.schemas import OnboardingStatus
from gfmstudio.amo.utils import (
    delete_job_on_completion,
    invoke_model_onboarding_handler,
    prepare_model_files,
)
from gfmstudio.config import settings
from gfmstudio.exceptions import PresignedLinkExpired
from gfmstudio.log import logger

from .task_manager import amo_task_manager


async def prepare_model_artifacts(
    model_id: str,
    model_framework: str,
    model_configs_url: str,
    model_checkpoint_url: str,
    artifact_job_id: str,
    user: str,
):
    try:
        # Call prepare_model_files synchronously to get the jobid
        jobid = await prepare_model_files(
            model_id=model_id,
            configs_url=model_configs_url,
            checkpoint_url=model_checkpoint_url,
            destination_bucket=settings.AMO_FILES_BUCKET,
            artifact_job_id=artifact_job_id,
        )

        # The jobid can be used for further tasks, e.g., monitoring or deletion
        await delete_job_on_completion(model_id, jobid)
    except PresignedLinkExpired:
        logger.debug("Checkpoint or config presigned link provided is expired.")
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.PRESIGNED_URL_EXPIRED,
        )
    except HTTPException as http_exc:
        # Catching HTTPException, any propagated presigned URL status updates are returned
        http_exc_detail = str(http_exc.detail)
        if "PRESIGNED URL" not in http_exc_detail.upper():
            amo_task_manager.set_task_status(
                task_id=model_id,
                status=OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
            )
        raise http_exc
    except Exception:
        # Catch any other Exceptions
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
        )
        logger.exception(
            "Internal server error occurred during internal script execution."
        )


async def invoke_deploy_models_with_caikit(
    model_id: str,
    model_framework: str,
    model_configs_url: str,
    model_checkpoint_url: str,
    deploy_image: str,
    user: str,
    deployment_type: str = None,
    model_deploy_token: str = None,  # TODO: Deprecate
    artifact_job_id: str = None,
    resources: dict = None,
    gpuResources: dict = None,
):
    await prepare_model_artifacts(
        model_id=model_id,
        model_framework=model_framework,
        model_configs_url=model_configs_url,
        model_checkpoint_url=model_checkpoint_url,
        artifact_job_id=artifact_job_id,
        user=user,
    )
    logger.info(f"{model_id}: Model onboarding starting 🏁 🏁 🏁")
    task_status = amo_task_manager.get_task_status(task_id=model_id, user=user)
    if task_status in [
        OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
        OnboardingStatus.ARTIFACT_TRANSFER_REQUEST_SUBMITTED,
        OnboardingStatus.ARTIFACT_TRANSFER_STARTED,
    ]:
        logger.error(f"{model_id}: Model artifacts not ready for onboarding 🛑")
        return

    try:
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_DEPLOY_REQUEST_SUBMITTED,
            user=user,
        )
        await invoke_model_onboarding_handler(
            model_framework,
            model_id,
            model_deploy_token,
            deployment_type,
            resources,
            gpuResources,
            deploy_image,
        )
        logger.info(f"Started model onboarding for {model_framework}:{model_id}")
        return {"message": "Model onboarding request submitted"}
    except Exception:
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_DEPLOY_FAILED,
            user=user,
        )
        logger.exception(
            "Internal server error occurred during internal script execution."
        )
