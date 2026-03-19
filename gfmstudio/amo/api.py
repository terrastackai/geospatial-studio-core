# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gfmstudio.amo.task_manager import amo_task_manager
from gfmstudio.auth import authorizer
from gfmstudio.celery_worker import invoke_model_offboarding, invoke_model_onboarding
from gfmstudio.common.api import utils
from gfmstudio.log import logger

from .schemas import OnboardingStatus, OnboardModelRequest

app = APIRouter(dependencies=[Depends(authorizer.auth_handler)])


def validate_model_id(name: str):
    if len(name) > 30:
        raise HTTPException(
            status_code=422, detail="Model ID must not exceed 30 characters."
        )
    if any(c.isupper() for c in name):
        raise HTTPException(status_code=422, detail="Model ID must be lowercase.")
    pattern = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
    if not re.match(pattern, name):
        raise HTTPException(
            status_code=422,
            detail="Model ID contains invalid characters. Use only [a-z0-9-] and do not start or end with '-'.",
        )


@app.get("/amo-tasks/{model_id}", tags=["Inference / Models"])
async def retrieve_amo_task(
    model_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(authorizer.auth_handler),
):
    """Check the status of a background task."""
    user = auth[0]
    model_id = model_id.strip()
    sanitized_model_id = validate_model_id(model_id)
    sanitized_model_id = model_id

    task = amo_task_manager._check_availability(
        db=db,
        task_id=f"amo-{sanitized_model_id}",
        user=user,
    )
    if not task:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": task.task_id,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


@app.post("/amo-tasks", tags=["Inference / Models"])
async def onboard_inference_model(
    request: Request,
    item: OnboardModelRequest,
    db: Session = Depends(utils.get_db),
    auth=Depends(authorizer.auth_handler),
):
    user = auth[0]
    sanitized_model_id = None
    # Validate that model_id complies with DNS naming conventions
    model_id = item.model_id.strip()
    validate_model_id(model_id)
    sanitized_model_id = model_id

    sanitized_model_framework = item.model_framework.strip().replace("_", "-").lower()

    task_id = f"amo-{sanitized_model_id}"
    task_status = amo_task_manager.get_task_status(
        db=db,
        task_id=task_id,
        user=user,
    )
    if task_status and (
        task_status
        not in [
            OnboardingStatus.MODEL_OFFBOARDING_COMPLETE,
            OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
            OnboardingStatus.PRESIGNED_URL_EXPIRED,
            OnboardingStatus.PRESIGNED_URL_FAILED,
        ]
    ):
        msg = (
            f"model_id: {item.model_id} already in use."
            "You can only reuse a model_id for successfully offboarded models and models that fail onboarding."
        )
        raise HTTPException(
            status_code=422,
            detail={"message": msg},
        )
    else:
        amo_task_manager.set_task_status(
            db=db,
            task_id=task_id,
            status=OnboardingStatus.ARTIFACT_TRANSFER_REQUEST_SUBMITTED,
            user=user,
        )

    # Background tasks
    job_id = f"amo-artifacts-{model_id.lower()}-{str(uuid.uuid4())[:6]}"
    invoke_model_onboarding.delay(
        model_id=sanitized_model_id,
        model_framework=sanitized_model_framework,
        model_configs_url=str(item.model_configs_url),
        model_checkpoint_url=str(item.model_checkpoint_url),
        deployment_type=item.deployment_type,
        deploy_image=item.inference_container_image,
        artifact_job_id=job_id,
        user=user,
    )

    return {
        "model_framework": sanitized_model_framework,
        "model_id": sanitized_model_id,
        "jobid": job_id,
        "message": "Prepare model artifact started",
    }


@app.delete("/amo-tasks/{model_id}", tags=["Inference / Models"])
async def offboard_inference_model(
    model_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(authorizer.auth_handler),
):
    user = auth[0]
    model_id = model_id.strip()
    validate_model_id(model_id)
    sanitized_model_id = model_id

    task = amo_task_manager._check_availability(
        db=db,
        task_id=f"amo-{sanitized_model_id}",
        user=user,
    )
    if not task:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if task.status in [
        OnboardingStatus.MODEL_OFFBOARDING_REQUEST_SUBMITTED,
        OnboardingStatus.MODEL_OFFBOARDING_STARTED,
        OnboardingStatus.MODEL_OFFBOARDING_COMPLETE,
    ]:
        return {
            "message": f"An existing Model offboarding task for {model_id} already in progress or complete."
        }

    try:
        invoke_model_offboarding.delay(model_id=sanitized_model_id, user=user)
        amo_task_manager.set_task_status(
            task_id=sanitized_model_id,
            status=OnboardingStatus.MODEL_OFFBOARDING_REQUEST_SUBMITTED,
            user=user,
        )
        return {"message": "Model offboarding request submitted"}
    except Exception:
        amo_task_manager.get_task_status(task_id=sanitized_model_id, user=user)
        logger.exception(
            "Internal server error occured during internal script execution."
        )
        raise HTTPException(status_code=500, detail="Internal server error occured.")
