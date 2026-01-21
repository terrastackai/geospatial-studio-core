# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import os
from typing import Any, Dict, List, Optional

from botocore.client import ClientError
from fastapi import HTTPException
from sqlalchemy.orm import Session
from terrakit.download.geodata_utils import list_data_connectors

from gfmstudio.celery_worker import invoke_v2_inference_pipelines_task
from gfmstudio.common.api import crud
from gfmstudio.config import settings
from gfmstudio.cos_client import get_cos_client
from gfmstudio.inference.types import EventDetailType, EventStatus, ModelStatus
from gfmstudio.inference.v2 import schemas
from gfmstudio.inference.v2.models import Model, Task, GenericProcessor
from gfmstudio.log import logger

model_crud = crud.ItemCrud(model=Model)
task_crud = crud.ItemCrud(model=Task)
generic_processor_crud = crud.ItemCrud(model=GenericProcessor)

pipelines_bucket_name = settings.PIPELINES_V2_COS_BUCKET
EXPERIMENTAL_MODEL_NAMING = "sandbox"


def is_model_inference_ready(model_obj: Model) -> bool:
    """
    Determine if the given model is ready for inference.

    Parameters
    ----------
    model_obj : Model
        The model to validate.

    Returns
    -------
    bool
        True if model is ready; raises HTTPException otherwise.
    """

    status = model_obj.status
    if status in {
        ModelStatus.DEPLOY_IN_PROGRESS,
        ModelStatus.DEPLOY_FAILED,
        ModelStatus.DEPLOY_ERROR,
        # ModelStatus.PENDING,
    }:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": f"Model in state: {status} cannot be used for inference. "
                "Ensure it is successfully deployed first OR "
                "Deploy another instance of the fine-tuned model."
            },
        )

    if EXPERIMENTAL_MODEL_NAMING in model_obj.internal_name:
        return True

    if not model_obj.model_url:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Cannot run inference > model missing model_url. Ensure model was deployed successfully."
            },
        )

    return True


def save_inference_config(
    inference_id: str,
    inference_config: dict,
):
    """
    Save the inference configuration to a file in the specified S3 bucket.

    Parameters
    ----------
    inference_id : str
        The unique identifier for the inference.
    inference_config : dict
        The configuration data to save.
    inference_root_folder : str
        The root folder in the S3 bucket where the config will be saved.

    """
    inference_root_folder = settings.PIPELINES_V2_INFERENCE_ROOT_FOLDER
    pipeline_s3_client = get_cos_client()
    # TODO: Use commented out code after merging 👇
    # https://github.ibm.com/geospatial-studio/geospatial-studio-core/pull/334
    # if settings.ENVIRONMENT.lower() != "local":
    if not inference_root_folder:
        try:
            pipeline_s3_client.put_object(
                Body=json.dumps(inference_config, indent=4),
                Bucket=pipelines_bucket_name,
                Key=f"{inference_id}/{inference_id}_config.json",
                ContentType="application/json",
            )
            pipeline_s3_client.put_object(
                Key=f"{inference_id}/{inference_id}-task_planning/",
                Bucket=pipelines_bucket_name,
            )
        except ClientError as e:
            logger.error(f"Failed to save inference config to S3: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save inference configuration to S3 bucket.",
            )
        logger.info(
            f"Saved inference config to S3 bucket {pipelines_bucket_name} "
            f"at {inference_id}/{inference_id}_config.json"
        )
    else:
        config_filename = f"{inference_id}_config.json"

        base_path = os.path.join(inference_root_folder, inference_id)
        task_path = os.path.join(base_path, f"{inference_id}-task_planning")
        os.makedirs(task_path, exist_ok=True)

        config_file_path = os.path.join(base_path, config_filename)
        with open(config_file_path, "w") as fp:
            json.dump(inference_config, fp, indent=4)

        logger.info(f"Saved inference config to local path {config_file_path}")


def get_inference_model(
    db: Session, inference: schemas.InferenceCreateInput, user: str
):
    """Retrieve model by ID or display name."""
    if inference.model_id:
        model_obj = model_crud.get_by_id(
            db=db, item_id=inference.model_id, shared=True, user=user
        )
    else:
        models = model_crud.get_all(
            db=db,
            filters={"display_name": inference.model_display_name, "latest": True},
            shared=True,
            user=user,
        )
        model_obj = models[0] if models else None
        if model_obj:
            inference.model_id = model_obj.id

    if not model_obj:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Ensure it exists before retrying.",
        )

    return model_obj


def get_data_connector_config(
    inference: schemas.InferenceCreateInput, model_data_spec: list
) -> list:
    """Get data connector configuration from inference or data sources catalogue."""
    if inference.data_connector_config:
        return inference.data_connector_config

    data_sources = list_data_connectors(as_json=True)
    data_connector_config = []

    for spec in model_data_spec:
        data_source = next(
            (
                item
                for item in data_sources
                if item["connector"] == spec["connector"]
                and item["collection_name"] == spec["collection"]
            ),
            None,
        )
        if data_source:
            data_connector_config.append(data_source)

    return data_connector_config


def get_pipeline_steps(inference: schemas.InferenceCreateInput, model_obj) -> list:
    """Get and adjust pipeline steps based on spatial domain URLs."""
    pipeline_steps = inference.pipeline_steps or model_obj.pipeline_steps or []

    if inference.spatial_domain.urls:
        # Replace specific connectors with url-connector for URL-based domains
        pipeline_steps = [
            {
                k: (
                    "url-connector"
                    if v in ["sentinelhub-connector", "terrakit-data-fetch"]
                    else v
                )
                for k, v in step.items()
            }
            for step in pipeline_steps
        ]

    if inference.generic_processor_id:
        # Append generic processor step if provided
        pipeline_steps.append(
            {
                "status": "WAITING",
                "process_id": "generic-python-processor",
                "step_number": len(pipeline_steps),
                # Executable file?
            }
        )
    return pipeline_steps


def build_inference_config(
    inference: schemas.InferenceCreateInput,
    model_obj,
    data_connector_config: List[Dict],
    model_data_spec: List[Dict],
    geoserver_push: Dict,
    pipeline_steps: List[Dict],
    generic_processor: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build the complete inference configuration dictionary."""
    return {
        **inference.model_dump(),
        "data_connector_config": data_connector_config,
        "model_input_data_spec": model_data_spec,
        "post_processing": inference.post_processing
        or model_obj.postprocessing_options,
        "geoserver_push": geoserver_push,
        "model_access_url": model_obj.model_url,
        "pipeline_steps": pipeline_steps,
        "generic_processor": generic_processor,
    }


def handle_pipeline_integration(
    db: Session,
    inference: schemas.InferenceCreateInput,
    model_obj,
    created_inference,
    inference_config: dict,
    user: str,
):
    """Handle pipeline integration based on configuration type."""
    if settings.PIPELINES_V2_INTEGRATION_TYPE == "database":
        handle_database_integration(
            db, inference, model_obj, created_inference, inference_config, user
        )
    elif settings.PIPELINES_V2_INTEGRATION_TYPE == "api":
        handle_api_integration(inference, created_inference, user)


def build_inference_pipelines_v2_payload(
    model_obj,
    inference_config: dict,
    inference_id: str,
    user: str,
):
    return schemas.V2PipelineCreate(
        model_internal_name=model_obj.internal_name,
        model_id=model_obj.internal_name,
        tune_id=str(inference_config.get("fine_tuning_id")),
        inference_id=inference_id,
        description=inference_config.get("description"),
        location=inference_config.get("location"),
        model_access_url=model_obj.model_url,
        user=user,
        spatial_domain=inference_config.get("spatial_domain") or {},
        temporal_domain=inference_config.get("temporal_domain"),
        maxcc=inference_config.get("maxcc"),
        model_input_data_spec=inference_config.get("model_input_data_spec"),
        data_connector_config=inference_config.get("data_connector_config"),
        geoserver_push=inference_config.get("geoserver_push"),
        pipeline_steps=inference_config.get("pipeline_steps"),
        post_processing=inference_config.get("post_processing"),
        inferencing=inference_config.get("inferencing", {}),
        generic_processor=inference_config.get("generic_processor", {})
    )


def handle_database_integration(
    db: Session,
    model_obj,
    created_inference,
    inference_config: dict,
    user: str,
):
    """Handle database-based pipeline integration."""
    # Build V2 payload
    inference_v2_payload = build_inference_pipelines_v2_payload(
        model_obj=model_obj,
        inference_config=inference_config,
        inference_id=str(created_inference.id),
        user=user,
    )

    # Save config with renamed keys
    pipeline_config = inference_v2_payload.model_dump()
    pipeline_config["pipeline-steps"] = pipeline_config.pop("pipeline_steps")
    save_inference_config(str(created_inference.id), pipeline_config)

    # Create planning task
    task_crud.create(
        db=db,
        item=schemas.TaskCreate(
            task_id=f"{created_inference.id}-task_planning",
            status="READY",
            pipeline_steps=[
                {"status": "READY", "process_id": "inference-planner", "step_number": 0}
            ],
            inference_id=created_inference.id,
            inference_folder=f"/data/{created_inference.id}",
            created_by=user,
        ),
        user=user,
    )


def handle_api_integration(
    inference: schemas.InferenceCreateInput, created_inference, user: str
):
    """Handle API-based pipeline integration."""
    logger.info("Invoking v2 scaled pipelines to run inference.")

    inference_v2_payload = schemas.V2PipelineCreate(
        model_internal_name=created_inference.model.internal_name,
        model_id=str(created_inference.model.id),
        tune_id=inference.fine_tuning_id,
        inference_id=str(created_inference.id),
        description=created_inference.description,
        location=created_inference.location,
        **created_inference.inference_config,
    )

    invoke_v2_inference_pipelines_task.delay(
        payload=inference_v2_payload.model_dump(),
        notify=True,
        user=user,
        inference_id=str(created_inference.id),
        channel=f"geoinf:event:{created_inference.id}",
        detail_type=EventDetailType.TASK_COMPLETE,
        status=EventStatus.COMPLETED,
    )


def get_generic_processor(generic_processor_object: GenericProcessor) -> dict:

    """Retrieve generic processor content.

    Returns
    -------
    dict
        Contents of the generic python processor
    """
    return {
            "name": generic_processor_object["name"],
            "description": generic_processor_object["description"],
            "processor_parameters": generic_processor_object["processor_parameters"],
            "processor_file_path": generic_processor_object["processor_file_path"],
            "status": generic_processor_object["status"],
            "processor_presigned_url": generic_processor_object["processor_presigned_url"],
        }