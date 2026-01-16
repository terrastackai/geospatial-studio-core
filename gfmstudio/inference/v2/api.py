# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import datetime
import os
import uuid
from typing import Optional

import boto3
import httpx
from botocore.client import ClientError, Config
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session
from sse_starlette import EventSourceResponse
from terrakit import DataConnector
from terrakit.download.geodata_utils import list_data_connectors

from gfmstudio.amo.schemas import OnboardModelRequest
from gfmstudio.auth.authorizer import auth_handler
from gfmstudio.celery_worker import (
    invoke_cancel_inference,
    invoke_dataset_webhooks_handlers,
    invoke_fine_tuning_webhook_handlers,
)
from gfmstudio.common.api import crud, utils
from gfmstudio.config import settings
from gfmstudio.cos_client import get_cos_client
from gfmstudio.data_advisor.data_advisor import (
    data_advisor_list_collections,
    invoke_data_advisor_service,
)
from gfmstudio.inference import redis_handler
from gfmstudio.inference.errors import errors
from gfmstudio.inference.integration_adaptors.utils import (
    generate_download_presigned_url,
    generate_upload_presigned_url,
)
from gfmstudio.inference.types import (
    EventDetailType,
    InferenceStatus,
    ModelStatus,
    transition_to,
)
from gfmstudio.inference.v2 import helpers, schemas
from gfmstudio.inference.v2.models import (
    Inference,
    Model,
    Notification,
    Task,
    GenericProcessor,
)
from gfmstudio.inference.v2.schemas import DataAdvisorRequestSchema
from gfmstudio.inference.v2.services import (
    cleanup_autodeployed_model_resources,
    notify_inference_webhook_events,
    send_websocket_notification,
    update_inference_status,
    update_inference_webhook_events,
)
from gfmstudio.log import logger

router = APIRouter(dependencies=[Depends(auth_handler)])

model_crud = crud.ItemCrud(model=Model)
task_crud = crud.ItemCrud(model=Task)
inference_crud = crud.ItemCrud(model=Inference)
notification_crud = crud.ItemCrud(model=Notification)
generic_processor_crud = crud.ItemCrud(model=GenericProcessor)

EXPERIMENTAL_MODEL_NAMING = "sandbox"
pipelines_bucket_name = settings.PIPELINES_V2_COS_BUCKET


# ***************************************************
# Model
# ***************************************************
@router.post(
    "/models",
    response_model=schemas.ModelGetResponse,
    tags=["Inference / Models"],
    status_code=201,
)
async def create_model(
    model: schemas.ModelCreateInput,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    existing_models = model_crud.get_all(
        db=db,
        filters={"display_name": model.display_name},
        user=user,
    )
    model = schemas.ModelCreate(**model.model_dump())
    if existing_models:
        model.version = existing_models[0].version + 1
    model.internal_name = utils.generate_internal_name(
        model.display_name, version=model.version
    )
    model.latest = model.latest or True
    new_model = model_crud.create(db, model, user=user)

    # Mark old model as not latest
    if existing_models:
        existing_model = existing_models[0]
        existing_model.latest = False
        db.commit()

    return new_model


@router.post(
    "/models/{model_id}/deploy",
    tags=["Inference / Models"],
    status_code=200,
)
async def deploy_permanent_model(
    request: Request,
    model_id: uuid.UUID,
    model: schemas.ModelOnboardingInputSchema,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    existing_model = model_crud.get_by_id(
        db=db,
        item_id=model_id,
        user=user,
    )
    if not existing_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # return model_crud.update(db, item_id=model_id, item=model, user=user)
    if existing_model.status in [
        ModelStatus.DEPLOY_IN_PROGRESS,
        # ModelStatus.DEPLOY_REQUESTED,
        ModelStatus.COMPLETED,
        ModelStatus.UNAVAILABLE,
    ]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Model in state: {existing_model.status} cannot be redeployed. Create a new Model."
            },
        )

    model_configs = existing_model.model_onboarding_config or {}
    checkpoint_url = model.model_checkpoint_url or model_configs.get(
        "model_checkpoint_url"
    )
    configs_url = model.model_configs_url or model_configs.get("model_configs_url")

    if not (checkpoint_url and configs_url):
        msg = (
            "Both model_checkpoint_url and model_configs_url should be provided in the "
            f"`{existing_model.display_name}` model to run inference on a fine-tuned model."
        )
        raise HTTPException(
            status_code=422,
            detail=[{"msg": msg}],
        )

    dict_item = model.model_dump()
    dict_item["status"] = ModelStatus.DEPLOY_REQUESTED
    updated_item = model_crud.update(db, model_id, dict_item, user=user)

    from gfmstudio.amo.api import onboard_inference_model

    # Invoke the model-onboarding microservice: Celery task started and runs in the background.
    await onboard_inference_model(
        request=request,
        item=OnboardModelRequest(
            model_id=str(updated_item.internal_name),
            model_name=str(updated_item.internal_name),
            model_configs_url=str(updated_item.model_configs_url),
            model_checkpoint_url=str(updated_item.model_checkpoint_url),
        ),
        db=db,
        auth=auth,
    )

    return {
        "id": existing_model.id,
        "internal_name": existing_model.internal_name,
        "display_name": existing_model.display_name,
        "status": updated_item.status,
    }


@router.patch(
    "/models/{model_id}",
    response_model=schemas.ModelGetResponse,
    tags=["Inference / Models"],
    status_code=201,
)
async def update_model(
    model_id: uuid.UUID,
    model: schemas.ModelUpdateInput,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    existing_model = model_crud.get_by_id(
        db=db,
        item_id=model_id,
        user=user,
    )
    if not existing_model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model_crud.update(db, item_id=model_id, item=model, user=user)


@router.get(
    "/models", response_model=schemas.ModelListResponse, tags=["Inference / Models"]
)
async def list_models(
    db: Session = Depends(utils.get_db),
    internal_name: Optional[str] = Query(
        None, description="Filter by the internal name of the model."
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    auth=Depends(auth_handler),
):
    user = auth[0]
    filters = {}
    search_filters = {}
    if internal_name:
        search_filters["internal_name"] = internal_name
    count, items = model_crud.get_all(
        db=db,
        limit=limit,
        skip=skip,
        search=search_filters,
        user=user,
        shared=True,
        filters=filters,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@router.get(
    "/models/{model_id}",
    response_model=schemas.ModelGetResponse,
    tags=["Inference / Models"],
)
async def get_model(
    model_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = model_crud.get_by_id(db, model_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Model not found")

    return item


@router.delete("/models/{model_id}", tags=["Inference / Models"], status_code=204)
async def delete_model(
    model_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = model_crud.get_by_id(db, model_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Model not found")

    model_crud.soft_delete(db, item_id=model_id, user=user)
    return item


# ***************************************************
# Inference
# ***************************************************
@router.post(
    "/inference/dry-run",
    response_model=schemas.V2PipelineCreate,
    tags=["Inference / Inference"],
    status_code=201,
)
async def dry_run_inference(
    request: Request,
    inference: schemas.InferenceCreateInput,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    return await create_inference(
        request=request,
        inference=inference,
        auth=auth,
        dry_run=True,
        db=db,
    )


@router.post(
    "/inference",
    response_model=schemas.InferenceGetResponse,
    tags=["Inference / Inference"],
    status_code=201,
)
async def create_inference(
    request: Request,
    inference: schemas.InferenceCreateInput,
    dry_run: bool = Query(False, include_in_schema=False),
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]

    # Validate and retrieve model
    model_obj = helpers.get_inference_model(db, inference, user)

    # Check if model is ready for inference
    helpers.is_model_inference_ready(model_obj=model_obj)
    if (
        EXPERIMENTAL_MODEL_NAMING in model_obj.internal_name
    ) and not inference.fine_tuning_id:
        raise HTTPException(
            status_code=422,
            detail={"msg": "`fine_tuning_id` required to try out experimental model."},
        )

    # Prepare inference configuration
    # Get the data spec for the model
    model_data_spec = inference.model_input_data_spec or model_obj.model_input_data_spec
    geoserver_push = inference.geoserver_push or model_obj.geoserver_push

    if not (model_data_spec and geoserver_push):
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing geoserver_push and model_input_data_spec in the model metadata. Please"
                " provide them in the request or update them in the model metadata."
            ),
        )

    # Get the data source details from the catalogue
    data_connector_config = helpers.get_data_connector_config(
        inference, model_data_spec
    )
    pipeline_steps = helpers.get_pipeline_steps(inference, model_obj)

    # Build inference configuration
    inference_config = helpers.build_inference_config(
        inference,
        model_obj,
        data_connector_config,
        model_data_spec,
        geoserver_push,
        pipeline_steps,
    )

    # Create inference record
    inference_obj = schemas.InferenceCreate(
        **inference.model_dump(), inference_config=inference_config
    )
    if dry_run is True:
        inference_v2_payload = helpers.build_inference_pipelines_v2_payload(
            model_obj=model_obj,
            inference_config=inference_config,
            inference_id=str(uuid.uuid4()),
            user=user,
        )
        return inference_v2_payload

    created_inference = inference_crud.create(db, item=inference_obj, user=user)
    created_inference.maxcc = inference_config["maxcc"]

    # Handle pipeline integration
    if settings.PIPELINES_V2_INTEGRATION_TYPE == "database":
        helpers.handle_database_integration(
            db, model_obj, created_inference, inference_config, user
        )
    elif settings.PIPELINES_V2_INTEGRATION_TYPE == "api":
        helpers.handle_api_integration(inference, created_inference, user)

    return created_inference


@router.post(
    "/data-advice/{data_connector}",
    status_code=200,
    tags=["Inference / Inference"],
)
async def check_data_availability(
    data_connector: str,
    item: DataAdvisorRequestSchema,
    auth=Depends(auth_handler),
):
    """
    Query data-advisor service to check data availability before running an inference.
    """
    # Missing credentials
    if data_connector in ["sentinelhub", "nasa_earthdata"]:
        credentials = {
            "sentinelhub": {
                "credentials": ["SH_CLIENT_ID", "SH_CLIENT_SECRET"],
            },
            "nasa_earthdata": {
                "credentials": ["NASA_EARTH_BEARER_TOKEN"],
            },
        }
        connector = credentials[data_connector]
        if all(cred not in os.environ for cred in connector["credentials"]):
            logger.warning(
                f"Error: Missing credentials {' and '.join(connector['credentials'])}. \
                    Please update .env with correct credentials."
            )
            raise HTTPException(
                status_code=401,
                detail=[{"msg": "Error: Missing credentials"}],
            )

    # Invalid collection
    dc = DataConnector(connector_type=data_connector)
    collections = dc.connector.list_collections()
    for collection in item.collections:
        if collection not in collections:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "msg": f"Invalid collection '{collection}'. \
                            Please choose from one of the following collection {collections}"
                    }
                ],
            )

    try:
        resp = await invoke_data_advisor_service(
            data_connector=data_connector, **item.model_dump()
        )
        return resp
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=[{"msg": f"{e}"}],
        )


@router.get(
    "/data-advice/{data_connector}",
    status_code=200,
    tags=["Inference / Inference"],
)
async def list_data_source_collections(
    data_connector: str,
    auth=Depends(auth_handler),
):
    """Query data-advisor service to list the collections available for a specific data source."""
    try:
        resp = await data_advisor_list_collections(data_connector=data_connector)
    except httpx.HTTPError:
        raise HTTPException(
            status_code=500,
            detail=[{"msg": "Could not connect to data-advisor service."}],
        )

    if resp:
        return resp
    else:
        raise HTTPException(status_code=resp.status_code, **resp)


@router.get(
    "/inference",
    response_model=schemas.InferenceListResponse,
    tags=["Inference / Inference"],
)
async def list_inferences(
    db: Session = Depends(utils.get_db),
    model_id: Optional[uuid.UUID] = Query(None, description="Inference model option."),
    tune_id: Optional[str] = Query(None, description="Tune id option."),
    created_by: Optional[str] = Query(
        None, description="Email of user who created inference"
    ),
    location: Optional[str] = Query(
        None, description="Location where the insference use "
    ),
    saved: Optional[bool] = Query(
        None, description="Filter pre-computed demo examples"
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    auth=Depends(auth_handler),
):
    user = auth[0]
    filters = {}
    filter_expr = None
    filter_expr_list = []
    if location:
        filters["location"] = location
    if saved is not None:
        filter_expr_list.append(Inference.demo["demo"].astext == str(saved).lower())
    if model_id:
        filters["model_id"] = model_id
    if created_by:
        filters["created_by"] = created_by
    if tune_id:
        filter_expr_list.append(
            Inference.inference_config["fine_tuning_id"].astext == str(tune_id).lower()
        )
    if filter_expr_list:
        filter_expr = and_(*filter_expr_list)

    count, items = inference_crud.get_all(
        db=db,
        limit=limit,
        skip=skip,
        user=user,
        filters=filters,
        filter_expr=filter_expr,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@router.get(
    "/inference/{inference_id}",
    response_model=schemas.InferenceGetResponse,
    tags=["Inference / Inference"],
)
async def retrieve_inference(
    inference_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = inference_crud.get_by_id(db, inference_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Inference not found")

    return item


@router.get(
    "/inference/{inference_id}/tasks",
    response_model=schemas.InferenceTasksListResponse,
    tags=["Inference / Inference"],
)
async def get_inference_tasks(
    inference_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = inference_crud.get_by_id(db, inference_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Inference not found")

    inference_tasks = task_crud.get_all(
        db, user=user, filters={"inference_id": inference_id}
    )

    return schemas.InferenceTasksListResponse(
        inference_id=inference_id,
        status=item.status,
        tasks=inference_tasks,
    )


@router.delete(
    "/inference/{inference_id}", tags=["Inference / Inference"], status_code=204
)
async def delete_inference(
    inference_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = inference_crud.get_by_id(db, inference_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Inference not found")

    inference_crud.soft_delete(db, item_id=inference_id, user=user)
    return item


@router.post(
    "/inference/{inference_id}/cancel", tags=["Inference / Inference"], status_code=202
)
async def cancel_inference(
    inference_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = inference_crud.get_by_id(db, inference_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Inference not found")
    if item.status in [
        InferenceStatus.COMPLETED,
        InferenceStatus.COMPLETED_WITH_ERRORS,
        InferenceStatus.FAILED,
        InferenceStatus.STOPPED,
    ]:
        return {"msg": f"Inference in {item.status} end state cannot be stopped."}
    invoke_cancel_inference.delay(str(inference_id), user)
    logger.info(f"Stopping inference {inference_id}")
    return JSONResponse(
        status_code=202,
        content={
            "msg": "Cancel inference request accepted.Inference tasks will be terminated gracefully."
        },
    )


# ***************************************************
# Generic Processor component
# ***************************************************
@router.post(
    "/generic-processor/",
    response_model=schemas.GenericProcessorGetResponse,
    tags=["Tasks / Generic processor"],
    status_code=201,
)
async def create_generic_processor(
    request: Request,
    generic_processor: schemas.GenericProcessorCreate,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]

    print("Generic" ,generic_processor)

    created_generic_processor = generic_processor_crud.create(
        db, item=generic_processor, user=user
    )

    return created_generic_processor


@router.get(
    "/generic-processor/{generic_processor_id}",
    response_model=schemas.GenericProcessorGetResponse,
    tags=["Tasks / Generic processor"],
)
async def retrieve_generic_processor(
    generic_processor_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = generic_processor_crud.get_by_id(db, generic_processor_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Generic Processor component not found")

    return item


@router.get(
    "/generic-processor",
    response_model=schemas.GenericProcessorListResponse,
    tags=["Tasks / Generic processor"],
)
async def list_generic_processors(
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]

    count, items = generic_processor_crud.get_all(
        db=db,
        user=user,
        total_count=True,
    )

    if not items:
        raise HTTPException(status_code=404, detail="Generic Processor components not found")

    return {"results": items, "total_records": count}


@router.delete(
    "/generic-processor/{generic_processor_id}",
    tags=["Tasks / Generic processor"],
    status_code=204,
)
async def delete_generic_processor(
    generic_processor_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    item = generic_processor_crud.get_by_id(db, generic_processor_id, user=user)
    if not item:
        raise HTTPException(
            status_code=404, detail="Generic Processor component not found"
        )

    generic_processor_crud.soft_delete(db, item_id=generic_processor_id, user=user)
    return item


# ***************************************************
# Task
# ***************************************************
@router.get("/tasks/{task_id}/output", tags=["Inference / Inference"])
async def get_tasks_output_url(
    task_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
    cos_client=Depends(get_cos_client),
):
    user = auth[0]
    task = task_crud.get_all(db, filters={"task_id": task_id}, user=user)
    if not task:
        raise HTTPException(status_code=404, detail={"msg": "Task not found"})

    if task[0].status not in ["FINISHED"]:
        raise HTTPException(
            status_code=409,
            detail={"msg": "Task is not completed. Output URL is not available."},
        )

    inference_id = task_id.split("-task")[0]
    object_key = f"{inference_id}/{task_id}/archive.zip"
    try:
        url = cos_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": pipelines_bucket_name, "Key": object_key},
            ExpiresIn=302400,
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "task_id": task_id,
        "output_url": url,
    }


@router.get("/tasks/{task_id}/logs/{step_id}", tags=["Inference / Inference"])
async def get_task_step_logs(
    task_id: str,
    step_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
    cos_client=Depends(get_cos_client),
):
    user = auth[0]
    task = task_crud.get_all(db, filters={"task_id": task_id}, user=user)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    filtered_step = [
        step for step in task[0].pipeline_steps if step["process_id"] == step_id
    ]
    if not filtered_step:
        raise HTTPException(
            status_code=404, detail={"msg": f"Step {step_id} not found in task"}
        )

    if filtered_step[0].get("status") not in ["FINISHED", "FAILED", "ERROR"]:
        raise HTTPException(
            status_code=409,
            detail={"msg": f"Step {step_id} is not completed. Logs are not available."},
        )

    inference_id = task_id.split("-task")[0]
    object_key = f"{inference_id}/{task_id}/{task_id}-{step_id}-stdout.log"
    try:
        url = cos_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": pipelines_bucket_name, "Key": object_key},
            ExpiresIn=302400,
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"task_id": task_id, "step_id": step_id, "step_log_url": url}


# ***************************************************
# DataSource
# ***************************************************
@router.get(
    "/data-sources",
    response_model=schemas.DataSourceListResponse,
    tags=["Inference / Data Sources"],
)
async def list_data_sources(
    connector: Optional[str] = Query(
        None, description="Filter by data connector type."
    ),
    collection: Optional[str] = Query(None, description="Filter by collection."),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    auth=Depends(auth_handler),
):
    data_sources = list_data_connectors(as_json=True)
    if connector:
        data_sources = [item for item in data_sources if item["connector"] == connector]
    if collection:
        data_sources = [
            item for item in data_sources if item["collection_name"] == collection
        ]
    data_sources = data_sources[skip : skip + limit]

    return {"results": data_sources}


# ***************************************************
# Notification
# ***************************************************
@router.get(
    "/notifications/{event_id}",
    response_model=schemas.NotificationListResponse,
    tags=["Studio / Notifications"],
)
async def list_webhooks(
    event_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    source: Optional[str] = Query(
        None, description="The service that generated the webhook event."
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    auth=Depends(auth_handler),
):
    user = auth[0]
    qp_filters = {"event_id": event_id}
    if source:
        qp_filters["source"] = source

    items = notification_crud.get_all(
        db=db, filters=qp_filters, limit=limit, skip=skip, user=user
    )
    return {"results": items}


async def handle_inference_webhooks(
    event: schemas.NotificationCreate, db: Session, user: str, inference_obj
):
    webhook_obj = None
    current_status = inference_obj.status
    if event.detail_type == EventDetailType.LAYER_READY:
        # only update inference with layers and send webhook to UI
        await update_inference_webhook_events(
            db=db,
            user=user,
            inference_id=inference_obj.id,
            event_detail=event.detail,
        )

    elif event.detail_type == EventDetailType.TASK_PENDING_GEOSERVER:
        try:
            transition_to(current_status, InferenceStatus.PUBLISHING_RESULTS)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        await update_inference_webhook_events(
            db=db,
            user=user,
            inference_id=inference_obj.id,
            event_detail=event.detail,
        )

    elif event.detail_type == EventDetailType.MSG_NOTIFY:
        warning_message = ""
        if "error_code" in event.detail:
            code = event.detail["error_code"]
            warning_message = errors.get(code, {}).get("uiMessage")

        await notify_inference_webhook_events(
            channel=f"geoinf:event:{event.event_id}",
            message=warning_message if warning_message else event.detail,
            detail_type=EventDetailType.MSG_NOTIFY,
            status=current_status,
        )

    elif event.detail_type in [EventDetailType.TASK_FAILED, EventDetailType.TASK_ERROR]:
        # only send webhook to UI
        message = "An error occured when running inference."
        if "error_code" in event.detail:
            code = event.detail["error_code"]
            message = errors.get(code, {}).get("uiMessage") or message
        elif (
            "show_service_error" in event.detail
            and event.detail["show_service_error"] == "show"
        ):
            message = event.detail["error"]

        await notify_inference_webhook_events(
            channel=f"geoinf:event:{event.event_id}",
            message={"error": message},
            detail_type=EventDetailType.TASK_ERROR,
            status=current_status,
        )

    elif event.detail_type == EventDetailType.TASK_MODELS_CLEANUP:
        await cleanup_autodeployed_model_resources(
            db=db,
            user=user,
            model_id=inference_obj.model_usecase.id,
            model_name=inference_obj.model_usecase.name,
        )

    elif event.detail_type == EventDetailType.TASK_LAYERS_UPDATED:
        try:
            transition_to(current_status, InferenceStatus.COMPLETED)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        await update_inference_webhook_events(
            db=db,
            user=user,
            inference_id=inference_obj.id,
            event_detail=event.detail,
        )

    elif event.detail_type == EventDetailType.TASK_UPDATED:
        # reconcile status of inference. and update ui
        updated_item = await reconcile_inference_status(
            db=db, inference_id=inference_obj.id, user=user
        )
        await send_websocket_notification(
            updated_item=updated_item,
            channel=f"geoinf:event:{event.event_id}",
            detail_type=event.detail_type,
            status=updated_item.status,
        )

    return webhook_obj.id if webhook_obj else webhook_obj


@router.post(
    "/notifications",
    response_model=schemas.NotificationGetResponse,
    status_code=201,
    tags=["Studio / Notifications"],
)
async def receive_webhook(
    item: schemas.NotificationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    user = auth[0]
    # Do not save events with empty details:
    if not item.detail:
        raise HTTPException(
            status_code=412, detail="Event with empty details not processed."
        )

    # Check if the event is a completed task response and update inference.
    if "inference" in item.source.lower():
        # Check inference associated with the event_id
        inferences = inference_crud.get_all(
            db=db,
            filters={"id": str(item.event_id)},
            user=user,
            ignore_user_check=True,
        )

        if not inferences:
            raise HTTPException(
                status_code=404,
                detail=f"Inference with event>({item.event_id}) Not Found.",
            )

        user = inferences[0].created_by or user
        item.inference_id = item.event_id

        await handle_inference_webhooks(
            event=item, inference_obj=inferences[0], db=db, user=user
        )
    elif "ftuning" in item.source.lower():
        invoke_fine_tuning_webhook_handlers.delay(
            event=item.model_dump(),
            user=user,
        )
    elif "data" in item.source.lower():
        invoke_dataset_webhooks_handlers.delay(
            event=item.model_dump(),
            user=user,
        )

    webhook_obj = notification_crud.create(db, item, user=user)

    return {
        "event_id": item.event_id,
        "id": webhook_obj.id,
    }


async def reconcile_inference_status(db, inference_id, user, event_status=None):
    inference_tasks = task_crud.get_all(
        db=db, user=user, filters={"inference_id": inference_id}
    )

    planning_tasks_status = None
    planning_tasks = [task for task in inference_tasks if "planning" in task.task_id]
    if len(planning_tasks) > 0:
        planning_tasks_status = planning_tasks[0].status.upper()

    total_non_planning_tasks = len(inference_tasks) - len(planning_tasks)

    final_status = None

    logger.debug(f"Total tasks: {len(inference_tasks)}")
    logger.debug(f"Total planning tasks: {len(planning_tasks)}")
    logger.debug(f"Total non planning tasks: {total_non_planning_tasks}")
    logger.debug(f"Planning task status: {planning_tasks_status}")

    # If inference has no tasks, nothing to reconcile; planning ongoing
    if total_non_planning_tasks == 0 and planning_tasks_status in ["READY", "RUNNING"]:
        return
    elif total_non_planning_tasks == 0 and planning_tasks_status == "FAILED":
        final_status = InferenceStatus.FAILED
    elif total_non_planning_tasks == 0 and planning_tasks_status == "STOPPED":
        final_status = InferenceStatus.STOPPED
    else:
        task_status_counts = {
            status: sum(
                1
                for task in inference_tasks
                if (task.status.upper() == status and "planning" not in task.task_id)
            )
            for status in ["FINISHED", "FAILED", "READY", "RUNNING", "STOPPED"]
        }
        logger.debug(f"Here is the task dictionary: {task_status_counts}")
        if task_status_counts["FINISHED"] == total_non_planning_tasks:
            final_status = InferenceStatus.COMPLETED
        elif task_status_counts["FAILED"] == total_non_planning_tasks:
            final_status = InferenceStatus.FAILED
        elif (
            task_status_counts["STOPPED"] > 0
            and (task_status_counts["FAILED"] + task_status_counts["FINISHED"]) >= 0
            and (task_status_counts["READY"] + task_status_counts["RUNNING"]) == 0
        ):
            final_status = InferenceStatus.STOPPED
        elif (
            task_status_counts["FINISHED"] > 0
            and task_status_counts["FAILED"] > 0
            and (task_status_counts["READY"] + task_status_counts["RUNNING"]) == 0
        ):
            final_status = InferenceStatus.COMPLETED_WITH_ERRORS
        elif task_status_counts["FINISHED"] > 0:
            final_status = InferenceStatus.PARTIALLY_COMPLETED
        elif task_status_counts["RUNNING"] > 0:
            final_status = InferenceStatus.RUNNING
        elif task_status_counts["READY"] > 0:
            final_status = InferenceStatus.RUNNING

    inference = inference_crud.get_by_id(db, inference_id, user=user)

    try:
        updated_item = None
        if final_status and inference.status != final_status:
            new_status = transition_to(inference.status, final_status)
            updated_item = await update_inference_status(
                inference_id=inference_id,
                status=new_status,
                event_error=None,
                user=user,
                db=db,
            )
        else:
            updated_item = inference
        return updated_item
    except ValueError:
        msg = f"INVALID TRANSITION cannot transition from {inference.status} to {final_status}"
        logger.error("\n\nIssue encountered when performing inference state transition")
        logger.exception(f"❌ {msg}")


# ##############################################
# ###---- Files-related endpoints
# ##############################################
@router.get(
    "/file-share", tags=["Studio / Files"], response_model=schemas.FilesShareOut
)
async def get_fileshare_presigned_urls(
    object_name: str = Query(
        description="Object name/id of the uploaded file",
        max_length=60,
        min_length=6,
        pattern=r"^[a-zA-Z0-9-_]+.[a-z]+$",
    ),
    auth=Depends(auth_handler),
):
    """Generate presigned urls for sharing files i.e uploading and downloading files."""
    upload_url = None
    download_url = None
    current_date = str(datetime.datetime.now(datetime.timezone.utc).date())
    object_key = f"{current_date}/{object_name}"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.OBJECT_STORAGE_KEY_ID,
        aws_secret_access_key=settings.OBJECT_STORAGE_SEC_KEY,
        endpoint_url=settings.OBJECT_STORAGE_ENDPOINT,
        config=Config(signature_version=settings.OBJECT_STORAGE_SIGNATURE_VERSION),
        verify=(settings.ENVIRONMENT.lower() != "local"),
    )
    try:
        upload_url = generate_upload_presigned_url(
            s3=s3,
            object_key=object_key,
            bucket_name=settings.TEMP_UPLOADS_BUCKET,
            expiration=28800,
        )
    except Exception:
        logger.exception("An error occured when generating upload presigned-url.")
        raise HTTPException(
            status_code=500, detail={"message": "Failed creating upload presigned url."}
        )

    try:
        download_url = generate_download_presigned_url(
            s3=s3,
            object_key=object_key,
            bucket_name=settings.TEMP_UPLOADS_BUCKET,
            expiration=28800,
        )
    except Exception:
        logger.exception("An error occured when generating download presigned-url.")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed creating download presigned url."},
        )

    return {
        "upload_url": upload_url,
        "download_url": download_url,
        "message": "Use upload_url to add data to a temporary storage. download_url to share data after upload.",
    }


# ***************************************************
# SSE APIs
# ***************************************************
@router.get(
    "/async/notify/{event_id}",
    tags=["Inference / Inference"],
    include_in_schema=False,
)
async def stream_server_sent_events(
    event_id: uuid.UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """Endpoint to stream server sent event notifications."""
    user = auth[0]
    inferences = inference_crud.get_all(
        db=db,
        filters={"id": str(event_id)},
        user=user,
    )
    if not inferences:
        raise HTTPException(
            status_code=404, detail=f"Inference with EventId ({event_id}) Not Found."
        )

    event_generator = await redis_handler.subscribe_to_channel(
        channel=f"geoinf:event:{event_id}"
    )

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )
