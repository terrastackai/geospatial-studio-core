# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import base64
import copy
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import jsonschema
import yaml
from asyncer import asyncify
from botocore.exceptions import ClientError
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import and_, literal_column
from sqlalchemy.orm import Session

from gfmstudio.auth import authorizer
from gfmstudio.auth.authorizer import auth_handler
from gfmstudio.celery_worker import deploy_hpo_tuning_celery_task, invoke_tune_upload
from gfmstudio.common.api import crud, utils
from gfmstudio.config import BASE_DIR, settings
from gfmstudio.fine_tuning import schemas
from gfmstudio.fine_tuning.core import object_storage
from gfmstudio.fine_tuning.core.iterate_utils import update_terratorch_iterate_config
from gfmstudio.fine_tuning.core.mlflow_logs import get_mlflow_metrics
from gfmstudio.fine_tuning.core.tuning_config_utils import (
    get_dataset_params,
    merge_nested_dicts,
)
from gfmstudio.fine_tuning.dataset_schemas import (
    DatasetsSummaryResponseSchema,
    GeoDatasetMetadataUpdateSchema,
    GeoDatasetPreScanRequestSchemaV2,
    GeoDatasetRequestSchemaV2,
    GeoDatasetsResponseSchemaV2,
)
from gfmstudio.fine_tuning.models import BaseModels, GeoDataset, Tunes, TuneTemplate
from gfmstudio.fine_tuning.utils import tune_handlers
from gfmstudio.fine_tuning.utils.dataset_handlers import (
    data_and_label_match,
    extract_bands_from,
    list_zipped_files,
    make_k8s_secret_literal,
    obtain_band_descriptions_from,
    validate_and_transform_data_sources,
    validate_and_transform_options,
)
from gfmstudio.fine_tuning.utils.tune_handlers import get_rendered_tuning_template
from gfmstudio.inference.v2.models import Inference
from gfmstudio.inference.v2.models import Model as InferenceModel
from gfmstudio.inference.v2.schemas import InferenceCreateInput, InferenceGetResponse
from gfmstudio.log import logger, loglevel_name
from gfmstudio.fine_tuning.utils.dataset_handlers import capture_and_upload_job_log
from gfmstudio.fine_tuning.core.kubernetes import collect_pod_logs
from gfmstudio.fine_tuning.utils.webhook_event_handlers import upload_logs_cos
from datetime import datetime


BASEDIR = os.path.dirname(os.path.abspath(__file__))
pipeline_api_url = (
    settings.DATA_PIPELINE_BASE_URL
    if "dataset-factory-api" in os.getenv("HOSTNAME", "")
    else settings.DATA_PIPELINE_BASE_URL.replace("-internal-", "-")
)


def is_base64_encoded(data: str) -> bool:
    try:
        # Check if the string length is a multiple of 4 (Base64 requirement)
        if len(data) % 4 == 0:
            # Try decoding the string using base64
            base64.b64decode(data, validate=True)
            return True  # It's valid base64-encoded data
        return False
    except (base64.binascii.Error, ValueError):
        return False  # Invalid base64-encoded data


def parse_hpo_form(tune_metadata: str = Form(...)) -> schemas.TuneSubmitHPO:
    try:
        return schemas.TuneSubmitHPO(**json.loads(tune_metadata))
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=str(e))


app = APIRouter(dependencies=[Depends(authorizer.auth_handler)])

tunes_crud = crud.ItemCrud(model=Tunes)
tunes_models_crud = crud.ItemCrud(model=Tunes)
bases_crud = crud.ItemCrud(model=BaseModels)
tune_template_crud = crud.ItemCrud(model=TuneTemplate)
dataset_crud = crud.ItemCrud(model=GeoDataset)
inference_crud = crud.ItemCrud(model=Inference)
inference_model_crud = crud.ItemCrud(model=InferenceModel)

"""
Create file folders if they don't exist already
"""
if settings.ENVIRONMENT.lower() == "local":
    tasks_dir = os.path.join(settings.TUNE_BASEDIR, "tune-tasks")
    if os.path.isdir(tasks_dir) is False:
        logger.info(f"Creating tasks directory: {tasks_dir}")
        os.makedirs(tasks_dir)

    trained_dir = os.path.join(settings.TUNE_BASEDIR, "pre-trained")
    if os.path.isdir(trained_dir) is False:
        logger.info(f"Creating pre-trained directory: {trained_dir}")
        os.makedirs(trained_dir)


###############################################
# ---- Get token endpoint
###############################################
def grab_urls(s3, bucket_name, tune_id, config_file_ext="yaml"):
    """Function to grab checkpoint and config urls

    Parameters
    ----------
    s3 : str
        S3 client
    bucket_name : str
        bucket_name
    tune_id : str
        Unique tune id
    config_file_ext : str, optional
        config file extension, by default "yaml"

    Returns
    -------
    tuple   (config_url, checkpoint_url)
        Presigned urls for the checkpoint and config
    """
    logger.info("🪣 %s %s %s", s3, bucket_name, tune_id)

    contents = object_storage.detailed_prefix_list(
        s3=s3, bucket=bucket_name, tunes_path="tune-tasks", tune_id=tune_id
    )
    # logger.info(f"{tune_id} has this files ({len(contents)}): {contents}")
    config_and_checkpoint = object_storage.find_config_and_checkpoint(
        contents=contents,
    )
    logger.info(f"Config and checkpoint {config_and_checkpoint}")

    return config_and_checkpoint.signed_urls(
        client=s3, bucket=bucket_name, expires_in=172800
    )


def grab_tune_file_presigned_url(
    bucket_name: str,
    file_key: str,
    s3: object_storage.object_storage_client,
    file_type: str = None,
):
    """Function to grab pre-signed urls for tune failed logs

    Parameters
    ----------
    bucket_name : str
        bucket name
    file_key : str
        file path to the logs in COS
    s3 : object_storage.object_storage_client
        S3 client

    Returns
    -------
    str
        presigned logs url
    """
    try:
        if object_storage.check_s3_file_exists(
            s3=s3, bucket_name=bucket_name, file_key=file_key
        ):
            # create the presigned url for log file
            return object_storage.generate_presigned_url(
                s3=s3, bucket_name=bucket_name, file_key=file_key, expiration=3600
            )
        else:
            logger.info(
                f"File {file_key} for the  {file_type} does not exist in bucket {bucket_name}."
            )
            return None

    except Exception:
        logger.exception(
            f"Something went wrong with grabbing presigned url for the {file_type} {file_key}"
        )


def create_model_if_not_exist(db_session, user: str, name: str = "sandbox-base-model"):
    # Look up a sandbox model or create one if it doesn't exist
    base_models = bases_crud.get_all(
        db=db_session, filters={"name": name}, ignore_user_check=True
    )
    logger.debug(f"Available base models:{base_models}")

    if not base_models:
        data = schemas.BaseModelsIn(
            **{
                "name": "sandbox-base-model",
                "description": "base model",
            }
        )
        created_model = bases_crud.create(db=db_session, item=data, user=user)

    base_model_id = base_models[0].id if base_models else created_model.id
    return base_model_id


def create_dataset_entry_if_not_exist(
    db_session, user: str, name: str = "sandbox-dataset"
):
    # Look up a sandbox dataset or create one if it doesn't exist
    datasets = dataset_crud.get_all(
        db=db_session,
        filters={"dataset_name": name},
        ignore_user_check=True,
    )
    if not datasets:
        data = GeoDatasetRequestSchemaV2(
            **{
                "dataset_name": "sandbox-dataset",
                "label_suffix": ".mask.tif",
                "dataset_url": "https://example.com",
                "purpose": "Segmentation",
                "description": "sandbox dataset",
                "data_sources": [],
            }
        )

        created_dataset = dataset_crud.create(db=db_session, user=user, item=data)
    dataset_id = datasets[0].id if datasets else created_dataset.id
    return dataset_id


###############################################
# ---- Tune-related endpoints
###############################################


@app.get(
    "/tunes-and-models",
    tags=["FineTuning / Tunes"],
    response_model=schemas.TunesAndInferenceModels,
)
async def list_tunes_and_models(
    auth=Depends(auth_handler),
    type: Literal["Tune", "Model"] = Query(
        None, description="Filter by Tune or Model."
    ),
    shared: Optional[bool] = Query(None, description="Filter by shared tunes/models."),
    name: Optional[str] = Query(
        None, description="Filter by the name of the tune/model."
    ),
    status: Optional[list] = Query(
        None, description="Filter by more than one status of the tune/model in a list."
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to list fine tuning jobs

    \f
    Parameters
    ----------
    request : Request
        Request
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)
    type : Optional[str], optional
        Whether to return Tunes or Models or both
    name : Optional[str], optional
        Name of tune/model to filter by, by default Query(None, description="Filter by the name of the tune/model.")
    status : Optional[str], optional
        Status of tunes/models to filter by, by default
        Query(None, description="Filter by the status of the tune/model.")
    limit : Optional[int], optional
        Max no of items to retrieve, by default Query(25, description="The maximum number of items to retrieve.")
    skip : Optional[int], optional
        Number of items to skip, by default Query(0, description="The number of items to skip.")
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TunesAndInferenceModels
        Dictionary of tunes/models found
    """
    user = auth[0]
    qp_filters = {}
    search_filters = {}
    ignore_user_check = False
    filter_expr = None
    if name:
        search_filters["name"] = name
        search_filters["display_name"] = name
    if shared is not None:
        # TODO: Add logic to mark tunes as sharable and update this filter
        # we currently assume that for a tune to be shared it was
        # created_by the system user.
        qp_filters["created_by"] = settings.DEFAULT_SYSTEM_USER if shared else user
        ignore_user_check = True

    UNION_FIELD_NAMES = list(schemas.TuneAndInferenceModel.model_fields.keys())

    union_config = []
    tune_union_config = {
        "model": Tunes,
        "filter_expr": Tunes.status.in_(status) if status else None,
        "columns": [
            Tunes.id.label(UNION_FIELD_NAMES[0]),
            Tunes.name.label(UNION_FIELD_NAMES[1]),
            Tunes.status.label(UNION_FIELD_NAMES[2]),
            Tunes.description.label(UNION_FIELD_NAMES[4]),
            literal_column("'Tune'").label(UNION_FIELD_NAMES[5]),
            Tunes.active.label("active_"),
            Tunes.created_by.label("created_by_"),
            Tunes.created_at.label("created_at_"),
            Tunes.updated_at.label("updated_at_"),
        ],
    }
    model_union_config = {
        "model": InferenceModel,
        "filter_expr": InferenceModel.status.in_(status) if status else None,
        "columns": [
            InferenceModel.internal_name.label(UNION_FIELD_NAMES[0]),
            InferenceModel.display_name.label(UNION_FIELD_NAMES[1]),
            InferenceModel.status.label(UNION_FIELD_NAMES[2]),
            InferenceModel.description.label(UNION_FIELD_NAMES[4]),
            literal_column("'Model'").label(UNION_FIELD_NAMES[5]),
            InferenceModel.active.label("active_"),
            InferenceModel.created_by.label("created_by_"),
            InferenceModel.created_at.label("created_at_"),
            InferenceModel.updated_at.label("updated_at_"),
        ],
    }

    if type == "Tune":
        union_config = [tune_union_config]
    elif type == "Model":
        union_config = [model_union_config]
    else:
        union_config = [tune_union_config, model_union_config]

    count, items = tunes_models_crud.union_all(
        union_config=union_config,
        db=db,
        filters=qp_filters,
        filter_expr=filter_expr,
        search=search_filters,
        limit=limit,
        skip=skip,
        user=user,
        ignore_user_check=ignore_user_check,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@app.get("/tunes", tags=["FineTuning / Tunes"], response_model=schemas.TunesOut)
async def list_tunes(
    auth=Depends(auth_handler),
    shared: Optional[bool] = Query(None, description="Filter by shared tunes."),
    name: Optional[str] = Query(None, description="Filter by the name of the tune."),
    status: Optional[str] = Query(
        None, description="Filter by the status of the tune."
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to list fine tuning jobs

    \f
    Parameters
    ----------
    request : Request
        Request
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)
    name : Optional[str], optional
        Name of tune to filter by, by default Query(None, description="Filter by the name of the tune.")
    status : Optional[str], optional
        Status of tunes to filter by, by default Query(None, description="Filter by the status of the tune.")
    limit : Optional[int], optional
        Max no of items to retrieve, by default Query(25, description="The maximum number of items to retrieve.")
    skip : Optional[int], optional
        Number of items to skip, by default Query(0, description="The number of items to skip.")
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TunesOut
        Dictionary of tunes found
    """
    user = auth[0]
    qp_filters = {}
    search_filters = {}
    ignore_user_check = False
    if name:
        search_filters["name"] = name
    if status:
        qp_filters["status"] = status
    if shared is not None:
        # TODO: Add logic to mark tunes as sharable and update this filter
        # we currently assume that for a tune to be shared it was
        # created_by the system user.
        qp_filters["created_by"] = settings.DEFAULT_SYSTEM_USER if shared else user
        ignore_user_check = True

    count, items = tunes_crud.get_all(
        db=db,
        filters=qp_filters,
        search=search_filters,
        limit=limit,
        skip=skip,
        user=user,
        ignore_user_check=ignore_user_check,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@app.get(
    "/tunes/{tune_id}",
    tags=["FineTuning / Tunes"],
    response_model=schemas.TuneStatusOut,
)
async def retrieve_tune(
    tune_id: str,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to retrieve tune by id.
    If the tune's status is Failed, a pre-signed url for the logs is generated.

    \f
    Parameters
    ----------
    tune_id : str
        Unique tune_id
    request : Request
        Request
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TuneStatusOut
        Found tune

    Raises
    ------
    HTTPException
        404: Tune not Found
    """
    user = auth[0]
    item = tunes_crud.get_by_id(db=db, item_id=tune_id, user=user)
    if not item:
        raise HTTPException(status_code=404, detail="Tune not found")
    if item.status != "Failed" and item.status != "Finished":
        logs = await collect_pod_logs(tune_id=tune_id)
        if logs:
            # Push log file to COS
            current_date = datetime.now().strftime("%Y-%m-%d")
            full_s3_log_file_path = f"ftlogs/{current_date}/{tune_id}.log"
            await upload_logs_cos(logs, full_s3_log_file_path)
    try:
        tunes_crud.update(
            db=db,
            item_id=tune_id,
            item={"logs": full_s3_log_file_path},
            protected=False,
        )
    except Exception:
        logger.exception("Tune status was not updated.")
    # create pre-signed url for the logs
    if item.logs or logs:
        s3 = object_storage.object_storage_client()

        try:
            logs_pre_signed_url = grab_tune_file_presigned_url(
                bucket_name=settings.TUNES_FILES_BUCKET,
                file_key=item.logs or logs,
                s3=s3,
                file_type="logs",
            )

            # Since downloading artifacts does not allow the config to be downloaded if tune not successful
            # Add the config to the tune response if failed.

            #  For backward compatibility, allow this to be optional
            tuning_config_presigned_url = ""
            if item.tuning_config:
                tuning_config_presigned_url = grab_tune_file_presigned_url(
                    bucket_name=settings.TUNES_FILES_BUCKET,
                    file_key=item.tuning_config,
                    s3=s3,
                    file_type="tuning config",
                )

            updated_dict = item.__dict__

            updated_dict["logs_presigned_url"] = logs_pre_signed_url
            updated_dict["tuning_config_presigned_url"] = tuning_config_presigned_url

            return updated_dict

        except Exception:
            logger.exception(
                f"{tune_id} Error generating presigned url for {item.logs}"
            )

    return item


@app.patch("/tunes/{tune_id}", tags=["FineTuning / Tunes"])
async def update_tune(
    tune_id: str,
    data: schemas.TuneUpdateIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to update tune in the database

    \f
    Parameters
    ----------
    tune_id : str
        Unique tune id
    data : schemas.TuneUpdateIn
        Schema to validate by
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Dict of updated tune

    Raises
    ------
    HTTPException
        404: Tune not found.
    HTTPException
        500: Could not rename tune.
    """
    user = auth[0]
    tune = tunes_crud.get_by_id(db=db, item_id=tune_id, user=user)
    if not tune:
        raise HTTPException(status_code=404, detail=f"Tune {tune_id} not found")

    if data.train_options:
        train_options = copy.deepcopy(tune.train_options)
        data.train_options = merge_nested_dicts(
            dict1=train_options, dict2=data.train_options
        )

    try:
        tunes_crud.update(db=db, item_id=tune_id, item=data, user=user)
    except:  # noqa: E722
        logger.exception("Update failed for " + tune_id)
        detail = f"Could not update tune: {tune_id}"
        raise HTTPException(status_code=500, detail=detail) from None

    return {"message": "Tune successfully updated."}


@app.delete("/tunes/{tune_id}", tags=["FineTuning / Tunes"], status_code=204)
async def delete_tune(
    tune_id: str,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to delete a tune

    \f
    Parameters
    ----------
    tune_id : str
        Unique tune id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Message of successfully deleted tune
    """
    user = auth[0]
    tunes_crud.soft_delete(db=db, item_id=tune_id, user=user)
    # Add deleting from COS/S3 as well....
    return {"message": f"successfully deleted tune-{tune_id}"}


@app.post(
    "/submit-tune/dry-run",
    tags=["FineTuning / Tunes"],
    status_code=200,
    # response_model=schemas.TuneDryRunOut,
)
async def submit_tune_dry_run(
    data_in: schemas.TuneSubmitIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Validate tune configuration and return rendered YAML without submitting.

    \f
    This endpoint performs all validation steps and generates the tune configuration
    YAML template, but does NOT:
    - Create a tune record in the database
    - Save configuration to storage (local or COS)
    - Submit the tune job for execution

    Use this endpoint to:
    - Preview the generated configuration
    - Validate tune parameters before submission
    - Debug configuration issues
    - Test template rendering

    Parameters
    ----------
    data_in : schemas.TuneSubmitIn
        Tune submission parameters
    auth : Authentication token
    db : Database session

    Returns
    -------
    schemas.TuneDryRunOut
        Validation result with rendered YAML template

    Raises
    ------
    HTTPException
        404: Template or base model not found
        422: Invalid configuration
    """
    user = auth[0]
    logger.info("User (%s) performing dry run for tune: %s", user, data_in.name)

    # Validate tune template
    tune_task = tune_handlers.validate_tune_template(db, data_in.tune_template_id, user)
    tune_type = (
        data_in.train_options.get("tune_type", "") if data_in.train_options else ""
    )

    # Validate base model compatibility (if not user-defined)
    if tune_type != "user-defined":
        tune_handlers.validate_base_model_compatibility(
            db, data_in.base_model_id, tune_task, user
        )

    # Check name uniqueness
    data_in.name = tune_handlers.ensure_unique_tune_name(db, data_in.name)

    # Prepare tune data
    model_params = tune_handlers.normalize_model_parameters(data_in.model_parameters)
    tune_dataset = await tune_handlers.prepare_tune_dataset(
        db, data_in.dataset_id, data_in.train_options, user
    )

    # Generate a temporary tune_id for template rendering
    temp_tune_id = f"dry-run-{data_in.name.lower()}"

    # Generate tune configuration (without creating database record)
    rendered_template = await tune_handlers.generate_tune_template(
        db=db,
        user=user,
        tune_id=temp_tune_id,
        data_in=data_in,
        model_params=model_params,
        tune_dataset=tune_dataset,
        tune_type=tune_type,
    )

    # Validate runtime image is configured
    # Create a mock tune object for runtime image validation
    mock_tune = type(
        "obj",
        (object,),
        {
            "tune_template": type(
                "obj", (object,), {"extra_info": tune_task.extra_info or {}}
            )()
        },
    )()

    tune_handlers.get_runtime_image(data_in, mock_tune)

    return Response(content=rendered_template, media_type="text/plain")


@app.post(
    "/submit-tune",
    tags=["FineTuning / Tunes"],
    status_code=201,
    response_model=schemas.TuneSubmitOut,
)
async def submit_tune(
    data_in: schemas.TuneSubmitIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Submit a tuning task.

    \f

    Parameters
    ----------
    data_in : schemas.TuneSubmitIn
        Tune submission parameters
    dry_run : bool, optional
        If True, returns rendered template without submitting
    auth : Authentication token
    db : Database session

    Returns
    -------
    schemas.TuneSubmitOut
        Submission result with tune_id and status

    Raises
    ------
    HTTPException
        404: Template or base model not found
        422: Invalid configuration or missing runtime image
        500: COS upload failure
    """
    user = auth[0]
    logger.info("User (%s) submitting tuning task", user)

    # Validate and prepare tune configuration
    tune_task = tune_handlers.validate_tune_template(db, data_in.tune_template_id, user)
    tune_type = (
        data_in.train_options.get("tune_type", "") if data_in.train_options else ""
    )

    if tune_type != "user-defined":
        tune_handlers.validate_base_model_compatibility(
            db, data_in.base_model_id, tune_task, user
        )

    # Ensure unique tune name
    data_in.name = tune_handlers.ensure_unique_tune_name(db, data_in.name)

    # Prepare tune data
    model_params = tune_handlers.normalize_model_parameters(data_in.model_parameters)
    tune_dataset = await tune_handlers.prepare_tune_dataset(
        db, data_in.dataset_id, data_in.train_options, user
    )

    # Create tune record
    created_tune = tunes_crud.create(db=db, item=data_in, user=user)
    tune_id = created_tune.id

    # Generate tune configuration
    rendered_template = await tune_handlers.generate_tune_template(
        db=db,
        user=user,
        tune_id=tune_id,
        data_in=data_in,
        model_params=model_params,
        tune_dataset=tune_dataset,
        tune_type=tune_type,
    )

    # Save configuration to storage
    bucket_key = f"tune-tasks/{tune_id}/{tune_id}_config.yaml"
    config_path = await tune_handlers.save_tune_config(
        tune_id, rendered_template, bucket_key
    )

    # Submit tune job
    tune_runtime_image = tune_handlers.get_runtime_image(data_in, created_tune)
    ftune_job_id, status, detail = await tune_handlers.submit_tune_job(
        tune_id=tune_id,
        config_path=config_path,
        runtime_image=tune_runtime_image,
    )

    # Update tune with job details
    if ftune_job_id:
        tunes_crud.update(
            db=db,
            item_id=tune_id,
            item={
                "mcad_id": ftune_job_id,
                "tuning_config": bucket_key,
                "status": status,
            },
            user=user,
        )

    return {
        "tune_id": tune_id,
        "mcad_id": ftune_job_id,
        "status": status,
        "message": detail,
    }


@app.post(
    "/upload-completed-tunes",
    tags=["FineTuning / Tunes"],
    status_code=201,
)
async def upload_tune(
    data_in: schemas.UploadTuneInput,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    user = auth[0]
    logger.debug(f"User {user} submitting tuning task: \n {data_in.model_dump()}")

    base_model_id = create_model_if_not_exist(db_session=db, user=user)
    dataset_id = create_dataset_entry_if_not_exist(db_session=db, user=user)

    # Lookup a default tune-template
    tune_template = tune_template_crud.get_all(
        db=db,
        user=user,
        filters={"purpose": str(schemas.TaskPurposeEnum.SEGMENTATION)},
        ignore_user_check=True,
    )

    if not tune_template:
        raise HTTPException(
            status_code=404,
            detail={
                "msg": "Missing a default tune template. Please create one to proceed"
            },
        )

    # curate train_options
    train_options = {}
    if data_in.geoserver_push:
        train_options["geoserver_push"] = data_in.geoserver_push
    if data_in.model_input_data_spec:
        train_options["model_input_data_spec"] = data_in.model_input_data_spec
    if data_in.data_connector_config:
        train_options["data_connector_config"] = data_in.data_connector_config
    if data_in.post_processing:
        train_options["post_processing"] = data_in.post_processing

    # Create a Tunes model instance
    status = "Pending"
    tune_instance = schemas.TuneSubmitIn(
        **{
            "name": data_in.name,
            "description": data_in.description,
            "status": status,
            "dataset_id": dataset_id,
            "base_model_id": base_model_id,
            "tune_template_id": tune_template[0].id,
            "train_options": train_options,
        }
    )

    created_tune = tunes_crud.create(
        db=db,
        item=tune_instance,
        user=user,
    )
    tune_id = created_tune.id
    logger.debug(f"Created tuning record with ID: {tune_id}")

    logger.info(f"{tune_id}: Submitted async tune upload task")
    invoke_tune_upload.delay(
        tune_config_url=data_in.tune_config_url,
        tune_checkpoint_url=data_in.tune_checkpoint_url,
        tune_id=str(tune_id),
        user=user,
    )
    return {"message": "Upload started", "tune_id": tune_id}


@app.post(
    "/submit-hpo-tune",
    tags=["FineTuning / Tunes"],
    status_code=201,
)
async def submit_hpo_tune_yaml(
    tune_metadata: str = Depends(parse_hpo_form),
    config_file: UploadFile = File(...),
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Submit HPO tuning job using YAML configuration file."""
    user = auth[0]
    ALLOWED_CONFIG_FILE_EXTENSIONS = {".yaml", ".yml"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # Dataset exists?
    tune_dataset = dataset_crud.get_by_id(
        db=db, item_id=tune_metadata.dataset_id, user=user
    )
    if not tune_dataset:
        raise HTTPException(
            status_code=404,
            detail={
                "msg": f"Dataset {tune_metadata.dataset_id} not found. Ensure dataset is onboarded and retry."
            },
        )

    # Validate file type
    file_extension = Path(config_file.filename).suffix.lower()
    if file_extension not in ALLOWED_CONFIG_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only YAML config files (.yaml, .yml) are allowed",
        )

    config_content = await config_file.read()

    # Validate file size
    if len(config_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes",
        )

    # Validate YAML syntax
    try:
        yaml.safe_load(config_content.decode("utf-8"))
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")

    # Add a tune-template entry:
    template_name = f"{tune_metadata.name}-generated-template"
    tune_templates = tune_template_crud.get_all(
        db=db, filters={"name": template_name}, user=user
    )
    if not tune_templates:
        tune_template_data = schemas.TaskIn(
            name=template_name,
            description=tune_metadata.description,
            content=base64.b64encode(config_content),
        )
        created_template = tune_template_crud.create(
            db=db, item=tune_template_data, user=user
        )
    else:
        created_template = tune_templates[0]

    try:
        # Create entry in tunes db
        base_model_id = create_model_if_not_exist(db_session=db, user=user)
        tune_payload = tune_metadata.model_dump()
        tune_payload.update(
            {
                "base_model_id": base_model_id,
                "dataset_id": tune_metadata.dataset_id,
                "tune_template_id": created_template.id,
            }
        )
        created_tune = tunes_crud.create(
            db=db,
            item=schemas.TuneSubmitIn(**tune_payload),
            user=user,
        )
        tune_id = created_tune.id

        logger.info(
            f"YAML configuration validated successfully for hpo-tune: {tune_id}"
        )
        # Prepare job data for Celery task
        bucket_key = f"tune-tasks/{tune_id}/{tune_id}_config.yaml"
        tune_dir = os.path.join(settings.TUNE_BASEDIR, f"tune-tasks/{tune_id}")
        if os.path.isdir(tune_dir) is False:
            os.mkdir(tune_dir)

        bucket_dir = os.path.join(settings.TUNE_BASEDIR, bucket_key)
        config_data = yaml.load(config_content, Loader=yaml.SafeLoader)
        config_data = update_terratorch_iterate_config(
            config=config_data,
            experiment_name=tune_id,
            mlflow_url=settings.MLFLOW_URL,
            artifact_dir=settings.FILES_MOUNT,
        )
        with open(bucket_dir, "w", encoding="utf-8") as config_file:
            yaml.dump(
                config_data, config_file, default_flow_style=False, sort_keys=False
            )

        # Submit Celery task
        deploy_hpo_tuning_celery_task.apply_async(
            kwargs={
                "ftune_id": tune_id,
                "ftune_config_file": bucket_dir,
                "ftuning_runtime_image": settings.FT_HPO_IMAGE,
                "tune_type": schemas.TuneOptionEnum.K8_JOB,
            },
            task_id=tune_id,
        )

        logger.info(f"Celery task submitted for hpo-tune: {tune_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing YAML submission.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return {
        "tune_id": tune_id,
        "job_id": tune_id,
        "status": created_tune.status,
        "message": "HPO tune submitted.",
    }


@app.post(
    "/tunes/{tune_id}/try-out",
    tags=["FineTuning / Tunes"],
    response_model=InferenceGetResponse,
)
async def try_tuned_model(
    request: Request,
    tune_id: str,
    inference_data: schemas.TryOutTuneInput,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to try-out inference on a tune without deploying the model.

    \f
    Parameters
    ----------
    request : Request
        Request
    tune_id : str
        Unique tune id
    data_in : schemas.TunedModelDeployIn
        schemas.TunedModelDeployIn
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Dict with deployed model details

    Raises
    ------
    HTTPException
        404: Tune not found
    HTTPException
        500: Errors getting the model artifacts presinged urls
    HTTPException
        400: No checkpoint file or size too small
    """
    user, token, _ = auth
    tune_meta = tunes_crud.get_by_id(db=db, item_id=tune_id, user=user)
    if not tune_meta:
        raise HTTPException(status_code=404, detail="Tune not found")

    tune_train_options = tune_meta.train_options
    data_connector_config = (
        inference_data.data_connector_config
        or tune_train_options.get("data_connector_config")
    )
    model_input_data_spec = (
        inference_data.model_input_data_spec
        or tune_train_options.get("model_input_data_spec")
    )
    geoserver_push = inference_data.geoserver_push or tune_train_options.get(
        "geoserver_push"
    )
    post_processing = inference_data.post_processing or tune_train_options.get(
        "post_processing"
    )
    inferencing = getattr(
        inference_data, "inferencing", None
    ) or tune_train_options.get("inferencing")
    # TODO; Refactor try-out-models i.e Add shared helper methods for validating and
    # creating inferences.
    sandbox_models = inference_model_crud.get_all(
        db=db, filters={"display_name": "geofm-sandbox-models"}
    )
    if sandbox_models:
        sandbox_model = sandbox_models[0]
        geoserver_push = geoserver_push or sandbox_model.geoserver_push
        model_input_data_spec = (
            model_input_data_spec or sandbox_model.model_input_data_spec
        )

    if not (geoserver_push and model_input_data_spec):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing geoserver_push and model_input_data_spec in the train_options of {tune_id} tune. Please"
                " provide them in the request or update the tune to add them in your tune_options."
            ),
        )

    inference_request = {
        "model_display_name": "geofm-sandbox-models",
        "description": inference_data.description,
        "location": inference_data.location,
        "fine_tuning_id": tune_id,
        "spatial_domain": inference_data.spatial_domain.model_dump(),
        "temporal_domain": inference_data.temporal_domain,
        "model_input_data_spec": model_input_data_spec,
        "geoserver_push": geoserver_push,
        "maxcc": inference_data.maxcc,
    }

    tune_train_options["model_input_data_spec"] = model_input_data_spec
    if geoserver_push:
        tune_train_options["geoserver_push"] = [
            gp if isinstance(gp, dict) else gp.model_dump() for gp in geoserver_push
        ]

    if data_connector_config:
        inference_request["data_connector_config"] = data_connector_config
        tune_train_options["data_connector_config"] = data_connector_config

    if post_processing:
        inference_request["post_processing"] = post_processing
        tune_train_options["post_processing"] = post_processing

    if inferencing:
        inference_request["inferencing"] = inferencing
        tune_train_options["inferencing"] = inferencing
    from gfmstudio.inference.v2.api import create_inference

    inference_updated_request = inference_data.model_dump()
    inference_updated_request.update(inference_request)

    resp = await create_inference(
        request=request,
        inference=InferenceCreateInput(**inference_updated_request),
        auth=auth,
        db=db,
    )

    # Update the tune with user updated train options.
    if user == tune_meta.created_by:
        tunes_crud.update(
            db=db,
            item_id=str(tune_id),
            item={"train_options": tune_train_options},
            user=user,
        )

    return resp


@app.get(
    "/tunes/{tune_id}/download",
    tags=["FineTuning / Tunes"],
    response_model=schemas.TuneDownloadOut,
)
async def download_tune(
    tune_id: str,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to download tune

    \f
    Parameters
    ----------
    tune_id : str
        Unique tune id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TuneDownloadOut
        Dictionary with tune details including presigned urls to download the artifacts.

    Raises
    ------
    HTTPException
        404: Tune not found
    HTTPException
        500: Errors getting the model artifacts presinged urls
    HTTPException
        400: No checkpoint file or size too small
    """
    user = auth[0]
    tune_meta = tunes_crud.get_by_id(db=db, item_id=tune_id, user=user)
    if not tune_meta:
        raise HTTPException(status_code=404, detail="Tune not found")

    s3 = object_storage.object_storage_client()
    try:
        config_url, checkpoint_url = await asyncify(grab_urls)(
            s3=s3, bucket_name=settings.TUNES_FILES_BUCKET, tune_id=tune_id
        )
    except object_storage.NoCheckpointOrTooSmallFileException:
        logger.exception(
            "Checkpoint file too small or doesn't exist in COS bucket. "
            "See MIN_CHECKPOINT_SIZE in config."
        )
        raise HTTPException(400, detail="No checkpoint or size too small") from None
    except Exception:
        logger.exception("Errors getting the URLs")
        raise HTTPException(500, detail="Errors getting the presinged urls") from None

    response = {
        "id": tune_id,
        "name": tune_meta.name,
        "description": tune_meta.description,
        "config_url": config_url,
        "checkpoint_url": checkpoint_url,
    }

    return response


@app.get(
    "/tunes/{tune_id}/metrics",
    tags=["FineTuning / Tunes"],
    response_model=schemas.TunedModelMlflowMetrics,
)
async def mlflow_metrics(
    tune_id: str,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to get mlflow metrics per tune

    \f
    Parameters
    ----------
    tune_id : str
        Unique tune id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TunedModelMlflowMetrics
        Dictionary with metrics for a single tune

    Raises
    ------
    HTTPException
        404: Tune not Found
    HTTPException
        404: No run or experiment found for the Tune
    HTTPException
        500: Something went wrong in the MLflow server.
    """
    user = auth[0]
    tune_meta = tunes_crud.get_by_id(db=db, item_id=tune_id, user=user)
    if not tune_meta:
        raise HTTPException(status_code=404, detail="Tune not found")

    status, runs, details = get_mlflow_metrics(tune_id=tune_id)

    if status in ("FINISHED", "RUNNING"):
        response = {
            "id": tune_id,
            "status": status,
            "runs": runs,
            "details": details,
        }

        return response
    elif status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=details)

    else:
        raise HTTPException(status_code=500, detail=details)


###############################################
# ---- Base model-related endpoints
###############################################
@app.get(
    "/base-models",
    tags=["FineTuning / Base models"],
    response_model=schemas.BaseModelsOut,
)
async def get_bases(
    auth=Depends(auth_handler),
    shared: Optional[bool] = Query(None, description="Filter by shared tunes."),
    name: Optional[str] = Query(
        None, description="Filter by the name of the base model."
    ),
    status: Optional[list] = Query(
        None, description="Filter by the status of the base model."
    ),
    model_category: Optional[schemas.ModelCategory] = Query(
        None,
        description="Filter by model_category. Options[prithvi, terramind, clay, dofa, resnet, convnext]",
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to list all base models

    \f
    Parameters
    ----------
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    name : Optional[str], optional
        Filter by name of the base model, by default Query(None, description="Filter by the name of the base model.")
    status : Optional[str], optional
        Status of base model to filter by, by default Query(None, description="Filter by the status of the tune.")
    limit : Optional[int], optional
        Max number of models to retrieve, by default Query(25, description="The maximum number of items to retrieve.")
    skip : Optional[int], optional
        Number of items to skip, by default Query(0, description="The number of items to skip.")
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.BaseModelsOut
        dict of found base models
    """
    user = auth[0]
    qp_filters = {}
    search_filters = {}
    filter_expr = None
    filter_expr_list = []
    ignore_user_check = False
    if name:
        search_filters["name"] = name
    if status:
        filter_expr_list.append(BaseModels.status.in_(status))
    if model_category:
        filter_expr_list.append(
            BaseModels.model_params["model_category"].astext
            == str(model_category).lower()
        )
    if shared is not None:
        # TODO: Add logic to mark tunes as sharable and update this filter
        # we currently assume that for a tune to be shared it was
        # created_by the system user.
        qp_filters["created_by"] = settings.DEFAULT_SYSTEM_USER if shared else user
        ignore_user_check = True
    if filter_expr_list:
        filter_expr = and_(*filter_expr_list)
    count, items = bases_crud.get_all(
        db=db,
        filters=qp_filters,
        search=search_filters,
        filter_expr=filter_expr,
        limit=limit,
        skip=skip,
        user=user,
        ignore_user_check=ignore_user_check,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@app.post("/base-models", tags=["FineTuning / Base models"], status_code=201)
async def create_base_model(
    data: schemas.BaseModelsIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to create base model

    \f
    Parameters
    ----------
    data : schemas.BaseModelsIn
        _description_
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Dictinary with the created base model id
    """
    user = auth[0]
    created_model = bases_crud.create(db=db, item=data, user=user)
    return {"id": created_model.id}


@app.get(
    "/base-models/{base_id}",
    tags=["FineTuning / Base models"],
    response_model=schemas.BaseModelOut,
)
async def get_base_by_id(
    base_id: str,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to get base model by id

    \f
    Parameters
    ----------
    base_id : str
        Base model id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.BaseModelOut
        The Found base model

    Raises
    ------
    HTTPException
        404: Base model not found
    """
    user = auth[0]
    data = bases_crud.get_by_id(db=db, item_id=base_id, user=user)
    if not data:
        raise HTTPException(404, detail=f"Base Model {base_id} not found")

    return data


@app.patch(
    "/base-models/{base_id}/model-params",
    tags=["FineTuning / Base models"],
    response_model=schemas.BaseModelParamsOut,
)
async def get_base_model_params(
    base_id: str,
    data: schemas.BaseModelParamsIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to get base model params

    \f
    Parameters
    ----------
    base_id : str
        Base Model id
    data : ModelBaseParams
        Base Model Parameters
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.BaseModelParamsOut
        Dict with the Base Model id and model params

    Raises
    ------
    HTTPException
        404: Base Model not Found
    """
    user = auth[0]
    params_dict = json.loads(json.dumps(data.dict()))
    base_model = bases_crud.get_by_id(db=db, item_id=base_id, user=user)
    if not base_model:
        raise HTTPException(404, detail=f"BaseModel {base_id} not found")

    bases_crud.update(
        db=db, item_id=base_id, item={"model_params": params_dict}, user=user
    )

    return {"id": base_id, "model_params": data}


###############################################
# ---- Tune Template-related endpoints
###############################################
@app.get(
    "/tune-templates", tags=["FineTuning / Templates"], response_model=schemas.TasksOut
)
async def list_tune_templates(
    auth=Depends(auth_handler),
    name: Optional[str] = Query(
        None, description="Filter by the name of the tune template."
    ),
    model_category: Optional[schemas.ModelCategory] = Query(
        None,
        description="Filter by model_category. Options[prithvi, terramind, clay, dofa, resnet, convnext]",
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
    purpose: Optional[str] = Query(
        None, description="Filter by this tune template purpose"
    ),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to get tuning templates

    \f
    Parameters
    ----------
    request : Request
        Request
    auth : _type_, optional
        Authenitication Key, by default Depends(auth_handler)
    name : Optional[str], optional
        Filter by tune template nam, by default Query(None, description="Filter by the name of the tune template.")
    limit : Optional[int], optional
        Max number of tune templates to retrieve, by default
         Query(25, description="The maximum number of items to retrieve.")
    skip : Optional[int], optional
        Number of items to skip, by default Query(0, description="The number of items to skip.")
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TasksOut
        Dict of lists of tune_templates
    """
    user = auth[0]
    qp_filters = {}
    search_filters = {}
    filter_expr = None
    if name:
        search_filters["name"] = name
    if model_category:
        filter_expr = (
            TuneTemplate.extra_info["model_category"].astext
            == str(model_category).lower()
        )
    if purpose:
        qp_filters["purpose"] = purpose

    count, items = tune_template_crud.get_all(
        db=db,
        filters=qp_filters,
        filter_expr=filter_expr,
        search=search_filters,
        limit=limit,
        skip=skip,
        user=user,
        total_count=True,
    )
    return {"results": items, "total_records": count}


@app.post("/tune-templates", tags=["FineTuning / Templates"], status_code=201)
async def create_tasks(
    data: schemas.TaskIn,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to create a task

    \f
    Parameters
    ----------
    data : schemas.TaskIn
        The task data to be added
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Dict of created task id
    """
    user = auth[0]
    # Do check if 'Other' purpose
    if data.purpose == schemas.TaskPurposeEnum.OTHER:
        if data.dataset_id is not None:
            res = await get_dataset_params(
                dataset_id=data.dataset_id,
                user=user,
                dataset_crud=dataset_crud,
                db=db,
            )
            res = res["training_params"]
        else:
            raise HTTPException(
                status_code=412,
                detail="Validation Error: If purpose is Other, dataset_id should not be empty.",
            )

    # check if data.content contents are base64 encoded
    if is_base64_encoded(data.content):
        created_task = tune_template_crud.create(db=db, item=data, user=user)
        return {"id": created_task.id}
    else:
        raise HTTPException(
            status_code=412,
            detail="Validation Error: data.content should be base64 encoded.",
        )


@app.get(
    "/tune-templates/{task_id}",
    tags=["FineTuning / Templates"],
    response_model=schemas.TaskOut,
)
async def retrieve_task(
    task_id: uuid.UUID,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to retrieve a task

    \f
    Parameters
    ----------
    task_id : str
        The task id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    schemas.TaskOut
        A dict of the found task

    Raises
    ------
    HTTPException
        404: Task not found
    """
    user = auth[0]
    task = tune_template_crud.get_by_id(db=db, item_id=task_id, user=user)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task


@app.get("/tune-templates/{task_id}/template", tags=["FineTuning / Templates"])
async def get_task_content_template(
    task_id: uuid.UUID,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint that shows the template string as Python file with placeholders like `${varname}`

    \f
    Parameters
    ----------
    task_id : str
        The task id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database sessiin, by default Depends(utils.get_db)

    Returns
    -------
    Response
        Stringified version of the template

    Raises
    ------
    HTTPException
        404: Task not found
    """
    user = auth[0]
    data = tune_template_crud.get_by_id(db=db, item_id=task_id, user=user)
    if not data:
        raise HTTPException(404, detail=f"Task {task_id} not found")
    content = base64.b64decode(data.content or "")
    return Response(content=content, media_type="application/yaml")


@app.put("/tune-templates/{task_id}/schema", tags=["FineTuning / Templates"])
async def update_task_schema(
    task_id: uuid.UUID,
    task_schema: Annotated[Any, Body()],
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint that allows to update the JSONSchema of a task.

    \f
    Parameters
    ----------
    task_id : str
        The task is
    task_schema : Annotated[Any, Body
        The task schema to be updated
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Dict with task id and message

    Raises
    ------
    HTTPException
        404: Task not Found
    HTTPException
        412: Validation Error: Input data should be a valid JSON object.
    HTTPException
        412: Validation Error: Other validation errors
    """
    user = auth[0]
    task = tune_template_crud.get_by_id(db=db, item_id=task_id, user=user)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        task_schema = (
            json.loads(task_schema) if isinstance(task_schema, str) else task_schema
        )
    except json.decoder.JSONDecodeError:
        detail = "Validation Error: Input data should be a valid JSON object."
        raise HTTPException(status_code=412, detail=detail) from None

    # Validate the json schema
    validator = jsonschema.Draft202012Validator(task_schema)
    try:
        # Validate the JSON schema against the metaschema
        validator.validate(task_schema)
    except jsonschema.exceptions.ValidationError as exc:
        detail = f"Validation error: {exc}"
        raise HTTPException(status_code=412, detail=detail) from None

    tune_template_crud.update(
        db=db, item_id=task_id, user=user, item={"model_params": task_schema}
    )
    return {"task_id": task_id, "message": "Task updated successfully"}


@app.put("/tune-templates/{task_id}/template", tags=["FineTuning / Templates"])
async def update_task_contents_with_file(
    task_id: uuid.UUID,
    file: Annotated[bytes, File()],
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to upload a task file to the database

    \f
    Parameters
    ----------
    task_id : str
        The task id
    file : Annotated[bytes, File
        Base64 encoded file to be uploaded to the db
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Success upload message
    """
    user = auth[0]
    task_contents = base64.b64encode(file).decode()
    from cheap_repr import cheap_repr

    logger.info(cheap_repr(task_contents))
    tune_template_crud.update(
        db=db, item_id=task_id, user=user, item={"content": task_contents}
    )
    return {"message": f"Task {task_id} updated successfully"}


@app.get("/tune-templates/{task_id}/test-render", tags=["FineTuning / Templates"])
async def check_task_content_rendered_with_defaults(
    task_id: uuid.UUID,
    dataset_id,
    base_model,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint that checks that the the task renders correctly

    \f
    Parameters
    ----------
    task_id : _type_
        The task id
    dataset_id : _type_
        The dataset id
    base_model : _type_
        The Base Model id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    Response
        Rendered template
    """
    user = auth[0]
    user_defined_params = {}

    tune_dataset = await get_dataset_params(
        dataset_id=dataset_id,
        user=user,
        dataset_crud=dataset_crud,
        db=db,
    )
    default_dataset_configs = tune_dataset["training_params"]
    default_dataset_configs["seg_map_suffix"] = f"*{tune_dataset['label_suffix']}"

    _, rendered_tune_template = await get_rendered_tuning_template(
        db=db,
        user=user,
        dataset_id=dataset_id,
        base_model=base_model,
        task_id=task_id,
        model_parameters=user_defined_params,
        default_dataset_configs=default_dataset_configs,
    )

    return Response(content=rendered_tune_template, media_type="text/plain")


@app.get(
    "/tune-templates/{task_id}/test-render-user-defined-task",
    tags=["FineTuning / Templates"],
)
async def check_user_defined_task_content_rendered_with_defaults(
    task_id: uuid.UUID,
    dataset_id,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint that checks that the user defined task renders correctly

    \f
    Parameters
    ----------
    task_id : _type_
        The task id
    dataset_id : _type_
        The dataset id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    Response
        Rendered template
    """
    user = auth[0]
    user_defined_params = {}
    _, rendered_tune_template = (
        await tune_handlers.get_rendered_user_defined_tuning_template(
            db=db,
            user=user,
            dataset_id=dataset_id,
            task_id=task_id,
            model_parameters=user_defined_params,
        )
    )

    return Response(content=rendered_tune_template, media_type="text/plain")


@app.delete(
    "/tune-templates/{task_id}", tags=["FineTuning / Templates"], status_code=204
)
async def delete_tune_template(
    task_id: uuid.UUID,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Endpoint to delete a task

    \f
    Parameters
    ----------
    task_id : str
        Unique tune id
    request : Request
        Request
    auth : _type_, optional
        Authentication Key, by default Depends(auth_handler)
    db : Session, optional
        The database session, by default Depends(utils.get_db)

    Returns
    -------
    dict
        Message of successfully deleted tune
    """
    user = auth[0]
    tune_template_crud.soft_delete(db=db, item_id=task_id, user=user)
    # Add deleting from COS/S3 as well....
    return {"message": f"successfully deleted task-{task_id}"}


###############################################
# ---- Dataset-related endpoints
###############################################


@app.get(
    "/datasets",
    tags=["FineTuning / Datasets"],
    response_model=Union[GeoDatasetsResponseSchemaV2, DatasetsSummaryResponseSchema],
)
async def list_datasets(
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
    dataset_name: Optional[str] = Query(
        None, description="Filter by this dataset name."
    ),
    purpose: Optional[list] = Query(
        None, description="Filter by this dataset purpose."
    ),
    status: Optional[str] = Query(None, description="Filter by this dataset status"),
    summary: bool = Query(
        False, description="Flag to return summary of datasets or full view"
    ),
    limit: Optional[int] = Query(
        25, description="The maximum number of items to retrieve."
    ),
    skip: Optional[int] = Query(0, description="The number of items to skip."),
):
    user = auth[0]
    filter_fields = {"version": "v2"}
    filter_expr = None
    search_filters = {}
    if dataset_name:
        search_filters["dataset_name"] = dataset_name
    if purpose:
        filter_expr = GeoDataset.purpose.in_(purpose)
    if status:
        filter_fields["status"] = status

    count, items = dataset_crud.get_all(
        db,
        user=user,
        ignore_user_check=False,
        limit=limit,
        skip=skip,
        filters=filter_fields,
        filter_expr=filter_expr,
        search=search_filters,
        total_count=True,
    )

    response_cls = (
        DatasetsSummaryResponseSchema if summary else GeoDatasetsResponseSchemaV2
    )
    return response_cls(results=items, total_records=count)


@app.post("/datasets/pre-scan", tags=["FineTuning / Datasets"], status_code=201)
async def pre_scan_dataset(
    item: GeoDatasetPreScanRequestSchemaV2,
    auth=Depends(auth_handler),
):
    """Endpoint to pre_scan dataset and return bands and band descriptions.
    This endpoint only supports .zip files.

    \f
    Parameters
    ----------
    request : Request
        Request
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)

    Returns
    -------
    extracted_bands : list
        list of extracted band indices and description

    Raises
    ------
    HTTPException
        422: Unprocessable entity
    """
    try:
        tif_files = list_zipped_files(item.dataset_url)
    except Exception as error:
        tif_files = None
        logger.exception(f"URL or suffixes invalid: {error}")
        raise HTTPException(status_code=422, detail="URL or suffixes invalid")

    extracted_bands = {}
    for training_data_suffix in item.training_data_suffixes:
        data_files = sorted([X for X in tif_files if training_data_suffix in X])
        label_files = sorted([X for X in tif_files if item.label_suffix in X])

        if data_and_label_match(
            data_files, training_data_suffix, label_files, item.label_suffix
        ):
            logger.info("Data and labels matched")
        else:
            logger.debug("Error: Data and labels don't match, based on filenames")
            raise HTTPException(
                status_code=422,
                detail="Data and labels don't match, according to the filenames",
            )

        try:
            band_descriptions = obtain_band_descriptions_from(
                dataset_url=item.dataset_url, sample_data=data_files[0]
            )
        except Exception as error:
            logger.exception(f"Error obtaining band descriptions: {error}")
            raise HTTPException(
                status_code=422, detail="Unable to read sample tif image"
            )

        bands = extract_bands_from(band_descriptions=band_descriptions)

        extracted_bands[training_data_suffix] = bands

    return extracted_bands


@app.get("/datasets/{dataset_id}/sample", tags=["FineTuning / Datasets"])
async def get_sample_images(
    dataset_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
    sample_count: int = Query(
        5, description="The number of images and labels returned"
    ),
) -> Response:
    """Endpoint for obtaining urls to a selection of onboarded images.

    \f
    Parameters
    ----------
    dataset_id : str
        Id for the dataset from which the example file urls are obtained
    request : Request
        Request
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)
    sample_count : int
        the number of samples the user would like, the default is 5

    Returns
    -------
    dict
        a dictionary which contains the urls to example images

    Raises
    ------
    HTTPException
        500: Internal server error
    """
    user = auth[0]
    label_path = dataset_id + "/labels"
    training_data_path = dataset_id + "/training_data"
    s3 = object_storage.object_storage_client()

    try:
        sample_images_response = object_storage.get_item_download_links(
            s3=s3,
            count=sample_count,
            bucket_name=settings.DATASET_FILES_BUCKET,
            directory_path=training_data_path,
        )
        sample_labels_response = object_storage.get_item_download_links(
            s3=s3,
            count=sample_count,
            bucket_name=settings.DATASET_FILES_BUCKET,
            directory_path=label_path,
        )
    except ClientError:
        logger.exception("Could not generate sample images")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal error - Could not generate sample images"},
        )

    if (
        sample_images_response["status_code"] == 200
        and sample_labels_response["status_code"] == 200
    ):
        item = dataset_crud.get_by_id(db=db, item_id=dataset_id, user=user)
        return {
            "id": dataset_id,
            "dataset_name": item.dataset_name,
            "custom_bands": item.custom_bands,
            "label_categories": item.label_categories,
            "sample_images": sample_images_response["items"],
            "sample_label": sample_labels_response["items"],
        }
    else:
        status_code = (
            sample_images_response["status_code"]
            if sample_images_response["status_code"] != 200
            else sample_labels_response["status_code"]
        )
        error_message = (
            sample_images_response["message"]
            if sample_images_response["status_code"] != 200
            else sample_labels_response["message"]
        )
        return {
            "id": dataset_id,
            "status_code": status_code,
            "message": error_message,
        }


@app.patch("/datasets/{dataset_id}", tags=["FineTuning / Datasets"])
async def update_dataset_metadata(
    dataset_id: str,
    item: GeoDatasetMetadataUpdateSchema,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """Endpoint for editing the metadata

    \f
    Parameters
    ----------
    dataset_id : str
        Id for the dataset from which the example file urls are obtained
    request : Request
        Request
    item : dataset_schemas.GeoDatasetMetadataUpdateSchema
        A dictionary of the metadata fields which need update
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)

    Returns
    -------
    dict
        a dictionary which contains the dataset id and update status
    Raises
    ------
    HTTPException
        404: Dataset Not Found
        406: Updated values not Acceptable
    """
    user = auth[0]
    dataset = dataset_crud.get_by_id(db=db, item_id=dataset_id, user=user)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset-{dataset_id} not found")

    updated_metadata = {}

    if item.dataset_name:
        updated_metadata["dataset_name"] = item.dataset_name

    if item.description:
        updated_metadata["description"] = item.description

    if item.label_categories:
        current_label_categories = dataset.label_categories
        # Update old label categories
        for current_label_category in current_label_categories:
            for new_label_category in item.label_categories:
                if current_label_category["id"] == new_label_category["id"]:
                    current_label_category.update(new_label_category)
                    item.label_categories.remove(new_label_category)
        # Add new label categories
        for additional_label_category in item.label_categories:
            current_label_categories.append(additional_label_category)
        updated_metadata["label_categories"] = current_label_categories

    if item.custom_bands:
        current_custom_bands = dataset.custom_bands
        # Update old bands
        for current_band in current_custom_bands:
            for new_band in item.custom_bands:
                if current_band["id"] == new_band["id"]:
                    current_band.update(new_band)
                    item.custom_bands.remove(new_band)
        if len(item.custom_bands) > 0:
            raise HTTPException(
                status_code=406,
                detail=(
                    "It's not allowed to add new custom_bands after onboarding, "
                    "so please re-onboard your data to avoid the metadata and dataset being out of sync"
                ),
            )
        updated_metadata["custom_bands"] = current_custom_bands

    dataset_crud.update(db=db, item_id=dataset_id, user=user, item=updated_metadata)

    return {
        "id": dataset_id,
        "message": "Metadata has been updated accordingly",
    }


@app.post("/datasets/onboard", tags=["FineTuning / Datasets"], status_code=201)
async def onboard_dataset(
    item: GeoDatasetRequestSchemaV2,
    auth=Depends(auth_handler),
    db: Session = Depends(utils.get_db),
):
    """Onboard dataset to the studio

    \f
    Parameters
    ----------
    request : Request
        Request
    item : dataset_schemas.GeoDatasetRequestSchemaV2
        A dictionary of onboarding payload
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)

    Returns
    -------
    dict
        a dictionary which indicates the onboarding request is successfully submitted for a particular dataset

    Raises
    ------
    HTTPException
        404: Dataset Not Found
        406: Updated values not Acceptable
    """
    # Add dataset to DB
    user = auth[0]
    item.version = "v2"
    created_item = dataset_crud.create(db=db, item=item, user=user)

    current_dir = f"{BASE_DIR}/gfmstudio/fine_tuning"
    kjob_tpl = (
        f"{BASE_DIR}/deployment/jobs/onboarding-v2-pipeline-{created_item.id}.yaml"
    )
    # fmt: off
    create_job_deployment_file_command = (
        f"cp {current_dir}/deployment/k8-dataset-onboarding-v2-pipeline.tpl.yaml {kjob_tpl}"
    )
    # fmt: on
    k8s_create_secret_command = (
        "kubectl create secret generic dataset-onboarding-v2-pipeline-params-"
        + created_item.id
    )
    k8s_start_job_command = f"kubectl apply -f {kjob_tpl}"
    replace_dataset_id_command = (
        f"sed -i "
        f"'s|dataset-id|{created_item.id}|g; "
        f"s|DATA_PVC_NAME|{settings.DATA_PVC}|g; "
        f"s|DATASET_PIPELINE_IMAGE|{settings.DATASET_PIPELINE_IMAGE}|g; "
        f"s|IMAGE_PULL_SECRET|{settings.FT_IMAGE_PULL_SECRETS}|g' "
        f"{kjob_tpl}"
    )

    try:
        secret_literals = [
            ("LOGLEVEL", make_k8s_secret_literal(loglevel_name)),
            (
                "DATA_SOURCES",
                make_k8s_secret_literal(
                    validate_and_transform_data_sources(item.data_sources)
                ),
            ),
            ("LABEL_SUFFIX", make_k8s_secret_literal(item.label_suffix)),
            (
                "ONBOARDING_OPTIONS",
                make_k8s_secret_literal(
                    validate_and_transform_options(item.onboarding_options)
                ),
            ),
            ("DATASET_ID", make_k8s_secret_literal(created_item.id)),
            ("DATASET_URL", make_k8s_secret_literal(created_item.dataset_url)),
            (
                "OBJECT_STORAGE_SEC_KEY",
                make_k8s_secret_literal(settings.OBJECT_STORAGE_SEC_KEY),
            ),
            (
                "OBJECT_STORAGE_KEY_ID",
                make_k8s_secret_literal(settings.OBJECT_STORAGE_KEY_ID),
            ),
            ("DF_WEBHOOK_URL", make_k8s_secret_literal(settings.GEOFT_WEBHOOK_URL)),
            ("DF_APIKEY", make_k8s_secret_literal(settings.FT_API_KEY)),
            ("BUCKET_NAME", make_k8s_secret_literal(settings.DATASET_FILES_BUCKET)),
            (
                "OBJECT_STORAGE_REGION",
                make_k8s_secret_literal(settings.OBJECT_STORAGE_REGION),
            ),
            (
                "OBJECT_STORAGE_ENDPOINT",
                make_k8s_secret_literal(settings.OBJECT_STORAGE_ENDPOINT),
            ),
        ]

        k8s_create_secret_command += " " + " ".join(
            f"--from-literal={key}={value}" for key, value in secret_literals
        )
    except ValueError as e:
        logger.exception(k8s_create_secret_command)
        dataset_crud.delete(db=db, item_id=created_item.id, user=user)
        raise HTTPException(status_code=422, detail=e)

    try:
        create_secret_output = subprocess.check_output(
            k8s_create_secret_command, shell=True
        )
        logger.info(create_secret_output)
    except subprocess.CalledProcessError as exc:
        dataset_crud.delete(db=db, item_id=created_item.id, user=user)
        error_message = str(exc.output)
        raise HTTPException(status_code=500, detail=error_message)

    try:
        create_deployment_file_output = subprocess.check_output(
            create_job_deployment_file_command, shell=True
        )
        logger.info("Job deployment file created " + str(create_deployment_file_output))
        replace_id_output = subprocess.check_output(
            replace_dataset_id_command, shell=True
        )
        logger.info("Job deployment file edited" + str(replace_id_output))
        create_job_output = subprocess.check_output(k8s_start_job_command, shell=True)
        logger.info(create_job_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        raise HTTPException(status_code=500, detail=error_message)

    logger.debug(f"Created new dataset: {created_item.id}")
    return {
        "Dataset": "submitting - adding dataset and labels",
        "dataset_id": created_item.id,
        "dataset_url": item.dataset_url,
    }


@app.get("/datasets/{dataset_id}", tags=["FineTuning / Datasets"])
async def retrieve_dataset(
    dataset_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """Endpoint for obtaining dataset metadata

    \f
    Parameters
    ----------
    dataset_id : str
        Id for the dataset we'd like to get metadata
    request : Request
        Request
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)

    Returns
    -------
    dict
        a dictionary which contains the dataset's metadata

    Raises
    ------
    HTTPException
        404: Dataset Not Found
    """
    user = auth[0]
    item = dataset_crud.get_by_id(db=db, item_id=dataset_id, user=user)
    if not item:
        raise HTTPException(
            status_code=404, detail={"msg": f"Dataset {dataset_id} Not Found"}
        )
    
    # Also get logs if dataset succeeded? -- Yes. -- to it in the webhook event handlers section.
    if item.status == "Pending":
        cos_log_path = capture_and_upload_job_log(dataset_id, "v2")
        # dataset_crud.update(
        #     db=db,
        #     item_id=dataset_id,
        #     item={
        #         "logs": cos_log_path,
        #     },
        #     protected=False,
        # )
        item.logs = cos_log_path

    if item.logs: #==failed and item.logs:
        s3 = object_storage.object_storage_client()

        try:
            logs_pre_signed_url = grab_tune_file_presigned_url(
                bucket_name=settings.DATASET_FILES_BUCKET,
                file_key=item.logs,
                s3=s3,
                file_type="dataset logs",
            )

            updated_dict = item.__dict__

            updated_dict["logs_presigned_url"] = logs_pre_signed_url

            return updated_dict

        except Exception:
            logger.exception(
                f"{dataset_id} Error generating presigned url for {item.logs}"
            )

    return item


@app.delete("/datasets/{dataset_id}", tags=["FineTuning / Datasets"])
async def delete_dataset(
    dataset_id: str, db: Session = Depends(utils.get_db), auth=Depends(auth_handler)
) -> Response:
    """Endpoint for deleting the dataset

    \f
    Parameters
    ----------
    dataset_id : str
        Id for the dataset which we'd like to delete
    request : Request
        Request
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    auth : _type_, optional
        Authentication key, by default Depends(auth_handler)

    Returns
    -------
    dict
        a dictionary which indicates the outcome of the deletion

    Raises
    ------
    HTTPException
        500: Internal server error (usually indicates a problem on the S3 side)
    """
    user = auth[0]
    s3 = object_storage.object_storage_client()
    cos_deletion_response = object_storage.remove_from_cos(
        s3=s3, bucket_name=settings.DATASET_FILES_BUCKET, directory_path=dataset_id
    )
    if "Errors" in cos_deletion_response.keys():
        return JSONResponse(
            content={
                "message": "The following objects were not successfully deleted: "
                + str(cos_deletion_response["Errors"])
                + " Please try again.",
                "status_code": cos_deletion_response["ResponseMetadata"][
                    "HTTPStatusCode"
                ],
            }
        )
    else:
        dataset_to_delete = dataset_crud.get_by_id(db=db, item_id=dataset_id, user=user)
        if dataset_to_delete:
            dataset_crud.delete(db=db, item_id=dataset_id, user=user)
        return JSONResponse(
            content={
                "message": "All objects in the dataset have been successfully deleted",
                "status_code": cos_deletion_response["ResponseMetadata"][
                    "HTTPStatusCode"
                ],
            }
        )
