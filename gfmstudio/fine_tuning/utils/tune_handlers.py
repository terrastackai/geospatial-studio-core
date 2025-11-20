# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""Helper functions for tune submission and management."""

import base64
import logging
import os
import random
import string
from typing import Any, Dict, Optional, Tuple

import yaml
from asyncer import asyncify
from fastapi import HTTPException
from jinja2 import BaseLoader, Environment, runtime
from sqlalchemy.orm import Session

from gfmstudio.celery_worker import deploy_tuning_job_celery_task
from gfmstudio.common.api import crud
from gfmstudio.config import settings
from gfmstudio.fine_tuning import schemas
from gfmstudio.fine_tuning.core import object_storage, tunes
from gfmstudio.fine_tuning.core.kubernetes import deploy_tuning_job
from gfmstudio.fine_tuning.core.schema import TuneTemplateParameters
from gfmstudio.fine_tuning.core.tuning_config_utils import (
    convert_to_jinja2_compatible_braces,
    fix_yaml_indentation,
    get_base_params,
    get_dataset_params,
    get_task_params,
    merge_nested_dicts,
    populate_tuning_template,
    replace_data_block,
    update_mlflow_logger,
    update_trainer_dir_paths,
    validate_template_blocks,
)
from gfmstudio.fine_tuning.models import BaseModels, GeoDataset, Tunes, TuneTemplate
from gfmstudio.fine_tuning.utils.geoserver_handlers import convert_to_geoserver_sld

logger = logging.getLogger(__name__)

tunes_crud = crud.ItemCrud(model=Tunes)
bases_crud = crud.ItemCrud(model=BaseModels)
tune_template_crud = crud.ItemCrud(model=TuneTemplate)
dataset_crud = crud.ItemCrud(model=GeoDataset)


def to_yaml(data):
    """Convert to yaml

    Parameters
    ----------
    data : _type_
        data

    Returns
    -------
    yaml
        converted yaml
    """
    if isinstance(data, runtime.Undefined):
        return data
    return yaml.dump(data)


"""
Initialize jinja2 environment
"""
jinja_env = Environment(loader=BaseLoader())  # noqa: S701
jinja_env.filters.update({"to_yaml": to_yaml})


async def get_rendered_tuning_template(
    db,
    user: str,
    task_id: str,
    dataset_id: str,
    base_model: str,
    model_parameters: dict,
    tune_id: str = None,
    default_dataset_configs: dict = None,
):
    """Endpoint to get rendered tuning template

    \f
    Parameters
    ----------
    db : Session = Depends(utils.get_db),
        The database session.
    user : str
        The logged in user.
    task_id : str
        Task id used to render template.
    dataset_id : str
        Dataset id used to render template.
    base_model : str
        Base model for the rendered template.
    model_parameters : dict
        The model parameters used for the rendered template.
    tune_id : str, optional
        The Unique tune id, by default None
    auth_headers : dict, optional
        Authentication Key, by default None
    default_dataset_configs : dict, optional

    Returns
    -------
    tuple
        Tuple with model configs and the rendered tune template

    Raises
    ------
    HTTPException
        412: Selected task is either missing task_content or task_schema.
    """
    # From Task: Retrieve template and default tuning configs
    task_content, default_task_configs = await get_task_params(
        db=db, tasks_crud=tune_template_crud, task_id=task_id, user=user
    )

    task_content = None if task_content == "None" else task_content
    if not (task_content or default_task_configs):
        raise HTTPException(
            status_code=412,
            detail="Selected task is either missing task_content or task_schema.",
        )

    task_content = convert_to_jinja2_compatible_braces(
        base64.b64decode(task_content).decode("utf-8")
    )

    # From Dataset: Retrieve dataset default tuning configs
    if not default_dataset_configs:
        tune_dataset = await get_dataset_params(
            dataset_id=dataset_id,
            user=user,
            db=db,
            dataset_crud=dataset_crud,
        )
        default_dataset_configs = tune_dataset["training_params"]
        default_dataset_configs["seg_map_suffix"] = f"*{tune_dataset['label_suffix']}"

    # From base models: Retrieve base-model default configs
    base_models_obj, default_base_configs = await get_base_params(
        db=db, base_models_crud=bases_crud, base_id=base_model
    )

    # Merge all tuning configs
    dataset_and_base_configs = merge_nested_dicts(
        dict1=default_base_configs, dict2=default_dataset_configs
    )

    dataset_basemodel_and_task_config = merge_nested_dicts(
        dict1=dataset_and_base_configs, dict2=default_task_configs
    )

    user_defined_configs = model_parameters
    final_tuning_config = merge_nested_dicts(
        dict1=dataset_basemodel_and_task_config, dict2=user_defined_configs
    )

    # Construct final tuning config
    tune_id = tune_id or tunes.generate_tune_id()
    backbone_model_path = f"{settings.BACKBONE_MODELS_MOUNT}gfm_models/{base_models_obj.checkpoint_filename}"
    # Get name from model_params['backbone'] instead of checkpoint_file.
    try:
        backbone_model_name = base_models_obj.model_params.get("backbone", {})
    except KeyError as e:
        raise HTTPException(
            status_code=422, detail=(f"Missing backbone key for base_models: {e}")
        )
    mlflow_tag = {"email": user, "name": user.split("@")[0]}

    # Add a check if the pretrained model is a terramind version that the
    # modalities are the expected ones. [ RGB, S2L2A, S2L1C, S1GRD, S1RTC, DEM ]
    # https://ibm-research.slack.com/archives/C08SEP2RFFF/p1749217069201179?thread_ts=1749201002.491059&cid=C08SEP2RFFF

    terramind_supported_modalities = ["RGB", "S2L2A", "S2L1C", "S1GRD", "S1RTC", "DEM"]
    try:
        image_modality = final_tuning_config["image_modalities"]
        if backbone_model_name.startswith("terramind"):
            if all(item in terramind_supported_modalities for item in image_modality):
                model_configs_obj = TuneTemplateParameters(
                    tune_id=tune_id,
                    mount_root=settings.FILES_MOUNT,
                    data_root=settings.DATA_MOUNT,
                    pretrained_weights_path=backbone_model_path,
                    pretrained_model_name=backbone_model_name,
                    backbone_model_root=settings.BACKBONE_MODELS_MOUNT,
                    data_id=dataset_id,
                    mlflow_tags=mlflow_tag,
                    **final_tuning_config,
                )

                logger.debug("final_tuning_config: %s", model_configs_obj.model_dump())

                # Prepare tuning template
                tune_template = jinja_env.from_string(task_content)
                rendered_tune_template = populate_tuning_template(
                    template=tune_template, params=model_configs_obj
                )

                return model_configs_obj, rendered_tune_template
            else:
                msg = f"{backbone_model_name} modalities expected to be one of: {terramind_supported_modalities}"
                raise HTTPException(
                    status_code=422,
                    detail={msg: msg},
                )

        else:
            # not a terramind model
            model_configs_obj = TuneTemplateParameters(
                tune_id=tune_id,
                mount_root=settings.FILES_MOUNT,
                data_root=settings.DATA_MOUNT,
                pretrained_weights_path=backbone_model_path,
                pretrained_model_name=backbone_model_name,
                backbone_model_root=settings.BACKBONE_MODELS_MOUNT,
                data_id=dataset_id,
                mlflow_tags=mlflow_tag,
                **final_tuning_config,
            )
            logger.debug("final_tuning_config: %s", model_configs_obj.model_dump())

            # Prepare tuning template
            tune_template = jinja_env.from_string(task_content)
            rendered_tune_template = populate_tuning_template(
                template=tune_template, params=model_configs_obj
            )

            return model_configs_obj, rendered_tune_template

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=(f"Missing required key for datasets in v2 pipelines: {e}"),
        )

    except Exception as e:
        logger.exception("Exception when constructing tuning template...")
        raise HTTPException(status_code=422, detail=f"{e}")


def validate_tune_template(db: Session, template_id: str, user: str):
    """Validate that tune template exists.

    Parameters
    ----------
    db : Session
        Database session
    template_id : str
        Tune template identifier
    user : str
        User identifier

    Returns
    -------
    TuneTemplate
        The validated tune template

    Raises
    ------
    HTTPException
        404 if template not found
    """
    tune_task = tune_template_crud.get_by_id(db=db, item_id=template_id, user=user)
    if not tune_task:
        raise HTTPException(
            status_code=404,
            detail=f"TuneTemplate: {template_id} not found",
        )
    return tune_task


def validate_base_model_compatibility(
    db: Session,
    base_model_id: str,
    tune_task,
    user: str,
):
    """Validate base model exists and is compatible with tune template.

    Parameters
    ----------
    db : Session
        Database session
    base_model_id : str
        Base model identifier
    tune_task : TuneTemplate
        The tune template to validate against
    user : str
        User identifier

    Raises
    ------
    HTTPException
        404 if base model not found
        422 if model category incompatible with template
    """
    base_model = bases_crud.get_by_id(db=db, item_id=base_model_id, user=user)
    if not base_model:
        raise HTTPException(
            status_code=404,
            detail=f"BaseModel {base_model_id} not found",
        )

    base_model_params = base_model.model_params or {}
    tune_extra_info = tune_task.extra_info or {}

    base_category = base_model_params.get("model_category")
    template_category = tune_extra_info.get("model_category")

    if base_category != template_category:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Template model_category '{template_category}' is incompatible with "
                f"base model category '{base_category}'. Please select a compatible template."
            ),
        )


def ensure_unique_tune_name(db: Session, name: str) -> str:
    """Ensure tune name is unique by appending suffix if needed.

    Parameters
    ----------
    db : Session
        Database session
    name : str
        Desired tune name

    Returns
    -------
    str
        Unique tune name (lowercase)
    """
    items = tunes_crud.get_all(db=db, filters={"name": name}, ignore_user_check=True)

    if items:
        # Append random suffix to make unique
        alphabet = string.ascii_lowercase + string.digits
        suffix = "".join(random.choices(alphabet, k=8))
        name = f"{name}-{suffix}"

    return name.lower()


def normalize_model_parameters(model_parameters: Any) -> Dict[str, Any]:
    """Normalize model parameters to dictionary format.

    Parameters
    ----------
    model_parameters : Any
        Model parameters (dict or pydantic model)

    Returns
    -------
    Dict[str, Any]
        Normalized parameter dictionary
    """
    if isinstance(model_parameters, dict):
        return model_parameters
    return model_parameters.dict()


async def prepare_tune_dataset(
    db: Session,
    dataset_id: str,
    train_options: Optional[Dict[str, Any]],
    user: str,
) -> Dict[str, Any]:
    """Prepare dataset parameters and merge with train options.

    Parameters
    ----------
    db : Session
        Database session
    dataset_id : str
        Dataset identifier
    train_options : Optional[Dict[str, Any]]
        Training options from request
    user : str
        User identifier

    Returns
    -------
    Dict[str, Any]
        Complete dataset configuration
    """
    # Retrieve dataset from factory
    tune_dataset = await get_dataset_params(
        dataset_id=dataset_id,
        user=user,
        dataset_crud=dataset_crud,
        db=db,
    )

    # Process label categories for geoserver
    train_options = train_options or {}
    label_categories = train_options.get("label_categories") or tune_dataset.get(
        "label_categories", {}
    )

    geoserver_sld = {}
    if label_categories:
        # Convert label categories to geoscerver style
        geoserver_sld = convert_to_geoserver_sld(
            label_categories=label_categories,
            task_type=tune_dataset.get("purpose", "segmentation").lower(),
        )

    if not train_options.get("geoserver_push"):
        train_options["geoserver_push"] = geoserver_sld

    if not train_options.get("model_input_data_spec"):
        train_options["model_input_data_spec"] = tune_dataset.get("data_sources", [])

    return tune_dataset


async def generate_tune_template(
    db: Session,
    user: str,
    tune_id: str,
    data_in: schemas.TuneSubmitIn,
    model_params: Dict[str, Any],
    tune_dataset: Dict[str, Any],
    tune_type: str,
) -> str:
    """Generate rendered tune template YAML.

    Parameters
    ----------
    db : Session
        Database session
    user : str
        User identifier
    tune_id : str
        Tune identifier
    data_in : schemas.TuneSubmitIn
        Tune submission input
    model_params : Dict[str, Any]
        Normalized model parameters
    tune_dataset : Dict[str, Any]
        Dataset configuration
    tune_type : str
        Type of tune (user-defined or standard)

    Returns
    -------
    str
        Rendered YAML template

    Raises
    ------
    HTTPException
        422 if template rendering fails
    """
    # Prepare dataset config with label suffix
    tune_dataset_config = tune_dataset["training_params"]
    tune_dataset_config["seg_map_suffix"] = f"*{tune_dataset['label_suffix']}"

    try:
        if tune_type == "user-defined":
            _, rendered_template = await get_rendered_user_defined_tuning_template(
                db=db,
                user=user,
                tune_id=tune_id,
                task_id=data_in.tune_template_id,
                dataset_id=data_in.dataset_id,
                model_parameters=model_params,
                default_dataset_configs=tune_dataset_config,
            )
        else:
            _, rendered_template = await get_rendered_tuning_template(
                db=db,
                user=user,
                tune_id=tune_id,
                task_id=data_in.tune_template_id,
                dataset_id=data_in.dataset_id,
                base_model=data_in.base_model_id,
                model_parameters=model_params,
                default_dataset_configs=tune_dataset_config,
            )

        return rendered_template

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


async def get_rendered_user_defined_tuning_template(
    db,
    user: str,
    task_id: str,
    dataset_id: str,
    model_parameters: dict,
    tune_id: str = None,
    default_dataset_configs: dict = None,
):
    """Endpoint to get rendered tuning template

    \f
    Parameters
    ----------
    db : Session = Depends(utils.get_db),
        The database session.
    user : str
        The logged in user.
    task_id : str
        Task id used to render template.
    dataset_id : str
        Dataset id used to render template.
    base_model : str
        Base model for the rendered template.
    model_parameters : dict
        The model parameters used for the rendered template.
    tune_id : str, optional
        The Unique tune id, by default None
    auth_headers : dict, optional
        Authentication Key, by default None
    default_dataset_configs : dict, optional

    Returns
    -------
    tuple
        Tuple with model configs and the rendered tune template

    Raises
    ------
    HTTPException
        412: Selected task is either missing task_content or task_schema.
    """
    # From Task: Retrieve template and default tuning configs
    task_content, default_task_configs = await get_task_params(
        db=db, tasks_crud=tune_template_crud, task_id=task_id, user=user
    )

    task_content = None if task_content == "None" else task_content
    if not (task_content or default_task_configs):
        raise HTTPException(
            status_code=412,
            detail="Selected task is either missing task_content or task_schema.",
        )

    task_content = convert_to_jinja2_compatible_braces(
        base64.b64decode(task_content).decode("utf-8")
    )

    # From Dataset: Retrieve dataset default tuning configs
    if not default_dataset_configs:
        all_dataset_configs = await get_dataset_params(
            dataset_id=dataset_id,
            user=user,
            db=db,
            dataset_crud=dataset_crud,
        )

        # Add the label_suffix to the training_params.
        default_dataset_configs = all_dataset_configs["training_params"]
        # Append wildcard * to labels.
        default_dataset_configs["seg_map_suffix"] = (
            f"*{all_dataset_configs['label_suffix']}"
        )

    # Merge all tuning configs
    dataset_basemodel_and_task_config = merge_nested_dicts(
        dict1=default_dataset_configs, dict2=default_task_configs
    )

    user_defined_configs = model_parameters
    final_tuning_config = merge_nested_dicts(
        dict1=dataset_basemodel_and_task_config, dict2=user_defined_configs
    )

    # Construct final tuning config
    tune_id = tune_id or tunes.generate_tune_id()
    mlflow_tag = {"email": user, "name": user.split("@")[0]}
    model_configs_obj = TuneTemplateParameters(
        tune_id=tune_id,
        mount_root=settings.FILES_MOUNT,
        data_root=settings.DATA_MOUNT,
        data_id=dataset_id,
        mlflow_tags=mlflow_tag,
        **final_tuning_config,
    )

    logger.debug("final_tuning_config: %s", model_configs_obj.model_dump())

    # Prepare tuning template
    tune_template = jinja_env.from_string(task_content)
    rendered_tune_template = populate_tuning_template(
        template=tune_template, params=model_configs_obj
    )

    # update trainer block paths
    rendered_tune_template_with_paths = update_trainer_dir_paths(
        tune_id=tune_id, template=rendered_tune_template
    )
    # update trainer.logger block with mlflow
    rendered_tune_template_with_mlflow = update_mlflow_logger(
        tune_id=tune_id, template=rendered_tune_template_with_paths, user=user
    )
    # update data block with dataset from dataset factory

    # if Multimodal dataset, We cannot check if the base model is terramind at this point.
    # The user provides a custom backbone and the best we can check is how many modalities
    # present in the data and update the data block.

    updated_rendered_tune_template_with_data = replace_data_block(
        template=rendered_tune_template_with_mlflow,
        dataset_params=default_dataset_configs,
    )
    # else Multimodal, format the dicts
    # validate that all sections are present in the config
    validated_rendered_tune_template = validate_template_blocks(
        template=updated_rendered_tune_template_with_data
    )
    # Fix yaml indentation for list items
    indented_rendered_tune_template = fix_yaml_indentation(
        validated_rendered_tune_template
    )

    return model_configs_obj, indented_rendered_tune_template


async def save_tune_config(
    tune_id: str,
    rendered_template: str,
    bucket_key: str,
) -> str:
    """Save tune configuration to local storage and COS.

    Parameters
    ----------
    tune_id : str
        Tune identifier
    rendered_template : str
        Rendered YAML template
    bucket_key : str
        COS bucket key path

    Returns
    -------
    str
        Local file path to saved configuration

    Raises
    ------
    HTTPException
        500 if COS upload fails
    """
    # Upload to COS if not local environment
    if settings.ENVIRONMENT.lower() != "local":
        s3 = object_storage.object_storage_client()
        try:
            await asyncify(s3.put_object)(
                Bucket=settings.TUNES_FILES_BUCKET,
                Body=rendered_template,
                Key=bucket_key,
            )
            logger.debug(
                f"Config uploaded to COS: {settings.TUNES_FILES_BUCKET}/{bucket_key}"
            )
        except Exception as exc:
            logger.exception("Failed to upload config to COS")
            raise HTTPException(
                status_code=500, detail="Failed to upload configuration to storage"
            ) from exc

    # Save to local storage
    tune_dir = os.path.join(settings.TUNE_BASEDIR, f"tune-tasks/{tune_id}")
    os.makedirs(tune_dir, exist_ok=True)

    config_path = os.path.join(settings.TUNE_BASEDIR, bucket_key)
    with open(config_path, "w") as f:
        f.write(rendered_template)

    logger.debug(f"Config saved locally: {config_path}")
    return config_path


def get_runtime_image(data_in: schemas.TuneSubmitIn, created_tune) -> str:
    """Get runtime image for tune job.

    Parameters
    ----------
    data_in : schemas.TuneSubmitIn
        Tune submission input
    created_tune : Tune
        Created tune database object

    Returns
    -------
    str
        Runtime image identifier

    Raises
    ------
    HTTPException
        422 if no valid runtime image configured
    """
    runtime_image = None

    if data_in.train_options:
        runtime_image = data_in.train_options.get("image")

    if not runtime_image and created_tune.tune_template.extra_info:
        runtime_image = created_tune.tune_template.extra_info.get("runtime_image")

    if not runtime_image:
        raise HTTPException(
            status_code=422,
            detail="Task must be configured with a valid runtime image",
        )

    return runtime_image


async def submit_tune_job(
    tune_id: str,
    config_path: str,
    runtime_image: str,
) -> Tuple[Optional[str], str, Optional[Dict[str, str]]]:
    """Submit tune job for execution.

    Parameters
    ----------
    tune_id : str
        Tune identifier
    config_path : str
        Path to configuration file
    runtime_image : str
        Runtime image for job

    Returns
    -------
    Tuple[Optional[str], str, Optional[Dict[str, str]]]
        (job_id, status, error_detail)
    """
    ftune_job_id = None
    status = "Pending"
    detail = None

    try:
        if settings.CELERY_TASKS_ENABLED:
            # Submit via Celery
            deploy_tuning_job_celery_task.apply_async(
                kwargs={
                    "ftune_id": tune_id,
                    "ftune_config_file": config_path,
                    "ftuning_runtime_image": runtime_image,
                    "tune_type": schemas.TuneOptionEnum.K8_JOB,
                },
                task_id=tune_id,
            )
            ftune_job_id = f"kjob-{tune_id}".lower()
            status = "In_progress"
        else:
            # Submit directly
            ftune_job_id, updated_status = await deploy_tuning_job(
                ftune_id=tune_id,
                ftune_config_file=config_path,
                ftuning_runtime_image=runtime_image,
                tune_type=schemas.TuneOptionEnum.K8_JOB,
            )
            status = updated_status or "Submitted"

        logger.info(f"Tune job {ftune_job_id} submitted with status: {status}")

    except Exception:
        message = "Tune entry saved, but job submission failed"
        logger.exception(message)
        detail = {"info": message}  # {"error": str(exc)}
        status = "Failed"

    return ftune_job_id, status, detail
