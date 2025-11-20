# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import copy
import json
import re
from ast import literal_eval

import structlog
import yaml
from fastapi import HTTPException

# from gfmstudio.fine_tuning.core.dataset_factory import query_dataset_factory
from gfmstudio.config import settings
from gfmstudio.fine_tuning.core import schema

logger = structlog.get_logger()


class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def get_object_from_jsonschema(schema: dict) -> dict:
    """Function to get object from jsonschema

    Parameters
    ----------
    schema : dict
        json schema

    Returns
    -------
    dict
        dictionary with json objects
    """
    defaults = {}
    if not schema:
        return {}

    def recursive_traverse(schema, current_path=[]):
        if "type" in schema:
            if schema["type"] == "object" and "properties" in schema:
                for key, value in schema["properties"].items():
                    current_path.append(key)
                    recursive_traverse(value, current_path)
                    current_path.pop()
            elif schema["type"] == "array" and "items" in schema:
                current_path.append(schema["items"])
                recursive_traverse(schema["items"], current_path)
                current_path.pop()

            if "default" in schema:
                current_dict = defaults
                for key in current_path[:-1]:
                    current_dict = current_dict.setdefault(key, {})
                current_dict[current_path[-1]] = schema["default"]

    recursive_traverse(schema)
    return defaults


def merge_nested_dicts(dict1: dict, dict2: dict):
    """Function that merges two nested dicts with the second dict updating nested values

    Parameters
    ----------
    dict1 : dict
        dict 1 to be merged
    dict2 : dict
        dict 2 to be merged

    Returns
    -------
    dict
        dict with merged the two dicts contents
    """

    merged_dict = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if key in merged_dict:
            if isinstance(value, dict):
                merged_dict[key] = merge_nested_dicts(merged_dict[key], value)
            else:
                merged_dict[key] = value
        else:
            merged_dict[key] = value
    return merged_dict


def convert_to_jinja2_compatible_braces(string_template: str) -> str:
    """Function to convert to jinja2 compatible braces

    Parameters
    ----------
    string_template : str
        stringified version of template

    Returns
    -------
    str
        stringified version of the template with jinja2 compatible braces
    """
    return re.sub(r"\${(.*?)}", r"{{\1}}", string_template)


def populate_tuning_template(
    template,
    params: schema.TuneTemplateParameters,
) -> str:
    """Function to populate tuning template with means and stds

    Parameters
    ----------
    template :
        template to be populated
    params : schema.TuneTemplateParameters
        Schema used to validate template

    Returns
    -------
    str
        stringified template with means and stds populated
    """

    # select norm_means, norm_stds from data.bands
    # norm_means = [params.norm_means[i] for i in params.bands]
    # norm_stds = [params.norm_stds[i] for i in params.bands]
    norm_means = params.norm_means
    norm_stds = params.norm_stds

    original_params = json.loads(params.json())
    template_params = {
        "norm_stds": norm_stds or [],
        "norm_means": norm_means or [],
    }
    original_params.update(template_params)
    return template.render(**original_params)


def fix_yaml_indentation(template) -> str:
    """Function to fix yaml indentation for list items

    Parameters
    ----------
    template : _type_
        Template

    Returns
    -------
    str
        Stringified version of yaml template
    """
    yaml_template = yaml.safe_load(template)
    indented_yaml_template = yaml.dump(
        yaml_template, default_flow_style=False, sort_keys=False, Dumper=MyDumper
    )

    return indented_yaml_template


def validate_template_blocks(template) -> str:
    """Function to validate the template has these blocks
        ['seed_everything', 'trainer', 'data', 'model', 'optimizer', 'lr_scheduler']

    Parameters
    ----------
    template : _type_
        Template

    Returns
    -------
    str
        Stringified version of the template

    Raises
    ------
    HTTPException
        412, All expected blocks not in the template
    """
    yaml_template = yaml.safe_load(template)
    template_blocks = list(yaml_template.keys())
    expected_blocks = [
        "seed_everything",
        "trainer",
        "data",
        "model",
        "optimizer",
        "lr_scheduler",
    ]

    if all(val in template_blocks for val in expected_blocks):
        return template
    else:
        error = f"Template expected to have these sections {expected_blocks} "
        raise HTTPException(
            status_code=412,
            detail=error,
        )


def update_trainer_dir_paths(tune_id: str, template) -> str:
    """Function to update trainer dir paths

    Parameters
    ----------
    tune_id : str
        Tune ID
    template : _type_
        Template

    Returns
    -------
    str
        Stringified version of updated template
    """

    yaml_template = yaml.safe_load(template)

    # Update path for trainer
    yaml_template["trainer"][
        "default_root_dir"
    ] = f"{settings.FILES_MOUNT}tune-tasks/{tune_id}"

    # Enable Checkpointing
    yaml_template["trainer"]["enable_checkpointing"] = True

    # ToDo Make sure that a checkpoint is provided

    model_checkpoint_callback = False
    # Saving checkpoints paths
    for callback in yaml_template["trainer"]["callbacks"]:
        if "ModelCheckpoint" == callback["class_path"]:
            model_checkpoint_callback = True
            callback_args = callback["init_args"]
            callback_args["dirpath"] = f"{settings.FILES_MOUNT}tune-tasks/{tune_id}/"
            callback_args["filename"] = "best-state_dict-{epoch:02d}"
            callback_args["save_weights_only"] = True

        elif "StateDictAwareModelCheckpoint" == callback["class_path"]:
            model_checkpoint_callback = True
            callback_args = callback["init_args"]
            if (
                "save_weights_only" in callback_args
                and callback_args["save_weights_only"]
            ):
                callback_args["filename"] = (
                    f"{settings.FILES_MOUNT}tune-tasks/{tune_id}/best-state_dict-{{epoch:02d}}"
                )
            else:
                callback_args["filename"] = (
                    f"{settings.FILES_MOUNT}tune-tasks/{tune_id}/{{epoch}}"
                )

    # Create ModelCheckpoint callback
    # Add logger here to check output when the callbacks don't exist.

    if not model_checkpoint_callback:
        yaml_template["trainer"]["callbacks"].append(
            {
                "class_path": "ModelCheckpoint",
                "init_args": {
                    "dirpath": f"{settings.FILES_MOUNT}tune-tasks/{tune_id}/",
                    "filename": "best-state_dict-{epoch:02d}",
                    "save_weights_only": True,
                    "mode": "min",
                    "monitor": "val/loss",
                },
            }
        )

    return yaml.safe_dump(yaml_template, sort_keys=False, default_flow_style=False)


def update_mlflow_logger(tune_id: str, template, user: str) -> str:
    """Function to update mlflow logger section

    Parameters
    ----------
    tune_id : str
        Tune ID
    template : _type_
        Template
    user : str
        User logged in

    Returns
    -------
    str
        Stringified version of the template
    """
    yaml_template = yaml.safe_load(template)

    yaml_template["trainer"]["logger"] = {
        "class_path": "lightning.pytorch.loggers.mlflow.MLFlowLogger",
        "init_args": {
            "experiment_name": tune_id,  # Future version, change this to user / email
            "run_name": "Train",  # Future version, change this to tune_id
            "tracking_uri": settings.MLFLOW_URL,
            "save_dir": f"{settings.FILES_MOUNT}tune-tasks/{tune_id}/mlflow",
            "tags": {"email": user, "name": user.split("@")[0]},
        },
    }

    return yaml.safe_dump(yaml_template, sort_keys=False, default_flow_style=False)


def replace_data_block(template, dataset_params) -> str:
    """Function to replace the data block section in template with values from dataset factory

    Parameters
    ----------
    template : _type_
        Template
    dataset_params : dict
        Dataset parameters from dataset factory

    Returns
    -------
    str
        Stringified version of the template
    """

    # update paths, means, bands, classes
    key_mapping = {
        "bands": "dataset_bands",
        "rgb_band_indices": "rgb_indices",
        "norm_means": "means",
        "norm_stds": "stds",
        "train_labels_dir": "train_label_data_root",
        "test_labels_dir": "test_label_data_root",
        "val_labels_dir": "val_label_data_root",
        "train_data_dir": "train_data_root",
        "test_data_dir": "test_data_root",
        "val_data_dir": "val_data_root",
        "train_split_path": "train_split",
        "test_split_path": "test_split",
        "val_split_path": "val_split",
        "img_suffix": "img_grep",
        "seg_map_suffix": "label_grep",
    }

    # Update dataset_params by mapping old keys to new keys
    for old_key, new_key in key_mapping.items():
        if old_key in dataset_params:
            dataset_params[new_key] = dataset_params.pop(old_key)

    dataset_params["num_classes"] = len(dataset_params["classes"])
    try:
        # Remove the modality key if present as an outside key in dataset_params
        if "image_modalities" in dataset_params:
            image_modalities = dataset_params["image_modalities"]
            logger.info(f"\nImage modalities: {image_modalities} \n")

            if len(image_modalities) > 0:
                for key in image_modalities:
                    dataset_params.pop(key, None)
        # If dataset is unimodal, remove multimodal parameters and flatten the dict values
        if dataset_params["num_modalities"] == 1:
            # Flatten the dicts to grab the modality values
            dataset_params = flatten_dict(dataset_params)
            # logger.info(f"Updated Dataset params: {dataset_params}")

            # Remove these parameters, not needed in templates using the unimodal datasets .
            if "num_modalities" in dataset_params:
                dataset_params.pop("num_modalities")
            if "image_modalities" in dataset_params.keys():
                dataset_params.pop("image_modalities")
            if "rgb_modality" in dataset_params.keys():
                dataset_params.pop("rgb_modality")

        # Not all datasets have classes in them. If present, update to num_classes
        if "classes" in dataset_params:
            dataset_params["num_classes"] = len(dataset_params["classes"])
            dataset_params.pop("classes")

        # Remove these values as not supported in the datamodules
        if "constant_multiply" in dataset_params.keys():
            dataset_params.pop("constant_multiply")
        if "class_weights" in dataset_params.keys():
            dataset_params.pop("class_weights")

        # # Regex to match the existing `data:` block
        data_block_pattern = re.compile(r"data:\s*(.*?)(?=\n(?:\w+:|\Z))", re.DOTALL)

        # Update new_data_block with other values in the user defined template
        user_data_block = re.search(data_block_pattern, template).group(0)
        user_data_dict = yaml.safe_load(user_data_block)

        user_data_dict["data"]["init_args"].update(dataset_params)

        # Append data_root to data paths
        paths = [
            "train_data_root",
            "test_data_root",
            "val_data_root",
            "train_label_data_root",
            "test_label_data_root",
            "val_label_data_root",
            "train_split",
            "test_split",
            "val_split",
        ]
        for k in paths:
            if not isinstance(user_data_dict["data"]["init_args"][k], dict):
                user_data_dict["data"]["init_args"][
                    k
                ] = f"{settings.DATA_MOUNT}{user_data_dict['data']['init_args'][k]}"
            else:
                for key in user_data_dict["data"]["init_args"][k].keys():
                    user_data_dict["data"]["init_args"][k][
                        key
                    ] = f"{settings.DATA_MOUNT}{user_data_dict['data']['init_args'][k][key]}"

        # stringify output & cleanup pyaml anchor and aliases in the final yaml
        class VerboseSafeDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        updated_user_data_yaml = yaml.dump(
            user_data_dict,
            sort_keys=False,
            default_flow_style=False,
            Dumper=VerboseSafeDumper,
        )

        updated_user_data_yaml = f"\n{updated_user_data_yaml.strip()}"

        # Replace the old `data` block with the new one
        modified_yaml = re.sub(data_block_pattern, updated_user_data_yaml, template)

        return modified_yaml

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=(f"Missing required key for datasets in v2 pipelines: {e}"),
        )


def flatten_dict(dataset_params: dict):
    """Flatten dict to update dataset params

    Parameters
    ----------
    dataset_params : dict
        Dataset params dict

    Returns
    -------
    dict
        Updated dataset params dict
    """

    for key, value in dataset_params.items():
        if isinstance(value, dict):
            vals = list(value.values())
            if len(vals) == 1 and isinstance(vals, list):
                dataset_params[key] = vals[0]
            elif len(vals) == 1:
                dataset_params[key] = vals[0]
            else:
                dataset_params[key] = vals
    return dataset_params


async def get_task_params(db, tasks_crud, task_id: str, user: str) -> dict:
    """Function to get task params

    Parameters
    ----------
    db : Depends(utils.get_db)
        The database session
    tasks_crud : crud.ItemCrud(model=Tasks)
        Task crud schema
    task_id : str
        The task id
    user : str
        logged in user

    Returns
    -------
    dict
        Task params

    Raises
    ------
    HTTPException
        500: Internal Server fetching task
    HTTPException
        404: Task not Found
    HTTPException
        500: Task schema is not a valid Json Schema
    """
    try:
        tune_task = tasks_crud.get_by_id(db=db, item_id=task_id, user=user)
    except Exception:
        detail = f"Internal Server Error when fetching task-{task_id}"
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail)

    if not tune_task:
        raise HTTPException(
            status_code=404, detail={"message": f"Task-{task_id} not found"}
        )

    try:
        task_schema = (
            literal_eval(tune_task.model_params)
            if isinstance(tune_task.model_params, str)
            else tune_task.model_params
        )
    except ValueError:
        detail = "Task-{} schema is not a valid Json Schema."
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail)

    default_tuning_config = get_object_from_jsonschema(schema=task_schema)
    return tune_task.content, default_tuning_config


async def get_base_params(db, base_models_crud, base_id: str) -> tuple:
    """Function to get base model params

    Parameters
    ----------
    db : Session = Depends(utils.get_db)
        The database session
    base_models_crud : crud.ItemCrud(model=BaseModels)
        Base model CRUD request
    base_id : str
        The Base Model id

    Returns
    -------
    tuple
        BaseModel Object, Base Model Params

    Raises
    ------
    HTTPException
        404: Base Model not Found
    HTTPException
        500: Internal Server Error when fetching Base Model
    """
    try:
        base_model = base_models_crud.get_by_id(db=db, item_id=base_id)
    except Exception:
        detail = f"Internal Server Error when fetching base-{base_id}"
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail)

    if not base_model:
        raise HTTPException(
            status_code=404,
            detail={"message": f"BaseModel-{base_id} not found"},
        )

    model_params = base_model.model_params
    return base_model, model_params if isinstance(model_params, dict) else {}


def populate_v2_training_params(data) -> dict:
    data = data.to_dict()
    training_params = data.get("training_params")
    data_sources = data.get("data_sources")
    params_from_source = {
        "num_modalities": len(data_sources),
        "image_modalities": [],
        "bands": {},
        "norm_means": {},
        "norm_stds": {},
        "train_data_dir": {},
        "test_data_dir": {},
        "val_data_dir": {},
        "img_suffix": {},
        "rgb_band_indices": [],
    }

    for source in data_sources:
        modality = source.get("modality_tag")
        params_from_source["image_modalities"].append(modality)
        if not source.get("rgb_modality"):
            for idx, band in enumerate(source.get("bands", [])):
                band["index"] = int(band["index"])
                if band.get("RGB_band"):
                    params_from_source["rgb_modality"] = modality
                    params_from_source["rgb_band_indices"].append(idx)

        for key in [
            "bands",
            "norm_means",
            "norm_stds",
            "train_data_dir",
            "test_data_dir",
            "val_data_dir",
        ]:
            params_from_source[key][modality] = training_params.get(modality, {}).get(
                key
            )

        params_from_source["train_labels_dir"] = (
            training_params.get(modality, {})
            .get("train_labels_dir", "")
            .split(modality)[0]
        )
        params_from_source["test_labels_dir"] = (
            training_params.get(modality, {})
            .get("test_labels_dir", "")
            .split(modality)[0]
        )
        params_from_source["val_labels_dir"] = (
            training_params.get(modality, {})
            .get("val_labels_dir", "")
            .split(modality)[0]
        )
        params_from_source["img_suffix"][modality] = training_params.get(
            modality, {}
        ).get("file_suffix", "")

    params_from_source["output_bands"] = params_from_source["bands"]
    training_params.update({**params_from_source})
    data["training_params"] = training_params
    return data


async def get_dataset_params(
    dataset_id: str,
    user,
    db,
    dataset_crud,
) -> dict:
    """Function to get dataset params

    Parameters
    ----------
    dataset_id : str
        The dataset id
    dataset_factory_headers : dict
        The dataset factory payload headers

    Returns
    -------
    dict
        dataset parameters

    Raises
    ------
    HTTPException
        500: Internal Server Error in dataset factory service.
    HTTPException
        3/400s : Error from Dataset Factory
    HTTPException
        404: Dataset Not Found in the dataset-factory
    HTTPException
        412: Dataset missing required training params for Fine-tuning
    """
    tune_dataset = dataset_crud.get_by_id(db=db, item_id=dataset_id, user=user)
    if not tune_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset: {dataset_id} not Found")

    if not tune_dataset.training_params:
        raise HTTPException(
            status_code=412,
            detail={
                "message": f"Dataset-{dataset_id} missing required training params for Fine-tuning"
            },
        )

    return populate_v2_training_params(tune_dataset)
