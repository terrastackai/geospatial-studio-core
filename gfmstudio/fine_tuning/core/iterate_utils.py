# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""Terratorch iterate utils"""

from gfmstudio.log import logger


def _deep_set(dic, keys, value):
    """Set a value deep inside nested dict/list given dot notation keys."""
    key = keys[0]
    if key.isdigit():  # list index
        key = int(key)
    if len(keys) == 1:
        dic[key] = value
    else:
        _deep_set(dic[key], keys[1:], value)


def update_terratorch_iterate_config(
    config: dict,
    experiment_name: str,
    mlflow_url: str,
    artifact_dir: str,
) -> dict:
    """Update a loaded YAML config dict with new values using dot notation keys."""
    dynamic_fields = {
        "defaults.trainer_args.logger.init_args.tracking_uri": mlflow_url,
        "defaults.trainer_args.logger.init_args.save_dir": f"{artifact_dir}tune-tasks/{experiment_name}/mlflow",
        "defaults.trainer_args.logger.init_args.experiment_name": experiment_name,
        "experiment_name": experiment_name,
        "tasks.0.name": experiment_name,
        "storage_uri": f"{artifact_dir}tune-tasks/{experiment_name}/",
        # "storage_uri": mlflow_url,
    }

    for dotted_key, new_value in dynamic_fields.items():
        keys = dotted_key.split(".")
        try:
            _deep_set(config, keys, new_value)
        except KeyError:
            logger.debug(
                f"HPO Tune {experiment_name}: config with missing keys left unchanged."
            )

    return config
