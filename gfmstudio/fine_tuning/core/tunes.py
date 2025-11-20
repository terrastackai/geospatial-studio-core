# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import string
from datetime import datetime

import shortuuid


def generate_config(project_name: str, conf: dict, template_file):
    """Function to generate config

    Parameters
    ----------
    project_name : str
        The project name
    conf : dict
        Configuration with values to be replaced
    template_file : yaml file
        file

    Returns
    -------
    tuple
        Tuple with experiment name and experiment filepath
    """

    experiment_name = (
        project_name + "-exp-" + datetime.now().strftime("%Y%m%d-%H%M")
    )  # noqa: DTZ005
    conf["exp_name"] = experiment_name

    experiment_filepath = (
        "/opt/app-root/src/data/"
        + project_name
        + "/configs/"
        + experiment_name
        + "_config.yaml"
    )

    conf["iter_per_eval"] = str(
        5 * int(conf["number_training_files"] / int(conf["batch_size"]))
    )
    conf["num_iterations"] = str(
        conf["num_epochs"]
        * int(conf["number_training_files"] / int(conf["batch_size"]))
    )

    logging.info(conf)

    with open(template_file) as t:
        template = string.Template(t.read())

    final_output = template.substitute(**conf)

    with open(experiment_filepath, "w") as output:
        output.write(final_output)

    return experiment_name, experiment_filepath


def generate_tune_id():
    """Function to generate tune id

    Returns
    -------
    str
        generated tune id
    """
    return "geotune-" + shortuuid.uuid().lower()


def generate_dataset_id():
    """Function to generate dataset id

    Returns
    -------
    str
        generated dataset id
    """
    return "geodata-" + shortuuid.uuid().lower()
