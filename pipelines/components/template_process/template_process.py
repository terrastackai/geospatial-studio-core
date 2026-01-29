# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This starter component script
"""

# Dependencies
# pip install ibm-cos-sdk requests tenacity

import os
import json


# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")


if __name__ == "__main__":
    ######################################################################################################
    ###  Parse the inference and task configs from file
    ######################################################################################################

    inference_config_path = f"{inference_folder}/{inference_id}_config.json"
    with open(inference_config_path, "r") as fp:
        inference_dict = json.load(fp)

    task_folder = f"{inference_folder}/{task_id}"

    task_config_path = f"{task_folder}/{task_id}_config.json"
    with open(task_config_path, "r") as fp:
        task_dict = json.load(fp)

    ######################################################################################################
    ###  Add your processing code here
    ######################################################################################################

    ######################################################################################################
    ###  (optional) if you want to pass on information to later stages of the pipelines,
    ###             add information to the task config file which will be read later
    ######################################################################################################

    # with open(task_config_path, 'r') as fp:
    #     task_dict = json.load(fp)
    # task_dict['model_output_image'] = model_output_image

    # with open(task_config_path, 'w') as fp:
    #     json.dump(task_dict, fp, indent=4)
