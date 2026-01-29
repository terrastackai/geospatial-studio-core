# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This component reads the output from preprocessing and puts together a request to the inference service.
"""

# Dependencies
# pip install ibm-cos-sdk requests tenacity opentelemetry-distro opentelemetry-exporter-otlp

import os
import json
import time
import requests
from zipfile import ZipFile
import glob
from tenacity import retry, wait_random, stop_after_attempt
from gfm_data_processing.exceptions import GfmDataPipelineException
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.metrics import MetricManager

from inference_helper.util import (
    search_key_in_bucket,
    generate_presigned_urls_for_preprocessed_files,
    generate_presigned_url_for_upload_archive,
    upload_file,
    get_s3_client,
)

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

# model id header param
model_id_header_param = os.getenv("model_id_header_param", "grpc-metadata-mm-model-id")

# inference_headers_text
inference_headers_text = os.getenv(
    "inference_headers_text",
    '{"content-type": "application/json", "accept": "application/json", "grpc-metadata-mm-model-id": ""}',
)

push_model_input = os.getenv("push_model_input", "False")

mounted_pvc = os.getenv("mounted_pvc", "False")

number_of_retries_for_inference_status = int(os.getenv("number_of_retries_for_inference_status", "30"))

cos = get_s3_client()

process_id = os.getenv("process_id", "run-inference")

metric_manager = MetricManager(component_name=process_id)


@retry(wait=wait_random(min=30, max=60), stop=stop_after_attempt(6), reraise=True)
def make_inference_post_request(model_access_url, inference_headers, inference_payload, cos_object_upload_key_full):
    response = requests.post(
        model_access_url,
        headers=inference_headers,
        data=json.dumps(inference_payload),
        verify=False,
    )

    if response.status_code == 200:
        # Parse JSON response
        logger.info("Request completed")
        logger.info(response.text)
        object_search_result = None
        while not object_search_result:
            time.sleep(5)
            object_search_result = search_key_in_bucket(cos_object_upload_key_full)

        if push_model_input == "True":
            time.sleep(10)
            try:
                bucket_name = os.environ["bucket_name"]
            except:
                bucket_name = os.environ["temp_bucket_name"]

            task_folder = f"{inference_folder}/{task_id}"

            cos.download_file(bucket_name, cos_object_upload_key_full, f"{task_folder}/model_output.zip")

        with ZipFile(f"{task_folder}/model_output.zip", "r") as zObject:
            zObject.extractall(path=task_folder)

        os.system(f"mv {task_folder}/outputs/*.tif {task_folder}/")
        os.system(f"rm -fr {task_folder}/outputs")
        os.system(f"rm -fr {task_folder}/inputs")

    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        logger.error(response.text)
        response.raise_for_status()


@retry(wait=wait_random(min=30, max=60), stop=stop_after_attempt(6), reraise=True)
def make_inference_post_request_mounted(model_access_url, inference_headers, inference_payload):
    logger.info(f"About to send the payload: {inference_payload}")
    response = requests.post(
        model_access_url,
        headers=inference_headers,
        data=json.dumps(inference_payload),
        verify=False,
    )

    if response.status_code == 200:
        # Parse JSON response
        logger.info("Request completed")
        logger.info(response.text)

    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        logger.error(response.text)
        response.raise_for_status()


def check_if_output_is_ready(task_folder):
    status = check_response_status(task_folder)
    logger.info(f"Response status: {status}")
    logger.info("Waiting for inference output")

    if status == "COMPLETED":
        files_generated = glob.glob(f"{task_folder}/*.tif")
        model_output_image = [X for X in files_generated if "_pred" in X]
        if len(model_output_image) > 0:
            model_output_image = model_output_image[0]
            logger.info(f"Found inference output: {model_output_image}")
            return model_output_image
        else:
            raise GfmDataPipelineException("Inference output is not ready")
    else:
        raise GfmDataPipelineException("Inference output is not available. Status: {status}")


@retry(wait=wait_random(min=5, max=15), stop=stop_after_attempt(number_of_retries_for_inference_status), reraise=True)
def check_response_status(task_folder):
    logger.info("Checking inference status")
    task_parts = task_folder.rsplit("/", 1)
    complete_response_file = f"{task_parts[0]}/completed/{task_parts[1]}.json"

    inf_dir_contents = os.listdir(f"{task_parts[0]}")
    inf_dir_completed_contents = os.listdir(f"{task_parts[0]}/completed")
    logger.debug(f"Inference dir list: {inf_dir_contents}")
    logger.debug(f"Inference completed dir list: {inf_dir_completed_contents}")

    output_status_file = glob.glob(complete_response_file)

    if os.path.exists(complete_response_file) or output_status_file:
        logger.info(f"Found inference output: {complete_response_file}")
        with open(complete_response_file, "r") as fp:
            response_dict = json.load(fp)
        inference_status = response_dict["status"]
        return inference_status
    else:
        logger.warning(f"Inference output file not found: {complete_response_file}")
        raise GfmDataPipelineException("Inference output is not ready")


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def run_inference():
    try:
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

        logger.debug("We are about the send inference request")
        logger.debug(task_dict["imputed_input_image"])

        ######################################################################################################
        ###  for the input image, get a URL
        ######################################################################################################

        if mounted_pvc == "True":
            if isinstance(task_dict["imputed_input_image"], str):
                input_images = [task_dict["imputed_input_image"]]
            elif isinstance(task_dict["imputed_input_image"], list):
                input_images = task_dict["imputed_input_image"]

            inputs_object_for_inference_request = str(
                {
                    "inputs": input_images,
                    "event_id": task_id,
                    "model_id": inference_dict["model_internal_name"],
                    "run_async_inference": "true",
                    "use_shared_storage": "true",
                }
            )
            # "output": f"{task_folder}/model_output.zip",

            inference_payload = {"text": inputs_object_for_inference_request}
            inference_headers = json.loads(inference_headers_text)
            inference_headers[model_id_header_param] = inference_dict["model_internal_name"]

            notify_gfmaas_ui(
                event_id=inference_id,
                task_id=task_id,
                event_status="Inferencing ..",
            )

            logger.debug(f"Running inference now: {inference_payload}")
            make_inference_post_request_mounted(
                inference_dict["model_access_url"], inference_headers, inference_payload
            )

        else:
            if push_model_input == "True":
                input_object_name = "/".join(task_dict["imputed_input_image"].split("/")[-2:])
                print(input_object_name)

                print(os.environ["temp_bucket_name"])

                upload_file(cos, task_dict["imputed_input_image"], os.environ["temp_bucket_name"], input_object_name)

                download_presigned_url_list = generate_presigned_urls_for_preprocessed_files([input_object_name])
            else:
                download_presigned_url_list = generate_presigned_urls_for_preprocessed_files(
                    [task_dict["imputed_input_image"].replace("/data/", "")]
                )

            cos_object_upload_key_full, upload_presigned_url = generate_presigned_url_for_upload_archive(
                f"{inference_id}/{task_id}/model_output.zip"
            )

            inputs_object_for_inference_request = str(
                {
                    "inputs": download_presigned_url_list,
                    "output": upload_presigned_url,
                    "event_id": task_id,
                    "model_id": inference_dict["model_internal_name"],
                    "run_async_inference": "true",
                }
            )

            inference_payload = {"text": inputs_object_for_inference_request}
            inference_headers = json.loads(inference_headers_text)
            inference_headers[model_id_header_param] = inference_dict["model_internal_name"]

            notify_gfmaas_ui(
                event_id=inference_id,
                task_id=task_id,
                event_status="Inferencing ..",
            )

            print(f"Running inference now: {inference_payload}")

            make_inference_post_request(
                inference_dict["model_access_url"],
                inference_headers,
                inference_payload,
                cos_object_upload_key_full,
            )

        model_output_image = check_if_output_is_ready(task_folder)

        with open(task_config_path, "r") as fp:
            task_dict = json.load(fp)
        task_dict["model_output_image"] = model_output_image

        with open(task_config_path, "w") as fp:
            json.dump(task_dict, fp, indent=4)
    except GfmDataPipelineException as gfm_ex:
        raise gfm_ex
    except Exception as ex:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1042",
            message=f"Inference request to service has an error: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=False,
        )
        raise ex


if __name__ == "__main__":
    run_inference()
