# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0


"""
This component reads the output from preprocessing and puts together a request to the inference service.
"""

# Dependencies
# pip install ibm-cos-sdk requests tenacity aiohttp opentelemetry-distro opentelemetry-exporter-otlp

import asyncio
import aiohttp
import os
import json
import logging
from pathlib import Path
from gfm_data_processing.exceptions import GfmDataPipelineException
from gfm_data_processing.common import notify_gfmaas_ui, report_exception
from gfm_data_processing.metrics import MetricManager
from tenacity import retry, wait_random, stop_after_attempt

from inference_helper.util import (
    generate_presigned_urls_for_preprocessed_files,
    upload_file,
    get_s3_client,
)

logger = logging.getLogger(__name__)

# inference folder
inference_folder = Path(os.environ.get("inference_folder", ""))

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

push_model_input = os.getenv("push_model_input", "False").lower() == "true"

mounted_pvc = os.getenv("mounted_pvc", "False").lower() == "true"

number_of_retries_for_inference_status = int(
    os.getenv("number_of_retries_for_inference_status", "30")
)

cos = get_s3_client()

temp_bucket_name = os.getenv("temp_bucket_name")

process_id = os.getenv("process_id", "run-inference")

metric_manager = MetricManager(component_name=process_id)


def _move_predicted_image(source, destination):
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Path(source).rename(dest)
    return str(dest)


@retry(wait=wait_random(min=30, max=60), stop=stop_after_attempt(6), reraise=True)
async def make_inference_post_request(
    task_id: str,
    inference_id: str,
    model_id: str,
    inference_url: str,
    inference_input: str,
    inference_input_format: str,
    inference_output_path: str,
) -> dict:
    inference_payload = {
        "data": {
            "data": inference_input,
            "data_format": inference_input_format,
            "out_data_format": inference_input_format,
            "output_path": inference_output_path,
            "image_format": "tiff",
        },
        "model": model_id,
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                inference_url,
                json=inference_payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            ) as inference_response:
                if inference_response.status != 200:
                    error_body = await inference_response.text()
                    logger.error(f"vLLM Error Response: {error_body}")
                    inference_response.raise_for_status()

                inference_response_dict = (
                    await inference_response.json()
                )
        except aiohttp.ClientError as e:
            logger.error(
                f"HTTP request failed for task {task_id} - inference_id {inference_id}: {str(e)}"
            )
            raise GfmDataPipelineException(f"Inference request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON response for task {task_id} - inference_id {inference_id}: {str(e)}"
            )
            raise GfmDataPipelineException(
                f"Invalid inference response format: {str(e)}"
            ) from e
        except KeyError as e:
            logger.error(
                f"Missing 'status' key in response for task {task_id} - inference_id {inference_id}"
            )
            raise GfmDataPipelineException(
                f"Malformed inference response: missing {str(e)}"
            ) from e
        except asyncio.TimeoutError as e:
            logger.error(
                f"Inference request timeout for task {task_id} - inference_id {inference_id}"
            )
            raise GfmDataPipelineException(
                "Inference request timed out after 300s"
            ) from e

    prediction_output = inference_response_dict["data"]["data"]
    predicted_dest = _move_predicted_image(
        source=prediction_output,
        destination=f"{inference_input}_pred.tif",
    )
    # returns only the output image from the inference resposne
    return predicted_dest


def _load_json_config(file_path: Path) -> dict:
    """
    Load and parse JSON configuration file.

    Args:
        file_path: Path to the JSON configuration file

    Returns:
        dict: Parsed JSON configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with file_path.open() as fp:
        return json.load(fp)


def _prepare_inference_inputs(
    task_dict: dict, mounted_pvc: bool, push_model_input: bool
) -> tuple[list[str], str]:
    """
    Prepare inference inputs based on storage configuration.

    Args:
        task_dict: Task configuration dictionary
        mounted_pvc: Whether PVC is mounted
        push_model_input: Whether to push model input to temp bucket

    Returns:
        tuple: (inference_inputs, inference_input_format)
    """
    if mounted_pvc:
        # Handle mounted PVC case - use file paths
        imputed_input = task_dict["imputed_input_image"]
        if isinstance(imputed_input, str):
            input_images = [imputed_input]
        elif isinstance(imputed_input, list):
            input_images = imputed_input
        else:
            raise ValueError(
                f"Input image must be string or list of strings: {imputed_input}"
            )
        return input_images, "path"

    # Handle cloud storage case - use URLs
    if push_model_input:
        input_object_name = "/".join(task_dict["imputed_input_image"].split("/")[-2:])
        logger.debug(
            f"Uploading object {input_object_name} to bucket {temp_bucket_name}"
        )
        upload_file(
            cos, task_dict["imputed_input_image"], temp_bucket_name, input_object_name
        )
        preprocessed_items = [input_object_name]
    else:
        preprocessed_items = [task_dict["imputed_input_image"].replace("/data/", "")]

    download_presigned_url_list = generate_presigned_urls_for_preprocessed_files(
        preprocessed_items=preprocessed_items
    )
    return download_presigned_url_list, "url"


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
async def run_inference_vllm():
    """
    Execute inference requests for preprocessed data using vLLM service.

    This function orchestrates the complete inference workflow:
    1. Loads inference and task configurations from JSON files
    2. Prepares inference inputs based on storage configuration (PVC or S3)
    3. Sends asynchronous inference requests to the vLLM service
    4. Updates task configuration with inference results

    The function supports both mounted PVC (file paths) and cloud storage (URLs)
    for input data, and can handle multiple inference inputs concurrently.

    Raises:
        GfmDataPipelineException: If inference requests fail or critical errors occur
        Exception: For unexpected errors during execution

    Environment Variables:
        inference_folder: Directory containing inference configurations
        inference_id: Unique identifier for the inference job
        task_id: Unique identifier for the specific task
        mounted_pvc: Whether PVC is mounted (affects input format)
        push_model_input: Whether to push model input to temp bucket
    """
    try:
        ######################################################################################################
        ###  Parse the inference and task configs from file
        ######################################################################################################
        inference_config = inference_folder / f"{inference_id}_config.json"
        inference_dict = _load_json_config(inference_config)

        task_folder = inference_folder / task_id
        task_config = task_folder / f"{task_id}_config.json"
        task_dict = _load_json_config(task_config)

        logger.debug("We are about the send inference request")
        logger.debug(task_dict["imputed_input_image"])

        ######################################################################################################
        ###  Prepare inference inputs based on storage configuration
        ######################################################################################################
        inference_inputs, inference_input_format = _prepare_inference_inputs(
            task_dict, mounted_pvc, push_model_input
        )

        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Inferencing ..",
        )

        tasks = [
            make_inference_post_request(
                inference_id=inference_id,
                task_id=task_id,
                model_id=inference_dict["model_internal_name"],
                inference_input=inference_input,
                inference_url=inference_dict["model_access_url"],
                inference_output_path=str(task_folder),
                inference_input_format=inference_input_format,
            )
            for inference_input in inference_inputs
        ]

        # Execute all inference requests concurrently with graceful error handling
        inference_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and separate successes from failures
        successful_results = []
        failed_count = 0

        for idx, result in enumerate(inference_results):
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(
                    f"Inference failed for input {idx} (task: {task_id}): "
                    f"{type(result).__name__}: {result}"
                )
            else:
                logger.info(f"Inference result {idx}: {result}")
                successful_results.append(result)

        # Validate that at least one inference succeeded
        if not successful_results:
            raise GfmDataPipelineException(
                f"All {len(inference_results)} inference requests failed for task {task_id}"
            )

        if failed_count > 0:
            logger.warning(
                f"Partial success: {len(successful_results)}/{len(inference_results)} "
                f"inference requests succeeded for task {task_id}"
            )

        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Inferencing done ..",
        )

        # Update task config with inference results (using first successful result)
        task_dict["model_output_image"] = successful_results[0]

        with task_config.open("w") as fp:
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
    asyncio.run(main=run_inference_vllm())
