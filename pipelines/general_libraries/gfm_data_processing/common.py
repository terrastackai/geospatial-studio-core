# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import json
import requests
import datetime
from urllib.parse import urljoin
from gfm_logger.glogger import configure_logger
from gfm_data_processing.exceptions import GfmDataPipelineException


# log level
log_level = os.getenv("log_level", "INFO")

# gfmaas api base url
gfmaas_api_base_url = os.getenv("gfmaas_api_base_url", "")

# gfmaas api key
gfmaas_api_key = os.getenv("gfmaas_api_key", "")


# Configure logger
logger = configure_logger(log_level)


def get_unique_id(inputs_folder):
    """
    Pull unique_id from inputs_folder

    Args:
        inputs_folder (str): path to inputs folder

    Output:
        str: unique_id for event tracking
    """
    return inputs_folder.split("/")[3]


def set_up_folders(event_id, inputs_folder_prefix, outputs_folder_prefix):
    """
    Initial creation of folders

    Args:
        event_id (str): event_id for inference request
        inputs_folder_prefix (str): Prefix of path to inputs folder
        outputs_folder_prefix (str): Prefix of path to outputs folder

    Output:
        inputs_folder (str): Path to inputs folder
        outputs_folder (str): Path to outputs folder

    """

    # Make inputs and outputs directories
    inputs_folder = f"{inputs_folder_prefix}/{event_id}/inputs/"
    outputs_folder = f"{outputs_folder_prefix}/{event_id}/outputs/"
    try:
        os.makedirs(inputs_folder)
    except Exception as exc:
        exc_full = str(exc)
        if "transport endpoint is not connected" in exc_full.lower():
            error_code = "1009"
            error_message = f"Error type: {exec_type}. S3FS error with attached COS bucket. \n Full Stacktrace: {exc}"
        else:
            exec_type = type(exc)
            error_code = "1005"
            error_message = f"Event_id {event_id} already in use. Error type: {exec_type} \n Full Stacktrace: {exc}"

        report_exception(
            event_id=event_id,
            error_code=error_code,
            message=f"Preprocessing error: {error_message}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=True,
        )

    os.makedirs(outputs_folder)

    return inputs_folder, outputs_folder


def read_preprocess_result(file_location):
    """
    read preprocess result to use for data processing

    Args:
        file location (str): path to file with preprocess result

    Output:
        data (dict): preprocess result as a json
    """
    with open(file_location, "r") as f:
        data = json.load(f)
    return data


def notify_gfmaas_ui(
    event_id: str,
    task_id: str = None,
    event_status: str = None,
    event_detail_type: str = None,
    event_processed_result: dict = None,
):
    """Helper method to notify Gfmaas-UI
    Parameters
    ----------
    event_id : str
        The event_id sent from the client when the inference was started.
    task_id : str
        The task_id for specific task related to inference; an inference is segmented to multiple tasks based on bboxes/dates.
    event_status : str
        The current status of the event with `event_id` in the inference server
    event_detail_type : str
        The event detail type of the event with `event_id` in the inference server
    event_processed_result : dict
        The dict of the result for the event with `event_id` in the inference server
    """
    gfmaas_webhooks_url = urljoin(gfmaas_api_base_url, "notifications")
    gfmaas_webhooks_headers = {
        "Content-Type": "application/json",
        "X-API-KEY": gfmaas_api_key,
    }

    event_data = {
        "event_id": event_id,
        "detail_type": event_detail_type or "Inf:Task:Notify",
        "source": "com.inference-v2-pipelines-service.ibm",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "detail": event_processed_result or {"message": event_status, "task_id": task_id},
    }

    # try-except to capture a failed connection attempt, we want to continue with the inference regardless...
    try:
        response = requests.post(
            url=gfmaas_webhooks_url,
            headers=gfmaas_webhooks_headers,
            json=event_data,
            timeout=29,
            verify=False,
        )
        if response.status_code not in (200, 201):
            logger.error(
                "Failed to send task status. Reason: (%s)> %s",
                response.status_code,
                response.text,
            )
        else:
            logger.info("Sent a notification to calling service")
    except Exception as ex:
        logger.error("Failed to send task status. Reason: (%s)", ex)


def report_exception(
    event_id,
    task_id,
    error_code,
    message,
    event_detail_type="Inf:Task:Notify",
    verbose=False,
    full_details="",
    raise_exception=False,
    show_service_error=False,
):
    """
    Function used to report exceptions either through logger and webhooks
    with capability to raise GfmDataPipelineException

    Args:
        event_id (str(uuid)): event_id for tracking
        task_id: str,
        error_code (int): error code for exception
        message (str): Message
        event_detail_type (str): event type e.g. Notify, Error or Fail
        verbose (bool): If full exception is printed to logs
        full_details (str): full details of exception
        raise_exception (bool): if we raise a GfmDataPipelineException
        show_service_error (bool): if we pass error message to user
    """
    output_text = f"{str(error_code)} : {message}"
    full_details_prefix = "Full details: "
    if verbose:
        logger.error(
            f"{event_id}: An error occurred: {output_text}. {full_details_prefix + full_details if full_details else ''}"
        )

    event_processed_result = {}
    event_processed_result["error"] = message
    # when showing service error; do not pass the error code
    if show_service_error:
        event_processed_result["show_service_error"] = "show"
    else:
        event_processed_result["error_code"] = error_code
    notify_gfmaas_ui(
        event_id=event_id,
        task_id=task_id,
        event_detail_type=event_detail_type,
        event_processed_result=event_processed_result,
    )

    if raise_exception:
        raise GfmDataPipelineException(message, error_code, event_id)
