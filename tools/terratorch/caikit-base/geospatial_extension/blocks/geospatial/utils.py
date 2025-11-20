import functools
import io
import json
import logging
import os
import re
import time
import re

from datetime import datetime
import requests
import uuid

from geospatial_extension.blocks.geospatial.settings import APPLICATION_DATA_DIR
from webhooks import webhooks


logger = logging.getLogger(__name__)


def days_between_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    delta = end_date - start_date
    # Add 1 to include both start and end dates
    return delta.days + 1


def timeit(description=None, custom_logger=None):
    """Decorator to time functions.

    Usage Example
    -------------
    @timeit
    def my_function():
        # Function body
        time.sleep(2)

    @timeit("custom-name")
    def my_function():
        # Function body
        time.sleep(2)

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = custom_logger or logger
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            if description:
                logger.info(f"{description} took {elapsed_time_ms} ms to execute")
            else:
                logger.info(
                    f"Function {func.__name__} took {elapsed_time_ms} ms to execute"
                )
            return result

        return wrapper

    return decorator


def save_inference_output(
    output_data: dict,
    file_path: str,
    notify: bool = False,
    event_id: uuid.UUID = None,
):
    """
    Saves the output data to a file located at the specified file path and sends it as a webhook.

    Parameters
    ----------
    output_data (dict):
        The output data to be saved and sent.
    file_path (str):
        The file path where the output will be saved.
    notify (boolean):
        Boolean enables webhook notifications after saving data.
    event_id: (str)
        Unique id associated with the inference.

    """
    try:
        parent_dir = file_path.split(event_id)[0]
        response_file = os.path.join(parent_dir, "completed", f"{event_id}.json")
        with open(response_file, "w") as file:
            json.dump(output_data, file, indent=2)
        logger.info(f"{event_id} - Inference result saved to: {response_file}")
    except Exception as e:
        logger.warning(f"An error occurred while saving json output to file: {str(e)}")

    if notify:
        webhooks.notify_gfmaas_ui(event_id, event_details=output_data)


def cleanup_complete_inference_files(event_id: str, output_dir=APPLICATION_DATA_DIR):
    uuid_file_re = re.compile(
        r".*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.(zip|json)$"
    )
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if (
            os.path.isfile(file_path)
            and (event_id in file_path)
            and uuid_file_re.match(file_path)
        ):
            try:
                os.remove(file_path)
                logger.debug(
                    f"{event_id} - File removed after complete inference run: {file_path}"
                )
            except Exception:
                logger.debug(
                    f"{event_id} - Error removing file after inference run: {file_path}"
                )


def report_exception(
    event_id,
    error_code,
    message,
    verbose=False,
    full_details="",
    extra_details: dict = None,
):
    """
    Function used to report exceptions through logger, webhooks, and UI

    Args:
        event_id (str(uuid)): event_id for tracking
        error_code (int): error code for exception
        message (str): Message
        verbose (bool): If full exception is printed to logs
        full_details (str): full details of exception
    """
    output_text = f"{error_code}:{message}"
    logger.debug(output_text)
    if verbose:
        logger.debug(full_details)
    webhooks.notify_gfmaas_ui(event_id, output_text, event_details=extra_details)


def validate_presigned_urls(presigned_url: str, method: str = "GET") -> tuple:
    """
    Validates a presigned URLs.

    Args:
        presigned_url (str): The presigned URL to validate.

    Returns:
        tuple:
            bool: True if the presigned URL is valid (not expired and accessible), False otherwise.
            str: Error message if there was an error and empty otherwise.
    """
    try:
        if method.lower() == "put":
            # Create an in-memory file-like object with the file data (NOT a valid zip)
            # TODO: Update to use a valid zip file for testing.
            zip_file_obj = io.BytesIO(b"{'status': 'PENDING'}")
            response = requests.put(presigned_url, files={"file": zip_file_obj})
        else:
            response = requests.get(presigned_url)

        error = ""
        if response.status_code != 200:
            error = "Url provided is either invalid or expired."
        else:
            return True, None
        return False, f"Invalid url: {error}"
    except requests.exceptions.RequestException as e:
        logger.exception(
            "An error occured when attempting to validate the presigned URL."
        )
        return False, str(e)

def update_grep_config_in_file(config_path: str, new_img_pattern: str):
    """Function to update img_grep in config

    Parameters
    ----------
    config_path : str
        Config file path
    new_img_pattern : str
        New img_grep pattern
    """

    with open(config_path, 'r') as file:
        config = file.read()
    
    # Find the current img_grep pattern (this assumes there is one img_grep line)
    current_img_pattern_match = re.search(r"img_grep:\s*'(.*?)'", config)
    
    # If the img_grep line exists, update it with the new img_pattern
    if current_img_pattern_match:
        config = re.sub(r"img_grep:\s*'.*'", f"img_grep: '{new_img_pattern}'", config)
    
    # Write the updated config back to the file
    with open(config_path, 'w') as file:
        file.write(config)
