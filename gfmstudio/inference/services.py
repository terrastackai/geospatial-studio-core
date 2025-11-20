# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import datetime
import json
import os
import shutil
import uuid

import backoff
import boto3
import requests
from botocore.client import ClientError, Config
from redis.asyncio.client import Redis

from gfmstudio.common.api import crud, utils
from gfmstudio.config import settings
from gfmstudio.fine_tuning.models import Tunes
from gfmstudio.inference import redis_handler
from gfmstudio.inference.types import InferenceStatus
from gfmstudio.log import logger

from ..inference.v2.models import Inference, Task

tunes_crud = crud.ItemCrud(model=Tunes)

task_crud = crud.ItemCrud(model=Task)
inference_crud = crud.ItemCrud(model=Inference)


def data_advisor_choice_lookup(collection: str) -> str:
    """Deduce the data collection from string."""
    collection = collection.lower()
    if "hls" in collection:
        return "sentinelhub:hls_l30"
    elif "harmonized" in collection:
        return "sentinelhub:hls_l30"
    return "sentinelhub:s2_l2a"


async def notify_inference_webhook_events(
    *,
    channel: str,
    message: dict,
    detail_type: str,
    status: str,
    redis: Redis = None,
):
    """Push notifications to redis.

    Parameters
    ----------
    channel : str
        redis channel
    message : dict
        notification message
    detail_type : str
        Event detail type
    status : str
        status of the event
    redis : Redis, optional
        redis client instance, by default None
    """
    await redis_handler.publish_to_channel(
        redis_conn=redis,
        channel=channel,
        message=json.dumps(
            {
                "detail_type": detail_type,
                "status": status,
                "data": message,
            }
        ),
    )


def presigned_url_expires(url, expiry_threshold: int = 0):
    """Validate if an S3 pre-signed URL has expired or will expire in under 2 minutes.

    Parameters
    ==========

    url: str
        The S3 pre-signed URL to validate.
    expiry_threshold: int
        Seconds after which the url will expire. O=url is expired, 120=urls expires in 120sec

    """
    try:
        expiration_time = datetime.datetime.fromtimestamp(
            int(url.split("Expires=")[1].split("&")[0])
        )
        current_time = datetime.datetime.utcnow()
        time_remaining = expiration_time - current_time

        if time_remaining.total_seconds() < expiry_threshold:
            return True, "Expires soon or is expired"
        else:
            return False, "Active url"
    except Exception as e:
        return True, f"Invalid URL: {str(e)}"


async def invoke_tune_upload_handler(
    tune_config_url, tune_checkpoint_url, tune_id, user, db=None
):
    db = db or next(utils.get_db())
    tune_config_response = None
    tune_checkpoint_response = None
    tune_config_deploy_bucket_key = f"tune-tasks/{tune_id}/config_deploy.yaml"
    tune_config_bucket_key = f"tune-tasks/{tune_id}/{tune_id}_config.yaml"
    tune_checkpoint_bucket_key = f"tune-tasks/{tune_id}/best_state_dict_epochN.ckpt"

    try:
        logger.info("Downloading tune checkpoint and tune config yaml")
        if tune_config_url[0:4] == "http":
            tune_config_response = download_with_backoff(tune_config_url)
        if tune_checkpoint_url[0:4] == "http":
            tune_checkpoint_response = download_with_backoff(tune_checkpoint_url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logger.exception(f"Network error during download:{e}")
        raise

    if settings.ENVIRONMENT.lower() == "local":
        tune_dir = os.path.join(settings.TUNE_BASEDIR, f"tune-tasks/{tune_id}")
        if os.path.isdir(tune_dir) is False:
            os.mkdir(tune_dir)

        tune_config_deploy_bucket_dir = os.path.join(
            settings.TUNE_BASEDIR, tune_config_deploy_bucket_key
        )
        tune_config_bucket_dir = os.path.join(
            settings.TUNE_BASEDIR, tune_config_bucket_key
        )
        tune_checkpoint_bucket_dir = os.path.join(
            settings.TUNE_BASEDIR, tune_checkpoint_bucket_key
        )

        if tune_config_url[0:4] == "http":
            with open(tune_config_deploy_bucket_dir, "w") as config_file:
                config_file.write(tune_config_response.text)
            with open(tune_config_bucket_dir, "w") as config_file:
                config_file.write(tune_config_response.text)

        elif tune_config_url[0:4] == "file":
            if not os.path.exists(tune_config_deploy_bucket_dir):
                shutil.copyfile(tune_config_url[7:], tune_config_deploy_bucket_dir)
            if not os.path.exists(tune_config_bucket_dir):
                shutil.copyfile(tune_config_url[7:], tune_config_bucket_dir)

        if tune_checkpoint_url[0:4] == "http":
            with open(tune_checkpoint_bucket_dir, "wb") as checkpoint_file:
                checkpoint_file.write(tune_checkpoint_response.content)
        elif tune_checkpoint_url[0:4] == "file":
            if not os.path.exists(tune_checkpoint_bucket_dir):
                shutil.copyfile(tune_checkpoint_url[7:], tune_checkpoint_bucket_dir)

    else:
        pipelines_bucket_name = settings.TUNES_FILES_BUCKET
        try:
            logger.info(f"Connect to  cos bucket{pipelines_bucket_name}")
            pipeline_s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.OBJECT_STORAGE_KEY_ID,
                aws_secret_access_key=settings.OBJECT_STORAGE_SEC_KEY,
                endpoint_url=settings.OBJECT_STORAGE_ENDPOINT,
                config=Config(
                    signature_version=settings.OBJECT_STORAGE_SIGNATURE_VERSION
                ),
            )
        except ValueError as exc:
            logger.error(f"pipeline_s3_client Misconfiguration: {str(exc)}")

        try:
            logger.info("Uploading tune config file")
            # Uploading both config_deploy.yaml and {tune_id}_config.yaml
            pipeline_s3_client.upload_fileobj(
                tune_config_response.raw,
                Bucket=pipelines_bucket_name,
                Key=tune_config_deploy_bucket_key,
            )
            pipeline_s3_client.upload_fileobj(
                tune_config_response.raw,
                Bucket=pipelines_bucket_name,
                Key=tune_config_bucket_key,
            )
            logger.info("Uploading tune config Succeeded")

            logger.info("uploading tune checkpoint file")
            pipeline_s3_client.upload_fileobj(
                tune_checkpoint_response.raw,
                Bucket=pipelines_bucket_name,
                Key=tune_checkpoint_bucket_key,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.exception(f"Network error during download:{e}")
            raise
        except ClientError as e:
            logger.exception(f"Failed to upload files to cos: {e}")
            raise
    logger.info("Updating the tune with {tune_id} status to Finished")
    tunes_crud.update(
        db=db,
        item_id=tune_id,
        item={"status": "Finished"},
        user=user,
    )
    return {"msg": "Upload complete"}


def backoff_hdlr(details):
    logger.info(
        "Backing off {wait:0.1f}s after {tries} tries "
        "call:{target.__name__} with args:{args} and kwargs "
        "{kwargs}".format(**details)
    )


def giveup_hdlr(details):
    """
    Raises:
        Connection error: If connection fails and retries are exhausted.
    """
    logger.error(
        f"Giving up after {details['tries']} tries for args: {details['args']}"
    )
    raise requests.exceptions.ConnectionError(
        "Max retries reached in dowload_with_backoff"
    )


def fatal_code(e):
    # check if response exists and has a status_code attribute
    if hasattr(e, "response") and e.response is not None:
        return 400 <= e.response.status_code < 500
    return False


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
    max_time=180,
    jitter=None,
    factor=10,
    giveup=fatal_code,
    on_backoff=backoff_hdlr,
    on_giveup=giveup_hdlr,
)
def download_with_backoff(url):
    """
    Attempt to download content from the given URL with exponential backoff retries.

    This function makes an HTTP GET request to the specified URL with streaming enabled.
    It retries on ConnectionError and Timeout exceptions using exponential backoff,
    with the following behavior:

    - Retries continue up to a maximum total elapsed time of 180 seconds (`max_time=180`).
    - The wait time between retries grows exponentially, multiplied by a factor of 10.
    - No jitter is added to backoff delays (jitter=None).
    - Retries are aborted early (give up) if a fatal HTTP status code (4xx) is returned, as determined by `fatal_code`.
    - Logging callbacks `backoff_hdlr` and `giveup_hdlr` are called on each retry and on giving up, respectively.
    - Each individual HTTP request times out after 30 seconds to avoid hanging indefinitely.

    Parameters:
        url (str): The URL to send the GET request to.

    Returns:
        requests.Response: The response object from the successful HTTP request.

    Raises:
        requests.exceptions.ConnectionError: If the connection fails and retries are exhausted.
        requests.exceptions.Timeout: If the request times out and retries are exhausted.
    """
    return requests.get(url, stream=True, timeout=30)


async def invoke_cancel_inference_handler(
    inference_id: uuid.UUID,
    user: str,
    db_session=None,
):
    """
    Cancels running inference tasks associated with a given item.

    This asynchronous function handles the cancellation of inference tasks by:
    1. Retrieving all tasks related to the specified item.
    2. Changing the status of non-running, non-completed, and non-failed tasks to 'STOPPED'.
    3. Waiting for running tasks to complete, periodically checking their status with exponential backoff.
    4. Updating the overall inference status to 'STOPPED'.
    5. Committing changes to the database.

    Args:
        inference id: The  identifier for which inference tasks are to be stopped.
        user: The user associated with the inference tasks.

    Returns:
        None
    """
    session = db_session or next(utils.get_db())

    existing_inference = inference_crud.get_by_id(
        db=session, item_id=inference_id, user=user
    )
    if not existing_inference:
        logger.warning(f"Inference {inference_id} not found for user")
        return {
            "status": "not_found",
            "message": "Inference not found or no permission",
        }

    # Check tasks under the inference
    inference_tasks = task_crud.get_all(
        db=session, user=user, filters={"inference_id": inference_id}
    )
    # Change status of certain tasks to stopped
    running_tasks = []
    for task in inference_tasks:
        if task.status in ["RUNNING"]:
            running_tasks.append(task)
        elif task.status not in ["FINISHED", "DONE", "FAILED", "RUNNING", "STOPPED"]:
            status = "STOPPED"
            logger.info(
                "Updating the status for Inference-%s, user-%s", inference_id, user
            )
            update_item = {"status": status}
            task_crud.update(
                db=session,
                item_id=task.id,
                item=update_item,
                user=user,
            )
    # Wait for the running tasks to complete and mark them as stopped
    backoff = 2
    max_backoff = 30
    while running_tasks:
        session.expire_all()

        running_tasks = [
            task
            for task in running_tasks
            if task is not None
            and task_crud.get_by_id(session, item_id=task.id, user=user).status
            not in ["FINISHED", "DONE", "FAILED", "STOPPED"]
        ]

        if not running_tasks:
            break

        logger.info(
            f"Running tasks detected. Waiting for {backoff} seconds before checking again."
        )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)

    TERMINAL_STATUS = {"FINISHED", "DONE", "FAILED", "STOPPED"}
    session.expire_all()
    inference_tasks = task_crud.get_all(
        session, user=user, filters={"inference_id": inference_id}
    )
    if all(task.status in TERMINAL_STATUS for task in inference_tasks):
        inference_crud.update(
            db=session,
            item_id=inference_id,
            item={"status": InferenceStatus.STOPPED},
            user=user,
        )

        logger.info(f"Inference {inference_id} status updated to STOPPED.")
    else:
        logger.info(
            f"Inference {inference_id} not fully stopped.some tasks still incomplete."
        )
