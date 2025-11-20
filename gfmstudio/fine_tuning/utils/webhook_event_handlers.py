# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import subprocess
import time
from datetime import datetime
from typing import Union

from asyncer import asyncify
from fastapi import HTTPException
from sqlalchemy.orm import Session

from gfmstudio.common.api import crud, utils
from gfmstudio.config import BASE_DIR, settings
from gfmstudio.fine_tuning.core import kubernetes, object_storage
from gfmstudio.fine_tuning.core.kubernetes import collect_pod_logs
from gfmstudio.fine_tuning.models import GeoDataset, Tunes
from gfmstudio.fine_tuning.utils.dataset_handlers import (
    capture_and_upload_job_log,
    transform_error_message,
)
from gfmstudio.inference.v2.schemas import NotificationCreate
from gfmstudio.log import logger

tune_crud = crud.ItemCrud(model=Tunes)
dataset_crud = crud.ItemCrud(model=GeoDataset)


async def free_k8s_resources(tune_id: str):
    """Function that checks status of job and if in terminal state, deletes the job, pvc, configMap
        Complete : Job run successfully
        Failed: Job failed
        '' : In progress


    Parameters
    ----------
    tune_id : str
        Unique tune_id
    """

    k8s_job_status, job_id = await kubernetes.check_k8s_job_status(tune_id)
    logger.info(f"{tune_id} Webhook: Job status: {k8s_job_status}")

    # delete resources
    while k8s_job_status not in ["Complete", "Failed"]:
        # update status
        time.sleep(3)
        k8s_job_status, job_id = await kubernetes.check_k8s_job_status(tune_id)

    # delete resources; job, pvc, ConfigMap
    try:
        await kubernetes.delete_k8s_job_resources(tune_id=job_id)
        logger.info(f"{tune_id} Webhook: Resources deleted")

    except Exception:
        logger.exception(f"{tune_id} Error deleting resources.")


async def free_k8s_resources_by_label(tune_id: str):
    """Function that checks status of job and if in terminal state, deletes the all resources by label
        Complete : Job run successfully
        Failed: Job failed
        '' : In progress

    Parameters
    ----------
    tune_id : str
        Unique tune_id
    """

    k8s_job_status, _ = await kubernetes.check_k8s_job_status(tune_id)
    logger.info(f"{tune_id} Webhook: Job status: {k8s_job_status}")

    # delete resources
    while k8s_job_status not in ["Complete", "Failed"]:
        # update status
        time.sleep(3)
        k8s_job_status, _ = await kubernetes.check_k8s_job_status(tune_id)

    # append kjob to tune-id
    label = f"app=kjob-{tune_id}".lower()
    try:
        # delete all resources; all, pvc, ConfigMap
        await kubernetes.delete_k8s_resources_by_label(label_selector=label)
        logger.info(f"{tune_id} Webhook: Resources deleted")

    except Exception as e:
        logger.info(f"{tune_id} Error deleting resources: {e}")


async def upload_logs_cos(errored_logs_str: str, full_s3_log_file_path: str):
    """Upload log file to COS bucket.

    Parameters
    ----------
    errored_logs_str : str
         Stringified version of logs
    full_s3_log_file_path : str
        Full path to save the logs.
        format: "ftlogs/{date}/{tune-id}.log"

    Raises
    ------
    HTTPException
        500 if not able to upload logs to COS
    """
    # create boto3 client
    s3 = object_storage.object_storage_client()
    try:
        await asyncify(s3.put_object)(
            Bucket=settings.TUNES_FILES_BUCKET,
            Body=errored_logs_str,
            Key=full_s3_log_file_path,
        )
        logger.info(
            f"Log file successfully uploaded to s3://{settings.TUNES_FILES_BUCKET}/{full_s3_log_file_path}"
        )
    except Exception:
        detail = "Could not put pod logs to COS bucket."
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail) from None


async def handle_fine_tuning_webhooks(
    event: Union[NotificationCreate, dict], user: str, db: Session = None
):
    """Handle fine tuning service webhook events.

    For Tuning tasks, if status is;
        Failed: Pod logs are collected and pushed to COS & All tuning resources deleted
        Finished: All tuning resources deleted
        Errored: Logger (At this point all tuning resources are already deleted because no pod started.)

    Parameters
    ----------
    event : schemas.NotificationCreate, dict
        event object
    db : Session, optional
        The database session, by default Depends(utils.get_db)
    user : str
        User email

    Returns
    -------
    notification_id: str
        Created notification id

    """
    session = db or next(utils.get_db())
    event = NotificationCreate(**event) if isinstance(event, dict) else event
    tune_id = str(event.detail["tune_id"])
    notification_id = None

    full_s3_log_file_path = ""
    # If detail_type is Ftune:Task:JobNotifications, delete resources
    if event.detail_type == "Ftune:Task:JobNotifications":
        logger.debug(f"Ftune:Task:JobNotifications: {event.detail}")
        # if status is Failed, push the collected logs to COS
        if event.detail["status"] == "Failed":
            logger.debug(f"{tune_id}: Tuning task failed. Sending pod logs to COS")

            logs = await collect_pod_logs(tune_id=tune_id)

            if logs:
                # Push log file to COS
                current_date = datetime.now().strftime("%Y-%m-%d")
                full_s3_log_file_path = f"ftlogs/{current_date}/{tune_id}.log"
                await upload_logs_cos(logs, full_s3_log_file_path)

            await free_k8s_resources(tune_id)

        elif event.detail["status"] == "Finished":

            logger.debug(f"{tune_id}: Tuning Task finished successfully")
            await free_k8s_resources(tune_id)

        elif event.detail["status"] == "Error":
            logger.debug(
                f"{tune_id}: Tuning Task Errored and resources already deleted."
            )

    try:
        tune_id = str(event.detail["tune_id"])
        tunes = tune_crud.get_by_id(db=session, item_id=tune_id)
        if tunes:
            user = tunes.created_by or user
            tune_crud.update(
                db=session,
                item_id=tune_id,
                item={"status": event.detail["status"], "logs": full_s3_log_file_path},
                protected=False,
            )
    except Exception:
        logger.exception("Tune status was not updated.")

    if settings.CELERY_TASKS_ENABLED:
        # Inline import to avoid circular dependency
        from gfmstudio.celery_worker import celery_app

        # Send signal to kill celery job.
        celery_app.control.revoke(tune_id, terminate=True, signal="SIGTERM")

    return notification_id


async def handle_dataset_factory_webhooks(
    event: Union[NotificationCreate, dict], user: str, db: Session = None
):
    """Handle dataset factory service webhook events.

    For Dataset Factory tasks, if status is;
        Failed: Pod logs are collected and pushed to COS & All dataset and workflow resources deleted
        Success: Pod logs are collected and pushed to COS & All workflow resources deleted
        Pending: Pod logs are collected and pushed to COS & no workflow nor dataset resources are deleted

    Parameters
    ----------
    event : schemas.NotificationCreate, dict
        event object
    db : Session
        The database session, by default Depends(utils.get_db)
    user : str
        User email

    Returns
    -------
    dataset_id: str
        Id of the dataset that's updated

    """
    session = db or next(utils.get_db())
    event = NotificationCreate(**event) if isinstance(event, dict) else event

    dataset_id = str(event.detail["dataset_id"])
    logger.info(f"Incoming dataset-{dataset_id} update after onboarding")
    dataset = dataset_crud.get_by_id(db=session, item_id=dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Missing Dataset-{dataset_id} not updated."},
        )
    cos_log_path = capture_and_upload_job_log(dataset_id, "v2")
    k8s_delete_job_command = f"kubectl delete job onboarding-v2-pipeline-{dataset_id}"
    k8s_delete_secret_command = (
        f"kubectl delete secret dataset-onboarding-v2-pipeline-params-{dataset_id}"
    )
    remove_job_deployment_file_command = (
        f"rm {BASE_DIR}/deployment/jobs/onboarding-v2-pipeline-{dataset_id}.yaml"
    )

    try:
        delete_job_output = subprocess.check_output(k8s_delete_job_command, shell=True)
        logger.info(delete_job_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        logger.error("Unable to remove the job.  Error - " + error_message)

    try:
        delete_secret_output = subprocess.check_output(
            k8s_delete_secret_command, shell=True
        )
        logger.info(delete_secret_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        logger.error("Unable to remove secrets from the job. Error - " + error_message)

    try:
        delete_deployment_file_output = subprocess.check_output(
            remove_job_deployment_file_command, shell=True
        )
        logger.info(delete_deployment_file_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        logger.error("Unable to remove deployment file, Error - " + error_message)

    logger.info(f"Retrieved dataset for webhook update: {dataset.id}")
    user = dataset.created_by or user
    try:
        if event.detail["status"] == "Failed":
            dataset_crud.update(
                db=session,
                item_id=dataset_id,
                item={
                    "status": event.detail["status"],
                    "error": transform_error_message(
                        event.detail["error_code"], event.detail["error_message"]
                    ),
                    "logs": cos_log_path,
                },
                protected=False,
            )
        else:
            updated_training_params = dataset.training_params or {}

            # The label suffix should be prefixed with * for fine-tuning
            label_suffix = dataset.label_suffix
            label_suffix = label_suffix if "*" in label_suffix else f"*{label_suffix}"

            # Populate Training Params
            training_params = event.detail["training_params"]

            # Update the params that came from the user.
            updated_training_params.update(training_params)

            # Add classes from label-categories
            if categories := dataset.label_categories:
                classes = [i["id"] for i in categories if i.get("id")]
                if classes:
                    updated_training_params["classes"] = classes

                class_weights = [i["weight"] for i in categories if i.get("weight")]
                if class_weights:
                    updated_training_params["class_weights"] = class_weights

            dataset_crud.update(
                db=session,
                item_id=dataset_id,
                item={
                    "status": event.detail["status"],
                    "size": event.detail["size"],
                    "error": transform_error_message(
                        event.detail["error_code"], event.detail["error_message"]
                    ),
                    "training_params": updated_training_params,
                },
                protected=False,
            )
    except Exception:
        logger.exception("Dataset status was not updated.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Internal server error occurred. Dataset-{dataset_id} not updated."
            },
        )

    logger.info(f"Dataset status and details has been updated for {dataset_id}")
    return dataset_id
