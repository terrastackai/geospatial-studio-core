# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import json
import os
import re
import shlex
import uuid
from typing import Optional

import requests as req
from fastapi import HTTPException
from kubernetes import client, config

from gfmstudio.amo.task_manager import amo_task_manager
from gfmstudio.config import BASE_DIR, settings
from gfmstudio.exceptions import PresignedLinkExpired
from gfmstudio.log import logger

from .schemas import ModelFramework, OnboardingStatus


def generate_release_name(model_id: str) -> str:
    # Remove invalid characters and limit length, similar to shell script behavior
    sanitized_name = re.sub(r"[^a-z0-9-]", "", model_id.lower())
    release_name = sanitized_name[:53]  # Limit to 53 characters
    if release_name.endswith("-"):
        release_name = release_name.rstrip("-")  # Ensure it does not end with a dash
    return f"amo-{release_name}-release"


async def invoke_load_model_artifacts(
    bucket_name: str,
    config_presigned_url: str,
    checkpoint_presigned_url: str,
    model_id: str = None,
    kubernetes_job_name: str = None,
    pvc_name: str = None,
):
    # Load Kubernetes configuration
    config.load_incluster_config()
    api_instance = client.CoreV1Api()
    batch_api = client.BatchV1Api()

    checkpoint_filename = f"{model_id}-bestEpoch.ckpt"
    config_filename = f"{model_id}-config.yaml"
    model_id_directory_name = f"{model_id}/"
    model_outputs_directory_name = f"{model_id}/output/"

    # Define the Framework-Specific PVC
    pvc_name = pvc_name or f"{bucket_name.lower()}-pvc"
    pvc_body = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=client.V1ObjectMeta(
            name=pvc_name,
            namespace=f"{settings.NAMESPACE}",
            annotations={
                "ibm.io/auto-create-bucket": "true",
                "ibm.io/bucket": bucket_name.lower(),  # model_framework
                "ibm.io/secret-name": f"{settings.COS_CREDENTIALS_SECRET_NAME}",
                "ibm.io/endpoint": f"{settings.OBJECT_STORAGE_ENDPOINT}",
                "ibm.io/region": f"{settings.OBJECT_STORAGE_REGION}",
            },
            labels={"amo": model_id},
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=[f"{settings.AMO_API_PVC_ACCESS_MODE}"],
            resources=client.V1ResourceRequirements(
                requests={"storage": f"{settings.AMO_API_PVC_STORAGE_CAPACITY}"}
            ),
            storage_class_name=f"{settings.AMO_API_PVC_STORAGE_CLASS}",
        ),
    )

    # Create the Framework-Specific PVC
    try:
        logger.info("Try creating pvc & bucket for model framework")
        api_instance.create_namespaced_persistent_volume_claim(
            namespace=f"{settings.NAMESPACE}", body=pvc_body
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            logger.info(f"PVC & Bucket already exists: {e}")
        elif e.status != 409:
            logger.error("Failed to create pvc / bucket for model framework: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create PVC: {str(e)}"
            )

    # Create model id directory
    download_commands = [f"mkdir /mnt/data/{model_id_directory_name}"]

    # Create model outputs folder
    download_commands.append(f"mkdir -p /mnt/data/{model_outputs_directory_name}")

    # Define the download commands for each config file and checkpoint file
    download_commands.append(
        f"curl -o /mnt/data/{model_id_directory_name}{config_filename} '{config_presigned_url}'"
    )

    # download for checkpoint file
    download_commands.append(
        f"curl -o /mnt/data/{model_id_directory_name}{checkpoint_filename} '{checkpoint_presigned_url}'"
    )

    create_caikit_config = {
        "artifact": ".",
        "module_id": "089e235a-ae91-4c67-8d0c-3f0fa1c0e06e",
        "model_id": model_id,
        "config_path": config_filename,
        "checkpoint_file": checkpoint_filename,
    }
    create_caikit_config_file = (
        f"echo {json.dumps(create_caikit_config)} "
        f"| yq -P . > /mnt/data/{model_id_directory_name}config.yml"
    )

    download_commands.append(create_caikit_config_file)

    # Define and create the job with imagePullPolicy set to Always
    default_jobid = f"amo-download-{model_id.lower()}-{str(uuid.uuid4())[:12]}"
    jobid = kubernetes_job_name or default_jobid
    job_body = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": jobid, "labels": {"amo": model_id}},
        "spec": {
            "template": {
                "spec": {
                    "imagePullSecrets": [{"name": f"{settings.FT_IMAGE_PULL_SECRETS}"}],
                    "containers": [
                        {
                            "name": "downloader",
                            "image": f"{settings.AMO_API_MODEL_ARTIFACTS_DOCKER_IMAGE_URL}",  # Use the latest tag
                            "imagePullPolicy": "Always",  # Ensure the latest image is always pulled
                            "command": [
                                "sh",
                                "-c",
                                " && ".join(download_commands)
                                + " && echo 'Download complete'",
                            ],
                            "volumeMounts": [
                                {"name": "data-volume", "mountPath": "/mnt/data"}
                            ],
                        }
                    ],
                    "restartPolicy": "Never",
                    "volumes": [
                        {
                            "name": "data-volume",
                            "persistentVolumeClaim": {"claimName": pvc_name},
                        }
                    ],
                }
            }
        },
    }

    try:
        batch_api.create_namespaced_job(
            namespace=f"{settings.NAMESPACE}", body=job_body
        )
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.ARTIFACT_TRANSFER_STARTED,
        )
        return jobid  # Return the job ID
    except client.exceptions.ApiException:
        logger.exception("Failed to create Job")
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
        )
        raise HTTPException("Failed to create Job.")


async def invoke_model_onboarding_handler(
    model_framework: str,
    model_id: str,
    token: str,
    deploymentType: str = "gpu",
    resources: dict = None,
    gpuResources: dict = None,
    user_select_deploy_image: Optional[str] = None,  # New optional parameter
):
    # Sanitize inputs to avoid injection attacks
    safe_token = shlex.quote(token)
    safe_model_id = shlex.quote(model_id)
    safe_model_framework = shlex.quote(model_framework)
    release_name = generate_release_name(model_id)

    # Prepare environment variables and paths
    # TODO: put variables in configmap

    inference_image = settings.INFERENCE_SVC_CONTAINER_IMAGE
    if safe_model_framework.lower() == ModelFramework.TERRATORCH_V2.lower():
        inference_image = settings.INFERENCE_SVC_TERRATORCH_V2_IMAGE

    # Override the default image if user supplied one
    deploy_image = (
        user_select_deploy_image if user_select_deploy_image else inference_image
    )

    env_vars = {
        "MODEL_FRAMEWORK": safe_model_framework,
        "MODEL_ID": safe_model_id,
        "NAMESPACE": f"{settings.NAMESPACE}",
        "RESOURCE_NAME": f"{settings.AMO_RESOURCE_NAME}",
        "DEPLOY_IMAGE": f"{deploy_image}",
        "SERVICE_ACCOUNT_NAME": f"{settings.SERVICE_ACCOUNT_NAME}",
        "IMAGE_PULL_SECRET_NAME": f"{settings.FT_IMAGE_PULL_SECRETS}",
        "COS_BUCKET_LOCATION": f"{settings.OBJECT_STORAGE_REGION}",
        "COS_BUCKET_NAME": f"{settings.AMO_FILES_BUCKET}",
        "COS_ENDPOINT_URL": f"{settings.OBJECT_STORAGE_ENDPOINT}",
        "COS_CREDENTIALS_SECRET_NAME": f"{settings.COS_CREDENTIALS_SECRET_NAME}",
        "GATEWAY_URL": f"{settings.GEOFT_WEBHOOK_URL}",
        "GFMAAS_API_KEY_SECRET_NAME": f"{settings.GATEWAY_SECRET_NAME}",
        "GFMAAS_API_CRED_KEY": f"{settings.GFMAAS_API_CRED_KEY}",
        "CONFIGURE_INFERENCE_TOLERATION": f"{settings.CONFIGURE_INFERENCE_TOLERATION}",
        "MODEL_REG_API_TOKEN": safe_token,
        "DEPLOYMENT_TYPE": deploymentType,
        "AMO_INFERENCE_SHARED_PVC": f"{settings.AMO_INFERENCE_SHARED_PVC}",
    }

    # Add CPU and memory resources
    if resources:
        env_vars["REQUESTS_CPU"] = resources.get("requests", {}).get("cpu", "")
        env_vars["REQUESTS_MEMORY"] = resources.get("requests", {}).get("memory", "")
        env_vars["LIMITS_CPU"] = resources.get("limits", {}).get("cpu", "")
        env_vars["LIMITS_MEMORY"] = resources.get("limits", {}).get("memory", "")

    # Add GPU resources if applicable
    if deploymentType == "gpu" and gpuResources:
        env_vars["REQUESTS_GPU"] = gpuResources.get("requests", {}).get(
            "nvidia.com/gpu", ""
        )
        env_vars["LIMITS_GPU"] = gpuResources.get("limits", {}).get(
            "nvidia.com/gpu", ""
        )

    helm_create_script = f"{BASE_DIR}/gfmstudio/amo/scripts/helm/model-helm-create.sh"
    helm_deploy_script = f"{BASE_DIR}/gfmstudio/amo/scripts/helm/model-helm-deploy.sh"
    model_chart_path = f"{BASE_DIR}/amo/{safe_model_id}-chart"

    # Constructing the command string with the release name
    command_string = f"""
    cd /app &&
    {helm_create_script} &&
    {helm_deploy_script} {model_chart_path} {release_name} {env_vars['NAMESPACE']}
    """

    amo_task_manager.set_task_status(
        task_id=model_id,
        status=OnboardingStatus.MODEL_DEPLOY_STARTED,
    )

    command_to_run = ["/bin/bash", "-c", command_string]
    install_process = await asyncio.create_subprocess_exec(
        *command_to_run,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **env_vars},
    )

    install_stdout, _ = await install_process.communicate()

    if install_process.returncode == 0:
        # With check set to True in subprocess run; the following section should execute
        #  if the helm scripts complete with success exit status
        logger.info(install_stdout.decode())
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_DEPLOY_COMPLETE,
        )
        logger.info("All scripts executed successfully")
    else:
        logger.error(install_stdout.decode())
        logger.error(
            "model onboarding exited with a non-zero code, validate if resources were correctly provisioned"
        )
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_DEPLOY_FAILED,
        )


async def invoke_model_offboarding_handler(
    model_id: str, user: str = settings.DEFAULT_SYSTEM_USER
):
    # Sanitize inputs to avoid injection attacks
    safe_model_id = shlex.quote(model_id)
    release_name = generate_release_name(model_id)

    # Prepare environment variables and paths
    # TODO: put variables in configmap

    env_vars = {
        "NAMESPACE": f"{settings.NAMESPACE}",
    }

    helm_uninstall_script = (
        f"{BASE_DIR}/gfmstudio/amo/scripts/helm/model-helm-uninstall.sh"
    )

    # Constructing the command string with the release name
    command_string = f"cd /app && {helm_uninstall_script} {release_name} {safe_model_id} {env_vars['NAMESPACE']}"

    amo_task_manager.set_task_status(
        task_id=model_id,
        status=OnboardingStatus.MODEL_OFFBOARDING_STARTED,
    )

    command_to_run = ["/bin/bash", "-c", command_string]
    uninstall_process = await asyncio.create_subprocess_exec(
        *command_to_run,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **env_vars},
    )

    uninstall_stdout, _ = await uninstall_process.communicate()

    if uninstall_process.returncode == 0:
        # With check set to True in subprocess run; the following section should execute
        #  if the helm scripts complete with success exit status
        logger.info(uninstall_stdout.decode())
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_OFFBOARDING_COMPLETE,
        )
        logger.info("All scripts executed successfully")
    else:
        logger.error(uninstall_stdout.decode())
        logger.error(
            "offboarding exited with a non-zero code, validate if resources were correctly deprovisioned"
        )
        amo_task_manager.set_task_status(
            task_id=model_id,
            status=OnboardingStatus.MODEL_OFFBOARDING_FAILED,
        )


def check_presigned_url_expiry(url: str, model_id: str) -> bool:
    """
    Checks if a given presigned URL has expired by making a request and analyzing the response.
    If the response indicates that the URL is expired, returns True, otherwise returns False.

    Args:
        url (str): The presigned URL to check.
        model_id (str): The ID of the model to update the task status.

    Returns:
        bool: True if the presigned URL is expired, False otherwise.
    """
    try:
        # set a request timeout for connection and read timeout;
        # for successful url we don't need to wait for full download of artifact
        # which may take long
        response = req.get(url=url, timeout=(3, 3))
        if response.status_code == 403 and "ExpiredRequest" in response.text:
            logger.warning(f"Presigned URL for model {model_id} has expired")
            raise PresignedLinkExpired(
                message="One or more presigned URLs have expired.",
            )
        elif response.status_code in range(400, 411) or response.status_code in range(
            500, 510
        ):
            logger.warning(
                f"Presigned URL for model {model_id} failed due to {response.text}."
            )
            raise PresignedLinkExpired(
                message="One or more presigned URLs have failed.",
            )
        else:
            return False
    except req.exceptions.Timeout:
        logger.info("Timed out request; url seems not to have an error")
        return False
    except req.RequestException as e:
        logger.error(
            f"Error while checking presigned URL for model {model_id}: {str(e)}"
        )
        raise PresignedLinkExpired(
            message=f"Error while checking presigned URL: {str(e)}"
        )


async def prepare_model_files(
    model_id: str,
    configs_url: str,
    checkpoint_url: str,
    destination_bucket: str,
    artifact_job_id: str = None,
):
    """
    Prepare model configuration and checkpoint files for a given model.
    This function initiates the upload of model files to a specified bucket and handles
    exceptions related to the process.

    Args:
        model_id (str): Unique identifier for the model.
        configs_url (str): Presigned URL for the model configuration file.
        checkpoint_url (str): Presigned URL for the model checkpoint file.
        destination_bucket (str): The destination bucket where files will be stored.

    Returns:
        str: The job ID for the created upload task.

    Raises:
        HTTPException: If there is an issue with creating the directory or the download/upload process.
    """
    # Check if the presigned URLs are expired
    try:
        _ = check_presigned_url_expiry(
            configs_url, model_id
        ) or check_presigned_url_expiry(checkpoint_url, model_id)
    except Exception:
        logger.warning("Onboarding could not proceed due to exceptions.")
        return

    # Initiate the job to upload files to the bucket
    jobid = await invoke_load_model_artifacts(
        bucket_name=destination_bucket,
        config_presigned_url=configs_url,
        checkpoint_presigned_url=checkpoint_url,
        model_id=model_id,
        kubernetes_job_name=artifact_job_id,
    )
    return jobid  # Return the job ID if successful


async def delete_job_on_completion(
    model_id: str, job_name: str, namespace: str = f"{settings.NAMESPACE}"
):
    """
    Polls the job status, and deletes the job once it has completed.

    Parameters:
    ----------
    job_name : str
        The name of the Kubernetes Job to be monitored and deleted.
    namespace : str
        The namespace where the Job is located.
    """
    batch_api = client.BatchV1Api()
    try:
        # Polling the job status
        while True:
            job_status = batch_api.read_namespaced_job_status(job_name, namespace)
            if job_status.status.succeeded or job_status.status.failed:
                if job_status.status.succeeded:
                    amo_task_manager.set_task_status(
                        task_id=model_id,
                        status=OnboardingStatus.ARTIFACT_TRANSFER_COMPLETE,
                    )
                    logger.info("Downloads completed for model %s", model_id)
                else:
                    amo_task_manager.set_task_status(
                        task_id=model_id,
                        status=OnboardingStatus.ARTIFACT_TRANSFER_FAILED,
                    )
                # If the job has completed, delete it
                batch_api.delete_namespaced_job(
                    name=job_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(propagation_policy="Background"),
                )
                logger.info(f"Job '{job_name}' deleted successfully.")
                break
            await asyncio.sleep(30)  # Poll every 30 seconds
    except client.exceptions.ApiException as e:
        logger.error(f"Error in job deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete Job: {str(e)}")
