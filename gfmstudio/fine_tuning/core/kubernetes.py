# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import re
import shlex
import subprocess
import uuid
from subprocess import PIPE, Popen

import backoff
import yaml
from jinja2 import Template
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from gfmstudio.config import BASE_DIR, settings
from gfmstudio.fine_tuning import schemas
from gfmstudio.fine_tuning.core.procs import ProcessError, check_output
from gfmstudio.log import logger

# This lock prevents two coros trying to run kubectl login at the same time
COMMAND = f"kubectl get job --namespace={settings.NAMESPACE}"


async def run_subprocess_cmds(command: list):
    """Function to run subprocess commands using popen

    Parameters
    ----------
    command : list
        The list of commands to be run

    Returns
    -------
    tuple
        Tuple with stdout, stderr, status_code of command output
    """
    result = subprocess.run(command, stdout=PIPE, stderr=PIPE, text=True)

    if result.returncode == 0:
        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        logger.debug(
            f"Stdout: {stdout_output}, Stderr: {stderr_output}, Return Code: {result.returncode}"
        )

        return stdout_output, stderr_output, result.returncode
    else:
        logger.exception(f"Error in run_subprocess_cmds: {result.stderr}")
        return


async def ensure_logged_in(command=COMMAND):
    """Function that ensures kubectl is able to communicate to Kubernetes control plane.
    This function will create a ~/.kube/config by calling kubectl login.

    Parameters
    ----------
    command : str, optional
        Command to run to check that logged in successfully, by default COMMAND
        COMMAND="kubectl get job --namespace=NAMESPACE"
    """
    try:
        logging.info("Checking login status with kubectl list...")
        args = shlex.split(command)
        output = await check_output(*args)
        # If this works, means that we're logged in
        logging.info(f"kubectl reports we're logged in: {output}")
        return
    except ProcessError:
        logging.info("Logging into the cluster")
        login_command = (
            "bash -xc '"
            "kubectl login --token=$SATOKEN "
            "--server=$K8S_SERVER "
            "--insecure-skip-tls-verify=true'"
        )
        await check_output(
            *shlex.split(login_command),
        )


def get_current_namespace():
    """Function to get the current namespace

    Returns
    -------
    str
        The current namespace in the cluster
    """
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            namespace = f.read().strip()
        return namespace
    except FileNotFoundError:
        # If running locally or outside of a Kubernetes cluster, handle it accordingly
        return None


def render_template(template_path: str, context: dict):
    """Render a Jinja2 template with the provided context.

    Parameters:
    ----------
    template_path : str
        Path to the Jinja2 template file.
    context : dict
        Dictionary containing the context variables for rendering.

    Returns:
    -------
    str
        The rendered template as a string.
    """
    with open(template_path) as file_:
        template = Template(file_.read())
    return template.render(context)


def apply_deployment(deployment_yaml: str, namespace: str):
    """Apply a Kubernetes deployment using the provided YAML file.

    Parameters:
    ----------
    deployment_yaml : str
        Path to the deployment YAML file.
    namespace : str
        The OpenShift namespace where the deployment will occur.

    Raises:
    -------
    ApiException
        If there is an error while applying the deployment.
    """
    config.load_kube_config()
    kube_client = client.AppsV1Api()
    with open(deployment_yaml) as f:
        dep = yaml.safe_load(f)

    try:
        resp = kube_client.create_namespaced_deployment(body=dep, namespace=namespace)
        logger.info("Deployment created. status='%s'" % resp.metadata.name)
    except ApiException as e:
        logger.exception(
            "Exception when calling AppsV1Api->create_namespaced_deployment: %s\n" % e
        )
        raise


async def deploy_hpo_tuning_job(
    ftune_id: str,
    ftune_config_file: str,
    ftuning_runtime_image: str = None,
    namespace: str = None,
    **kwargs,
):
    deployment_id = f"kjob-{ftune_id}".lower()
    ftune_api_key = settings.FT_API_KEY
    webhook_event_id = str(uuid.uuid4())
    await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")
    logger.info("Logged in. Creating kubernetes job ...")

    gpu_values = [v.strip() for v in settings.NODE_GPU_SPEC.split(",") if v.strip()]
    # NOTE: Keep the formatting for affinity as is, it's replaced in a yaml template.
    # Updating the formatting may result in an invalid yaml.
    node_affinity = (
        (
            f"""
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: {settings.NODE_SELECTOR_KEY}
                operator: In
                values: {gpu_values}"""
        )
        if settings.CONFIGURE_GPU_AFFINITY
        else ""
    )
    command = [
        "sh",
        f"{BASE_DIR}/gfmstudio/fine_tuning/deployment/scripts/run_geoft_job.sh",
        deployment_id,
        ftune_config_file,
        f"{BASE_DIR}/gfmstudio/fine_tuning/deployment/k8-hpo-tuning-jobs.tpl.yml",
        ftune_id,
        ftune_api_key,
        webhook_event_id,
        str(settings.GEOFT_WEBHOOK_URL),
        ftuning_runtime_image,
        str(settings.FT_IMAGE_PULL_SECRETS),
        str(settings.RESOURCE_LIMIT_CPU),
        str(settings.RESOURCE_LIMIT_Memory),
        str(settings.RESOURCE_LIMIT_GPU),
        str(settings.RESOURCE_REQUEST_CPU),
        str(settings.RESOURCE_REQUEST_Memory),
        str(settings.RESOURCE_REQUEST_GPU),
        str(settings.RUN_TERRATORCH_TEST),
        str(node_affinity),
    ]
    logger.info(f"Executing command: {command}")

    process = Popen(command, stdin=PIPE, stderr=PIPE)
    stdoutdata, stderrdata = process.communicate()
    if process.returncode != 0:
        # send webhook of failure of job creation to UI
        logger.exception(
            f"Error creating resources for job {deployment_id}: {stderrdata}"
        )
        status = "Error"

    else:
        # update status to In Progress
        logger.info(f"In Progress for job {deployment_id}:  {stdoutdata}")
        status = "In_progress"

    logger.info("Deployment initiated and script executed successfully")
    if settings.CELERY_TASKS_ENABLED and status == "In_progress":
        # For celery tasks, wait untill the kubernetes job is complete before exiting.
        await monitor_k8_job_completion(f"{deployment_id}-hpo")

    return deployment_id, status


async def deploy_tuning_job(
    ftune_id: str,
    ftune_config_file: str,
    ftuning_runtime_image: str,
    ocp_token: str = None,
    namespace: str = None,
    tune_type: str = None,
    **kwargs,
):
    """
    Deploy a Kubernetes resource and run a Python script.

    Parameters:
    ----------
    ftune_id: str
        Id related to a tune.
    ftuning_runtime_image : str
        The container image to be used in the deployment.
    ftune_config_file: str
        Path to the fine-tuning config yaml that is used to kick off fine-tuning.
    ocp_token : str
        The OpenShift token used for authentication. Defaults to None
    namespace : str
        The OpenShift namespace where the deployment will occur. Defaults to None
    tune_type: str
        The type of tune to be run. Defaults to None

    Returns:
    -------
    dict
        A dictionary containing a success message and script output.

    Raises:
    -------
    HTTPException
        If there is an error during deployment or script execution.
    """
    if tune_type == schemas.TuneOptionEnum.K8_JOB:
        deployment_id = f"kjob-{ftune_id}".lower()
        ftune_api_key = settings.FT_API_KEY
        webhook_event_id = str(uuid.uuid4())
        await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")
        logger.info("Logged in. Creating kubernetes job ...")
        command = [
            "sh",
            f"{BASE_DIR}/gfmstudio/fine_tuning/deployment/scripts/run_geoft_job.sh",
            deployment_id,
            ftune_config_file,
            f"{BASE_DIR}/gfmstudio/fine_tuning/deployment/k8-tuning-jobs-deployment.tpl.yaml",
            ftune_id,
            ftune_api_key,
            webhook_event_id,
            str(settings.GEOFT_WEBHOOK_URL),
            ftuning_runtime_image,
            str(settings.FT_IMAGE_PULL_SECRETS),
            str(settings.RESOURCE_LIMIT_CPU),
            str(settings.RESOURCE_LIMIT_Memory),
            str(settings.RESOURCE_LIMIT_GPU),
            str(settings.RESOURCE_REQUEST_CPU),
            str(settings.RESOURCE_REQUEST_Memory),
            str(settings.RESOURCE_REQUEST_GPU),
            str(settings.RUN_TERRATORCH_TEST),
        ]
        logger.info(f"Executing command: {command}")

        process = Popen(command, stdin=PIPE, stderr=PIPE)
        stdoutdata, stderrdata = process.communicate()
        if process.returncode != 0:
            # send webhook of failure of job creation to UI
            logger.exception(
                f"Error creating resources for job {deployment_id}: {stderrdata}"
            )
            status = "Error"

        else:
            # update status to In Progress
            logger.info(f"In Progress for job {deployment_id}:  {stdoutdata}")
            status = "In_progress"

    elif tune_type == schemas.TuneOptionEnum.RAY_IO:
        deployment_id = f"rhoairay-{ftune_id}".lower()
        ftuning_runtime_image = "us.icr.io/essentials-cash/rhoairaytest:v1"
        if not namespace:
            try:
                namespace = get_current_namespace() or settings.NAMESPACE
            except Exception:
                namespace = settings.NAMESPACE

        if not ocp_token:
            ocp_token = settings.SATOKEN

        context = {
            "deployment_name": deployment_id,
            "image": ftuning_runtime_image,
            "ocp_token": ocp_token,
            "namespace": namespace,
        }
        template_file = (
            f"{BASE_DIR}/gfmstudio/fine_tuning/deployment/tune_deploy_template.yaml"
        )
        deployment_yaml = render_template(template_file, context)
        # Write rendered template to a temporary file.
        yaml_path = f"/tmp/{ftune_id}_deploy.yaml"
        with open(yaml_path, "w") as f:
            f.write(deployment_yaml)

        # Apply the deployment
        apply_deployment(yaml_path, namespace)

    else:
        raise Exception(
            f"Tune option is not recognize. Use one of {set(schemas.TuneOptionEnum)}"
        )

    logger.info("Deployment initiated and script executed successfully")
    if settings.CELERY_TASKS_ENABLED and status == "In_progress":
        # For celery tasks, wait untill the kubernetes job is complete before exiting.
        await monitor_k8_job_completion(ftune_id)

    return deployment_id, status


@backoff.on_exception(
    backoff.constant,
    ValueError,
    jitter=None,
    max_tries=settings.JOB_MAX_RETRY_COUNT,
    interval=settings.KJOB_MAX_WAIT_SECONDS,
    raise_on_giveup=False,
)
async def monitor_k8_job_completion(ftune_id: str):
    try:
        k8s_job_status, _ = await check_k8s_job_status(ftune_id)
    except Exception as exc:
        if "not found" in str(exc):
            return

    logger.info(f"{ftune_id} Waiting for job completion: Job status: {k8s_job_status}")
    if k8s_job_status not in ["Complete", "Failed"]:
        raise ValueError(f"{ftune_id}: Wait for task in PENDING state...")


async def check_k8s_job_status(tune_id: str, retry_label_lookup=True):
    """Function to check Kubernetes job status

    Parameters
    ----------
    tune_id : str
        Tune id
    retry_label_lookup: bool
        Wheather to retry lookup with labels.

    Returns
    -------
    tuple(str, str)
        A tuple of the status of the k8s job, and the job_id
    """

    kjob_id = tune_id if "kjob" in tune_id else f"kjob-{tune_id}-job".lower()

    # Log in
    await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")

    command = [
        "kubectl",
        "get",
        "job",
        kjob_id,
        "-o",
        "jsonpath={.status.conditions[*].type}",
    ]

    result = await run_subprocess_cmds(command=command)
    logger.info(f"kubectl run cmds result: {command} ---> {result}")

    if result and result[0] != "":
        # If job is not In Progress
        status = result[0].strip()
        logger.info(f"{kjob_id}: Status for job {status}")
        return status, kjob_id

    else:
        # Attempt retrieval Using labels:
        if retry_label_lookup is True:
            # Remove -job and -hpo suffix if present
            app_name = re.sub(r"-(job|hpo)$", "", kjob_id)
            get_by_lables_cmd = [
                "kubectl",
                "get",
                "jobs",
                "-l",
                f"app-name={app_name}",
                "-o",
                "name",
            ]
            logger.info(f"kubectl retry cmd: {get_by_lables_cmd}")
            result = await run_subprocess_cmds(command=get_by_lables_cmd)
            job_name = result[0].strip() if result else ""
            job_name = job_name.split("/")[-1]
            logger.info(f"kubectl retry job_name: {job_name}")
            if job_name:
                result = await check_k8s_job_status(job_name, retry_label_lookup=False)
                logger.info(f"kubectl retry result: {result}")
                return result if result else (None, job_name)
        logger.exception(f"{kjob_id}: Error in check_k8s_job_status: {result}")
        return None, tune_id


async def delete_k8s_job_resources(tune_id: str):
    """Function to delete Kubernetes job resources
        This function assumes that a pvc, pod, and configmap exists.

    Parameters
    ----------
    tune_id : str
        Tune id
    """

    # log in
    await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")

    # Delete the Job, ConfigMap, PVC in that order
    if "kjob" in tune_id:
        kjob_id = tune_id
        m = re.search(r"(geotune-[^-]+)", tune_id)
        tune_id = m.group(1)
    else:
        kjob_id = f"kjob-{tune_id}-job".lower()

    pvc_name = f"kjob-{tune_id}-config-pvc".lower()
    configMap_name = f"kjob-{tune_id}-config".lower()
    ftuning_script_config_map = f"ftuning-script-{tune_id}"

    resources = ["job", "pvc", "configmap", "configmap"]
    names = [kjob_id, pvc_name, configMap_name, ftuning_script_config_map]
    commands_list = [
        ["kubectl", "delete", resource, name]
        for resource, name in zip(resources, names)
    ]

    results = []
    for cmd in commands_list:
        result = await run_subprocess_cmds(command=cmd)
        results.append(result)

    # Some resources deleted but not all.
    if None in results and results.count(None) != 3:
        none_indices = [index for index, value in enumerate(results) if value is None]
        mapped_values = [
            commands_list[i][3] for i in none_indices if i < len(commands_list)
        ]
        logger.exception(
            f"Error encountered while deleting these resources: {mapped_values}. Check pod logs for more details."
        )
        return (
            "Failed",
            f"Error encountered while deleting these resources: {mapped_values}. Check pod logs for more details.",
        )

    # All resources not deleted.
    elif None in results and results.count(None) == 3:
        none_indices = [index for index, value in enumerate(results) if value is None]
        mapped_values = [
            commands_list[i][3] for i in none_indices if i < len(commands_list)
        ]
        logger.exception(
            f"Error encountered while deleting all the resources: {mapped_values}. Check pod logs for more details."
        )
        return (
            "Failed",
            f"Error encountered while deleting all the resources: {mapped_values}. Check pod logs for more details.",
        )

    # All resources successfully deleted.
    elif None not in results:
        mapped_values = [x[3] for x in commands_list]
        logger.debug(f"Successfully deleted resources: {mapped_values}")

        return "Success", f"Successfully deleted resources: {mapped_values}"

    # Something unexpected happened
    else:
        logger.exception(
            f"{kjob_id}: Error encountered. Check pod logs for more details."
        )

        return (
            "Failed",
            f"{kjob_id}: Error encountered. Check pod logs for more details.",
        )


async def collect_pod_logs(tune_id: str, retry_label_lookup=True):
    """Function to retrieve logs for all containers in a pod associated with a given job name.

    Parameters
    ----------
    tune_id : str
        Tune id

    Returns
    -------
    str
        The logs from the pod's containers or an error message.
    """
    kjob_id = f"kjob-{tune_id}-job".lower()

    # Log in
    await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")

    # get pods from job
    command_get_pod = [
        "kubectl",
        "get",
        "pods",
        "--selector=job-name=" + kjob_id,
        "--output=jsonpath={.items[*].metadata.name}",
    ]

    result_pod = await run_subprocess_cmds(command=command_get_pod)

    # If a pod exists
    if result_pod and result_pod[0] != "":
        logger.debug(f"POD: {result_pod[0]} found for job {kjob_id}")
        # get logs
        command_get_logs = ["kubectl", "logs", result_pod[0], "--all-containers=true"]
        result_logs = await run_subprocess_cmds(command=command_get_logs)
        if result_logs is None:
            return

        # If there are logs
        if result_logs and result_logs[0] != "":
            logs = result_logs[0]
            logger.debug(f"Logs for {result_pod[0]} found ")

            return logs

        else:
            if retry_label_lookup is True:
                # Remove -job and -hpo suffix if present
                app_name = re.sub(r"-(job|hpo)$", "", kjob_id)
                get_by_lables_cmd = [
                    "kubectl",
                    "get",
                    "pods",
                    "-l",
                    f"app-name={app_name}",
                    "-o",
                    "name",
                ]
                result = await run_subprocess_cmds(command=get_by_lables_cmd)
                job_name = result[0].strip() if result else ""
                job_name = job_name.split("/")[-1]
                logger.info(f"kubectl retry job_name: {job_name}")
                if job_name:
                    result = await check_k8s_job_status(
                        job_name, retry_label_lookup=False
                    )
                    logger.info(f"kubectl retry result: {result}")
                    return result if result else (None, job_name)

            logger.error(f"Error getting logs: {result_logs[1].decode('utf-8')}")
            return
    else:
        logger.error(f"No POD found with the specified job name: {kjob_id}")
        logs = None
        return


async def delete_k8s_resources_by_label(label_selector: str) -> str:
    """Function to delete resources in OpenShift that match the given label selector.

    This function excludes ReplicaSets from the deletion process, as they are managed by
    higher-level controllers like Deployments, StatefulSets, and DaemonSets.


    Parameters
    ----------
    label_selector : str
        The label selector used to filter the resources to be deleted.
        i.e "app=test-lifecycle-label"

    Returns
    -------
    str
        Returns "Success" if all resources were successfully deleted or if no resources were found. Returns "Failed"
        if an error occurred during the deletion process.
    """
    # log in
    await ensure_logged_in(f"kubectl get job --namespace={settings.NAMESPACE}")

    # Define the command to get resources with the matching label + configMaps, PVCs, and Service Accounts
    get_command = [
        "kubectl",
        "get",
        "all,cm,pvc,sa",
        "-l",
        label_selector,
        "-o",
        "name",
    ]

    try:
        # Run the 'kubectl get' command and capture the output
        resources = subprocess.check_output(get_command, text=True).strip().split("\n")
        logger.debug(f"Resources to be deleted: {resources}")

        # If there are any resources, delete them
        if resources and resources[0]:
            # Remove ReplicaSets from the list as they are controlled by higher-level controllers
            filtered_resources = [
                resource for resource in resources if "replicaset" not in resource
            ]
            # Define the command to delete all remaining resources
            delete_command = ["kubectl", "delete"] + filtered_resources
            subprocess.run(delete_command, check=True)
            logger.debug(f"Successfully deleted resources: {filtered_resources}")
            return "Success", f"Successfully deleted resources: {filtered_resources}"

        else:
            logger.debug(
                f"No non-ReplicaSet resources: {resources} found with the label: {label_selector}"
            )
            # No resources found, thus the operation is considered as success
            # No resources to delete means no failure
            return (
                "Success",
                f"No non-ReplicaSet found with the label: {label_selector}",
            )

    except subprocess.CalledProcessError as e:
        logger.exception("Error occurred while deleting resources")
        return "Failed", f"Error occurred while deleting resources: {e}"
