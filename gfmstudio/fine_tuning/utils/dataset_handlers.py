# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import os
import subprocess
from datetime import datetime

import rasterio
from remotezip import RemoteZip

from gfmstudio.config import settings
from gfmstudio.fine_tuning.core import object_storage
from gfmstudio.fine_tuning.utils.dataset_errors import error_stages, known_errors
from gfmstudio.log import logger


def list_zipped_files(url, just_tif=True):
    """list zipped files from an url without downloading the .zip"""
    with RemoteZip(url) as zip:
        il = zip.infolist()
    if just_tif is True:
        return [
            X.filename
            for X in il
            if (X.filename[-4:] == ".tif") | (X.filename[-5:] == ".tiff")
        ]
    else:
        return [X.filename for X in il]


def extract_bands_from(band_descriptions: list) -> list:
    """extract bands from a list of band descriptions taken from 'obtain_band_descriptions_from'

    Parameters
    ----------
    band_descriptions : list
        band_descriptions

    Returns
    -------
    bands: list
        a list with the band indices
    """
    c_bands = []

    for band in range(0, len(band_descriptions)):
        band_dict = {"id": str(band)}
        if band_descriptions[band] is not None:
            band_dict["value"] = band_descriptions[band]
        c_bands = c_bands + [band_dict]
    return c_bands


def obtain_band_descriptions_from(dataset_url: str, sample_data: str):
    """extract band description from remote .zip url

    Parameters
    ----------
    dataset_url : str
        remote .zip url to access the .zip file

    sample_data : str
        path to the particular data file within the .zip

    Returns
    -------
    bands: list
        a list band indices and descriptions
    """
    with RemoteZip(dataset_url) as zip:
        sample_image = zip.read(sample_data)

    with rasterio.MemoryFile(sample_image) as memfile:
        with memfile.open() as src:
            band_descriptions = src.descriptions

    return band_descriptions


def data_and_label_match(
    data_files: list, training_data_suffix: str, label_files: list, label_suffix: str
):
    """verify that the training data files and label files match

    Parameters
    ----------
    data_files : list
        list of all training data files

    training_data_suffix : str
        suffix of the training data files

    label_files : list
        list of all label data files


    label_suffix : str
        suffix of the label data files

    Returns
    -------
    _ : bool
        True if the files match, False otherwise
    """
    data_stems = sorted(
        [X.replace(training_data_suffix, "").split("/")[-1] for X in data_files]
    )
    label_stems = sorted(
        [X.replace(label_suffix, "").split("/")[-1] for X in label_files]
    )

    return data_stems == label_stems


def transform_error_message(error_code: str, error_message: str) -> str:
    """transform the error message to be more user-friendly

    Parameters
    ----------
    error_code : str
        error code which indicates from which stage an error has occurred during the onboarding workflow

    error_message : str
        the original error message returned by the workflow or the system

    Returns
    -------
    transformed_error_message: str
        the transformed user-friendly error message
    """
    if error_code == "0000":
        return "N/A"
    transformed_error_message = ""
    for key, value in known_errors.get(error_code, {}).items():
        if key in error_message:
            transformed_error_message = (
                error_message if value == "keep original" else value
            )
    if transformed_error_message == "":
        transformed_error_message = (
            "New error - "
            + error_stages.get(error_code)
            + ": "
            + error_message
            + ". Please notify the admin."
        )
    return transformed_error_message


def validate_and_transform_data_sources(data_sources: list) -> str:
    """Verify the data_sources which the user has provided is valid and
    transform it into the format the pipeline expects

    Parameters
    ----------
    data_sources : list
        the list of multimodal data sources which the passed in via the payload

    Returns
    -------
    data_sources: string
        a string which could be converted into json once it's in the pipeline
    """
    for data_source in data_sources:
        if "bands" not in data_source:
            raise ValueError("bands is a required field for a data source")
        if "modality_tag" not in data_source:
            raise ValueError("modality_tag is a required field for a data source")
        if "file_suffix" not in data_source:
            raise ValueError("file_suffix is a required field for a data source")
    return json.dumps(data_sources)


def validate_and_transform_options(options: dict) -> str:
    """verify the options which the user has provided is valid and transform it into the format the pipeline expects

    Parameters
    ----------
    options : dict
        the dict of onboarding options passed in via the payload

    Returns
    -------
    options: string
        a string which could be converted into json once it's in the pipeline
    """
    return json.dumps(options)


def make_k8s_secret_literal(original: str) -> str:
    """make k8s secret a literal for a kubectl command"""
    return f"'{original}'"


def capture_and_upload_job_log(dataset_id: str, version: str):
    """capture and upload k8s job log

    Parameters
    ----------
    dataset_id : str
        the dataset_id for the corresponding job log we'd like to upload

    version : str
        the pipeline version
    """
    try:
        job_name = f"onboarding-{version}-pipeline-{dataset_id}"
        pod_name = obtain_pod_name(job_name=job_name)
        local_log_path = obtain_job_log(pod_name, dataset_id)
        cos_log_path = upload_log_to_cos(local_log_path, dataset_id)
        remove_local_file(local_log_path)
    except Exception:
        logger.exception(f"Failed to upload logs to COS for dataset {dataset_id}")

    return cos_log_path


def remove_local_file(filepath: str):
    """remove a file given its filepath"""
    remove_file_command = f"rm {filepath}"
    try:
        remove_file_output = subprocess.check_output(remove_file_command, shell=True)
        logger.info(remove_file_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        logger.exception(error_message)


def obtain_pod_name(job_name: str) -> str:
    """given the job name for a pipeline, obtain the corresponding pod name

    Parameters
    ----------
    job_name : str
        the name of the job which is running the onboarding workflow

    Returns
    -------
    pod_name: str
        the name of the pod which is running the onboarding workflow
    """
    obtain_pod_command = f"kubectl get pods --selector=job-name={job_name} | sed -n '2 p' | sed 's/ .*//'"
    try:
        obtain_pod_output = subprocess.check_output(obtain_pod_command, shell=True)
        logger.info(obtain_pod_output)
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output).decode("utf-8")
        logger.exception(
            f"Unable to obtain the pod name.  Error message - {error_message}"
        )
    pod_name = obtain_pod_output.decode("utf-8").strip()
    return pod_name


def obtain_job_log(pod_name: str, dataset_id: str) -> str:
    """save job log to a local file

    Parameters
    ----------
    pod_name : str
        the name for the pod from which we'd like to capture logs

    dataset_id : str
        the dataset id for the dataset that's being onboarded

    Returns
    -------
    destination: str
        the filepath at which the job log is stored
    """
    logs_dir = "deployment/logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = f"{logs_dir}/{dataset_id}.log"
    try:
        with open(log_path, "w") as f:
            subprocess.run(
                ["kubectl", "logs", pod_name],
                check=True,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
            )
        logger.info(f"Saved pod logs to {log_path}")
    except subprocess.CalledProcessError as exc:
        error_message = str(exc.output)
        logger.exception(
            f"Unable to save the pod log for pod {pod_name}.  Error message - {error_message}"
        )
    return log_path


def upload_log_to_cos(source_path: str, dataset_id: str):
    """upload locally stored log to the COS bucket which stores dataset-factory logs

    Parameters
    ----------
    source_path : str
        local filepath which stores the job log

    dataset_id : str
        the dataset Id for the onboarded dataset
    """
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        s3 = object_storage.object_storage_client()
        bucket_name = settings.DATASET_FILES_BUCKET
        cos_destination_path = f"dflogs/{current_date}/{dataset_id}.log"
        s3.upload_file(source_path, bucket_name, cos_destination_path)
    except Exception as e:
        logger.exception(
            f"Error occurred when pushing logs for {dataset_id} to COS: {e}"
        )

    return cos_destination_path
