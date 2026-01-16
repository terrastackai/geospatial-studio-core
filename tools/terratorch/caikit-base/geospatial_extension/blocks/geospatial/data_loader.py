import os
import pathlib
import shutil
from urllib.parse import urlparse
from zipfile import ZipFile

import alog
import boto3
import requests

from geospatial_extension.blocks.geospatial.exceptions import (
    GeospatialInferenceException,
)
from geospatial_extension.blocks.geospatial.settings import APPLICATION_DATA_DIR, ENVIRONMENT
from geospatial_extension.blocks.geospatial.utils import report_exception
from webhooks import webhooks


logger = alog.use_channel("<BRT_TKN_EMBD>")


def get_unique_id(inputs_folder):
    """
    Pull unique_id from inputs_folder

    Args:
        inputs_folder (str): path to inputs folder

    Output:
        str: unique_id for event tracking
    """
    return inputs_folder.split("/")[3]


def set_up_folders(unique_id, extra_details: dict = None):
    """
    Initial creation of folders

    Args:
        unique_id (str(UUID)): unique id for tracing

    Output:
        inputs_folder (str): Path to inputs folder
        outputs_folder (str): Path to outputs folder

    """
    inputs_folder = os.path.join(APPLICATION_DATA_DIR, f"{unique_id}/inputs/")
    outputs_folder = os.path.join(APPLICATION_DATA_DIR, f"{unique_id}/outputs/")
    if not os.path.exists(inputs_folder):
        try:
            os.makedirs(inputs_folder)
        except Exception as exc:
            if "Transport endpoint is not connected" in str(exc):
                report_exception(
                    unique_id,
                    1031,
                    f"S3FS error with attached COS bucket.",
                    extra_details=extra_details,
                )
            else:
                exec_type = type(exc)
                report_exception(
                    unique_id,
                    1005,
                    f"Event_id/UUID {unique_id} already in use",
                    True,
                    f"Error type: {exec_type} \n Full Stacktrace: {exc}",
                    extra_details=extra_details,
                )
            raise

    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    return inputs_folder, outputs_folder


def check_url_input(url, unique_id, extra_details: dict = None):
    """
    Run through checks for URL input

    Args:
        url (str): pre-signed url
        unique_id (str(UUID)): unique id to track request

    Output:
        filename (str): name of file
        response (requests.Response()): request response

    """
    logger.debug(f"{unique_id}: {url}")
    try:
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        response = requests.get(url)
    except:
        report_exception(
            unique_id, 1011, f"Invalid url: {url}", extra_details=extra_details
        )
        raise
    else:
        if response.status_code not in [200, 201]:
            msg = f"Unable to get data from pre-signed url. Check authentication and expiration. {url}"
            report_exception(unique_id, 1006, msg, extra_details=extra_details)
            raise GeospatialInferenceException(msg)

    logger.debug(f"{unique_id} ({response.status_code}): filename is {filename}")
    if filename.rsplit(".", 1)[-1] not in ("zip", "tif", "tiff"):
        msg = f"Invalid data type from URL. Must be .zip, .tif, or .tiff."
        report_exception(unique_id, 1012, msg, extra_details=extra_details)
        raise GeospatialInferenceException(msg)

    return filename, response


def download_pre_signed_url(filename, response, inputs_folder):
    """
    Download content from pre-signed URL

    Args:
        filename (str): name of file
        response (requests.Response()): request response
        inputs_folder (str): path to output dir

    Output:
        output_files (list(str)): list of output file paths

    """
    unique_id = get_unique_id(inputs_folder)
    if not filename.endswith(".zip"):
        file_path = (
            inputs_folder
            + unique_id
            + "_"
            + filename.replace(" ", "_").rsplit(".", 1)[0]
            + ".tif"
        )
        logger.debug(f"{unique_id}: {file_path}")
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.debug(f"{unique_id}: output files are {file_path}")
        return [file_path]
    else:
        file_path = f"{inputs_folder}{filename}"
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.debug(f"{unique_id}: filepath is {file_path}")
        new_files = []
        with ZipFile(file_path, "r") as zObject:
            items = [file for file in zObject.namelist() if "_MACOSX/" not in file]
            for item in items:
                new_files.append(item)
                zObject.extract(item, path="output/" + unique_id + "/inputs/")
                # Could extractall then copy into the same folder
        logger.debug(f"{unique_id}: zip extraction completed")
        os.system(f"rm -r {file_path}")
        logger.debug(f"{unique_id}: new_files are {new_files}")
        output_files = []
        for file in new_files:
            os.rename(
                inputs_folder + file,
                inputs_folder
                + unique_id
                + "_"
                + file.replace(" ", "_").rsplit(".", 1)[0]
                + ".tif",
            )
            output_files.append(
                inputs_folder
                + unique_id
                + "_"
                + file.replace(" ", "_").rsplit(".", 1)[0]
                + ".tif"
            )
            logger.debug(f"{unique_id}: output files are {output_files}")
        return output_files


def url_request(urls, inputs_folder, extra_details: dict = None):
    """
    Pull data from url request type

    Args:
        urls (list(str)): list of pre-signed urls
        inputs_folder (str): Path to inputs folder

    Output:
        output_files (list(str)): list of output file paths
    """
    unique_id = get_unique_id(inputs_folder)
    logger.debug(f"{unique_id}: Decoding url...")
    output_files = []
    for url in urls:
        filename, response = check_url_input(
            url, unique_id, extra_details=extra_details
        )
        new_output_files = download_pre_signed_url(filename, response, inputs_folder)
        output_files.extend(new_output_files)

    return output_files


def create_presigned_url(object_key):
    """
    Create presigned url for data output to default COS bucket
    """
    # Use the geospatial-cos bucket
    logger.debug(f"Creating pre-signed URL for object key: {object_key}")
    endpoint_url = os.environ["OUTPUT_BUCKET_LOCATION"]
    # 'https://s3.us-south.cloud-object-storage.appdomain.cloud'  # Replace with the actual endpoint URL

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["gfm-inference-outputs-cos-access-key"],
        aws_secret_access_key=os.environ["gfm-inference-outputs-cos-secret-key"],
        endpoint_url=endpoint_url,
        verify=(ENVIRONMENT.lower() != "local"),
    )
    bucket_name = os.environ["OUTPUT_BUCKET"]
    params = {"Bucket": bucket_name, "Key": object_key}
    # Generate the pre-signed URL
    output_url = s3.generate_presigned_url("get_object", Params=params, ExpiresIn=43200)
    logger.debug(f"Output url: {output_url}")
    return output_url


def zip_inference_data(outputs_folder, inputs_folder, details: dict = None):
    """
    Zip data from inference run and upload to COS buckets

    Args:
        outputs_folder (str): path to outputs folder
        inputs_folder (str): path to inputs folder

    Output:
        zip_location (str): path to zip folder

    """
    # Remove lulc file:
    try:
        os.remove(outputs_folder + "lulc.tif")
    except:
        logger.debug("No lulc tile to delete")

    event_id = get_unique_id(inputs_folder)
    # zip the folder and move to completed location - should look like /app/output/completed/uuid
    webhooks.notify_gfmaas_ui(
        event_id, "Moving data to COS bucket", event_details=details
    )
    logger.debug(f"{event_id}: Moving data to COS bucket")
    zip_location = os.path.join(APPLICATION_DATA_DIR, f"completed/{event_id}.zip")
    root_dir = inputs_folder.replace("inputs/", "")
    directory = pathlib.Path(root_dir)
    with ZipFile(zip_location, mode="w") as archive:
        for file_path in directory.rglob("*"):
            archive.write(file_path, arcname=file_path.relative_to(directory))

    # Remove inputs and outputs folder now that they are saved in a zip
    try:
        shutil.rmtree(inputs_folder.replace("inputs/", ""))
    except Exception as ex:
        logger.debug(f"{event_id}: Failed to delete input/output dir: (%s)", ex)

    return zip_location


def save_to_output_url(zip_location, output_url, inputs_folder, details: dict = None):
    """
    Save inference data to provided output_url or generate pre-signed url for internal COS bucket

    Args:
        zip_location (str): path to zip folder
        output_url (str): pre-signed url provided with inference request
        inputs_folder (str): path to inputs folder

    Output:
        output_url (str): pre-signed url to access inference data
    """
    event_id = get_unique_id(inputs_folder)
    # push to output_url location and return output url
    uploaded = False
    if output_url:
        logger.debug(f"{event_id} - Uploading zip file to output-url: {zip_location}")
        try:
            with open(zip_location, "rb") as file:
                requests.request(
                    "PUT",
                    output_url,
                    data=file,
                    headers={"Content-Type": "application/octet-stream"},
                )
            uploaded = True
        except:
            report_exception(
                event_id, 1004, f"COS bucket upload error", extra_details=details
            )
            logger.exception("COS bucket upload error")
    # otherwise return the output url from the default COS bucket
    if (not output_url) or (not uploaded):
        webhooks.notify_gfmaas_ui(
            event_id, "Uploading to output pre-signed URL", event_details=details
        )
        output_url = create_presigned_url(
            object_key=f"completed/{inputs_folder.split('/')[3]}.zip"
        )

    return output_url
