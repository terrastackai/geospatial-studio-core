# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import re
import requests
from zipfile import ZipFile
from urllib.parse import urlparse
from gfm_data_processing.exceptions import GfmDataProcessingException
from gfm_data_processing.common import get_unique_id, logger, report_exception


def check_url_input(url, task_id, inference_id):
    """
    Run through checks for URL input

    Args:
        url (str): pre-signed url
        task_id (str(UUID)): unique id to track request

    Output:
        filename (str): name of file
        response (requests.Response()): request response

    """
    logger.debug(f"{task_id}: {url}")
    try:
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        response = requests.get(url)
    except:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1011",
            message=f"Preprocessing error: Invalid url: {url}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=True,
        )
    if response.status_code not in [200, 201]:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1006",
            message=f"Preprocessing error: Unable to get data from pre-signed url. Check authentication and expiration. {url}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=True,
        )

    logger.debug(f"{task_id}: {response.status_code}")
    logger.debug(f"{task_id}: filename is {filename}")
    if filename.rsplit(".", 1)[-1] not in ("zip", "tif", "tiff", "gpkg", "nc"):
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1012",
            message="Preprocessing error: Invalid data type from URL. Must be .zip, .tif, or .tiff.",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=True,
        )

    return filename, response


def has_all_mandatory_shapefile_ext(files):
    mandatory_extensions = {'.shp', '.shx', '.dbf'}
    found_extensions = {ext for filename in files for ext in mandatory_extensions if filename.lower().endswith(ext)}
    
    return mandatory_extensions.issubset(found_extensions)


def download_pre_signed_url(filename, response, date, inputs_folder):
    """
    Download content from pre-signed URL

    Args:
        filename (str): name of file
        response (requests.Response()): request response
        date (str): date when image was captured
        inputs_folder (str): path to output dir

    Output:
        output_files (list(str)): list of output file paths

    """
    unique_id = get_unique_id(inputs_folder)
    layer_type = filename.rsplit(".", 1)[-1]
    layer_type = layer_type.replace("tiff", "tif")
    regex = re.compile(".*[0-9]{4}-[0-9]{2}-[0-9]{2}")
    if layer_type != "zip":
        if layer_type == "tif" and not date and not regex.match(filename):
            raise GfmDataProcessingException("date element missing from tif file and date was not passed")
        date = "" if regex.match(filename) else date
        file_path = inputs_folder + unique_id + "_" + date + "_" + filename.replace(" ", "_").rsplit(".", 1)[0] + "." + layer_type
        file_path = file_path.replace("__", "_")
        logger.debug(f"{unique_id}: {file_path}")
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.debug(f"{unique_id}: output files are {file_path}")
        return [file_path]
        # output_files.append(file_path)
    else:
        file_path = inputs_folder + filename
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.debug(f"{unique_id}: filepath is {file_path}")
        new_files = []
        # try:
        with ZipFile(file_path, "r") as zObject:
            items = [file for file in zObject.namelist() if ("_MACOSX/" not in file and ".DS_Store" not in file)]
            if (has_all_mandatory_shapefile_ext(items)):
                return [file_path]
            for item in items:
                if any(tif in item for tif in [".tiff", ".tif"]) and (regex.match(item)):
                    new_files.append(item)
                    zObject.extract(item, path=inputs_folder)
                else:
                    logger.warning(f"{item}: File name lacks a valid date regex. Example date format is '2025-08-16'")
                    
        logger.debug(f"{unique_id}: zip extraction completed")
        os.system(f"rm -r {file_path}")
        logger.debug(f"{unique_id}: new_files are {new_files}")
        output_files = []
        # Add unique id:
        for file in new_files:
            os.rename(
                inputs_folder + file,
                inputs_folder + unique_id + "_" + file.rsplit("/", 1)[1].replace(" ", "_").rsplit(".", 1)[0] + ".tif",
            )
            output_files.append(inputs_folder + unique_id + "_" + file.rsplit("/", 1)[1].replace(" ", "_").rsplit(".", 1)[0] + ".tif")
            logger.debug(f"{unique_id}: output files are {output_files}")
        return output_files
