# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import time
import os, glob
import shutil

from gfm_data_processing.common import logger


def read_json_with_retries(filepath, max_retries=5, base_delay=0.5):
    """
    Tries to read a JSON file with retries in case of a JSONDecodeError.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.info(f"Attempt {attempt}: Failed to decode JSON - {e}")
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))  # Exponential backoff
        except FileNotFoundError as e:
            logger.info(f"Attempt {attempt}: File not found - {e}")
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))

def make_tmp_dir(path: str, tmp_dir:str):
    """Make tmp directory

    Parameters
    ----------
    path : str
        Original path
    tmp_dir : str
        new dir

    Returns
    -------
    str
        Path to tmp_dir
    """
    new_path = os.path.join(path, tmp_dir)
    os.makedirs(new_path, exist_ok=True)

    return new_path

def copy_tiffs(search_key: str, old_path:str, new_path:str):
    """Function to copy tiffs to a tmp folder

    Parameters
    ----------
    search_key : str
        Key to search tiff to copy
    old_path : str
        Old path where the tiff lives
    new_path : str
        New path to copy the tiff
    """
    for filename in os.listdir(old_path):
        if search_key in filename.lower() and filename.lower().endswith(("tif","tiff")):
            # Build full paths for the tiffs
            old_path = os.path.join(old_path, filename)
            new_path = os.path.join(new_path, filename)
            logger.info(f"Copying Image from {old_path} to {new_path}")
            shutil.copy2(src=old_path, dst=new_path)

def delete_tmp_dir(path:str):
    """Function to delete tmp dir if it exists

    Parameters
    ----------
    path : str
        Path to tmp_dir
    """
    if os.path.exists(path):
        ## Delete the tmp folder after inference
        shutil.rmtree(path)

def upload_prediction_tiff(output_folder, task_folder):
    s3_client = s3()

    instance = "geospatial-storage-example-data"
    bucket = "test-geo-inference-pipelines"
    predicted_file_path = glob.glob(f"{output_folder}/*_pred.tif")[0]
    basename = os.path.basename(predicted_file_path)
    cos_file_path = f"{task_folder}/{basename}"
    cos_file_path_updated = os.path.relpath(cos_file_path, "/data/")
    logger.info(
        f" Local file path: {predicted_file_path} .  \
            Cos file path {cos_file_path} ,\
        Cos file path updated with /data/ removed : {cos_file_path_updated} "
    )

    s3_client.upload_file(Filename=predicted_file_path, Bucket=bucket, Key=cos_file_path_updated)
    logger.info(" Uploaded file ")


def s3():
    return boto3.client(
        "s3",
        aws_access_key_id="",
        aws_secret_access_key="",
        endpoint_url="https://s3.us-east.cloud-object-storage.appdomain.cloud",
        config=Config(signature_version="s3v4"),
        region_name="us-east",
    )


def wait_for_file(
    task_folder,
    timeout=120,
    interval=15,
):
    """
    Waits until a specific file (key) exists in the given bucket.
    Retries every `interval` seconds, up to `timeout` seconds.
    """
    s3_client = s3()
    bucket = "test-geo-inference-pipelines"
    predicted_file_path = glob.glob(f"{task_folder}/*_pred.tif")[0]
    predicted_file_path_updated = os.path.relpath(predicted_file_path, "/data/")

    elapsed = 0
    while elapsed < timeout:
        try:
            s3_client.head_object(Bucket=bucket, Key=predicted_file_path_updated)
            print(f" File '{predicted_file_path_updated}' is now available in bucket '{bucket}'")
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f" File not yet found, retrying in {interval}s...")
            else:
                print(f" Unexpected error: {e}")
                return False
        time.sleep(interval)
        elapsed += interval

    print(f" Timeout reached. File '{predicted_file_path_updated}' not found in '{bucket}' after {timeout} seconds.")
    return False
