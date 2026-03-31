# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""
Curated upload operator for dataset onboarding.

This module handles the complete dataset onboarding pipeline including:
- Dataset download and extraction
- Image and label validation
- COG (Cloud Optimized GeoTIFF) conversion
- Train/test/val split generation
- Statistics calculation (mean/std for normalization)
- Upload to object storage

Requirements:
    pip install rio-cogeo wget scikit-learn>=1.3.0 rasterio boto3==1.35.82
    botocore numpy>=1.22.2 tqdm requests humanize
"""

import collections
import datetime
import glob
import json
import logging
import os
import re
import ssl
import subprocess
import sys
import tarfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from multiprocessing import Pool, Process, Queue, cpu_count
from pathlib import Path

import boto3
import humanize
import numpy as np
import rasterio
import requests
import wget
from botocore.client import Config
from rio_cogeo.cogeo import cog_validate
from sklearn.model_selection import train_test_split
from terrakit.chip.tiling import chip_and_label_data
from terrakit.download.download_data import download_data
from terrakit.transform.labels import process_labels
from tqdm import tqdm


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """
    Configure logging with proper formatting and level.
    
    This function sets up logging to handle both command-line parameters
    and environment variables for log level configuration.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from environment (default to INFO)
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Create module logger
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(log_level)
    
    return module_logger


# Initialize logging
logger = setup_logging()


# ============================================================================
# COMMAND-LINE PARAMETER PROCESSING
# ============================================================================

def process_command_line_parameters():
    """
    Process command-line parameters and set them as environment variables.
    
    Parses command-line arguments in the format key=value and sets them
    as environment variables for use throughout the script.
    """
    # Pattern to match valid parameter format: KEY=value
    param_pattern = re.compile(r"[A-Za-z0-9_]*=[.\/A-Za-z0-9]*")
    
    # Filter and extract parameters
    parameters = [
        arg for arg in sys.argv
        if "=" in arg and param_pattern.match(arg)
    ]
    
    # Set parameters as environment variables
    for parameter in parameters:
        key, value = parameter.split("=", 1)
        logger.info(f'Setting parameter: {key} = "{value}"')
        os.environ[key] = value
    
    # Update log level if specified
    new_log_level = os.environ.get("log_level", "").upper()
    if new_log_level and new_log_level != logger.level:
        logger.info(f"Updating log level to {new_log_level}")
        logging.getLogger().setLevel(new_log_level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(new_log_level)


# Process command-line parameters
process_command_line_parameters()


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Performance tuning constants
MAX_UPLOAD_WORKERS = int(os.getenv("MAX_UPLOAD_WORKERS", "10"))
MAX_COG_WORKERS = int(os.getenv("MAX_COG_WORKERS", str(max(1, cpu_count() - 1))))
CHUNK_SIZE = int(os.getenv("RASTER_CHUNK_SIZE", "1024"))

logger.info(f"Performance settings: MAX_UPLOAD_WORKERS={MAX_UPLOAD_WORKERS}, "
            f"MAX_COG_WORKERS={MAX_COG_WORKERS}, CHUNK_SIZE={CHUNK_SIZE}")


# ============================================================================
# CONFIGURATION
# ============================================================================

# API configuration
df_api_route = os.getenv(
    "df_api_route",
    "https://geoft-dataset-factory-api-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/",
)
df_api_key = os.getenv("DF_APIKEY", "some-api-key")

# Dataset configuration
dataset_url = os.getenv(
    "dataset_url",
    "https://ibm.box.com/shared/static/jaqwlc4hgg734xxum9mrhv5rcdn9cb7g.zip",
)
dataset_id = os.getenv("dataset_id", "geodata-someuuid")
label_suffix = os.getenv("label_suffix", ".mask.tif")
data_sources = os.getenv("data_sources", "{}")
onboarding_options = os.getenv("onboarding_options", "'{}'")

# Payload for tracking
payload = {
    "dataset_url": dataset_url,
    "label_suffix": label_suffix,
    "dataset_id": dataset_id,
    "data_sources": data_sources,
    "onboarding_options": onboarding_options,
}

# Error tracking
error = {"code": "0000", "message": "N/A"}

logger.info("Configuration loaded successfully")
logger.debug(f"Dataset ID: {dataset_id}")
logger.debug(f"API Route: {df_api_route}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def object_storage_client():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("OBJECT_STORAGE_ENDPOINT", ""),
        aws_access_key_id=os.getenv("OBJECT_STORAGE_KEY_ID", ""),
        aws_secret_access_key=os.getenv("OBJECT_STORAGE_SEC_KEY", ""),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("OBJECT_STORAGE_REGION", ""),
    )
    return s3


if "notifications" in df_api_route:
    df_webhooks_url = df_api_route
else:
    df_webhooks_url = df_api_route + "v2/webhooks"
df_webhooks_headers = {
    "Content-Type": "application/json",
    "X-API-KEY": df_api_key,
}


def notify_df_api(onboarding_details: dict = None):
    """Helper method to notify dataset-factory API.
    
    Args:
        onboarding_details: Dict containing dataset_id and status.
            For success, must also include "size" and "training_params".
            
    Returns:
        None
    """
    if onboarding_details is None:
        logger.error("onboarding_details cannot be None")
        return
    logger.info("Notify the dataset-factory API onboarding status")
    event_data = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "detail": onboarding_details,
        "detail_type": "FT:Data:Onboarding",
        "source": "com.ibm.dataset-factory-onboarding",
    }
    if "notifications" in df_webhooks_url and onboarding_details["status"] in [
        "Succeeded",
        "Failed",
    ]:
        event_data.update(
            {
                "detail_type": "FT:Data:Finished",
            }
        )

    try:
        response = requests.post(
            url=df_webhooks_url,
            headers=df_webhooks_headers,
            json=event_data,
            timeout=29,
            verify=False,
        )
        if response.status_code not in (200, 201):
            logger.error(
                "Failed to send task status. Reason: (%s)> %s",
                response.status_code,
                response.text,
                stack_info=True,
            )
        else:
            logger.info("Sent a notification to dataset-factory api")
    except Exception as ex:
        logger.error(
            "Failed to send task status. Reason: (%s)",
            ex,
            stack_info=True,
        )


default_error = {"code": "0000", "message": "N/A"}


def populate_onboarding_details(
    dataset_id: str,
    status: str,
    onboarding_details: dict = None,
    size: str = None,
    training_params: list = None,
    error: dict = None,
):
    """Populate onboarding details dictionary.
    
    Args:
        dataset_id: Dataset identifier
        status: Onboarding status (Succeeded/Failed/Onboarding)
        onboarding_details: Dict to populate (created if None)
        size: Dataset size string
        training_params: List of training parameters per modality
        error: Error dict with code and message
        
    Returns:
        None (modifies onboarding_details in place)
    """
    logger.info("Populate onboarding details to send back to the API webhook")
    
    if onboarding_details is None:
        onboarding_details = {}
    if error is None:
        error = default_error
        
    try:
        onboarding_details["dataset_id"] = dataset_id
        onboarding_details["status"] = status
        onboarding_details["error_code"] = error["code"]
        onboarding_details["error_message"] = error["message"]
        if status == "Succeeded":
            onboarding_details["training_params"] = {}
            onboarding_details["size"] = size
            stages = ["train", "test", "val"]
            for stage in stages:
                onboarding_details["training_params"][stage + "_split_path"] = (
                    "/" + dataset_id + "/split_files/" + stage + "_data.txt"
                )
            for single_modal_param in training_params:
                modality_tag = single_modal_param["modality_tag"]
                norm_means = single_modal_param["norm_means"]
                norm_stds = single_modal_param["norm_stds"]
                bands = single_modal_param["bands"]
                onboarding_details["training_params"][modality_tag] = {}
                onboarding_details["training_params"][modality_tag][
                    "norm_means"
                ] = norm_means
                onboarding_details["training_params"][modality_tag][
                    "norm_stds"
                ] = norm_stds
                onboarding_details["training_params"][modality_tag]["bands"] = bands
                onboarding_details["training_params"][modality_tag]["file_suffix"] = (
                    "*" + single_modal_param["file_suffix"]
                )
                for stage in stages:
                    onboarding_details["training_params"][modality_tag][
                        stage + "_data_dir"
                    ] = ("/" + dataset_id + "/training_data/" + modality_tag + "/")
                    onboarding_details["training_params"][modality_tag][
                        stage + "_labels_dir"
                    ] = ("/" + dataset_id + "/labels/")
    except Exception as e:
        error["code"] = "0010"
        logger.error(
            "Error occurred when populating onboarding details.  Error details: ",
            e,
            stack_info=True,
        )
        raise e


def obtain_file_stem(filepaths: list, suffix: str) -> list:
    """Extract file stems (names without suffix) from file paths.
    
    Args:
        filepaths: List of file paths
        suffix: File suffix to remove
        
    Returns:
        List of file stems
        
    Raises:
        Exception: If file stem extraction fails
    """
    logger.info(f"Extracting file stems for {len(filepaths)} files")
    try:
        if not filepaths:
            logger.warning("Empty file list provided")
            return []
        return [Path(filepath).name.replace(suffix, "") for filepath in filepaths]
    except Exception as e:
        error["code"] = "0004"
        logger.error(
            f"Error processing file stems: {e}",
            stack_info=True,
        )
        raise e


def create_and_save_split_files(
    local_source_path: str,
    label_splits: tuple,
    label_suffix: str,
):
    Path(local_source_path + "/split_files/").mkdir(parents=True, exist_ok=True)
    for stage, file_list in zip(["train", "test", "val"], label_splits):
        with open(local_source_path + "/split_files/" + stage + "_data.txt", "w") as fp:
            for X in sorted(file_list):
                stem = X.replace(label_suffix, "").split("/")[-1]
                fp.write(stem + "\n")


def create_and_upload_split_files(
    s3,
    bucket_name,
    local_source_path: str,
    cos_destination_path: str,
    label_splits: tuple,
    label_suffix: str,
):
    logger.info("Create split files and upload to COS")
    try:
        for stage, file_list in zip(["train", "test", "val"], label_splits):
            with open(local_source_path + "/" + stage + "_data.txt", "w") as fp:
                for X in sorted(file_list):
                    stem = X.replace(label_suffix, "").split("/")[-1]
                    fp.write(stem + "\n")

            response = s3.upload_file(
                local_source_path + "/" + stage + "_data.txt",
                bucket_name,
                cos_destination_path + "/split_files/" + stage + "_data.txt",
            )

    except Exception as e:
        error["code"] = "0006"
        logger.error(
            "An error occurred when uploading the split files. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def download_dataset(source_url: str, destination: str):
    logger.info(f"Downloading from {source_url}")
    try:
        if not os.path.exists(destination):
            os.makedirs(destination)
        ssl._create_default_https_context = ssl._create_unverified_context
        filename = wget.download(source_url, out=destination)  # might not be a zip here
        if zipfile.is_zipfile(filename):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(destination)
        elif tarfile.is_tarfile(filename):
            tar = tarfile.open(filename)
            tar.extractall(destination)
            tar.close()
        else:
            error["code"] = "0001"
            logger.error(
                "File type is unaccepted.  Please provide an url to a .zip or tar ball.  Error details: ",
                e,
                stack_info=True,
            )
    except Exception as e:
        error["code"] = "0001"
        logger.error(
            "Exception occurred when downloading the dataset. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def prepare_dataset(working_dir):
    logger.info("Processing labels to dataset")
    try:
        process_labels(
            dataset_name=payload["dataset_id"],
            working_dir=working_dir,
            labels_folder=working_dir,
        )
        queried_data = download_data(
            dataset_name=payload["dataset_id"],
            working_dir=working_dir,
        )
        chip_and_label_data(
            dataset_name=payload["dataset_id"],
            working_dir=working_dir,
            queried_data=queried_data,
            chip_label_suffix=label_suffix,
            keep_files=False,
        )
    except Exception as e:
        error["code"] = "0012"
        logger.error(
            "Exception occurred when preparing dataset. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def find_and_sort_from(filepath: str, suffix: str):
    logger.info("Sorting files")
    try:
        sorted_files = sorted(
            glob.glob(
                filepath + "/**/*" + suffix,
                recursive=True,
            )
        )
        return sorted_files
    except Exception as e:
        error["code"] = "0002"
        logger.error(
            "Exception occured when finding and sorting the images. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def _process_single_cog(filepath: str, dest: str) -> tuple:
    """
    Process a single file for COG validation and conversion.
    
    Args:
        filepath: Path to the file to process
        dest: Destination directory
        
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        filename = Path(filepath).name
        is_valid, errors, warnings = cog_validate(filepath)
        
        if not is_valid:
            if errors:
                logger.debug(f"COG validation errors for {filename}: {errors}")
            
            logger.info(f"Converting {filename} to COG format")
            
            # Convert to COG
            cog_output = filepath + ".cog.tif"
            subprocess.check_output(
                f"rio cogeo create --cog-profile lzw --use-cog-driver {filepath} {cog_output}",
                shell=True,
            )
            os.remove(filepath)
            os.rename(cog_output, filepath)
        
        # Move to destination
        dest_path = os.path.join(dest, filename)
        os.rename(filepath, dest_path)
        
        return (True, None)
        
    except Exception as e:
        error_msg = f"Error processing {filepath}: {str(e)}"
        logger.error(error_msg)
        return (False, error_msg)


def cog_validation_and_sort(filepaths: list, dest: str):
    """
    Validate and convert files to COG format using parallel processing.
    
    Args:
        filepaths: List of file paths to process
        dest: Destination directory
        
    Raises:
        RuntimeError: If COG processing fails for any files
    """
    logger.info(f"Validating {len(filepaths)} files for COG format using {MAX_COG_WORKERS} workers")
    
    try:
        os.makedirs(dest, exist_ok=True)
        
        # Use multiprocessing pool for parallel processing
        with Pool(processes=MAX_COG_WORKERS) as pool:
            # Create tasks
            tasks = [(filepath, dest) for filepath in filepaths]
            
            # Process files in parallel with progress bar
            results = []
            for filepath in filepaths:
                result = pool.apply_async(_process_single_cog, (filepath, dest))
                results.append((filepath, result))
            
            # Collect results with progress bar
            failed_files = []
            for filepath, result in tqdm(results, desc="Processing COGs"):
                success, error_msg = result.get()
                if not success:
                    failed_files.append((filepath, error_msg))
            
            if failed_files:
                logger.error(f"{len(failed_files)} files failed COG processing")
                for filepath, error_msg in failed_files[:5]:  # Show first 5 errors
                    logger.error(error_msg)
                error["code"] = "0008"
                raise RuntimeError(f"{len(failed_files)} files failed COG processing")
        
        logger.info("COG validation and conversion completed successfully")
        
    except Exception as e:
        error["code"] = "0008"
        logger.error(f"Error during COG validation: {e}", stack_info=True)
        raise e


def save_file(dest: str, filepath: str):
    try:
        logger.info(
            "Saving "
            + filepath.split("/")[-1]
            + " to the datasets split directory: "
            + dest
        )
        os.rename(filepath, dest + "/" + filepath.split("/")[-1])
    except Exception as e:
        error["code"] = "0012"
        logger.error(
            f"Error occurred when saving files to {dest}.  Error Details: ",
            e,
            stack_info=True,
        )
        raise e


def upload_labels(bucket_name: str, label_files: list, dataset_id: str):
    logger.info("Uploading labels")
    try:
        s3 = object_storage_client()
        for f in tqdm(label_files):
            logger.info("Uploading " + f.split("/")[-1] + " to the datasets COS")
            response = s3.upload_file(
                f, bucket_name, dataset_id + "/labels/" + f.split("/")[-1]
            )
    except Exception as e:
        error["code"] = "0012"
        logger.error(
            "Error occurred when uploading labels to COS.  Error Details: ",
            e,
            stack_info=True,
        )
        raise e


def upload_images(bucket_name, image_files: list, dataset_id: str, modality_tag: str):
    logger.info("Uploading images for modality - " + modality_tag)
    try:
        s3 = object_storage_client()
        for f in tqdm(image_files):
            logger.info("Uploading " + f.split("/")[-1] + " to the datasets COS")
            response = s3.upload_file(
                f,
                bucket_name,
                dataset_id + "/training_data/" + modality_tag + "/" + f.split("/")[-1],
            )

    except Exception as e:
        error["code"] = "0009"
        logger.error(
            "Error occurred when uploading training image for modality - "
            + modality_tag
            + " - to COS.  Error Details: ",
            e,
            stack_info=True,
        )
        raise e


def find_total_size(files: list) -> str:
    logger.info("Calculating total dataset size")
    size = 0
    for file in files:
        size += os.path.getsize(file)
    return humanize.naturalsize(size)


def cleanup_image(image: np.ndarray, bands: list) -> bool:
    logger.info("Cleaning up missing values from dataset")
    image[image <= -9999] = np.nan
    mean = np.longdouble(np.nanmean(image, axis=(1, 2)))
    if True in np.isnan(mean):
        return False
    for band in bands:
        np.nan_to_num(image[band], nan=mean[band], copy=False)
    return True


def run_paralleled_processes(processes: list) -> bool:
    for p in processes:
        p.start()
    hasFailed = False
    hasCompleted = False
    while not hasFailed and not hasCompleted:
        completed = []
        for p in processes:
            if p.exitcode is not None and p.exitcode != 0:
                print(f"PROCESS {p}'S EXIT CODE IS {p.exitcode}")
                hasFailed = True
            if p.exitcode is not None:
                completed.append(p)
        if hasFailed:
            for p in processes:
                p.terminate()
            hasCompleted = True
        if len(completed) == len(processes):
            break
    for p in processes:
        p.join()
    return not hasFailed


def find_mean_and_std(raw_bands: str, training_images: list) -> tuple:
    logger.info("Calculating training parameters")
    try:
        bands = [int(X["index"]) for X in raw_bands]
        training_data_size = len(training_images)
        sums = [None] * training_data_size
        sums_sqs = [None] * training_data_size
        count = 0
        for path in tqdm(training_images):
            with rasterio.open(path) as src:
                image = np.longdouble(src.read()[bands, :, :])
                is_successful = cleanup_image(image=image, bands=bands)
                if not is_successful:
                    continue
                sums[count] = np.longdouble(image.sum(axis=(1, 2)))
                sums_sqs[count] = (np.longdouble(image) ** 2).sum(axis=(1, 2))
                count += 1
        sums = [x for x in sums if x is not None]
        sums_sqs = [x for x in sums_sqs if x is not None]
        total_sum = sum(sums)
        total_sum_sqs = sum(sums_sqs)
        pixel_count = count * image.shape[1] * image.shape[2]
        total_mean = np.float64(total_sum / pixel_count)
        total_var = (total_sum_sqs / pixel_count) - (total_mean**2)
        total_std = np.float64(np.sqrt(total_var))
        return (total_mean, total_std, bands)
    except Exception as e:
        error["code"] = "0007"
        logger.error(
            "An error occurred when calculating training parameters. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def create_file_splits(
    split_weights: tuple, label_files: list, image_file_lists: list
) -> tuple:
    logger.info("Creating split files")
    if sum(split_weights) != 1:
        error["code"] = "0005"
        raise ValueError(
            "The split provide isn't valid, because the weights don't add up to 1."
        )
    try:
        image_lists_ndarray = np.array(image_file_lists)
        image_lists_ndarray_trans = np.transpose(image_lists_ndarray)
        train_size, test_size, val_size = split_weights
        test_val_size = 1 - train_size
        # calculate random splits
        (
            x_train_files,
            x_test_val_files,
            y_train_files,
            y_test_val_files,
        ) = train_test_split(
            image_lists_ndarray_trans,
            label_files,
            train_size=train_size,
            test_size=test_val_size,
            random_state=0,
        )
        intermediate_val_size = val_size / test_val_size
        intermediate_test_size = test_size / test_val_size
        x_test_files, x_val_files, y_test_files, y_val_files = train_test_split(
            x_test_val_files,
            y_test_val_files,
            train_size=intermediate_val_size,
            test_size=intermediate_test_size,
            random_state=0,
        )
        x_train_file_lists = np.transpose(x_train_files).tolist()
        x_test_file_lists = np.transpose(x_test_files).tolist()
        x_val_file_lists = np.transpose(x_val_files).tolist()
        return (
            (x_train_file_lists, y_train_files),
            (x_test_file_lists, y_test_files),
            (x_val_file_lists, y_val_files),
        )
    except Exception as e:
        error["code"] = "0005"
        logger.error(
            "An error occurred when splitting the data. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def verify_image_sizes(image_paths: list) -> tuple:
    logger.info("Verifying image dimensions")
    try:
        all_sizes = []
        for image_path in image_paths:
            with rasterio.open(image_path) as image:
                all_sizes.append(image.shape)
        unique_sizes = collections.Counter(all_sizes)
        if len(unique_sizes) > 1:
            error["code"] = "0003"
            majority_size, _ = unique_sizes.most_common()[0]
            outlier_indices = [
                index for index, size in enumerate(all_sizes) if size != majority_size
            ]
            outlier_image_paths = [image_paths[index] for index in outlier_indices]
            image_name_start_indices = [
                image_path.rfind("/") + 1 for image_path in outlier_image_paths
            ]
            length = min(len(outlier_image_paths), 10)
            outlier_image_names = [
                outlier_image_paths[i][image_name_start_indices[i] :]
                for i in range(length)
            ]
            logger.error("Inconsistent image dimensions.")
            raise ValueError(
                f"{outlier_image_names} do not have the same dimension as the other images.  All images onboarded need to follow the same dimension for fine-tuning and inference.  Please verify the dimension of ALL images before onboarding again."
            )
    except Exception as e:
        error["code"] = "0003"
        logger.error(
            "An error occurred when verifying image sizes. Error details: ",
            e,
            stack_info=True,
        )
        raise e


def onboard_one_modality(
    onboarding_info: dict,
    working_dir,
    dataset_id,
    file_split,
    output_queue: Queue,
):
    total_mean, total_std, bands = find_mean_and_std(
        raw_bands=onboarding_info["bands"], training_images=file_split[0]
    )
    logger.info(onboarding_info["modality_tag"])
    logger.info(total_mean)
    logger.info(total_std)
    Path(working_dir + "/training_data/" + onboarding_info["modality_tag"]).mkdir(
        parents=True, exist_ok=True
    )
    p1 = Process(
        target=cog_validation_and_sort,
        args=(
            onboarding_info["image_files"],
            working_dir + "/training_data/" + onboarding_info["modality_tag"],
        ),
    )
    processes = [p1]
    ran_successfully = run_paralleled_processes(processes=processes)
    norm_stds = total_std.tolist() if ran_successfully else None
    norm_means = total_mean.tolist() if ran_successfully else None
    training_params = {
        "modality_tag": onboarding_info["modality_tag"],
        "norm_stds": norm_stds,
        "norm_means": norm_means,
        "bands": bands,
        "file_suffix": onboarding_info["file_suffix"],
    }
    logger.debug(f"Training parms: {training_params}")
    output_queue.put(training_params)


def get_onboarding_options(options: str) -> dict:
    # Remove surrounding single quotation marks
    # options_striped = options[1:-1]
    # Return json
    # return json.loads(options_striped)
    return json.loads(options)


def main():
    s3 = object_storage_client()

    onboarding_details = {}

    working_path = "/data/" + payload["dataset_id"]

    dataset_bucket = os.getenv("DATA_BUCKET", "geoft-service-datasets")

    if "dataset_url" in payload:
        download_dataset(
            source_url=payload["dataset_url"],
            destination=working_path,
        )
    else:
        error["code"] = "0001"
        raise Exception("dataset_url is a required field.")

    if "onboarding_options" in payload:
        logger.debug(f"Onboarding_options: {payload['onboarding_options']}")
        onboarding_options = get_onboarding_options(payload["onboarding_options"])
        if (
            "from_labels" in onboarding_options
            and onboarding_options["from_labels"] is True
        ):
            prepare_dataset(working_path)

    multimodal_onboarding_info = []
    cos_info = {"instance": s3, "bucket_name": dataset_bucket}

    # -- Find all data and label files
    image_file_lists = []
    training_file_suffixes = []
    data_sources = json.loads(payload["data_sources"])
    for data_source in data_sources:
        file_suffix = data_source["file_suffix"]
        files = find_and_sort_from(filepath=working_path, suffix=file_suffix)
        split_weights = (0.6, 0.2, 0.2)
        onboarding_info = {
            "file_suffix": file_suffix,
            "image_files": files,
            "split_weights": split_weights,
            "bands": data_source["bands"],
            "modality_tag": data_source["modality_tag"],
        }
        multimodal_onboarding_info.append(onboarding_info)
        image_file_lists.append(files)
        training_file_suffixes.append(file_suffix)

    label_files = find_and_sort_from(
        filepath=working_path, suffix=payload["label_suffix"]
    )

    image_files = list(chain.from_iterable(image_file_lists))
    verify_image_sizes(image_files + label_files)

    image_stem_lists = []
    for onboarding_info in multimodal_onboarding_info:
        image_stems = obtain_file_stem(
            suffix=onboarding_info["file_suffix"],
            filepaths=onboarding_info["image_files"],
        )
        image_stem_lists.append(image_stems)

    label_stems = obtain_file_stem(
        suffix=payload["label_suffix"], filepaths=label_files
    )

    for image_stems in image_stem_lists:
        if image_stems != label_stems:
            error["code"] = "0004"
            logger.error("Error: Data and labels don't match, based on the filenames")
            logger.error(f"image_stems: {image_stems}")
            logger.error(f"label_stems: {label_stems}")
            raise Exception(
                "Error: Data and labels don't match, based on the filenames"
            )

    #######------- Handling or creating splits files
    file_splits = create_file_splits(
        split_weights=(0.6, 0.2, 0.2),
        label_files=label_files,
        image_file_lists=image_file_lists,
    )

    train_pair, test_pair, val_pair = file_splits
    y_splits = (train_pair[1], test_pair[1], val_pair[1])

    create_and_save_split_files(
        local_source_path=working_path,
        label_splits=y_splits,
        label_suffix=payload["label_suffix"],
    )

    training_params = Queue()

    multimodal_onboarding_processes = []
    for onboarding_info, train_file_list, test_file_list, val_file_list in zip(
        multimodal_onboarding_info, train_pair[0], test_pair[0], val_pair[0]
    ):
        file_split = (train_file_list, test_file_list, val_file_list)
        single_modal_onboarding_process = Process(
            target=onboard_one_modality,
            args=(
                onboarding_info,
                working_path,
                payload["dataset_id"],
                file_split,
                training_params,
            ),
        )
        multimodal_onboarding_processes.append(single_modal_onboarding_process)

    Path(working_path + "/labels").mkdir(parents=True, exist_ok=True)
    label_cog_validation = Process(
        target=cog_validation_and_sort, args=(label_files, working_path + "/labels")
    )
    multimodal_onboarding_processes.append(label_cog_validation)

    size = find_total_size(image_files + label_files)
    ran_successfully = run_paralleled_processes(
        processes=multimodal_onboarding_processes
    )

    if ran_successfully is False:
        size = "0MB"
    status = "Succeeded" if ran_successfully else "Failed"

    training_params_list = []

    while not training_params.empty():
        training_params_list.append(training_params.get())

    training_params = training_params_list

    # -- Uploading calculated properties to COS
    logger.info("Save calculated properties of the dataset to COS")
    try:
        with open(working_path + "/dataset_properties.json", "w") as f:
            json.dump(training_params, f, indent=4)
            # s3.put_object(
            #     Bucket=dataset_bucket,
            #     Body=json.dumps(training_params),
            #     Key=dataset_id + "/dataset_properties.json",
            # )
    except Exception as e:
        error["code"] = "0011"
        logger.error(
            f"An error occurred when saving training params to {working_path}. Error details: ",
            e,
            stack_info=True,
        )
        raise e

    # -- Write successful to working dir
    with open(working_path + "/data.json", "w") as f:
        json.dump({"Success": "True"}, f)

    #######------- Notify the dataset-factory api of onboarding results
    populate_onboarding_details(
        dataset_id=dataset_id,
        onboarding_details=onboarding_details,
        status=status,
        size=size,
        training_params=training_params,
    )

    print("*************************************")
    print("onboarding_details is - ")
    print(onboarding_details)
    print("*************************************")

    notify_df_api(onboarding_details)


if __name__ == "__main__":
    try:
        start_onboarding_details = {}
        start_onboarding_details["dataset_id"] = dataset_id
        start_onboarding_details["status"] = "Onboarding"
        start_onboarding_details["error_code"] = default_error["code"]
        start_onboarding_details["error_message"] = default_error["message"]
        notify_df_api(start_onboarding_details)
        main()
    except Exception as e:
        logger.error(
            "An exception occurred when onboarding the dataset", stack_info=True
        )
        logger.error("Exception - " + str(e))
        error["message"] = str(e)
        onboarding_details = {}
        populate_onboarding_details(
            dataset_id=dataset_id,
            onboarding_details=onboarding_details,
            status="Failed",
            size="0MB",
            error=error,
        )
        notify_df_api(onboarding_details)
