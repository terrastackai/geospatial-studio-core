# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging

# default code for each  operator
import os
import re
import sys

# init logger
root = logging.getLogger()
root.setLevel("INFO")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)
logging.basicConfig(level=logging.CRITICAL)

# get parameters from args
parameters = list(
    filter(
        lambda s: s.find("=") > -1
        and bool(re.match(r"[A-Za-z0-9_]*=[.\/A-Za-z0-9]*", s)),
        sys.argv,
    )
)

# set parameters to env variables
for parameter in parameters:
    variable = parameter.split("=")[0]
    value = parameter.split("=", 1)[-1]
    logging.info(f'Parameter: {variable} = "{value}"')
    os.environ[variable] = value

# update log level
log_level = os.environ.get("log_level", "INFO")
if log_level != "INFO":
    logging.info(f"Updating log level to {log_level}")
    root.setLevel(log_level)
    handler.setLevel(log_level)
"""
This is the curated upload operator.
"""

# You can add multiple comments if the packages require a specific order.
# pip install rio-cogeo wget scikit-learn>=1.3.0 rasterio boto3==1.35.82 botocore numpy>=1.22.2 tqdm requests humanize

import collections
import datetime
import glob
import json
import logging
import os
import subprocess
import sys
import tarfile
import uuid
import zipfile
from itertools import chain
from multiprocessing import Process, Queue
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

# Input string for curated upload proided by claimed
df_api_route = os.getenv(
    "df_api_route",
    "https://geoft-dataset-factory-api-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/",
)
dataset_url = os.getenv(
    "dataset_url",
    "https://ibm.box.com/shared/static/jaqwlc4hgg734xxum9mrhv5rcdn9cb7g.zip",
)
data_sources = os.getenv("data_sources", "{}")
label_suffix = os.getenv("label_suffix", ".mask.tif")
dataset_id = os.getenv("dataset_id", "geodata-someuuid")
df_api_key = os.getenv("DF_APIKEY", "some-api-key")
onboarding_options = os.getenv("onboarding_options", "'{}'")

payload = {}
payload["dataset_url"] = dataset_url
payload["label_suffix"] = label_suffix
payload["dataset_id"] = dataset_id
payload["data_sources"] = data_sources
payload["onboarding_options"] = onboarding_options

# This is the log level for the onboarding script
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()

"""
Logging settings.
"""
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)s)",
    level=LOGLEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
)

error = {"code": "0000", "message": "N/A"}


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
    """Helper method to notify dataset-factory API
    Parameters
    ----------
    onboarding_details : dict
        Must contain dataset_id and status regardless of whether onboarding pipeline succeeded/failed
        If the pipeline succeeds, the variable needs to also include, "size" and "training_params".
    """
    logger.info("Notify the dataset-factory API onboarding status")
    event_data = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "detail": onboarding_details,
    }
    if "notifications" in df_webhooks_url:
        event_data.update(
            {
                "event_id": str(uuid.uuid4()),
                "detail_type": "FT:Data:Finished",
                "source": "com.ibm.dataset-factory-onboarding",
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
    onboarding_details: dict = {},
    size: str = None,
    training_params: list = None,
    error: dict = default_error,
):
    logger.info("Populate onboarding details to send back to the API webhook")
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


def obtain_file_stem(filepaths, suffix):
    logger.info("Obtain file stems")
    try:
        return [filepath.replace(suffix, "").split("/")[-1] for filepath in filepaths]
    except Exception as e:
        error["code"] = "0004"
        logger.error(
            "An error occurred when processing file stems. Error details: ",
            e,
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


def cog_validation_and_sort(filepaths: list, dest):
    logger.info("Validating that images are COG")
    try:
        for f in tqdm(filepaths):
            is_valid, errors, warnings = cog_validate(f)
            if is_valid is False:
                if len(errors) > 0:
                    logger.info(
                        f"An error occurred processing {f.split('/')[-1]}: %s", errors
                    )
                logger.info(
                    "---> "
                    + f.split("/")[-1]
                    + " not valid COG - trying to change that :)"
                )
                subprocess.check_output(
                    "rio cogeo create --cog-profile lzw --use-cog-driver "
                    + f
                    + " "
                    + f
                    + ".cog.tif",
                    shell=True,
                )
                os.remove(f)
                os.rename(f + ".cog.tif", f)
            try:
                save_file(dest, f)
            except Exception as e:
                error["code"] = "0012"
                logger.error(
                    f"Error occurred when saving files to {dest}.  Error Details: ",
                    e,
                    stack_info=True,
                )
                raise e
    except Exception as e:
        error["code"] = "0008"
        logger.error(
            "An error occurred during COG validation. Error details: ",
            e,
            stack_info=True,
        )
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
