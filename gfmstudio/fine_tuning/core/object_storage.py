# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Optional

import boto3
import structlog
from botocore.client import Config
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.type_defs import ObjectTypeDef

from gfmstudio.config import Settings, get_settings, settings

logger = structlog.get_logger()


class BaseCOSException(Exception): ...


class NoCheckpointOrTooSmallFileException(BaseCOSException): ...


class NoConfigFileException(BaseCOSException): ...


def object_storage_client(settings_: Optional[Settings] = None) -> S3Client:
    """Function to create an object storage client

    Parameters
    ----------
    settings_ : Optional[Settings], optional
        param to grab env variables, by default None

    Returns
    -------
    S3Client
        The object storage client
    """
    if not settings_:
        settings_ = get_settings()
    logging.info(
        f"Initialized object storage client to {settings.OBJECT_STORAGE_ENDPOINT}"
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=settings_.OBJECT_STORAGE_ENDPOINT,
        aws_access_key_id=settings_.OBJECT_STORAGE_KEY_ID,
        aws_secret_access_key=settings_.OBJECT_STORAGE_SEC_KEY,
        config=Config(signature_version=settings.OBJECT_STORAGE_SIGNATURE_VERSION),
        region_name=settings_.OBJECT_STORAGE_REGION,
    )
    return s3


def run_bucket_client(setting: Optional[Settings]):
    """Function to run bucket client

    Parameters
    ----------
    setting : Optional[Settings]
        param to grab env variables

    Returns
    -------
    S3Client
        object storage client
    """
    return object_storage_client(settings=setting)


def upload(
    s3: S3Client,
    bucket,
    model_run_id,
    file_name,
    content,
    prefix="main",
    experiment="default",
):
    """Function to create object key and upload file contents to s3

    Parameters
    ----------
    s3 : S3Client
        S3 client
    bucket : _type_
        bucket name
    model_run_id : _type_
        model run id
    file_name : _type_
        The file name
    content : _type_
        The content to upload
    prefix : str, optional
        Prefix to the path, by default "main"
    experiment : str, optional
        Experiment name, by default "default"
    """
    path = experiment + "/" + model_run_id + "/" + prefix + "/" + file_name
    s3.put_object(Bucket=bucket, Body=content, Key=path)


def detailed_prefix_list(
    s3: S3Client,
    bucket,
    tunes_path,
    tune_id,
) -> list[ObjectTypeDef]:
    """Function to list a COS bucket prefix, and returns a list of contents.
    If no contents are found raises a bare Exception

    Parameters
    ----------
    s3 : S3Client
        S3 client
    bucket : _type_
        The bucket name
    tunes_path : _type_
        The tunes path
    tune_id : _type_
        The tune id

    Returns
    -------
    list[ObjectTypeDef]
        A list of contents in the path

    Raises
    ------
    ValueError
        If error occured listing objects in bucket
    Exception
        If the bucket and prefix are empty
    """
    prefix = f"{tunes_path}/{tune_id}"
    try:
        bucket_lst = s3.list_objects(Bucket=bucket, Prefix=prefix)
    except ClientError as base_error:
        msg = f"Error listing {bucket}: {base_error}"
        raise ValueError(msg) from base_error
    contents: list[ObjectTypeDef] = bucket_lst.get("Contents", [])
    if not contents:
        msg = (
            f"Bucket {bucket} path {prefix} is empty. "
            "Check if the experiment, model run id and prefix are correct."
        )
        raise Exception(msg)
    return contents


def signed_url(
    s3: S3Client,
    bucket: str,
    tunes_path: str,
    tune_id: str,
    file_name: str,
    expires_in: int = 3600,
):
    """Function to generate a signed url for a specific  tunes_path/tune_id/file_name

    Parameters
    ----------
    s3 :  S3Client
        s3 client
    bucket : str
        The bucket name
    tunes_path : str
        The tunes path
    tune_id : str
        The tune id
    file_name : str
        The file name
    expires_in : int, optional
        Expiration time in seconds for the link, by default 3600

    Returns
    -------
    str
        Signed url to the created object path
    """
    signed_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket,
            "Key": f"{tunes_path}/{tune_id}/{file_name}",
        },
        ExpiresIn=expires_in,
    )
    logging.info(f"Signed {bucket}://{tunes_path}/{tune_id}/{file_name}")

    return signed_url


async def generate_sample_presigned_urls(
    *,
    bucket_name: str,
    folder_name: str,
    s3_client=None,
    num_urls: int = 5,
    expiration: int = 3600,
):
    """
    Generate presigned URLs for objects in an S3 bucket's folder.

    Parameters
    ----------
    bucket_name (str):
        The name of the S3 bucket.
    folder_name (str):
        The name of the folder within the bucket.
    num_urls (int, optional):
        The number of presigned URLs to generate.
    expiration (int, optional):
        The expiration time for the presigned URLs in seconds.

    Returns
    -------
    List[str]:
        A list of generated presigned URLs.

    """
    s3_client = s3_client or boto3.client("s3")

    logger.info("Fetching items from %s bucket.", bucket_name)

    # List all objects in the specified folder
    # Using asyncio.to_thread to prevent blocking.
    objects = await asyncio.to_thread(
        s3_client.list_objects_v2, Bucket=bucket_name, Prefix=folder_name
    )

    if "Contents" not in objects:
        logger.info("No content found in bucket: %s", bucket_name)
        return []

    # Get a random selection of objects
    logger.info("Items found in bucket; Generating presigned URLs...")
    selected_objects = random.sample(
        objects["Contents"], min(num_urls, len(objects["Contents"]))
    )

    async def generate_presigned_url(obj):
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket_name,
                "Key": obj["Key"],
            },
            ExpiresIn=expiration,
        )

    # Generate presigned URLs for the random objects.
    # Gather multiple coroutines and run them concurrently to speed up this task.
    presigned_urls = await asyncio.gather(
        *[generate_presigned_url(obj) for obj in selected_objects]
    )

    return presigned_urls


@dataclass
class ObjectKeysPresign:
    """DataClass to Presign config and checkpoint files

    Attributes
    ----------
    config_file: str
        Path to the config file
    checkpoint_file: str
        Path to the checkpoint file

    Methods
    -------
    _sign()
        Signs a single file
    signed_urls()
        Signs config and checkpoint files

    """

    config_file: str
    checkpoint_file: str

    def _sign(self, client: S3Client, bucket: str, expires_in: int, key):
        """Function to generate presigned url

        Parameters
        ----------
        client : S3Client
            botocore client
        bucket : str
            bucket name
        expires_in : int
            time in seconds for link to expire
        key : _type_
            file path

        Returns
        -------
        str
            signed url
        """
        signed_url = client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
            },
            ExpiresIn=expires_in,
        )
        return signed_url

    def signed_urls(
        self, client: S3Client, bucket: str, expires_in: int = 3600
    ) -> tuple[str, str]:
        """Function to sign config and checkpoint

        Parameters
        ----------
        client : S3Client
            botocore client
        bucket : str
            bucket name
        expires_in : int, optional
            time in seconds for link to expire , by default 3600

        Returns
        -------
        tuple[str, str]
            Signed url for the config and checkpoint
        """
        return (
            self._sign(
                client=client,
                bucket=bucket,
                expires_in=expires_in,
                key=self.config_file,
            ),
            self._sign(
                client=client,
                bucket=bucket,
                expires_in=expires_in,
                key=self.checkpoint_file,
            ),
        )


def looks_like_checkpoint(element: ObjectTypeDef):
    """Function that receives a dictionary from boto3 list_object and
    evaluates if it belongs to a checkpoint file

    Parameters
    ----------
    element : ObjectTypeDef
        boto3 list_object

    Returns
    -------
    bool
        True if file is a checkpoint, else False
    """
    ok = False
    filename: str = element.get("Key", "")
    if filename.endswith(".pt"):
        ok = True
    if filename.endswith(".pth"):
        ok = True
    if filename.endswith(".ckpt"):
        ok = True
    if filename.endswith(".bin"):
        ok = True
    if not ok:
        return False
    size: int = element.get("Size", 0)

    if size < settings.MIN_CHECKPOINT_SIZE:
        logger.info(
            f"{filename} has the right extension but the size is too small "
            f"{size}<{settings.MIN_CHECKPOINT_SIZE}"
        )
        return False
    return True


def signed_score(key: str, score_table: list[str]) -> int:
    """Function that calculates a signed score.

        Calculates a score (or preference), indicating the reverse index over a list of suffixes.
        The first element of the score_table will be given the highest score, and sub-sequent
        entries will have a decreasing score reaching 0 for the last element of the list.
        Any key suffix which doesn't appear in the score table will return a -1.

    Parameters
    ----------
    key : str
        object key
    score_table : list[str]
        Score table indicating a reverse index over a list of suffixes.

    Returns
    -------
    int
        index of score table with the highest score
    """

    if "." not in key:
        return -1

    scores: dict[str, int] = dict(zip(score_table[::-1], range(len(score_table) + 1)))

    *_, detected_key_extension = key.split(".")

    if detected_key_extension not in score_table:
        return -1

    return scores[detected_key_extension]


def find_config_and_checkpoint(
    contents: list[ObjectTypeDef], config_file_ext_preference=None
) -> ObjectKeysPresign:
    """Function to find config and checkpoint in bucket

        Given an object list from botocore, will find the config file and the checkpoint
        in the contents.

    Parameters
    ----------
    contents : list[ObjectTypeDef]
        botocore object list
    config_file_ext_preference : str, optional
        file extension for the config. i.e 'yaml', by default None

    Returns
    -------
    ObjectKeysPresign
        instance of ObjectKeysPresign class

    Raises
    ------
    NoCheckpointOrTooSmallFileException
        If a checkpoint too small or doesn't exist
    NoCheckpointOrTooSmallFileException
        If a checkpoint too small or doesn't exist
    """
    config_file_ext_preference = settings.CONFIG_FILE_TYPES
    # NOTE: "_deploy.yaml" is terratorch specific config file for inference. Please update if
    # terratorch is changed or a different framework is in use.
    filenames = [entry["Key"] for entry in contents if "_deploy.yaml" in entry["Key"]]
    scored_filenames: dict[str, int] = {
        filename: signed_score(filename, score_table=config_file_ext_preference)
        for filename in filenames
    }
    valid_config_filenames = {k: v for k, v in scored_filenames.items() if v > -1}
    # Prefer the file with the highest score
    try:
        config_file, _ = max(
            valid_config_filenames.items(), key=lambda key_score: key_score[1]
        )
    except ValueError:
        raise NoCheckpointOrTooSmallFileException() from None

    checkpoint_files = [entry for entry in contents if looks_like_checkpoint(entry)]
    if not checkpoint_files:
        NoCheckpointOrTooSmallFileException()

    # Try to find a valid pytorch like file (.pt, .pth) that has a size at least
    # of the size of the settings.MIN_CHECKPOINT_SIZE

    try:
        # NOTE: Specific to terratorch, only the state-dict model checkpoint is needed for inference.
        checkpoint_files = [
            cfile["Key"] for cfile in checkpoint_files if "state_dict" in cfile["Key"]
        ]
        if checkpoint_files:
            checkpoint_file = checkpoint_files[0]
        else:
            raise NoCheckpointOrTooSmallFileException()
    except ValueError:
        raise NoCheckpointOrTooSmallFileException() from None

    return ObjectKeysPresign(config_file=config_file, checkpoint_file=checkpoint_file)


def check_s3_file_exists(s3, bucket_name: str, file_key: str):
    """
    Check if a file exists in an S3 bucket.

    Parameters
    ----------
    s3: S3Client
        Boto3 client to access S3.
    bucket_name: str
        Name of the S3 bucket.
    file_key: str
        The S3 object key (file path).

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    try:
        s3.head_object(Bucket=bucket_name, Key=file_key)
        logger.info(f"File {file_key} exists in bucket {bucket_name}.")
        return True
    except ClientError as e:
        # A 404 error indicates that the object does not exist.
        if e.response["Error"]["Code"] == "404":
            logger.exception(f"File {file_key} does not exist in bucket {bucket_name}.")
            return False
        else:
            # For other errors, raise the exception.
            raise e


def generate_presigned_url(
    s3,
    file_key,
    bucket_name: str = None,
    expiration: int = 3600,
):
    """
    Generate a presigned URL for an S3 object.

    Parameters
    ----------
    s3: S3Client
        Boto3 client to access S3.
    bucket_name: str
        Name of the S3 bucket.
    file_key: str
        The S3 object key (file path).
    expiration: int
        Time in seconds for the presigned URL to remain valid (default is 3600 seconds).

    Returns
    -------
    str
        The presigned URL as a string.
    """

    # Ensure inputs are strings
    if not isinstance(bucket_name, str):
        raise TypeError(f"bucket_name must be a string, got {type(bucket_name)}")
    if not isinstance(file_key, str):
        raise TypeError(f"file_key must be a string, got {type(file_key)}")

    try:
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": file_key},
            ExpiresIn=expiration,
        )
        logger.debug("Created download presigned URL.")
        return presigned_url
    except ClientError:
        logger.exception(f"Error encountered generating pre-signed url for {file_key}")


def generate_upload_presigned_url(
    s3,
    object_key: str,
    bucket_name: str = None,
    expiration: int = 28800,
) -> str:
    """Create presigned-URLs to upload data.

    Parameters
    ----------
    s3: S3Client
        Boto3 client to access S3.
    bucket_name: str
        Name of the S3 bucket.
    object_key: str
        The S3 object key (file path).
    expiration: int
        Time in seconds for the presigned URL to remain valid (default is 3600 seconds).

    Returns
    -------
    str
        The presigned URL as a string.
    """
    logger.debug(f"Creating upload pre-signed URL for object key: {object_key}")
    bucket_name = bucket_name or settings.TEMP_UPLOADS_BUCKET
    signedUrl = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=expiration,
    )
    return signedUrl


def get_item_download_links(
    s3: S3Client, count: int, bucket_name: str, directory_path: str
) -> dict:
    list_objects_response = s3.list_objects(Bucket=bucket_name, Prefix=directory_path)
    all_objects = list_objects_response.get("Contents", [])
    download_urls = []

    if not all_objects:
        return {
            "status_code": "404",
            "message": "Missing labels/images in the onboarded data store.",
        }
    try:
        for index in range(0, count):
            item = all_objects[index]["Key"]
            download_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": item},
                ExpiresIn=3600,
            )
            download_urls.append(download_url)
        return {
            "status_code": list_objects_response["ResponseMetadata"]["HTTPStatusCode"],
            "items": download_urls,
        }
    except IndexError as err:
        return {"status_code": 400, "message": err}
    except:  # noqa: E722
        return {"status_code": 500, "message": "Please contact admin for help"}


def remove_from_cos(s3: S3Client, bucket_name: str, directory_path: str) -> dict:
    list_objects_response = s3.list_objects(Bucket=bucket_name, Prefix=directory_path)
    objects_to_remove = list_objects_response.get("Contents", [])

    if len(objects_to_remove) > 0:
        hasMore = True
        while hasMore:
            objects = []
            for obj in objects_to_remove:
                objects.append({"Key": obj["Key"]})
            delete = {"Objects": objects}
            delete_objects_response = s3.delete_objects(
                Bucket=bucket_name, Delete=delete
            )
            list_objects_response = s3.list_objects(
                Bucket=bucket_name, Prefix=directory_path
            )
            objects_to_remove = list_objects_response.get("Contents", [])
            if objects_to_remove == []:
                hasMore = False
        return delete_objects_response
    else:
        delete_objects_response = {
            "ResponseMetadata": {
                "HTTPStatusCode": 204,
                "message": "The resource does not exist",
            }
        }
    return delete_objects_response
