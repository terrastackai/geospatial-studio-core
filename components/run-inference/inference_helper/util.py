# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from gfm_data_processing.common import logger


# Uncomment next 2 lines for local testing
import dotenv

dotenv.load_dotenv()

# presigned url expiration
url_expiration = int(os.getenv("url_expiration", 6000))

# cos/s3 endpoint
endpoint = os.getenv("endpoint", "https://s3.us-east.cloud-object-storage.appdomain.cloud")

# object key name
object_key_prefix = os.getenv("object_key_prefix", "")


def get_s3_client():
    cos = ibm_boto3.client(
        "s3",
        aws_access_key_id=os.getenv("access_key_id"),
        aws_secret_access_key=os.getenv("secret_access_key"),
        endpoint_url=os.getenv("endpoint", "https://s3.us-east.cloud-object-storage.appdomain.cloud"),
        config=Config(signature_version="s3v4"),
    )
    return cos


cos = get_s3_client()


def generate_presigned_url(http_method, bucket_name, object_key_name, expiration):
    signedUrl = cos.generate_presigned_url(
        http_method,
        Params={"Bucket": bucket_name, "Key": object_key_name},
        ExpiresIn=expiration,
    )
    return signedUrl


def generate_presigned_urls_for_preprocessed_files(preprocessed_items):
    try:
        bucket_name = os.environ["bucket_name"]
    except:
        bucket_name = os.environ["temp_bucket_name"]

    download_presigned_url_list = []
    for preprocess_item in preprocessed_items:
        # target_object_key_in_cos_arr = preprocess_item
        # preprocess_item_object_key_suffix_in_cos = "/".join(
        #     target_object_key_in_cos_arr
        # )
        preprocess_item_presigned_url = generate_presigned_url(
            "get_object",
            bucket_name,
            f"{preprocess_item}",
            url_expiration,
        )
        download_presigned_url_list.append(preprocess_item_presigned_url)
    return download_presigned_url_list


def generate_presigned_url_for_upload_archive(inference_result_location):
    try:
        bucket_name = os.environ["bucket_name"]
    except:
        bucket_name = os.environ["temp_bucket_name"]

    # data directory is mounted on a cos bucket, stripping the first directory will provide the key for the bucket object
    # cos_target_upload_key_prefix2_arr = inference_result_location.split("/")[1:]
    # cos_target_upload_key = "/".join(cos_target_upload_key_prefix2_arr)
    # cos_object_upload_key_full = f"{object_key_prefix}{cos_target_upload_key}"
    upload_presigned_url = generate_presigned_url("put_object", bucket_name, inference_result_location, url_expiration)
    return inference_result_location, upload_presigned_url


def search_key_in_bucket(inference_output_object_prefix_key):
    try:
        bucket_name = os.environ["bucket_name"]
    except:
        bucket_name = os.environ["temp_bucket_name"]

    logger.info("Retrieving bucket contents from: {0}".format(bucket_name))
    try:
        fileSize = None
        files = cos.list_objects_v2(
            Bucket=bucket_name,
            Prefix=inference_output_object_prefix_key,
        )
        for file in files.get("Contents", []):
            fileSize = file["Size"]
            if file["Size"] > 102400:
                logger.info("Item: {0} ({1} bytes).".format(file["Key"], file["Size"]))
                return True
            elif fileSize < 102400:
                logger.info("File not found")
                return False
            elif fileSize is None:
                logger.info("File not found")
                return False

            # Create the directory if it doesn't exist and download file
            # os.makedirs(os.path.dirname(inference_result_location), exist_ok=True)
            # cos.download_file(bucket_name, inference_output_object_prefix_key, inference_result_location)
            # return True

    except ClientError as be:
        logger.info("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        logger.info("Unable to retrieve bucket contents: {0}".format(e))


def upload_file(s3_client, file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True
