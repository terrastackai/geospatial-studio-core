# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import os

import boto3
from botocore.client import Config

from gfmstudio.config import settings

logger = logging.getLogger(__name__)


def generate_download_presigned_url(
    object_key,
    s3=None,
    bucket_name=None,
    expiration=43200,
):
    """
    Create presigned url for data output to default COS bucket
    """
    # Use the geospatial-cos bucket
    logger.debug(f"Creating pre-signed URL for object key: {object_key}")
    if not s3:
        endpoint_url = os.getenv(
            "OUTPUT_BUCKET_LOCATION",
            "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        )
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("gfm-inference-outputs-cos-access-key"),
            aws_secret_access_key=os.getenv("gfm-inference-outputs-cos-secret-key"),
            endpoint_url=endpoint_url,
            verify=(settings.ENVIRONMENT.lower() != 'local'),
        )
    bucket_name = bucket_name or os.getenv("OUTPUT_BUCKET", "dev-output-pv-storage")
    params = {"Bucket": bucket_name, "Key": object_key}
    # Generate the pre-signed URL
    output_url = s3.generate_presigned_url(
        "get_object", Params=params, ExpiresIn=expiration,
    )
    logger.debug(f"Output url: {output_url}")
    return output_url


def generate_upload_presigned_url(
    object_key,
    s3=None,
    bucket_name=None,
    expiration=14400,
):
    """Create presigned-URLs to upload data."""
    logger.debug(f"Creating pre-signed URL for object key: {object_key}")
    http_method = "put_object"
    bucket_name = bucket_name or os.getenv("OUTPUT_BUCKET", "dev-output-pv-storage")
    if not s3:
        cos_service_endpoint = os.getenv(
            "OUTPUT_BUCKET_LOCATION",
            "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        )
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("gfm-inference-outputs-cos-access-key"),
            aws_secret_access_key=os.getenv("gfm-inference-outputs-cos-secret-key"),
            endpoint_url=cos_service_endpoint,
            config=Config(signature_version=settings.OBJECT_STORAGE_SIGNATURE_VERSION),
            verify=(settings.ENVIRONMENT.lower() != "local"),
        )

    signedUrl = s3.generate_presigned_url(
        http_method,
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=expiration,
    )
    return signedUrl
