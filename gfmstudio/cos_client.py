# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from functools import lru_cache

import boto3
from botocore.config import Config
from fastapi import FastAPI, Request

from gfmstudio.config import settings
from gfmstudio.log import logger


def init_cos_client(app: FastAPI = None):
    """
    Initialize and attach global boto3 clients to the FastAPI app state.
    Called once during startup.
    """
    logger.info("🔌 Initializing object storage client...")
    try:
        cos_client = boto3.client(
            "s3",
            endpoint_url=settings.OBJECT_STORAGE_ENDPOINT,
            aws_access_key_id=settings.OBJECT_STORAGE_KEY_ID,
            aws_secret_access_key=settings.OBJECT_STORAGE_SEC_KEY,
            config=Config(signature_version=settings.OBJECT_STORAGE_SIGNATURE_VERSION),
            region_name=settings.OBJECT_STORAGE_REGION,
        )
        if app:
            app.state.cos_client = cos_client
    except ValueError as exc:
        logger.error(f"❌ Cos client Misconfiguration: {str(exc)}")
    finally:
        logger.info("✅ Cos client Initialized Successfully")

    if not app:
        return cos_client


@lru_cache()
def get_cos_client(request: Request = None):
    """
    Dependency injector for cos client.
    Ensures the same client instance is reused per app.
    """
    if request:
        return request.app.state.cos_client
    return init_cos_client()
