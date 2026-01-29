# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import urllib.parse

import requests

from gfmstudio.config import settings
from gfmstudio.log import logger


def invoke_pipelines_orchestration_service(
    *,
    payload: str,
    inference_id: str,
    deploy_model: bool = False,
    pipeline_version: str = None,
):
    data = {
        "name": inference_id,
        "service_payload": payload,
    }
    if pipeline_version == "v2":
        JOB_ENDPOINT = urllib.parse.urljoin(
            settings.INFERENCE_PIPELINE_V2_BASE_URL, "/submit_inference"
        )
        data = payload
    elif deploy_model:
        JOB_ENDPOINT = (
            f"{settings.INFERENCE_PIPELINE_BASE_URL}/pipelines"
            "/{settings.DEPLOY_FOR_INFERENCE_PIPELINE_ID}/jobs"
        )
    else:
        JOB_ENDPOINT = f"{settings.INFERENCE_PIPELINE_BASE_URL}/pipelines/{settings.INFERENCE_PIPELINE_ID}/jobs"

    headers = {"accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(JOB_ENDPOINT, headers=headers, json=data, verify=False)

    # Check the response status code
    if response.status_code // 100 == 2:
        logger.info("pre-processing POST request was successful.")
        logger.info("Response: %s", response.json())
    else:
        logger.error(
            "pre-processing POST request failed. Status code: %s", response.status_code
        )
        logger.error("Response: %s", response.text)
    return response
