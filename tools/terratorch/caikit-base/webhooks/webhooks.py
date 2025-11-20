# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import concurrent.futures
import datetime
import logging
import os
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


GFMAAS_API_BASE_URL = os.getenv("GFMAAS_API_BASE_URL")
GFMAAS_API_KEY = os.getenv("GFMAAS_API_KEY")
GFMAAS_WEBHOOKS_URL = (
    urljoin(GFMAAS_API_BASE_URL, "webhooks") if GFMAAS_API_BASE_URL else None
)
WATSONX_WEBHOOKS_URL = os.getenv("WATSONX_WEBHOOKS_URL")

# Temporary map because we still maintain a very small number of subscribers.
WEBHOOK_SUBSCRIBERS = [
    {
        "tenant_alias": "watsonx",
        "tenant_webhook_url": WATSONX_WEBHOOKS_URL,
        "tenant_http_headers": {
            "Content-Type": "application/json",
        },
    },
    {
        "tenant_alias": "geostudio-gateway",
        "tenant_webhook_url": GFMAAS_WEBHOOKS_URL,
        "tenant_http_headers": {
            "Content-Type": "application/json",
            "X-API-KEY": GFMAAS_API_KEY,
        },
    },
]


def send_webhook(subscriber, payload, event_id):
    url = subscriber["tenant_webhook_url"]
    if space := payload["detail"].get("space_id"):
        url = urljoin(url, f"?space_id={space}")
    elif project := payload["detail"].get("project_id"):
        url = urljoin(url, f"?project_id={project}")

    headers = subscriber["tenant_http_headers"]
    try:
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=29,
            verify=False,
        )
        if response.status_code not in (200, 201):
            logger.error(
                f"{event_id} - Failed to send task status. Reason: (%s)> %s",
                response.status_code,
                response.text,
            )
            return None
        return response.status_code
    except Exception as ex:
        logger.error(f"{event_id} - Failed to send task status. Reason: (%s)", ex)
        return None


def send_webhooks(payload, event_id):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(send_webhook, subscriber, payload, event_id)
            for subscriber in WEBHOOK_SUBSCRIBERS
            if subscriber.get("tenant_webhook_url")
        ]
        if not futures:
            logger.warning(
                f"{event_id} - No subscriber endpoints enabled for notifications."
            )
            return

        for future in concurrent.futures.as_completed(futures):
            status_code = future.result()
            if status_code is not None:
                logger.info(
                    f"{event_id} - Webhook request completed with status code {status_code}"
                )


def notify_gfmaas_ui(
    event_id: str, event_status: str = None, event_details: dict = None
):
    """Helper method to notify Gfmaas-UI

    Parameters
    ----------
    event_id : str
        The event_id sent from the client when the inference task was started.
    event_status : str
        The current status of the event with `event_id` in the inference server
    """
    event_details = event_details if isinstance(event_details, dict) else {}
    event_details["status"] = event_details.get("status", "IN_PROGRESS")
    event_details["message"] = event_details.get("message", event_status)
    event_data = {
        "event_id": event_id,
        "detail_type": "Inf:Task:Notify",
        "source": "com.inference-server.ibm",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "detail": event_details,
    }
    send_webhooks(payload=event_data, event_id=event_id)
