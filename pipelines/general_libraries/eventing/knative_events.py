# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""
Event publisher for publishing CloudEvents to Knative Event Broker.
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

try:
    from cloudevents.core.bindings.http import to_structured
    from cloudevents.core.formats.json import JSONFormat
    from cloudevents.core.v1.event import CloudEvent as CECloudEvent

    CLOUDEVENTS_AVAILABLE = True
except ImportError:
    CLOUDEVENTS_AVAILABLE = False

try:
    from gfm_data_processing.common import logger
except ImportError:
    logger = logging.getLogger(__name__)

from pipelines.general_libraries.eventing.cloudevents_schema import CloudEvent


class EventPublisher:
    """
    Publisher for sending CloudEvents to Knative Event Broker.
    """

    def __init__(
        self,
        broker_url: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize the event publisher.

        Args:
            broker_url: URL of the Knative Event Broker
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.broker_url = broker_url or os.getenv(
            "EVENT_BROKER_URL",
            "http://broker-ingress.knative-eventing.svc.cluster.local/default/default",
        )
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.enabled = os.getenv("EVENT_DRIVEN_MODE", "false").lower() == "true"

        if self.enabled:
            logger.info(f"Event publisher initialized with broker: {self.broker_url}")
        else:
            logger.info("Event publisher disabled (EVENT_DRIVEN_MODE=false)")

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish a CloudEvent to the broker.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Event publishing disabled, skipping event: {event.type}")
            return False

        try:
            if not CLOUDEVENTS_AVAILABLE:
                logger.error("cloudevents library not available, cannot publish event")
                return False

            # Convert our CloudEvent to cloudevents library format
            ce_event = CECloudEvent(
                attributes={
                    "specversion": event.specversion,
                    "type": event.type,
                    "source": event.source,
                    "id": event.id,
                    "time": event.time,
                    "datacontenttype": event.datacontenttype,
                    "subject": event.subject,
                },
                data=event.data,
            )

            # Convert to structured format (HTTP POST with JSON body)
            struct_event = to_structured(event=ce_event, event_format=JSONFormat())

            # Publish to broker
            response = requests.post(
                self.broker_url,
                headers=struct_event.headers,
                data=struct_event.body,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            if response.status_code in (200, 201, 202):
                logger.info(
                    f"Successfully published event: type={event.type}, "
                    f"id={event.id}, subject={event.subject}"
                )
                return True
            else:
                logger.error(
                    f"Failed to publish event: status={response.status_code}, "
                    f"response={response.text}, event_id={event.id}"
                )
                return False

        except Exception as ex:
            logger.error(
                f"Exception publishing event: {ex}, event_id={event.id}", exc_info=True
            )
            return False

    def publish_dict(self, event_dict: Dict[str, Any]) -> bool:
        """
        Publish an event from a dictionary.

        Args:
            event_dict: Dictionary containing event data

        Returns:
            True if published successfully, False otherwise
        """
        try:
            event = CloudEvent(**event_dict)
            return self.publish(event)
        except Exception as ex:
            logger.error(f"Failed to create CloudEvent from dict: {ex}", exc_info=True)
            return False


# Global event publisher instance
_event_publisher: Optional[EventPublisher] = None


def get_event_publisher() -> EventPublisher:
    """
    Get or create the global event publisher instance.

    Returns:
        EventPublisher instance
    """
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = EventPublisher()
    return _event_publisher


def publish_event(event: CloudEvent) -> bool:
    """
    Convenience function to publish an event using the global publisher.

    Args:
        event: CloudEvent to publish

    Returns:
        True if published successfully, False otherwise
    """
    publisher = get_event_publisher()
    return publisher.publish(event)


def publish_event_dict(event_dict: Dict[str, Any]) -> bool:
    """
    Convenience function to publish an event from a dictionary.

    Args:
        event_dict: Dictionary containing event data

    Returns:
        True if published successfully, False otherwise
    """
    publisher = get_event_publisher()
    return publisher.publish_dict(event_dict)


# Made with Bob
