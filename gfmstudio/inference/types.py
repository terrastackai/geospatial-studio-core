# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import enum
from typing import Dict, List


class EventDetailType(str, enum.Enum):
    MSG_NOTIFY = "Inf:Task:Notify"
    LAYER_READY = "Inf:Task:LayerReady"
    TASK_COMPLETE = "Inf:Task:Complete"
    TASK_UPDATED = "Inf:Task:Updated"
    TASK_PENDING_GEOSERVER = "Inf:Task:PendingGeoserver"
    TASK_ERROR = "Inf:Task:Error"
    TASK_FAILED = "Inf:Task:Failed"
    MSG_NOTIFY_AMO_MODELS = "Inf:Task:AmoNotify"
    TASK_MODELS_CLEANUP = "Inf:Task:AmoCleanUp"
    TASK_LAYERS_UPDATED = "Inf:Task:AddLayerComplete"


class EventStatus(str, enum.Enum):
    """Status of an inference event.

    PENDING: Event is pending.
    RUNNING: Event is running.
    COMPLETED: Event is completed.
    PENDING_GEOSERVER: Event is pending push to Geoserver.
    FAILED: Event failed.
    CANCELLED: Event is cancelled.

    # Flow of events
    PENDING -> RUNNING -> COMPLETED
    PENDING -> RUNNING -> FAILED

    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelStatus(enum.StrEnum):
    PENDING = "PENDING"
    DEPLOY_REQUESTED = "DEPLOY_REQUESTED"
    TOKEN_ERROR = "TOKEN_ERROR"
    DEPLOY_IN_PROGRESS = "DEPLOY_IN_PROGRESS"
    DEPLOY_ERROR = "DEPLOY_ERROR"
    COMPLETED = "COMPLETED"
    DEPLOY_FAILED = "DEPLOY_FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class InferenceStatus(enum.StrEnum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"  # Inference request has been received and registered
    RUNNING = "RUNNING"  # Running the ML model inference per tasK
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    COMPLETED = "COMPLETED"  # All tasks were successful; workflow is completed
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"  # Some tasks failed irrecoverably, others may have passed
    FAILED = "FAILED"  # Inference failed entirely or with unrecoverable error
    STOPPED = "STOPPED"
    PUBLISHING_RESULTS = (
        "PUBLISHING_RESULTS"  # Uploading results to external systems like GeoServer
    )


class GenericProcessorStatus(enum.StrEnum):
    PENDING = "PENDING"  # Script yet uploaded to COS
    FINISHED = "FINISHED"  # Script successfully uploaded to COS
    FAILED = "FAILED"  # Script upload to COS failed


INFERENCE_STATE_TRANSITIONS: Dict[InferenceStatus | str, List[InferenceStatus]] = {
    InferenceStatus.PENDING: [
        InferenceStatus.SUBMITTED,
        InferenceStatus.RUNNING,
    ],
    InferenceStatus.SUBMITTED: [InferenceStatus.RUNNING],
    InferenceStatus.RUNNING: [
        InferenceStatus.PARTIALLY_COMPLETED,
        InferenceStatus.COMPLETED,
        InferenceStatus.COMPLETED_WITH_ERRORS,
    ],
    InferenceStatus.PUBLISHING_RESULTS: [InferenceStatus.COMPLETED],
    InferenceStatus.PARTIALLY_COMPLETED: [
        InferenceStatus.COMPLETED,
        InferenceStatus.COMPLETED_WITH_ERRORS,
    ],
    "*": [InferenceStatus.FAILED, InferenceStatus.STOPPED],
}


# State machine code that we use to transition the infrence status states.
def can_transition(from_state: InferenceStatus, to_state: InferenceStatus) -> bool:
    allowed = INFERENCE_STATE_TRANSITIONS.get(from_state, [])
    wildcard = INFERENCE_STATE_TRANSITIONS.get("*", [])

    return to_state in allowed or to_state in wildcard


def transition_to(
    current_state: InferenceStatus, new_state: InferenceStatus
) -> InferenceStatus:
    if can_transition(current_state, new_state):
        return new_state
    # for scenarios where other properties need updating in the inference
    if current_state == new_state:
        return new_state
    raise ValueError(f"Invalid state transition: {current_state} -> {new_state}")
