# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """
    Event types for pipeline task lifecycle.

    Task Lifecycle:
    1. TASK_READY - Task is ready to be processed
    2. TASK_COMPLETED - Task finished successfully
    3. TASK_FAILED - Task failed with error
    """

    # Task lifecycle events
    TASK_READY = "com.geospatial.task.ready"
    TASK_COMPLETED = "com.geospatial.task.completed"
    TASK_FAILED = "com.geospatial.task.failed"


class TaskEventData(BaseModel):
    """
    Data payload for task events.
    Contains minimal required information for task processing.
    """

    # Core identifiers
    task_id: str = Field(..., description="Unique task identifier")
    inference_id: str = Field(..., description="Parent inference identifier")
    process_id: str = Field(
        ..., description="Process/component identifier (e.g., 'inference-planner')"
    )
    step_number: int = Field(..., description="Pipeline step number (0-based)")

    # Processing context
    inference_folder: str = Field(..., description="Path to inference data folder")
    pipeline_steps: List[Dict[str, Any]] = Field(
        ..., description="Complete pipeline configuration"
    )

    # Optional fields
    priority: int = Field(
        default=5, ge=1, le=10, description="Task priority (1=highest, 10=lowest)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if task failed"
    )
    start_time: Optional[str] = Field(
        default=None, description="Task start timestamp (ISO 8601)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Task end timestamp (ISO 8601)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CloudEvent(BaseModel):
    """
    CloudEvents v1.0 specification compliant event.

    Required attributes:
    - specversion: CloudEvents spec version (always "1.0")
    - type: Event type from EventType enum
    - source: URI identifying the event source
    - id: Unique event identifier (auto-generated UUID)

    Optional attributes:
    - time: Event timestamp (auto-generated ISO 8601)
    - datacontenttype: Content type of data (default: "application/json")
    - subject: Subject of the event (e.g., "task/123")
    - data: Event payload
    """

    # Required CloudEvents attributes
    specversion: str = Field(
        default="1.0", description="CloudEvents specification version"
    )
    type: EventType = Field(..., description="Event type")
    source: str = Field(..., description="Event source URI")
    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event identifier"
    )

    # Optional CloudEvents attributes
    time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp in ISO 8601 format",
    )
    datacontenttype: str = Field(
        default="application/json", description="Content type of data"
    )
    subject: Optional[str] = Field(default=None, description="Subject of the event")
    data: Any = Field(..., description="Event payload data")

    class Config:
        use_enum_values = True


# Event creation helpers


def create_task_ready_event(
    task_id: str,
    inference_id: str,
    process_id: str,
    step_number: int,
    inference_folder: str,
    pipeline_steps: List[Dict[str, Any]],
    priority: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> CloudEvent:
    """
    Create a task ready event when a task is ready to be processed.

    This event triggers the corresponding Knative service to process the task.

    Args:
        task_id: Unique task identifier
        inference_id: Parent inference identifier
        process_id: Process/component identifier (e.g., 'inference-planner', 'terrakit-data-fetch')
        step_number: Pipeline step number (0-based)
        inference_folder: Path to inference data folder
        pipeline_steps: Complete pipeline steps configuration
        priority: Task priority (1=highest, 10=lowest)
        metadata: Additional metadata (e.g., {"triggered_by": "db_trigger"})

    Returns:
        CloudEvent with type TASK_READY

    Example:
        >>> event = create_task_ready_event(
        ...     task_id="task-123",
        ...     inference_id="inf-456",
        ...     process_id="inference-planner",
        ...     step_number=0,
        ...     inference_folder="/data/inf-456",
        ...     pipeline_steps=[...],
        ...     priority=5
        ... )
    """
    data = TaskEventData(
        task_id=task_id,
        inference_id=inference_id,
        process_id=process_id,
        step_number=step_number,
        inference_folder=inference_folder,
        pipeline_steps=pipeline_steps,
        priority=priority,
        metadata=metadata or {},
    )

    return CloudEvent(
        type=EventType.TASK_READY,
        source=f"geospatial-studio/task/{task_id}",
        subject=f"task/{task_id}/process/{process_id}",
        data=data.dict(),
    )


def create_task_completed_event(
    task_id: str,
    inference_id: str,
    process_id: str,
    step_number: int,
    inference_folder: str,
    pipeline_steps: List[Dict[str, Any]],
    start_time: str,
    end_time: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> CloudEvent:
    """
    Create a task completed event when a task finishes successfully.

    This event is published after successful task completion and may trigger
    the next step in the pipeline.

    Args:
        task_id: Unique task identifier
        inference_id: Parent inference identifier
        process_id: Process/component identifier
        step_number: Pipeline step number
        inference_folder: Path to inference data folder
        pipeline_steps: Complete pipeline steps configuration
        start_time: Task start timestamp (ISO 8601)
        end_time: Task end timestamp (ISO 8601)
        metadata: Additional metadata (e.g., {"duration_seconds": 45.2})

    Returns:
        CloudEvent with type TASK_COMPLETED

    Example:
        >>> event = create_task_completed_event(
        ...     task_id="task-123",
        ...     inference_id="inf-456",
        ...     process_id="inference-planner",
        ...     step_number=0,
        ...     inference_folder="/data/inf-456",
        ...     pipeline_steps=[...],
        ...     start_time="2025-01-15T10:00:00Z",
        ...     end_time="2025-01-15T10:00:45Z"
        ... )
    """
    data = TaskEventData(
        task_id=task_id,
        inference_id=inference_id,
        process_id=process_id,
        step_number=step_number,
        inference_folder=inference_folder,
        pipeline_steps=pipeline_steps,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata or {},
    )

    return CloudEvent(
        type=EventType.TASK_COMPLETED,
        source=f"geospatial-studio/service/{process_id}",
        subject=f"task/{task_id}/process/{process_id}",
        data=data.dict(),
    )


def create_task_failed_event(
    task_id: str,
    inference_id: str,
    process_id: str,
    step_number: int,
    inference_folder: str,
    pipeline_steps: List[Dict[str, Any]],
    error_message: str,
    start_time: str,
    end_time: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> CloudEvent:
    """
    Create a task failed event when a task fails with an error.

    This event is published when task processing fails and may trigger
    error handling or retry logic.

    Args:
        task_id: Unique task identifier
        inference_id: Parent inference identifier
        process_id: Process/component identifier
        step_number: Pipeline step number
        inference_folder: Path to inference data folder
        pipeline_steps: Complete pipeline steps configuration
        error_message: Error message describing the failure
        start_time: Task start timestamp (ISO 8601)
        end_time: Task end timestamp (ISO 8601)
        metadata: Additional metadata (e.g., {"error_code": "DATA_FETCH_FAILED"})

    Returns:
        CloudEvent with type TASK_FAILED

    Example:
        >>> event = create_task_failed_event(
        ...     task_id="task-123",
        ...     inference_id="inf-456",
        ...     process_id="terrakit-data-fetch",
        ...     step_number=0,
        ...     inference_folder="/data/inf-456",
        ...     pipeline_steps=[...],
        ...     error_message="Failed to fetch data from Sentinel Hub",
        ...     start_time="2025-01-15T10:00:00Z",
        ...     end_time="2025-01-15T10:00:30Z"
        ... )
    """
    data = TaskEventData(
        task_id=task_id,
        inference_id=inference_id,
        process_id=process_id,
        step_number=step_number,
        inference_folder=inference_folder,
        pipeline_steps=pipeline_steps,
        error_message=error_message,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata or {},
    )

    return CloudEvent(
        type=EventType.TASK_FAILED,
        source=f"geospatial-studio/service/{process_id}",
        subject=f"task/{task_id}/process/{process_id}",
        data=data.dict(),
    )


# Made with Bob
