# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    lateral,
    literal_column,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import column_property, relationship

from ...common.models import AbstractBase, utc_now
from ..types import ModelStatus


class Model(AbstractBase):
    """ML models table.

    This class contains the metadata for ML models.
    """

    __tablename__ = "inf_model"

    internal_name = Column(String(63), nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text)

    model_url = Column(String)
    pipeline_steps = Column(JSON)
    geoserver_push = Column(JSON)
    model_input_data_spec = Column(JSON)
    postprocessing_options = Column(JSON)
    status = Column(
        String(50),
        default=ModelStatus.PENDING,
        nullable=False,
    )
    groups = Column(JSONB)
    sharable = Column(Boolean, default=True)
    model_onboarding_config = Column(JSONB)
    version = Column(Float)
    latest = Column(Boolean, default=False)

    def __str__(self):
        return f"{self.internal_name} > {self.id}"


class Task(AbstractBase):
    """Inference task table.

    This class contains the metadata for inference tasks.
    """

    __tablename__ = "inf_task"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(
        String(50),
        default=ModelStatus.PENDING,
        nullable=False,
    )
    pipeline_steps = Column(JSONB)
    inference_id = Column(
        UUID(as_uuid=True), ForeignKey("inf_inference.id"), nullable=True
    )
    task_id = Column(String(100), unique=True, nullable=False)
    inference_folder = Column(String(100))
    priority = Column(String(50), default=5)  # 1 (highest) to 10 (lowest)
    queue = Column(String(50))

    def __str__(self):
        return f"{self.model.name} > {self.id}"


class Inference(AbstractBase):
    """Inference table.

    This class contains the metadata for inferences.
    """

    __tablename__ = "inf_inference"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    description = Column(Text)
    location = Column(String(100))
    inference_config = Column(JSONB)
    geoserver_layers = Column(JSON)
    priority = Column(String(50))
    queue = Column(String(50))
    demo = Column(JSONB)
    # info = Column(JSON)
    status = Column(
        String(50),
        default=ModelStatus.PENDING,
        nullable=False,
    )
    model_id = Column(UUID(as_uuid=True), ForeignKey("inf_model.id"), nullable=False)
    inference_output = Column(JSON)

    # Relationships to other tables
    model = relationship("Model", backref="inferences", lazy="joined")
    tasks = relationship("Task", backref="inference", lazy="joined")

    tasks_count_total = column_property(
        select(func.sum(func.jsonb_array_length(Task.pipeline_steps)))
        .where(Task.inference_id == id)
        .correlate_except(Task)
        .scalar_subquery()
    )

    # Total number of steps across all tasks that completed
    # SELECT count(*)
    # FROM task
    # JOIN LATERAL jsonb_array_elements(task.pipeline_steps) AS step ON TRUE
    # WHERE task.inference_id = '904d1e13-ddd2-415f-a963-120d16a240f0'
    #   AND step->>'status' = 'FINISHED';
    step_alias = lateral(func.jsonb_array_elements(Task.pipeline_steps)).alias("step")
    tasks_count_success = column_property(
        select(func.count())
        .select_from(Task)
        .join(step_alias, literal_column("TRUE"))
        .where(Task.inference_id == id, literal_column("step->>'status'") == "FINISHED")
        .correlate_except(Task)
        .scalar_subquery()
    )

    tasks_count_failed = column_property(
        select(func.count())
        .select_from(Task)
        .join(step_alias, literal_column("TRUE"))
        .where(Task.inference_id == id, literal_column("step->>'status'") == "FAILED")
        .correlate_except(Task)
        .scalar_subquery()
    )

    tasks_count_stopped = column_property(
        select(func.count())
        .select_from(Task)
        .join(step_alias, literal_column("TRUE"))
        .where(Task.inference_id == id, literal_column("step->>'status'") == "STOPPED")
        .correlate_except(Task)
        .scalar_subquery()
    )

    tasks_count_waiting = column_property(
        select(func.count())
        .select_from(Task)
        .join(step_alias, literal_column("TRUE"))
        .where(
            Task.inference_id == id,
            or_(
                literal_column("step->>'status'") == "WAITING",
                literal_column("step->>'status'") == "READY",
            ),
        )
        .correlate_except(Task)
        .scalar_subquery()
    )

    def __str__(self):
        return f"{self.id} > {self.id}"


class Notification(AbstractBase):
    """Notification table.

    This class contains the metadata for notifications.
    """

    __tablename__ = "notification"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detail_type = Column(String(100))
    detail = Column(JSON)
    source = Column(String(100))
    timestamp = Column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text("timezone('utc', now())"),
        nullable=False,
    )
    inference_id = Column(
        UUID(as_uuid=True), ForeignKey("inf_inference.id"), nullable=True
    )

    inference = relationship("Inference", backref="notifications", lazy="joined")

    def __str__(self):
        return f"{self.source} > {self.event_id}"

class GenericProcessor(AbstractBase):
    """Generic Processor table.

    This class contains the metadata for generic processors.
    """

    __tablename__ = "inf_generic_processor"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    processor_file_path = Column(String)
    processor_parameters = Column(JSON)
    status = Column(
        String(50),
        default=ModelStatus.PENDING,
        nullable=False,
    )

    def __str__(self):
        return f"{self.name} > {self.id}"
