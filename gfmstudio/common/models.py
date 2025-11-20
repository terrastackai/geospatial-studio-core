# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from ..common.db import Base


def utc_now():
    """Get the current UTC time.

    Returns:
        datetime: The current UTC time.
    """
    return datetime.now(timezone.utc)


class AbstractBase(Base):
    """Base model class to mixin common fields to all subclassed models."""

    __abstract__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        onupdate=func.now(),
    )
    created_by = Column(String(100))
    updated_by = Column(String(100))
    active = Column(Boolean, default=True, nullable=False)
    deleted = Column(Boolean, default=False)

    def __str__(self):
        raise NotImplementedError("Model should define a __str__ function.")

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__table__.c}
