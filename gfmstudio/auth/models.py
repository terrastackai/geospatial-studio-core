# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, UUID, Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import as_declarative, declared_attr, relationship
from sqlalchemy.sql import func

from .api_key_utils import apikey_expiry_date


@as_declarative()
class AuthBase:
    id: Any
    __name__: str

    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


class AuthAbstractBase(AuthBase):
    """Auth Base model class to mixin common fields to all subclassed models."""

    __abstract__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
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


class Organization(AuthAbstractBase):
    """Organization table."""

    __tablename__ = "organization"

    name = Column(String(300), nullable=False)

    def __str__(self):
        return f"{self.name[:15]}"


class User(AuthAbstractBase):
    """System users table."""

    __tablename__ = "user"

    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(150), unique=True, nullable=False)
    data_usage_consent = Column(Boolean, default=False)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organization.id"), nullable=True
    )
    extra_data = Column(JSON)

    organization = relationship("Organization", backref="users", lazy="joined")

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class APIKey(AuthAbstractBase):
    """APIKey table."""

    value = Column(Text(), nullable=False)
    hashed_key = Column(String, unique=True, index=True)
    last_used_at = Column(DateTime(timezone=True))
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"))
    expires_on = Column(DateTime, default=apikey_expiry_date)

    user = relationship("User", backref="apikeys", lazy="joined")

    def __str__(self):
        return f"{self.value[:15]}"
