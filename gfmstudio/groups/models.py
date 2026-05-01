# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import enum
import uuid

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from gfmstudio.common.db import Base


class GroupRole(str, enum.Enum):
    """Enum for group member roles."""

    owner = "owner"
    member = "member"


class ArtifactType(str, enum.Enum):
    """Enum for artifact types that can be shared with groups."""

    dataset = "dataset"
    tune = "tune"
    backbone = "backbone"
    model = "model"
    task_template = "task_template"
    inference_run = "inference_run"


class Group(Base):
    """Group model for team-based artifact sharing."""

    __tablename__ = "groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_by = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    members = relationship(
        "GroupMember",
        back_populates="group",
        cascade="all, delete-orphan",
        lazy="joined",
    )
    permissions = relationship(
        "ArtifactPermission",
        back_populates="group",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __str__(self):
        return f"Group(id={self.id}, name={self.name})"


class GroupMember(Base):
    """Group membership model linking users to groups with roles."""

    __tablename__ = "group_members"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    user_email = Column(String(255), primary_key=True, nullable=False)
    role = Column(
        Enum(GroupRole, name="group_role"),
        nullable=False,
        server_default="member",
    )
    added_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    group = relationship("Group", back_populates="members")

    def __str__(self):
        return f"GroupMember(group_id={self.group_id}, user={self.user_email}, role={self.role})"


class ArtifactPermission(Base):
    """Artifact permission model tracking which artifacts are shared with which groups."""

    __tablename__ = "artifact_permissions"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    artifact_type = Column(
        Enum(ArtifactType, name="artifact_type_enum"),
        primary_key=True,
        nullable=False,
    )
    artifact_id = Column(String(255), primary_key=True, nullable=False)
    granted_by = Column(String(255), nullable=False)
    granted_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    group = relationship("Group", back_populates="permissions")

    def __str__(self):
        return (
            f"ArtifactPermission(group_id={self.group_id}, "
            f"type={self.artifact_type}, artifact_id={self.artifact_id})"
        )


# Made with Bob
