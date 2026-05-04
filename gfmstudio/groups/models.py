# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import enum

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from gfmstudio.common.models import AbstractBase


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


class Group(AbstractBase):
    """Group model for team-based artifact sharing."""

    __tablename__ = "groups"

    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)

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


class GroupMember(AbstractBase):
    """Group membership model linking users to groups with roles."""

    __tablename__ = "group_members"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_email = Column(String(255), nullable=False, index=True)
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

    # Table constraints
    __table_args__ = (
        UniqueConstraint("group_id", "user_email", name="uq_group_member_group_user"),
    )

    def __str__(self):
        return f"GroupMember(id={self.id}, group_id={self.group_id}, user={self.user_email}, role={self.role})"


class ArtifactPermission(AbstractBase):
    """Artifact permission model tracking which artifacts are shared with which groups."""

    __tablename__ = "artifact_permissions"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type = Column(
        Enum(ArtifactType, name="artifact_type_enum"),
        nullable=False,
        index=True,
    )
    artifact_id = Column(String(255), nullable=False, index=True)
    granted_by = Column(String(255), nullable=False)
    granted_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    group = relationship("Group", back_populates="permissions")

    def __str__(self):
        return (
            f"ArtifactPermission(id={self.id}, group_id={self.group_id}, "
            f"type={self.artifact_type}, artifact_id={self.artifact_id})"
        )


# Made with Bob
