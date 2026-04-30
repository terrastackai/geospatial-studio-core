# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from gfmstudio.groups.models import ArtifactType, GroupRole


class GroupCreate(BaseModel):
    """Schema for creating a new group."""

    name: str
    description: Optional[str] = None


class GroupMemberAdd(BaseModel):
    """Schema for adding a member to a group."""

    user_email: str
    role: GroupRole = GroupRole.member


class MemberRoleUpdate(BaseModel):
    """Schema for updating a member's role."""

    role: GroupRole


class ArtifactPermissionGrant(BaseModel):
    """Schema for granting artifact access to a group."""

    artifact_type: ArtifactType
    artifact_id: UUID


class MemberOut(BaseModel):
    """Schema for group member response."""

    user_email: str
    role: GroupRole
    added_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArtifactPermissionOut(BaseModel):
    """Schema for artifact permission response."""

    group_id: UUID
    artifact_type: ArtifactType
    artifact_id: UUID
    granted_by: str
    granted_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GroupOut(BaseModel):
    """Schema for group response."""

    id: UUID
    name: str
    description: Optional[str]
    created_by: str
    created_at: datetime
    members: list[MemberOut] = []

    model_config = ConfigDict(from_attributes=True)


# Made with Bob
