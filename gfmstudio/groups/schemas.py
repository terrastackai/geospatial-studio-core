# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime
from typing import Optional, Union
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
    artifact_id: Union[UUID, str]


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
    artifact_id: str
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


class GroupSharingInfo(BaseModel):
    """Information about a group an artifact is shared with."""

    group_id: UUID
    group_name: str
    granted_by: str
    granted_at: datetime
    user_role: Optional[GroupRole] = None  # Present if user is a member

    model_config = ConfigDict(from_attributes=True)


class ArtifactSharingDetail(BaseModel):
    """Detailed information about an artifact shared with a group."""

    artifact_id: str
    artifact_type: ArtifactType
    artifact_name: Optional[str] = None  # If available from artifact model
    granted_by: str
    granted_at: datetime
    created_by: str  # Artifact owner
    created_at: datetime  # When artifact was created

    model_config = ConfigDict(from_attributes=True)


class ArtifactSharingListResponse(BaseModel):
    """Paginated response for artifact sharing list."""

    total: int
    limit: int
    offset: int
    artifacts: list[ArtifactSharingDetail]


# Made with Bob
