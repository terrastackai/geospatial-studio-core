# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from gfmstudio.auth.authorizer import auth_handler
from gfmstudio.common.api import utils
from gfmstudio.fine_tuning.models import BaseModels, GeoDataset, Tunes, TuneTemplate
from gfmstudio.groups.models import (
    ArtifactPermission,
    ArtifactType,
    Group,
    GroupMember,
    GroupRole,
)
from gfmstudio.groups.schemas import (
    ArtifactPermissionGrant,
    ArtifactPermissionOut,
    GroupCreate,
    GroupMemberAdd,
    GroupOut,
    MemberRoleUpdate,
)
from gfmstudio.inference.v2.models import Inference, Model

router = APIRouter(prefix="/groups", tags=["Groups"])

# Mapping of artifact types to their SQLAlchemy model classes
ARTIFACT_TYPE_TO_MODEL = {
    ArtifactType.dataset: GeoDataset,
    ArtifactType.tune: Tunes,
    ArtifactType.backbone: BaseModels,
    ArtifactType.model: Model,
    ArtifactType.task_template: TuneTemplate,
    ArtifactType.inference_run: Inference,
}


def _require_group_owner(group_id: UUID, user_email: str, db: Session) -> GroupMember:
    """
    Helper function to verify that a user is an owner of a group.

    Parameters
    ----------
    group_id : UUID
        The group ID to check
    user_email : str
        The user's email address
    db : Session
        Database session

    Returns
    -------
    GroupMember
        The group member record if user is an owner

    Raises
    ------
    HTTPException
        403 if user is not an owner of the group
    """
    member = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == user_email,
            )
        )
        .first()
    )

    if not member or member.role != GroupRole.owner:
        raise HTTPException(
            status_code=403,
            detail="Only group owners can perform this action",
        )

    return member


def _require_group_member(group_id: UUID, user_email: str, db: Session) -> GroupMember:
    """
    Helper function to verify that a user is a member of a group.

    Parameters
    ----------
    group_id : UUID
        The group ID to check
    user_email : str
        The user's email address
    db : Session
        Database session

    Returns
    -------
    GroupMember
        The group member record

    Raises
    ------
    HTTPException
        403 if user is not a member of the group
    """
    member = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == user_email,
            )
        )
        .first()
    )

    if not member:
        raise HTTPException(
            status_code=403,
            detail="You must be a member of this group to perform this action",
        )

    return member


# ***************************************************
# Group Management
# ***************************************************
@router.post("/", response_model=GroupOut, status_code=201)
async def create_group(
    group: GroupCreate,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Create a new group. The creator is automatically added as an owner.

    Parameters
    ----------
    group : GroupCreate
        Group creation data
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    GroupOut
        The created group with members
    """
    user_email = auth[0]

    # Check if group name already exists
    existing_group = db.query(Group).filter(Group.name == group.name).first()
    if existing_group:
        raise HTTPException(
            status_code=400,
            detail=f"Group with name '{group.name}' already exists",
        )

    # Create the group
    new_group = Group(
        name=group.name,
        description=group.description,
        created_by=user_email,
    )
    db.add(new_group)
    db.flush()

    # Add creator as owner
    creator_member = GroupMember(
        group_id=new_group.id,
        user_email=user_email,
        role=GroupRole.owner,
    )
    db.add(creator_member)
    db.commit()
    db.refresh(new_group)

    return new_group


@router.get("/", response_model=List[GroupOut])
async def list_groups(
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    List all groups the current user belongs to.

    Parameters
    ----------
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    List[GroupOut]
        List of groups the user is a member of
    """
    user_email = auth[0]

    groups = (
        db.query(Group)
        .join(GroupMember)
        .filter(GroupMember.user_email == user_email)
        .all()
    )

    return groups


@router.get("/{group_id}", response_model=GroupOut)
async def get_group(
    group_id: UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Get group details including members. User must be a member.

    Parameters
    ----------
    group_id : UUID
        The group ID
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    GroupOut
        The group details with members
    """
    user_email = auth[0]

    # Verify user is a member
    _require_group_member(group_id, user_email, db)

    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    return group


@router.delete("/{group_id}", status_code=204)
async def delete_group(
    group_id: UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Delete a group and all associated data. Must be an owner.

    Parameters
    ----------
    group_id : UUID
        The group ID
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)
    """
    user_email = auth[0]

    # Verify user is an owner
    _require_group_owner(group_id, user_email, db)

    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    db.delete(group)
    db.commit()


# ***************************************************
# Member Management
# ***************************************************
@router.post("/{group_id}/members", response_model=GroupOut, status_code=201)
async def add_member(
    group_id: UUID,
    member: GroupMemberAdd,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Add a user to the group. Only owners can add members.

    Parameters
    ----------
    group_id : UUID
        The group ID
    member : GroupMemberAdd
        Member data to add
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    GroupOut
        The updated group with members
    """
    user_email = auth[0]

    # Verify user is an owner
    _require_group_owner(group_id, user_email, db)

    # Check if group exists
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Check if member already exists
    existing_member = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == member.user_email,
            )
        )
        .first()
    )
    if existing_member:
        raise HTTPException(
            status_code=400,
            detail=f"User {member.user_email} is already a member of this group",
        )

    # Add new member
    new_member = GroupMember(
        group_id=group_id,
        user_email=member.user_email,
        role=member.role,
    )
    db.add(new_member)
    db.commit()
    db.refresh(group)

    return group


@router.delete("/{group_id}/members/{user_email}", status_code=204)
async def remove_member(
    group_id: UUID,
    user_email: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Remove a member from the group.
    Owners can remove anyone; members can remove themselves.

    Parameters
    ----------
    group_id : UUID
        The group ID
    user_email : str
        Email of the user to remove
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)
    """
    current_user_email = auth[0]

    # Get the member to be removed
    member_to_remove = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == user_email,
            )
        )
        .first()
    )

    if not member_to_remove:
        raise HTTPException(status_code=404, detail="Member not found in this group")

    # Check permissions: must be owner or removing self
    current_member = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == current_user_email,
            )
        )
        .first()
    )

    if not current_member:
        raise HTTPException(
            status_code=403,
            detail="You must be a member of this group to perform this action",
        )

    # Allow if user is owner OR removing themselves
    if current_member.role != GroupRole.owner and user_email != current_user_email:
        raise HTTPException(
            status_code=403,
            detail="Only group owners can remove other members",
        )

    db.delete(member_to_remove)
    db.commit()


@router.put("/{group_id}/members/{user_email}/role", response_model=GroupOut)
async def update_member_role(
    group_id: UUID,
    user_email: str,
    role_update: MemberRoleUpdate,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Update a member's role. Only owners can update roles.
    Cannot demote the last remaining owner.

    Parameters
    ----------
    group_id : UUID
        The group ID
    user_email : str
        Email of the user whose role to update
    role_update : MemberRoleUpdate
        New role data
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    GroupOut
        The updated group with members
    """
    current_user_email = auth[0]

    # Verify current user is an owner
    _require_group_owner(group_id, current_user_email, db)

    # Get the member to update
    member = (
        db.query(GroupMember)
        .filter(
            and_(
                GroupMember.group_id == group_id,
                GroupMember.user_email == user_email,
            )
        )
        .first()
    )

    if not member:
        raise HTTPException(status_code=404, detail="Member not found in this group")

    # If demoting from owner to member, check if this is the last owner
    if member.role == GroupRole.owner and role_update.role == GroupRole.member:
        owner_count = (
            db.query(func.count(GroupMember.user_email))
            .filter(
                and_(
                    GroupMember.group_id == group_id,
                    GroupMember.role == GroupRole.owner,
                    GroupMember.user_email != user_email,
                )
            )
            .scalar()
        )

        if owner_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot demote the last owner. Promote another member to owner first.",
            )

    # Update the role
    member.role = role_update.role
    db.commit()

    # Refresh and return the group
    group = db.query(Group).filter(Group.id == group_id).first()
    db.refresh(group)

    return group


# ***************************************************
# Artifact Permissions
# ***************************************************
@router.get("/{group_id}/artifacts", response_model=List[ArtifactPermissionOut])
async def list_group_artifacts(
    group_id: UUID,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    List all artifacts shared with this group. User must be a member.

    Parameters
    ----------
    group_id : UUID
        The group ID
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    List[ArtifactPermissionOut]
        List of artifact permissions for this group
    """
    user_email = auth[0]

    # Verify user is a member
    _require_group_member(group_id, user_email, db)

    permissions = (
        db.query(ArtifactPermission)
        .filter(ArtifactPermission.group_id == group_id)
        .all()
    )

    return permissions


@router.post(
    "/{group_id}/artifacts", response_model=ArtifactPermissionOut, status_code=201
)
async def grant_artifact_permission(
    group_id: UUID,
    permission: ArtifactPermissionGrant,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Share an artifact with the group. The caller must own the artifact.
    Only owners can share artifacts.

    Parameters
    ----------
    group_id : UUID
        The group ID
    permission : ArtifactPermissionGrant
        Artifact permission data
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)

    Returns
    -------
    ArtifactPermissionOut
        The created artifact permission
    """
    user_email = auth[0]

    # Verify user is an owner
    _require_group_owner(group_id, user_email, db)

    # Verify group exists
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Get the model class for this artifact type
    model_class = ARTIFACT_TYPE_TO_MODEL.get(permission.artifact_type)
    if not model_class:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact type: {permission.artifact_type}",
        )

    # Verify artifact exists and is owned by current user
    artifact = (
        db.query(model_class).filter(model_class.id == permission.artifact_id).first()
    )

    if not artifact:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {permission.artifact_type} with id {permission.artifact_id}",
        )

    # Check ownership - artifact must be created by current user
    if hasattr(artifact, "created_by") and artifact.created_by != user_email:
        raise HTTPException(
            status_code=403,
            detail="You can only share artifacts that you own",
        )

    # Check if permission already exists
    existing_permission = (
        db.query(ArtifactPermission)
        .filter(
            and_(
                ArtifactPermission.group_id == group_id,
                ArtifactPermission.artifact_type == permission.artifact_type,
                ArtifactPermission.artifact_id == permission.artifact_id,
            )
        )
        .first()
    )

    if existing_permission:
        raise HTTPException(
            status_code=400,
            detail="This artifact is already shared with this group",
        )

    # Create the permission
    new_permission = ArtifactPermission(
        group_id=group_id,
        artifact_type=permission.artifact_type,
        artifact_id=permission.artifact_id,
        granted_by=user_email,
    )
    db.add(new_permission)
    db.commit()
    db.refresh(new_permission)

    return new_permission


@router.delete("/{group_id}/artifacts/{artifact_type}/{artifact_id}", status_code=204)
async def revoke_artifact_permission(
    group_id: UUID,
    artifact_type: ArtifactType,
    artifact_id: str,
    db: Session = Depends(utils.get_db),
    auth=Depends(auth_handler),
):
    """
    Revoke a group's access to an artifact. Only owners can revoke access.

    Parameters
    ----------
    group_id : UUID
        The group ID
    artifact_type : ArtifactType
        The type of artifact
    artifact_id : str
        The artifact ID
    db : Session
        Database session
    auth : tuple
        Authentication tuple (email, token, groups)
    """
    user_email = auth[0]

    # Verify user is an owner
    _require_group_owner(group_id, user_email, db)

    # Find the permission
    permission = (
        db.query(ArtifactPermission)
        .filter(
            and_(
                ArtifactPermission.group_id == group_id,
                ArtifactPermission.artifact_type == artifact_type,
                ArtifactPermission.artifact_id == artifact_id,
            )
        )
        .first()
    )

    if not permission:
        raise HTTPException(
            status_code=404,
            detail="Artifact permission not found",
        )

    db.delete(permission)
    db.commit()


# Made with Bob
