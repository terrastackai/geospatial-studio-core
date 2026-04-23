# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from gfmstudio.config import settings
from gfmstudio.groups.models import ArtifactPermission, ArtifactType, GroupMember


def build_visibility_filter(
    model_class, user_email: str, artifact_type: ArtifactType, db: Session
):
    """
    Build a SQLAlchemy filter clause for artifact visibility.

    An artifact is visible to a user if any of the following conditions are true:
    1. The user owns the artifact (created_by == user_email)
    2. The artifact is system-wide (created_by == system@ibm.com)
    3. The artifact is shared with a group the user belongs to

    Parameters
    ----------
    model_class : Type
        The SQLAlchemy model class for the artifact (e.g., Tunes, GeoDataset, Model)
    user_email : str
        The email of the current user
    artifact_type : ArtifactType
        The type of artifact being queried
    db : Session
        Database session

    Returns
    -------
    BinaryExpression
        A SQLAlchemy or_() clause that can be used in a filter

    Examples
    --------
    >>> from gfmstudio.fine_tuning.models import Tunes
    >>> from gfmstudio.groups.models import ArtifactType
    >>> filter_clause = build_visibility_filter(Tunes, "user@example.com", ArtifactType.tune, db)
    >>> tunes = db.query(Tunes).filter(filter_clause).all()
    """
    # Subquery to get artifact IDs shared with groups the user belongs to
    # This finds all artifact_ids of the given type that are shared with
    # any group where the user is a member
    shared_artifacts_subquery = (
        select(ArtifactPermission.artifact_id)
        .join(
            GroupMember,
            GroupMember.group_id == ArtifactPermission.group_id,
        )
        .where(
            GroupMember.user_email == user_email,
            ArtifactPermission.artifact_type == artifact_type,
        )
        .scalar_subquery()
    )

    # Build the visibility filter with three conditions
    visibility_filter = or_(
        # Condition 1: User owns the artifact
        model_class.created_by == user_email,
        # Condition 2: Artifact is system-wide
        model_class.created_by == settings.DEFAULT_SYSTEM_USER,
        # Condition 3: Artifact is shared with a group the user belongs to
        model_class.id.in_(shared_artifacts_subquery),
    )

    return visibility_filter


# Made with Bob
