# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""change artifact_id to string

Revision ID: b7c8d9e0f1a2
Revises: f6fbafa54a75
Create Date: 2026-04-30 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "b7c8d9e0f1a2"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Change artifact_id column from UUID to VARCHAR(255) in artifact_permissions table.
    This allows non-UUID artifact IDs to be used in permissions.
    """
    # Drop the existing index on artifact_type and artifact_id
    op.drop_index("ix_artifact_permissions_artifact", table_name="artifact_permissions")
    
    # Drop the primary key constraint
    op.drop_constraint(
        "artifact_permissions_pkey", "artifact_permissions", type_="primary"
    )
    
    # Alter the artifact_id column type from UUID to VARCHAR(255)
    # PostgreSQL will automatically cast UUID to text/varchar
    op.alter_column(
        "artifact_permissions",
        "artifact_id",
        type_=sa.String(255),
        existing_type=postgresql.UUID(),
        existing_nullable=False,
    )
    
    # Recreate the primary key constraint
    op.create_primary_key(
        "artifact_permissions_pkey",
        "artifact_permissions",
        ["group_id", "artifact_type", "artifact_id"],
    )
    
    # Recreate the index on artifact_type and artifact_id
    op.create_index(
        "ix_artifact_permissions_artifact",
        "artifact_permissions",
        ["artifact_type", "artifact_id"],
        unique=False,
    )


def downgrade() -> None:
    """
    Revert artifact_id column from VARCHAR(255) back to UUID.
    WARNING: This will fail if any artifact_id values are not valid UUIDs.
    """
    # Drop the index
    op.drop_index("ix_artifact_permissions_artifact", table_name="artifact_permissions")
    
    # Drop the primary key constraint
    op.drop_constraint(
        "artifact_permissions_pkey", "artifact_permissions", type_="primary"
    )
    
    # Alter the artifact_id column type from VARCHAR(255) back to UUID
    # This will fail if any values are not valid UUIDs
    op.alter_column(
        "artifact_permissions",
        "artifact_id",
        type_=postgresql.UUID(),
        existing_type=sa.String(255),
        existing_nullable=False,
        postgresql_using="artifact_id::uuid",
    )
    
    # Recreate the primary key constraint
    op.create_primary_key(
        "artifact_permissions_pkey",
        "artifact_permissions",
        ["group_id", "artifact_type", "artifact_id"],
    )
    
    # Recreate the index
    op.create_index(
        "ix_artifact_permissions_artifact",
        "artifact_permissions",
        ["artifact_type", "artifact_id"],
        unique=False,
    )


# Made with Bob