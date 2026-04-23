# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""add groups and artifact permissions

Revision ID: a1b2c3d4e5f6
Revises: 9f02b7f65a7d
Create Date: 2026-03-12 17:59:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "9f02b7f65a7d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ENUM types only if they don't exist
    conn = op.get_bind()
    
    # Check and create group_role enum
    result = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'group_role'"
    )).fetchone()
    if not result:
        group_role_enum = postgresql.ENUM("owner", "member", name="group_role")
        group_role_enum.create(conn)
    
    # Check and create artifact_type_enum
    result = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'artifact_type_enum'"
    )).fetchone()
    if not result:
        artifact_type_enum = postgresql.ENUM(
            "dataset",
            "tune",
            "backbone",
            "model",
            "task_template",
            "inference_run",
            name="artifact_type_enum",
        )
        artifact_type_enum.create(conn)

    # Create groups table
    op.create_table(
        "groups",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Create group_members table
    op.create_table(
        "group_members",
        sa.Column("group_id", sa.UUID(), nullable=False),
        sa.Column("user_email", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM("owner", "member", name="group_role"),
            nullable=False,
            server_default="member",
        ),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["groups.id"],
            name="fk_group_members_group_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("group_id", "user_email"),
    )
    op.create_index(
        "ix_group_members_user_email", "group_members", ["user_email"], unique=False
    )

    # Create artifact_permissions table
    op.create_table(
        "artifact_permissions",
        sa.Column("group_id", sa.UUID(), nullable=False),
        sa.Column(
            "artifact_type",
            postgresql.ENUM(
                "dataset",
                "tune",
                "backbone",
                "model",
                "task_template",
                "inference_run",
                name="artifact_type_enum",
            ),
            nullable=False,
        ),
        sa.Column("artifact_id", sa.UUID(), nullable=False),
        sa.Column("granted_by", sa.String(length=255), nullable=False),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["groups.id"],
            name="fk_artifact_permissions_group_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("group_id", "artifact_type", "artifact_id"),
    )
    op.create_index(
        "ix_artifact_permissions_artifact",
        "artifact_permissions",
        ["artifact_type", "artifact_id"],
        unique=False,
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index("ix_artifact_permissions_artifact", table_name="artifact_permissions")
    op.drop_table("artifact_permissions")

    op.drop_index("ix_group_members_user_email", table_name="group_members")
    op.drop_table("group_members")

    op.drop_table("groups")

    # Drop ENUM types
    artifact_type_enum = postgresql.ENUM(
        "dataset",
        "tune",
        "backbone",
        "model",
        "task_template",
        "inference_run",
        name="artifact_type_enum",
    )
    artifact_type_enum.drop(op.get_bind())

    group_role_enum = postgresql.ENUM("owner", "member", name="group_role")
    group_role_enum.drop(op.get_bind())

# Made with Bob
