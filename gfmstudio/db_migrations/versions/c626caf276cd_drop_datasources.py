# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""drop_datasources

Revision ID: c626caf276cd
Revises: f6fbafa54a75
Create Date: 2025-10-14 15:51:40.459290

"""
from alembic import op
import sqlalchemy as sa
import uuid
from sqlalchemy.dialects.postgresql import UUID, JSONB
import os


# revision identifiers, used by Alembic.
revision = "c626caf276cd"
down_revision = "f6fbafa54a75"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("inf_data_source")


def downgrade() -> None:
    op.create_table(
        "inf_data_source",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=100)),
        sa.Column("updated_by", sa.String(length=100)),
        sa.Column("active", sa.Boolean, nullable=False, default=True),
        sa.Column("deleted", sa.Boolean, default=False),
        sa.Column("collection_id", sa.String(length=100)),
        sa.Column("data_connector", sa.String(length=100)),
        sa.Column("data_connector_config", sa.JSON),
        sa.Column("groups", JSONB),
        sa.Column("sharable", sa.Boolean, default=True),
    )
    migration_dir = os.path.dirname(__file__)
    sql_file_path = os.path.join(
        migration_dir, "..", "seed_data", "00-inf-data-sources.sql"
    )
    sql_file_path = os.path.abspath(sql_file_path)
    with open(sql_file_path, "r") as f:
        sql_command = f.read()
    op.execute(sql_command)
