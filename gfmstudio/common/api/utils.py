# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from typing import Generator

from sqlalchemy.orm import Session

from gfmstudio.common.db.session import SessionLocal, engine
from gfmstudio.log import logger


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    logger.debug("Current Connection Pool Number: %s", engine.pool.checkedout())
    try:
        yield db
    finally:
        db.close()


def is_valid_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def generate_internal_name(display_name: str, version: str = None) -> str:
    """Generate an internal name based on the display name."""
    # Replace underscores and spaces with hyphens, and convert to lowercase
    # Truncate to 30 characters to fit within database constraints
    internal_name = display_name.replace("_", "-").replace(" ", "-").lower()
    internal_name = internal_name[:30]
    if version:
        internal_name += f"-v{int(version)}"
    return f"{internal_name}-{uuid.uuid4().hex[:8]}"
