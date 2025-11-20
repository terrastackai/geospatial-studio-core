# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Generator

from sqlalchemy.orm import Session

from .db_session import AuthSessionLocal


def get_auth_db() -> Generator[Session, None, None]:
    """Dependency for auth db"""
    try:
        db = AuthSessionLocal()
        yield db
    finally:
        db.close()
