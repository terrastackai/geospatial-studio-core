# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings

# Auth database (shared across apps)
auth_engine = create_engine(str(settings.AUTH_DATABASE_URI), pool_pre_ping=True)
AuthSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=auth_engine)
