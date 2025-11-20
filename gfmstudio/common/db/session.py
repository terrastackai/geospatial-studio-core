# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ...config import settings

engine = create_engine(str(settings.DATABASE_URI), pool_size=30, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
