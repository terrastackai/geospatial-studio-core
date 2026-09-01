# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from multiprocessing import pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ...config import settings

engine = create_engine(
    str(settings.DATABASE_URI),
    pool_size=30, 
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800, # Recycle faster than PgBouncer's server_lifetime
    
    # CRITICAL for PgBouncer (Transaction Mode):
    # This prevents 'Prepared Statement' errors
    execution_options={
        "compiled_cache": None 
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
