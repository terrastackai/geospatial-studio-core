# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import time
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from gfmstudio.common.api.utils import get_db
from gfmstudio.log import logger
from gfmstudio.redis_client import get_redis_client

router = APIRouter()
_health_cache = {"readyz": None, "livez": None}
HEALTH_CACHE_TTL = 2  # seconds


class StatusEnum(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class LiveCheckResponse(BaseModel):
    status: StatusEnum
    version: Optional[str] = "2.0.0"


class ReadyCheckResponse(BaseModel):
    status: StatusEnum


def check_postgres_readiness(db: Session) -> None:
    """Checks if the database is ready."""
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database readiness check failed")
        raise HTTPException(status_code=503, detail="Database not ready")


def check_redis_readiness() -> None:
    """Checks if Redis is ready."""
    try:
        redis_client = get_redis_client()
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis client not initialized")
        redis_client.ping()
    except Exception:
        logger.exception("Redis readiness check failed")
        raise HTTPException(status_code=503, detail="Redis not ready")


# Define the readiness and liveness endpoints
@router.get("/readyz", response_model=ReadyCheckResponse, include_in_schema=False)
async def readiness(db: Session = Depends(get_db)):
    """Checks the readiness of the application.

    Performs checks for the availability of the database and Redis.
    Returns a 503 status code if any of the checks fail.
    """
    current_time = time.time()

    # Check cache first
    cached = _health_cache.get("readyz")
    if cached and (current_time - cached["timestamp"]) < HEALTH_CACHE_TTL:
        resp = ReadyCheckResponse(status=cached["status"])
        if cached["status"] == StatusEnum.DOWN:
            raise HTTPException(status_code=503, detail=resp.model_dump())
        return resp

    # Cache miss or expired - perform actual checks
    all_ready = True

    try:
        # Check if the database is ready
        check_postgres_readiness(db)
    except HTTPException:
        all_ready = False
    except Exception:
        logger.exception("Unexpected error during database check")
        all_ready = False

    try:
        # Check if Redis is ready
        check_redis_readiness()
    except HTTPException:
        all_ready = False
    except Exception:
        logger.exception("Unexpected error during Redis check")
        all_ready = False

    # Determine overall status
    overall_status = StatusEnum.UP if all_ready else StatusEnum.DOWN

    # Update cache
    _health_cache["readyz"] = {"status": overall_status, "timestamp": current_time}

    resp = ReadyCheckResponse(status=overall_status)

    if not all_ready:
        raise HTTPException(status_code=503, detail=resp.model_dump())

    return resp


@router.get("/livez", response_model=LiveCheckResponse, include_in_schema=False)
async def liveness():
    """Checks the liveness of the application.

    Returns a 200 status code indicating that the application is alive.
    """
    current_time = time.time()
    cached = _health_cache.get("livez")
    if cached and (current_time - cached["timestamp"]) < HEALTH_CACHE_TTL:
        return LiveCheckResponse(status=cached["status"], version="2.0.0")

    _health_cache["livez"] = {"status": StatusEnum.UP, "timestamp": current_time}

    return LiveCheckResponse(status=StatusEnum.UP, version="2.0.0")
