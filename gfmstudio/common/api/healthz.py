# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from requests import Session
from sqlalchemy.sql import text

from gfmstudio.common.api.utils import get_db
from gfmstudio.log import logger

router = APIRouter()


class StatusEnum(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class LiveCheckResponse(BaseModel):
    status: StatusEnum


class ReadyCheckResponse(BaseModel):
    status: StatusEnum
    checks: dict[str, StatusEnum]


def check_postgres_readiness(db: Session) -> None:
    """Checks if the database is ready."""
    try:
        db.execute(text("SELECT id FROM model_usecase LIMIT 1;"))
        db.close()
    except Exception:
        raise HTTPException(status_code=503, detail="Database not ready")


def check_redis_readiness() -> None:
    """Checks if Redis is ready."""
    # TODO: Add check connecting to Redis instance.


# Define the readiness and liveness endpoints
@router.get("/readyz", response_model=ReadyCheckResponse)
async def readiness(db: Session = Depends(get_db)):
    """Checks the readiness of the application.

    Performs checks for the availability of the database and Redis.
    Returns a 503 status code if any of the checks fail.

    """
    resp = ReadyCheckResponse(
        **{
            "status": StatusEnum.DOWN,
            "checks": {
                "postgresql": StatusEnum.DOWN,
            },
        }
    )

    try:
        # Check if the database is ready
        check_postgres_readiness(db=db)
        resp.checks["postgresql"] = StatusEnum.UP
    except Exception:
        logger.error("Database not ready")

    try:
        # Check if Redis is ready
        check_redis_readiness()
    except Exception:
        logger.error("Redis not ready")

    if all(value == StatusEnum.UP for value in resp.checks.values()):
        resp.status = StatusEnum.UP

    return resp


@router.get("/livez", response_model=LiveCheckResponse)
async def liveness():
    """Checks the liveness of the application.

    Returns a 200 status code indicating that the application is alive.
    """
    return {"status": "UP"}
