# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import copy
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import ValidationError

from gfmstudio import websockets
from gfmstudio.amo import api as amo_apis
from gfmstudio.auth import routes as auth_routes
from gfmstudio.auth.rate_limit_middleware import RateLimitMiddleware
from gfmstudio.common.api import healthz
from gfmstudio.config import settings
from gfmstudio.cos_client import init_cos_client
from gfmstudio.fine_tuning import api as geoft_apis
from gfmstudio.inference.v2 import api as inference_apiv2
from gfmstudio.jira import jira_apis
from gfmstudio.log import logger

tags_metadata = [
    {
        "name": "Studio / Authentication",
        "description": "Obtain authentication token.",
    },
    {
        "name": "Inference / Models",
        "description": "View information on models available to user..",
    },
    {
        "name": "Inference / Inference",
        "description": "Operations to run inference pipeline.",
    },
    {
        "name": "Inference / Data Sources",
        "description": "Operations on layers available for an inference.",
    },
    {
        "name": "FineTuning / Tunes",
    },
    {
        "name": "FineTuning / Templates",
        "description": "Operations to run inference pipeline.",
    },
    {
        "name": "FineTuning / Datasets",
    },
    {
        "name": "FineTuning / Base models",
    },
    {
        "name": "Studio / Files",
    },
    {
        "name": "Studio / Notifications",
    },
    {
        "name": "Studio / Feedback",
    },
]


def initialize_redis():
    """Establish a connection to the Redis server."""
    logger.info("🔌 Initializing Redis ...")
    try:
        redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("✅ Redis Initialized Successfully")
        return redis_client
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"❌ Redis: Connection error: {settings.REDIS_URL} > {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"❌ Redis: An unexpected error occurred: {settings.REDIS_URL} > {str(e)}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.

    - Startup: Initialize Redis, COS client, and other resources
    - Shutdown: Cleanup connections and resources
    """
    logger.info("🚀 Application startup initiated")
    app.state.redis_client = initialize_redis()
    init_cos_client(app)
    logger.info("✅ Application startup complete")

    yield

    # Shutdown
    logger.info("🛑 Application shutdown initiated")
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        try:
            app.state.redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis connection: {e}")

    logger.info("✅ Application shutdown complete")


app = FastAPI(
    lifespan=lifespan,
    title="fm.geospatial inference APIs",
    version="2.0.0",
    summary="Geospatial Studio Inference Gateway APIs.",
    openapi_tags=tags_metadata,
    debug=settings.DEBUG,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

instrumentator = Instrumentator().instrument(
    app,
    metric_namespace="fmgeospatial",
    metric_subsystem="inference",
)
instrumentator.expose(app, include_in_schema=False)

RATE_LIMIT_CONFIG = {
    "default": {
        "/v2/async/inference/": {
            "POST": {
                "limit": settings.RATELIMIT_SENSITIVE_RESOURCE_LIMIT,
                "window": settings.RATELIMIT_SENSITIVE_RESOURCE_WINDOW,
            },
        },
        "/v2/submit-tune": {
            "POST": {
                "limit": settings.RATELIMIT_SENSITIVE_RESOURCE_LIMIT,
                "window": settings.RATELIMIT_SENSITIVE_RESOURCE_WINDOW,
            }
        },
        "/v2/datasets/onboard": {
            "POST": {
                "limit": settings.RATELIMIT_SENSITIVE_RESOURCE_LIMIT,
                "window": settings.RATELIMIT_SENSITIVE_RESOURCE_WINDOW,
            }
        },
        "/health/livez": {"GET": {"limit": 100, "window": 60}},
        "/health/readyz": {"GET": {"limit": 100, "window": 60}},
        #  "/v1/base-models": {
        #     "GET": {"limit": 5, "window": 60}
        # },
        "/": {"limit": settings.RATELIMIT_LIMIT, "window": settings.RATELIMIT_WINDOW},
    },
    # Placeholder before moving storage to DB
    # This configuration would be limiting rates of resource requests per-user.
    # The default rate-liming logic applies the smae limits to every user the same way
    # "Brian.Ogolla@ibm.com": {
    #     "/v1/submit_tune": {
    #         "GET": {"limit": 10, "window": 300}
    #     }
    # }
}


# Extend the config if custom RATE_LIMIT_CONFIG is defined via environment variable
if settings.RATE_LIMIT_CONFIG and isinstance(settings.RATE_LIMIT_CONFIG, dict):
    merged_config = copy.deepcopy(settings.RATE_LIMIT_CONFIG)
    for key in RATE_LIMIT_CONFIG:
        if key in merged_config:
            merged_config[key].update(RATE_LIMIT_CONFIG[key])
        else:
            merged_config[key] = RATE_LIMIT_CONFIG[key]
    RATE_LIMIT_CONFIG = merged_config

if settings.RATELIMIT_ENABLED is True:
    app.add_middleware(
        RateLimitMiddleware,
        redis_client=None,  # Will be set from app.state during requests
        rate_limit_config=RATE_LIMIT_CONFIG,
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Handler for all unhandled Validation Errors"""
    raise HTTPException(
        status_code=422,
        detail=exc.errors(),
    )


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


# Register health check endpoints (critical for Kubernetes probes)
app.include_router(healthz.router, prefix="/health", tags=["Health"])

# api_router.include_router(api_events.events_router)

app.include_router(auth_routes.router, prefix="/v2")
app.include_router(inference_apiv2.router, prefix="/v2")
app.include_router(geoft_apis.app, prefix="/v2")
app.include_router(amo_apis.app, prefix="/v2")
app.include_router(websockets.router)
app.include_router(jira_apis.jira_router)
