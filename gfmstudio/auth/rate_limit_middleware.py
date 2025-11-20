# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import time
from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from redis import Redis
from starlette.middleware.base import BaseHTTPMiddleware

from .authorizer import auth_handler

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate-limiting requests based on user or group identifiers.
    Limits are enforced using Redis and can be customized per user or group.
    """

    def __init__(
        self,
        app,
        redis_client: Redis,
        rate_limit_config: dict,
        excluded_paths: Optional[list] = [
            "/",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        ],
    ):
        """
        Middleware for rate-limiting requests based on user or group identifiers.

        Parameters
        ----------
        app : ASGIApp
            The FastAPI or Starlette application instance.
        redis_client : Redis
            The Redis client used for storing and checking rate limits.
        rate_limit_config : dict
            A dictionary containing rate limit configurations for users, groups,
            and defaults.
        excluded_paths : list, optional
            A list of paths to exclude from rate-limiting checks.
        """
        super().__init__(app)
        self.redis = redis_client
        self.rate_limit_config = rate_limit_config
        self.default_limit = (
            rate_limit_config.get("default", {}).get("/", {}).get("limit")
        )
        self.default_window = (
            rate_limit_config.get("default", {}).get("/", {}).get("window")
        )
        self.excluded_paths = excluded_paths or []

    async def dispatch(self, request: Request, call_next):
        """
        Main middleware logic for intercepting requests and enforcing rate limits for incoming requests.

        Parameters
        ----------
        request : Request
            The incoming HTTP request.
        call_next : Callable
            The function to call the next middleware or endpoint.

        Returns
        -------
        Response
            The HTTP response after processing the request.

        Raises
        ------
        HTTPException
            Raised with status code 429 if the rate limit is exceeded.
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        default_auth_details = [
            "default",
            "auth_value",
            [],
        ]
        api_key = request.headers.get("x-api-key")
        jwt_token = request.headers.get("authorization")
        auth_type = "api_key" if api_key else "jwt_token"
        try:
            auth_details = (
                await auth_handler(request, jwt_header=jwt_token, api_key=api_key)
                or default_auth_details
            )
        except Exception:
            response = await call_next(request)
            return response

        endpoint = request.url.path
        identifier = f"{auth_type}:{auth_details[0]}:{endpoint}:{request.method}"

        rate_limit, time_window = self.get_rate_limit(
            auth_details={
                "type": auth_type,
                "id": auth_details[0],
                "group": auth_details[0],
            },
            endpoint=endpoint,
            method=request.method,
        )

        if not rate_limit or not time_window:
            response = await call_next(request)
            return response

        if not await self.is_request_allowed(identifier, rate_limit, time_window):
            # Raise a 429 Too Many Requests if rate limit is exceeded
            return JSONResponse(
                status_code=429,
                content={
                    "detail": [
                        {
                            "msg": f"Rate limit exceeded. Allowed {rate_limit} requests in {time_window} seconds.",
                        }
                    ]
                },
                headers={
                    "X-Rate-Limit-Limit": str(rate_limit),
                    "X-Rate-Limit-Remaining": "0",
                    "X-Rate-Limit-Reset": str(time_window),
                },
                media_type="application/json",
            )

        response = await call_next(request)
        response.headers["X-Rate-Limit-Limit"] = str(rate_limit)
        response.headers["X-Rate-Limit-Remaining"] = "TBD"
        response.headers["X-Rate-Limit-Reset"] = "TBD"
        return response

    def get_rate_limit(
        self,
        auth_details: dict,
        endpoint: str = "/",
        method: str = None,
    ) -> Tuple[int, int]:
        """
        Retrieves the appropriate rate limit and time window for the given auth details.

        Parameters
        ----------
        auth_details : dict
            A dictionary containing authentication details.
        endpoint: str
            The endpont against which rate limiting is defined.
        method: str
            Request method.

        Returns
        -------
        tuple
            A tuple containing the rate limit and time window in seconds.
        """
        if auth_details["type"] in ["user", "api_key", "jwt_token"]:
            if auth_details["id"] in self.rate_limit_config:
                config = self.rate_limit_config[auth_details["id"]]
            else:
                config = self.rate_limit_config["default"]

            limit = (
                config.get(endpoint, {}).get(method, {}).get("limit")
                or self.default_limit
            )
            window = (
                config.get(endpoint, {}).get(method, {}).get("window")
                or self.default_window
            )
            return limit, window
        return self.default_limit, self.default_window

    async def is_request_allowed(
        self, identifier: str, rate_limit: int, time_window: int
    ) -> bool:
        """
        Checks whether the given identifier has exceeded the rate limit.

        Parameters
        ----------
        identifier : str
            The unique identifier for the user or group (e.g., `user:123` or `group_a`).
        rate_limit : int
            The maximum number of allowed requests within the time window.
        time_window : int
            The time window in seconds for the rate limit.

        Returns
        -------
        bool
            True if the request is allowed, False if the rate limit is exceeded.
        """
        current_time = int(time.time())
        window_start = current_time - time_window
        redis_key = f"rate_limit:{identifier}"

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zcard(redis_key)
        pipe.zadd(redis_key, {str(current_time): current_time})
        pipe.expire(redis_key, time_window)
        results = pipe.execute()
        current_count = results[1]

        return current_count < rate_limit
