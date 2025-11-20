# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import redis
from redis import asyncio as aioredis

from gfmstudio.config import settings
from gfmstudio.log import logger


async def get_async_redis_client(redis_url: str = settings.REDIS_URL):
    """Establish a connection to the Redis server."""
    try:
        return await aioredis.from_url(redis_url, decode_responses=True)
    except redis.exceptions.ConnectionError:
        logger.exception("❌ Redis: Connection error: %s", redis_url)
    except Exception:
        logger.exception("❌ Redis: An unexpected error occurred: %s", redis_url)


def get_redis_client(redis_url: str = settings.REDIS_URL):
    """Establish a connection to the Redis server."""
    try:
        return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    except redis.exceptions.ConnectionError:
        logger.exception("❌ Redis: Connection error: %s", redis_url)
    except Exception:
        logger.exception("❌ Redis: An unexpected error occurred: %s", redis_url)


redis_client = get_redis_client()
