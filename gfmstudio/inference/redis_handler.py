# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import backoff

from gfmstudio.redis_client import get_async_redis_client as redis_client


async def subscribe_to_channel(channel: str, redis_conn=None):
    """Subscribe to the specified Redis channel and process incoming messages."""
    if not redis_conn:
        redis_conn = await redis_client()

    async def event_generator():
        async with redis_conn.pubsub() as listener:
            await listener.subscribe(channel)
            while True:
                message = await listener.get_message()
                if message is None:
                    continue
                if message.get("type") == "message":
                    yield message.get("data", {})

    return event_generator


@backoff.on_exception(backoff.expo, (ConnectionError), max_tries=5)
async def publish_to_channel(channel: str, message: str, redis_conn=None):
    """Publish a message to the specified Redis channel with retry and backoff."""
    if not redis_conn:
        redis_conn = await redis_client()

    return await redis_conn.publish(channel, message)
