# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import uuid
from typing import List

from fastapi import APIRouter
from starlette.websockets import WebSocket

from gfmstudio.inference import redis_handler
from gfmstudio.log import logger

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_inference_task_status(self, event: str, websocket: WebSocket):
        """Sends a inference status messages to a room."""
        await websocket.send_json(event)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/inference/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: uuid.UUID):
    task_id = str(task_id)
    await manager.connect(websocket)
    logger.debug("🔌 ✅ Websocket connected: event-%s", task_id)
    event_generator = await redis_handler.subscribe_to_channel(
        channel=f"geoinf:event:{task_id}"
    )

    try:
        async for event in event_generator():
            data = json.loads(event)
            await manager.send_inference_task_status(event=data, websocket=websocket)
    except Exception:
        logger.error("🔌 ❌ Websocket Error: event-%s", task_id)
        await manager.disconnect(websocket)
