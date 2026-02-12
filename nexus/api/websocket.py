"""
NEXUS WebSocket Handler

Real-time updates for signals, positions, and system status.
"""

import logging
from typing import Set, Dict, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = {"signals", "positions", "status"}
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, channel: str = "all") -> None:
        disconnected = set()
        for ws in self.active_connections:
            subs = self.subscriptions.get(ws, set())
            if channel != "all" and channel not in subs:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def subscribe(self, websocket: WebSocket, channels: list) -> None:
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)

    def unsubscribe(self, websocket: WebSocket, channels: list) -> None:
        if websocket in self.subscriptions:
            self.subscriptions[websocket] -= set(channels)


manager = ConnectionManager()


async def websocket_handler(websocket: WebSocket) -> None:
    """Handle WebSocket connection with subscribe/unsubscribe/ping support."""
    await manager.connect(websocket)

    await manager.send_personal(websocket, {
        "type": "connected",
        "service": "nexus",
        "timestamp": datetime.utcnow().isoformat(),
        "subscriptions": list(manager.subscriptions.get(websocket, set())),
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await manager.send_personal(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            elif msg_type == "subscribe":
                channels = data.get("channels", [])
                manager.subscribe(websocket, channels)
                await manager.send_personal(websocket, {"type": "subscribed", "channels": channels})
            elif msg_type == "unsubscribe":
                channels = data.get("channels", [])
                manager.unsubscribe(websocket, channels)
                await manager.send_personal(websocket, {"type": "unsubscribed", "channels": channels})
            else:
                await manager.send_personal(websocket, {"type": "error", "message": f"Unknown: {msg_type}"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Broadcast helpers for other components
async def broadcast_signal(signal: dict) -> None:
    await manager.broadcast(
        {"type": "signal", "data": signal, "timestamp": datetime.utcnow().isoformat()},
        channel="signals",
    )


async def broadcast_position_update(position: dict) -> None:
    await manager.broadcast(
        {"type": "position_update", "data": position, "timestamp": datetime.utcnow().isoformat()},
        channel="positions",
    )


async def broadcast_status_update(status: dict) -> None:
    await manager.broadcast(
        {"type": "status_update", "data": status, "timestamp": datetime.utcnow().isoformat()},
        channel="status",
    )
