"""
WebSocket handler for real-time signals and updates.
"""

from typing import Any
import json


async def websocket_handler(websocket: Any) -> None:
    """Handle WebSocket connection; push signals and status updates."""
    await websocket.send(json.dumps({"type": "connected", "service": "nexus"}))
