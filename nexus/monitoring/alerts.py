"""
Monitoring alerts (e.g. threshold breaches, failures).
"""

from typing import Callable, Any, Optional


def register_alert_handler(
    event: str,
    handler: Callable[[Any], None],
) -> None:
    """Register a handler for an alert event."""
    pass


async def raise_alert(event: str, payload: Any) -> None:
    """Raise an alert to registered handlers."""
    pass
