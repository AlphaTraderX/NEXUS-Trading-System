"""
Format signals and alerts for delivery (Discord, Telegram).
"""

from typing import Any, Dict


def format_signal(signal: Dict[str, Any]) -> str:
    """Format a trading signal for human-readable delivery."""
    return str(signal)


def format_alert(alert_type: str, payload: Any) -> str:
    """Format an alert (e.g. circuit breaker, kill switch)."""
    return f"[{alert_type}] {payload}"
