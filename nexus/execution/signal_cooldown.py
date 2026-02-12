"""
Signal cooldown manager - prevents duplicate signals on same symbol/edge.

Matches backtest SIGNAL_COOLDOWN_BARS:
- gap_fill: 5 days
- overnight_premium: 1 day
- vwap_deviation: 5 days
- rsi_extreme: 5 days
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from nexus.core.enums import EdgeType

logger = logging.getLogger(__name__)


# Cooldown periods matching backtest engine (daily bar equivalent)
COOLDOWN_DAYS: Dict[EdgeType, int] = {
    EdgeType.GAP_FILL: 5,
    EdgeType.OVERNIGHT_PREMIUM: 1,
    EdgeType.VWAP_DEVIATION: 5,
    EdgeType.RSI_EXTREME: 5,
}


@dataclass
class SignalCooldownManager:
    """
    Tracks last signal time per (symbol, edge) to prevent duplicates.

    Without this, the paper runner could open 5 identical gap trades
    on the same symbol on consecutive cycles.
    """

    # Key: (symbol, edge_type) -> last signal datetime
    _last_signal: Dict[Tuple[str, EdgeType], datetime] = field(default_factory=dict)

    def can_signal(self, symbol: str, edge_type: EdgeType) -> bool:
        """Check if a new signal is allowed (cooldown expired)."""
        key = (symbol, edge_type)

        if key not in self._last_signal:
            return True

        last_time = self._last_signal[key]
        cooldown_days = COOLDOWN_DAYS.get(edge_type, 5)
        cooldown_expires = last_time + timedelta(days=cooldown_days)

        now = datetime.now(timezone.utc)

        if now >= cooldown_expires:
            return True

        remaining = (cooldown_expires - now).total_seconds() / 86400
        logger.debug(
            "Cooldown active: %s %s - %.1f days remaining",
            symbol, edge_type.value, remaining,
        )
        return False

    def record_signal(self, symbol: str, edge_type: EdgeType) -> None:
        """Record that a signal was taken."""
        key = (symbol, edge_type)
        self._last_signal[key] = datetime.now(timezone.utc)
        logger.debug("Recorded signal: %s %s", symbol, edge_type.value)

    def get_cooldown_status(
        self, symbol: str, edge_type: EdgeType
    ) -> Optional[int]:
        """Get remaining cooldown days, or None if no cooldown."""
        key = (symbol, edge_type)

        if key not in self._last_signal:
            return None

        last_time = self._last_signal[key]
        cooldown_days = COOLDOWN_DAYS.get(edge_type, 5)
        cooldown_expires = last_time + timedelta(days=cooldown_days)

        now = datetime.now(timezone.utc)

        if now >= cooldown_expires:
            return None

        return (cooldown_expires - now).days + 1

    def clear_expired(self) -> int:
        """Clear expired cooldowns to prevent memory growth. Returns count cleared."""
        now = datetime.now(timezone.utc)
        expired_keys = []

        for key, last_time in self._last_signal.items():
            _, edge_type = key
            cooldown_days = COOLDOWN_DAYS.get(edge_type, 5)
            if now >= last_time + timedelta(days=cooldown_days):
                expired_keys.append(key)

        for key in expired_keys:
            del self._last_signal[key]

        return len(expired_keys)

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            f"{symbol}|{edge.value}": ts.isoformat()
            for (symbol, edge), ts in self._last_signal.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SignalCooldownManager":
        """Deserialize from state."""
        manager = cls()
        for key_str, ts_str in data.items():
            try:
                symbol, edge_str = key_str.split("|", 1)
                edge = EdgeType(edge_str)
                ts = datetime.fromisoformat(ts_str)
                manager._last_signal[(symbol, edge)] = ts
            except (ValueError, KeyError) as e:
                logger.warning("Failed to restore cooldown %s: %s", key_str, e)
        return manager
