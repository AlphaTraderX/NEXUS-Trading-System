"""
Signal Cooldown Manager - prevents duplicate signals.

Rules:
- Same symbol + direction: 60 minute cooldown (default)
- Same symbol any direction: 15 minute cooldown
- Configurable per edge type
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from nexus.core.enums import EdgeType, Direction

logger = logging.getLogger(__name__)


@dataclass
class CooldownConfig:
    """Cooldown configuration per edge."""
    same_direction_minutes: int = 60
    any_direction_minutes: int = 15


# Edge-specific cooldowns (some edges need longer cooldowns)
EDGE_COOLDOWNS: Dict[EdgeType, CooldownConfig] = {
    EdgeType.TURN_OF_MONTH: CooldownConfig(same_direction_minutes=1440, any_direction_minutes=1440),  # 24h - once per day
    EdgeType.MONTH_END: CooldownConfig(same_direction_minutes=1440, any_direction_minutes=1440),
    EdgeType.INSIDER_CLUSTER: CooldownConfig(same_direction_minutes=1440, any_direction_minutes=720),  # 24h same, 12h any
    EdgeType.GAP_FILL: CooldownConfig(same_direction_minutes=240, any_direction_minutes=60),  # 4h same, 1h any
    EdgeType.OVERNIGHT_PREMIUM: CooldownConfig(same_direction_minutes=1440, any_direction_minutes=1440),
    # Default for others: 60min same direction, 15min any direction
}


class CooldownManager:
    """
    Manages signal cooldowns to prevent duplicate trading.

    Usage:
        cooldown = CooldownManager()

        # Check before generating signal
        if cooldown.can_signal(symbol, direction, edge_type):
            signal = generate_signal(...)
            cooldown.record_signal(symbol, direction, edge_type)
    """

    def __init__(self, default_config: Optional[CooldownConfig] = None):
        self.default_config = default_config or CooldownConfig()
        # Key: (symbol, direction) -> last signal time
        self._last_signals: Dict[Tuple[str, Direction], datetime] = {}
        # Key: symbol -> last signal time (any direction)
        self._last_symbol_signals: Dict[str, datetime] = {}

    def can_signal(
        self,
        symbol: str,
        direction: Direction,
        edge_type: EdgeType,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a signal can be generated.

        Returns:
            (can_signal: bool, reason: Optional[str])
        """
        now = current_time or datetime.utcnow()
        config = EDGE_COOLDOWNS.get(edge_type, self.default_config)

        # Normalize direction to enum
        if isinstance(direction, str):
            direction = Direction(direction)

        # Check same symbol + direction cooldown
        key = (symbol.upper(), direction)
        if key in self._last_signals:
            last_time = self._last_signals[key]
            cooldown_end = last_time + timedelta(minutes=config.same_direction_minutes)
            if now < cooldown_end:
                remaining = int((cooldown_end - now).total_seconds() / 60)
                return False, f"Same direction cooldown: {remaining}min remaining"

        # Check any direction cooldown
        symbol_upper = symbol.upper()
        if symbol_upper in self._last_symbol_signals:
            last_time = self._last_symbol_signals[symbol_upper]
            cooldown_end = last_time + timedelta(minutes=config.any_direction_minutes)
            if now < cooldown_end:
                remaining = int((cooldown_end - now).total_seconds() / 60)
                return False, f"Any direction cooldown: {remaining}min remaining"

        return True, None

    def record_signal(
        self,
        symbol: str,
        direction: Direction,
        edge_type: EdgeType,
        signal_time: Optional[datetime] = None,
    ) -> None:
        """Record a signal for cooldown tracking."""
        now = signal_time or datetime.utcnow()
        symbol_upper = symbol.upper()

        # Normalize direction to enum
        if isinstance(direction, str):
            direction = Direction(direction)

        self._last_signals[(symbol_upper, direction)] = now
        self._last_symbol_signals[symbol_upper] = now

        logger.debug(
            "Recorded signal cooldown: %s %s (%s)",
            symbol_upper, direction.value, edge_type.value,
        )

    def clear_expired(self, current_time: Optional[datetime] = None) -> int:
        """Clear expired cooldowns to prevent memory growth."""
        now = current_time or datetime.utcnow()
        max_cooldown = timedelta(hours=48)  # Nothing should be older than 48h

        cleared = 0

        # Clear direction-specific
        expired_keys = [
            k for k, v in self._last_signals.items()
            if now - v > max_cooldown
        ]
        for k in expired_keys:
            del self._last_signals[k]
            cleared += 1

        # Clear symbol-specific
        expired_symbols = [
            s for s, v in self._last_symbol_signals.items()
            if now - v > max_cooldown
        ]
        for s in expired_symbols:
            del self._last_symbol_signals[s]
            cleared += 1

        if cleared > 0:
            logger.debug("Cleared %d expired cooldowns", cleared)

        return cleared

    def get_active_cooldowns(self) -> Dict[str, dict]:
        """Get all active cooldowns for debugging."""
        now = datetime.utcnow()
        active = {}

        for (symbol, direction), last_time in self._last_signals.items():
            key = f"{symbol}_{direction.value}"
            active[key] = {
                "last_signal": last_time.isoformat(),
                "minutes_ago": int((now - last_time).total_seconds() / 60),
            }

        return active

    def reset(self) -> None:
        """Reset all cooldowns (for testing)."""
        self._last_signals.clear()
        self._last_symbol_signals.clear()


# Singleton instance
_cooldown_manager: Optional[CooldownManager] = None


def get_cooldown_manager() -> CooldownManager:
    """Get the global cooldown manager instance."""
    global _cooldown_manager
    if _cooldown_manager is None:
        _cooldown_manager = CooldownManager()
    return _cooldown_manager
