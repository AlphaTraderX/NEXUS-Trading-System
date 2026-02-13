"""
Signal Cooldown Manager - Prevents overtrading same instrument/edge.

Purpose:
- Prevent same signal firing repeatedly
- Allow time for trades to develop
- Reduce correlation risk from clustered entries
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from nexus.core.enums import EdgeType
from nexus.config.edge_config import get_edge_cooldown, get_edge_max_daily


@dataclass
class CooldownEntry:
    """Single cooldown entry."""

    symbol: str
    edge: EdgeType
    triggered_at: datetime
    cooldown_until: datetime

    def is_active(self, now: datetime) -> bool:
        """Check if cooldown is still active."""
        return now < self.cooldown_until


class SignalCooldownManager:
    """
    Manages cooldowns to prevent overtrading.

    Cooldown types:
    1. Per-symbol+edge cooldown: Same symbol+edge can't signal again for X minutes
    2. Per-symbol cooldown: 5-minute minimum between any signals on same symbol
    3. Daily edge limit: Max signals per edge per day
    """

    def __init__(self, default_cooldown_minutes: int = 60):
        self.default_cooldown = default_cooldown_minutes

        # Key: (symbol, edge_value) -> CooldownEntry
        self._symbol_edge_cooldowns: Dict[Tuple[str, str], CooldownEntry] = {}

        # Key: symbol -> last signal time
        self._symbol_cooldowns: Dict[str, datetime] = {}

        # Key: edge_value -> list of signal times (for rate limiting)
        self._edge_signal_times: Dict[str, List[datetime]] = defaultdict(list)

        # Key: edge_value -> daily signal count
        self._daily_edge_counts: Dict[str, int] = defaultdict(int)
        self._last_reset_date: Optional[datetime] = None

    def can_signal(
        self,
        symbol: str,
        edge: EdgeType,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a signal is allowed.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        now = now or datetime.utcnow()

        # Reset daily counts if new day
        self._maybe_reset_daily(now)

        edge_str = edge.value
        key = (symbol, edge_str)

        # Check 1: Symbol+edge specific cooldown
        if key in self._symbol_edge_cooldowns:
            entry = self._symbol_edge_cooldowns[key]
            if entry.is_active(now):
                remaining = int((entry.cooldown_until - now).total_seconds()) // 60
                return False, f"Cooldown active: {symbol}/{edge_str} ({remaining}m remaining)"

        # Check 2: Global symbol cooldown (minimum 5 minutes between any signals)
        if symbol in self._symbol_cooldowns:
            last_signal = self._symbol_cooldowns[symbol]
            if now - last_signal < timedelta(minutes=5):
                elapsed = int((now - last_signal).total_seconds())
                return False, f"Symbol cooldown: {symbol} signaled {elapsed}s ago"

        # Check 3: Daily edge limit
        max_daily = get_edge_max_daily(edge)
        if self._daily_edge_counts[edge_str] >= max_daily:
            return False, f"Daily limit reached: {edge_str} ({max_daily} signals)"

        return True, None

    def record_signal(
        self,
        symbol: str,
        edge: EdgeType,
        now: Optional[datetime] = None,
    ) -> None:
        """Record that a signal was generated."""
        now = now or datetime.utcnow()
        self._maybe_reset_daily(now)
        edge_str = edge.value
        key = (symbol, edge_str)

        # Get cooldown duration for this edge
        cooldown_minutes = get_edge_cooldown(edge)
        if cooldown_minutes == 0:
            cooldown_minutes = self.default_cooldown

        cooldown_until = now + timedelta(minutes=cooldown_minutes)

        # Record cooldowns
        self._symbol_edge_cooldowns[key] = CooldownEntry(
            symbol=symbol,
            edge=edge,
            triggered_at=now,
            cooldown_until=cooldown_until,
        )

        self._symbol_cooldowns[symbol] = now
        self._edge_signal_times[edge_str].append(now)
        self._daily_edge_counts[edge_str] += 1

    def _maybe_reset_daily(self, now: datetime) -> None:
        """Reset daily counts if it's a new day."""
        today = now.date()
        if self._last_reset_date != today:
            self._daily_edge_counts.clear()
            self._last_reset_date = today

    def cleanup_expired(self, now: Optional[datetime] = None) -> int:
        """Remove expired cooldown entries. Returns count removed."""
        now = now or datetime.utcnow()

        expired_keys = [
            key
            for key, entry in self._symbol_edge_cooldowns.items()
            if not entry.is_active(now)
        ]

        for key in expired_keys:
            del self._symbol_edge_cooldowns[key]

        return len(expired_keys)

    def get_status(self) -> Dict:
        """Get current cooldown status."""
        now = datetime.utcnow()

        active_cooldowns = [
            {
                "symbol": entry.symbol,
                "edge": entry.edge.value,
                "remaining_minutes": max(
                    0, int((entry.cooldown_until - now).total_seconds()) // 60
                ),
            }
            for entry in self._symbol_edge_cooldowns.values()
            if entry.is_active(now)
        ]

        return {
            "active_cooldowns": len(active_cooldowns),
            "daily_counts": dict(self._daily_edge_counts),
            "cooldowns": active_cooldowns[:10],
        }

    def clear_all(self) -> None:
        """Clear all cooldowns (for testing)."""
        self._symbol_edge_cooldowns.clear()
        self._symbol_cooldowns.clear()
        self._edge_signal_times.clear()
        self._daily_edge_counts.clear()
