"""Tests for CooldownManager."""

import pytest
from datetime import datetime, timedelta

from nexus.core.enums import EdgeType, Direction
from nexus.execution.cooldown_manager import (
    CooldownManager,
    CooldownConfig,
    EDGE_COOLDOWNS,
    get_cooldown_manager,
)


class TestCooldownManager:
    """Test cooldown functionality."""

    def test_no_cooldown_initially(self):
        """First signal should always be allowed."""
        mgr = CooldownManager()
        can_signal, reason = mgr.can_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME)
        assert can_signal is True
        assert reason is None

    def test_same_direction_cooldown(self):
        """Same symbol + direction should be blocked."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        # Record a signal
        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, now)

        # Try again immediately - should be blocked
        can_signal, reason = mgr.can_signal(
            "AAPL", Direction.LONG, EdgeType.RSI_EXTREME,
            now + timedelta(minutes=5),
        )
        assert can_signal is False
        assert "Same direction cooldown" in reason

    def test_opposite_direction_short_cooldown(self):
        """Opposite direction has shorter cooldown."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        # Record LONG signal
        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, now)

        # Try SHORT after 5 min - should be blocked (any direction cooldown)
        can_signal, _ = mgr.can_signal(
            "AAPL", Direction.SHORT, EdgeType.RSI_EXTREME,
            now + timedelta(minutes=5),
        )
        assert can_signal is False

        # Try SHORT after 20 min - should be allowed (past 15 min any-direction cooldown)
        can_signal, _ = mgr.can_signal(
            "AAPL", Direction.SHORT, EdgeType.RSI_EXTREME,
            now + timedelta(minutes=20),
        )
        assert can_signal is True

    def test_cooldown_expires(self):
        """Cooldown should expire after configured time."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        # Record a signal
        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, now)

        # After 65 minutes, same direction should be allowed (60 min default)
        can_signal, _ = mgr.can_signal(
            "AAPL", Direction.LONG, EdgeType.RSI_EXTREME,
            now + timedelta(minutes=65),
        )
        assert can_signal is True

    def test_different_symbol_no_cooldown(self):
        """Different symbol should not be affected."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, now)

        can_signal, _ = mgr.can_signal(
            "MSFT", Direction.LONG, EdgeType.RSI_EXTREME, now,
        )
        assert can_signal is True

    def test_tom_has_long_cooldown(self):
        """Turn of Month should have 24h cooldown."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        mgr.record_signal("SPY", Direction.LONG, EdgeType.TURN_OF_MONTH, now)

        # After 2 hours - should still be blocked
        can_signal, _ = mgr.can_signal(
            "SPY", Direction.LONG, EdgeType.TURN_OF_MONTH,
            now + timedelta(hours=2),
        )
        assert can_signal is False

        # After 25 hours - should be allowed
        can_signal, _ = mgr.can_signal(
            "SPY", Direction.LONG, EdgeType.TURN_OF_MONTH,
            now + timedelta(hours=25),
        )
        assert can_signal is True

    def test_clear_expired(self):
        """Expired cooldowns should be cleared."""
        mgr = CooldownManager()
        old_time = datetime.utcnow() - timedelta(hours=50)

        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, old_time)

        cleared = mgr.clear_expired()
        assert cleared >= 1

    def test_symbol_case_insensitive(self):
        """Symbol lookup should be case insensitive."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        mgr.record_signal("aapl", Direction.LONG, EdgeType.RSI_EXTREME, now)

        can_signal, _ = mgr.can_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME, now)
        assert can_signal is False

    def test_reset(self):
        """Reset should clear all cooldowns."""
        mgr = CooldownManager()
        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME)

        mgr.reset()

        can_signal, _ = mgr.can_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME)
        assert can_signal is True

    def test_get_active_cooldowns(self):
        """Should return active cooldowns."""
        mgr = CooldownManager()
        mgr.record_signal("AAPL", Direction.LONG, EdgeType.RSI_EXTREME)

        active = mgr.get_active_cooldowns()
        assert "AAPL_long" in active

    def test_gap_fill_4h_same_direction(self):
        """Gap fill should have 4h same direction cooldown."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        mgr.record_signal("TSLA", Direction.LONG, EdgeType.GAP_FILL, now)

        # After 2h - still blocked
        can_signal, _ = mgr.can_signal(
            "TSLA", Direction.LONG, EdgeType.GAP_FILL,
            now + timedelta(hours=2),
        )
        assert can_signal is False

        # After 5h - allowed
        can_signal, _ = mgr.can_signal(
            "TSLA", Direction.LONG, EdgeType.GAP_FILL,
            now + timedelta(hours=5),
        )
        assert can_signal is True

    def test_insider_cluster_any_direction_12h(self):
        """Insider cluster any direction cooldown is 12h."""
        mgr = CooldownManager()
        now = datetime.utcnow()

        mgr.record_signal("AAPL", Direction.LONG, EdgeType.INSIDER_CLUSTER, now)

        # SHORT after 6h - blocked (12h any direction)
        can_signal, _ = mgr.can_signal(
            "AAPL", Direction.SHORT, EdgeType.INSIDER_CLUSTER,
            now + timedelta(hours=6),
        )
        assert can_signal is False

        # SHORT after 13h - allowed
        can_signal, _ = mgr.can_signal(
            "AAPL", Direction.SHORT, EdgeType.INSIDER_CLUSTER,
            now + timedelta(hours=13),
        )
        assert can_signal is True

    def test_custom_default_config(self):
        """Custom default config should be used for unknown edges."""
        custom = CooldownConfig(same_direction_minutes=30, any_direction_minutes=10)
        mgr = CooldownManager(default_config=custom)
        now = datetime.utcnow()

        mgr.record_signal("SPY", Direction.LONG, EdgeType.BOLLINGER_TOUCH, now)

        # After 20 min - still blocked (30 min custom default)
        can_signal, _ = mgr.can_signal(
            "SPY", Direction.LONG, EdgeType.BOLLINGER_TOUCH,
            now + timedelta(minutes=20),
        )
        assert can_signal is False

        # After 35 min - allowed
        can_signal, _ = mgr.can_signal(
            "SPY", Direction.LONG, EdgeType.BOLLINGER_TOUCH,
            now + timedelta(minutes=35),
        )
        assert can_signal is True


class TestGlobalCooldownManager:
    """Test singleton pattern."""

    def test_singleton(self):
        """Should return same instance."""
        mgr1 = get_cooldown_manager()
        mgr2 = get_cooldown_manager()
        assert mgr1 is mgr2

    def test_singleton_is_usable(self):
        """Singleton should work for signal tracking."""
        mgr = get_cooldown_manager()
        mgr.reset()  # Clean state

        can_signal, _ = mgr.can_signal("TEST", Direction.LONG, EdgeType.RSI_EXTREME)
        assert can_signal is True
