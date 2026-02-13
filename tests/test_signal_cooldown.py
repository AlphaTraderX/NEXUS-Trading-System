"""Tests for signal cooldown manager."""

import pytest
from datetime import datetime, timedelta
from nexus.core.enums import EdgeType
from nexus.risk.signal_cooldown import SignalCooldownManager


class TestSignalCooldown:
    def setup_method(self):
        self.manager = SignalCooldownManager(default_cooldown_minutes=60)

    def test_first_signal_allowed(self):
        """First signal on any symbol should be allowed."""
        allowed, reason = self.manager.can_signal("AAPL", EdgeType.VWAP_DEVIATION)
        assert allowed
        assert reason is None

    def test_cooldown_blocks_repeat(self):
        """Same symbol+edge should be blocked during cooldown."""
        now = datetime.utcnow()

        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)

        # Immediately after should be blocked
        allowed, reason = self.manager.can_signal(
            "AAPL", EdgeType.VWAP_DEVIATION, now + timedelta(minutes=1)
        )
        assert not allowed
        assert "Cooldown active" in reason

    def test_cooldown_expires(self):
        """Signal should be allowed after cooldown expires."""
        now = datetime.utcnow()

        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)

        # After cooldown should be allowed (VWAP has 60min cooldown)
        allowed, reason = self.manager.can_signal(
            "AAPL", EdgeType.VWAP_DEVIATION, now + timedelta(minutes=120)
        )
        assert allowed

    def test_different_edge_allowed(self):
        """Different edge on same symbol should be allowed after symbol cooldown."""
        now = datetime.utcnow()

        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)

        # Different edge should be allowed (after 5min symbol cooldown)
        allowed, reason = self.manager.can_signal(
            "AAPL", EdgeType.RSI_EXTREME, now + timedelta(minutes=10)
        )
        assert allowed

    def test_symbol_cooldown_blocks_rapid_signals(self):
        """Same symbol with different edge should be blocked within 5 minutes."""
        now = datetime.utcnow()

        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)

        # Within 5 minutes, even different edge should be blocked
        allowed, reason = self.manager.can_signal(
            "AAPL", EdgeType.RSI_EXTREME, now + timedelta(minutes=1)
        )
        assert not allowed
        assert "Symbol cooldown" in reason

    def test_daily_limit_enforced(self):
        """Daily signal limit per edge should be enforced."""
        now = datetime.utcnow()

        # Record max signals for gap_fill (max 6)
        for i in range(6):
            self.manager.record_signal(f"SYM{i}", EdgeType.GAP_FILL, now)

        # Next should be blocked
        allowed, reason = self.manager.can_signal("SYM99", EdgeType.GAP_FILL, now)
        assert not allowed
        assert "Daily limit" in reason

    def test_daily_reset(self):
        """Daily counts should reset on new day."""
        now = datetime.utcnow()

        for i in range(6):
            self.manager.record_signal(f"SYM{i}", EdgeType.GAP_FILL, now)

        # Next day should be allowed
        tomorrow = now + timedelta(days=1)
        allowed, reason = self.manager.can_signal(
            "NEWSTOCK", EdgeType.GAP_FILL, tomorrow
        )
        assert allowed

    def test_cleanup_expired(self):
        """Cleanup should remove expired entries."""
        now = datetime.utcnow()

        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)
        self.manager.record_signal("MSFT", EdgeType.RSI_EXTREME, now)

        # Way in the future, all should be expired
        removed = self.manager.cleanup_expired(now + timedelta(hours=48))
        assert removed == 2

    def test_clear_all(self):
        """Clear should reset everything."""
        now = datetime.utcnow()
        self.manager.record_signal("AAPL", EdgeType.VWAP_DEVIATION, now)
        self.manager.clear_all()

        allowed, _ = self.manager.can_signal("AAPL", EdgeType.VWAP_DEVIATION, now)
        assert allowed

    def test_status_report(self):
        """Status should report active cooldowns."""
        status = self.manager.get_status()
        assert "active_cooldowns" in status
        assert "daily_counts" in status
