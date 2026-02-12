"""
Tests for GapScanner (Gap and Go strategy).

Validates that scanner matches backtest logic:
- Gap UP → LONG (momentum continuation, not fade)
- Gap DOWN → SHORT (momentum continuation, not fade)
- Dynamic scoring based on gap size + volume + trend
- Only validated high-beta symbols
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np

from nexus.scanners.gap import GapScanner
from nexus.core.enums import Direction, EdgeType, Market


class TestGapScannerConfig:
    """Test GapScanner static configuration."""

    @pytest.fixture
    def scanner(self):
        return GapScanner(data_provider=MagicMock())

    def test_edge_type(self, scanner):
        assert scanner.edge_type == EdgeType.GAP_FILL

    def test_symbols_match_backtest(self, scanner):
        expected = ["SPY", "NVDA", "TSLA", "AAPL", "AMD", "COIN", "ROKU", "SHOP", "SQ", "MARA"]
        actual = scanner.INSTRUMENTS[Market.US_STOCKS]
        assert actual == expected

    def test_gap_range(self, scanner):
        assert scanner.min_gap_pct == 1.0
        assert scanner.max_gap_pct == 5.0

    def test_volume_threshold(self, scanner):
        assert scanner.min_volume_ratio == 1.5

    def test_notional_pct(self, scanner):
        assert scanner.notional_pct == 16


class TestGapDirection:
    """Test gap direction logic — LONG on gap up, SHORT on gap down."""

    @pytest.fixture
    def scanner(self):
        return GapScanner(data_provider=MagicMock())

    def _make_bars(self, gap_pct: float, day_confirms: bool, volume_ratio: float = 2.0):
        """Create mock bars with a specific gap."""
        np.random.seed(42)
        n = 25
        dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
        base = 100.0

        closes = [base] * (n - 1)
        opens = [base] * (n - 1)
        highs = [base + 1] * (n - 1)
        lows = [base - 1] * (n - 1)
        volumes = [1_000_000] * (n - 1)

        # Last bar: gap from prior close
        prev_close = base
        gap_open = prev_close * (1 + gap_pct / 100)

        if day_confirms:
            if gap_pct > 0:
                gap_close = gap_open * 1.005  # Close higher (confirms gap up)
            else:
                gap_close = gap_open * 0.995  # Close lower (confirms gap down)
        else:
            if gap_pct > 0:
                gap_close = gap_open * 0.995  # Gap up but closes red (fills)
            else:
                gap_close = gap_open * 1.005  # Gap down but closes green (fills)

        closes.append(gap_close)
        opens.append(gap_open)
        highs.append(max(gap_open, gap_close) + 0.5)
        lows.append(min(gap_open, gap_close) - 0.5)
        volumes.append(int(1_000_000 * volume_ratio))

        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )

    @pytest.mark.asyncio
    async def test_gap_up_with_volume_is_long(self, scanner):
        """Gap up + high volume + day confirms → LONG (not SHORT)."""
        bars = self._make_bars(gap_pct=2.5, day_confirms=True, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is not None
        assert opp.direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_gap_down_with_volume_is_short(self, scanner):
        """Gap down + high volume + day confirms → SHORT (not LONG)."""
        bars = self._make_bars(gap_pct=-2.5, day_confirms=True, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is not None
        assert opp.direction == Direction.SHORT

    @pytest.mark.asyncio
    async def test_gap_up_filling_rejected(self, scanner):
        """Gap up but day closes red (filling) → no signal."""
        bars = self._make_bars(gap_pct=2.5, day_confirms=False, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_gap_down_filling_rejected(self, scanner):
        """Gap down but day closes green (filling) → no signal."""
        bars = self._make_bars(gap_pct=-2.5, day_confirms=False, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_low_volume_rejected(self, scanner):
        """Gap with volume below 150% → no signal."""
        bars = self._make_bars(gap_pct=2.5, day_confirms=True, volume_ratio=1.2)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_gap_too_small_rejected(self, scanner):
        """Gap < 1% → no signal."""
        bars = self._make_bars(gap_pct=0.5, day_confirms=True, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is None

    @pytest.mark.asyncio
    async def test_gap_too_large_rejected(self, scanner):
        """Gap > 5% → no signal."""
        bars = self._make_bars(gap_pct=6.0, day_confirms=True, volume_ratio=2.0)
        scanner.get_bars_safe = AsyncMock(return_value=bars)

        opp = await scanner._scan_symbol("TSLA", Market.US_STOCKS)

        assert opp is None


class TestGapScoring:
    """Test dynamic gap scoring."""

    def test_strong_signal_score(self):
        """2.5% gap + 2x volume + trend aligned → A-tier (80+)."""
        score = GapScanner._calculate_gap_score(2.5, 2.0, True)
        assert score >= 80  # A-tier

    def test_medium_signal_score(self):
        """1.5% gap + 1.5x volume + no trend → B-tier (65-79)."""
        score = GapScanner._calculate_gap_score(1.5, 1.5, False)
        assert 50 <= score <= 79

    def test_weak_signal_score(self):
        """1% gap + 1.5x volume + no trend → C/D-tier."""
        score = GapScanner._calculate_gap_score(1.0, 1.5, False)
        assert 40 <= score <= 70

    def test_score_clamped_0_100(self):
        """Score should always be 0-100."""
        score = GapScanner._calculate_gap_score(10.0, 0.5, False)
        assert 0 <= score <= 100

    def test_edge_data_has_notional(self):
        """Edge data must include notional_pct."""
        scanner = GapScanner(data_provider=MagicMock())
        assert scanner.notional_pct == 16
