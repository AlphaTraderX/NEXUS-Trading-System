"""Tests for StockTwits sentiment client and SentimentScanner."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import numpy as np

from nexus.data.stocktwits import (
    StockTwitsClient,
    SentimentData,
    SentimentSpike,
    get_stocktwits_client,
)
from nexus.scanners.sentiment import SentimentScanner
from nexus.core.enums import Direction, EdgeType, Market


# ---------------------------------------------------------------------------
# SentimentData
# ---------------------------------------------------------------------------

class TestSentimentData:
    """Test SentimentData dataclass."""

    def test_bullish_ratio_all_bullish(self):
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=20,
            bearish_count=0,
            total_count=20,
        )
        assert data.bullish_ratio == 1.0
        assert data.sentiment_score == 1.0

    def test_bullish_ratio_all_bearish(self):
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=0,
            bearish_count=20,
            total_count=20,
        )
        assert data.bullish_ratio == 0.0
        assert data.sentiment_score == -1.0

    def test_bullish_ratio_mixed(self):
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=15,
            bearish_count=5,
            total_count=20,
        )
        assert data.bullish_ratio == 0.75
        assert data.sentiment_score == pytest.approx(0.5)

    def test_bullish_ratio_neutral(self):
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=10,
            bearish_count=10,
            total_count=20,
        )
        assert data.bullish_ratio == 0.5
        assert data.sentiment_score == 0.0

    def test_bullish_ratio_empty(self):
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=0,
            bearish_count=0,
            total_count=0,
        )
        assert data.bullish_ratio == 0.5
        assert data.sentiment_score == 0.0


# ---------------------------------------------------------------------------
# SentimentSpike
# ---------------------------------------------------------------------------

class TestSentimentSpike:
    """Test SentimentSpike detection."""

    def test_is_extreme_high(self):
        spike = SentimentSpike(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_sentiment=0.8,
            historical_mean=0.1,
            historical_std=0.2,
            z_score=3.5,
            message_count=25,
            direction="bullish",
        )
        assert spike.is_extreme is True

    def test_is_extreme_negative(self):
        spike = SentimentSpike(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_sentiment=-0.7,
            historical_mean=0.1,
            historical_std=0.2,
            z_score=-4.0,
            message_count=30,
            direction="bearish",
        )
        assert spike.is_extreme is True

    def test_not_extreme(self):
        spike = SentimentSpike(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_sentiment=0.2,
            historical_mean=0.1,
            historical_std=0.2,
            z_score=0.5,
            message_count=15,
            direction="bullish",
        )
        assert spike.is_extreme is False

    def test_boundary_exactly_2(self):
        spike = SentimentSpike(
            symbol="TEST",
            timestamp=datetime.now(),
            current_sentiment=0.5,
            historical_mean=0.1,
            historical_std=0.2,
            z_score=2.0,
            message_count=15,
            direction="bullish",
        )
        assert spike.is_extreme is True

    def test_boundary_just_below_2(self):
        spike = SentimentSpike(
            symbol="TEST",
            timestamp=datetime.now(),
            current_sentiment=0.3,
            historical_mean=0.1,
            historical_std=0.2,
            z_score=1.99,
            message_count=15,
            direction="bullish",
        )
        assert spike.is_extreme is False


# ---------------------------------------------------------------------------
# StockTwitsClient - History & Detection (no network)
# ---------------------------------------------------------------------------

class TestStockTwitsClient:
    """Test StockTwits client offline methods."""

    @pytest.fixture
    def client(self):
        return StockTwitsClient()

    def test_seed_history(self, client):
        client.seed_history("AAPL", [0.1, 0.2, 0.15, 0.3, 0.1, 0.2])

        stats = client.get_historical_stats("AAPL")

        assert stats is not None
        assert stats["count"] == 6
        assert 0.1 <= stats["mean"] <= 0.3
        assert stats["std"] > 0

    def test_historical_stats_insufficient_data(self, client):
        """Should return None with < 5 readings."""
        client.seed_history("FEW", [0.1, 0.2, 0.3])
        assert client.get_historical_stats("FEW") is None

    def test_historical_stats_unknown_symbol(self, client):
        assert client.get_historical_stats("UNKNOWN") is None

    def test_seed_history_case_insensitive(self, client):
        client.seed_history("aapl", [0.1, 0.2, 0.15, 0.3, 0.1, 0.2])
        stats = client.get_historical_stats("AAPL")
        assert stats is not None
        assert stats["count"] == 6

    def test_detect_spike_no_history(self, client):
        data = SentimentData(
            symbol="NEW",
            timestamp=datetime.now(),
            bullish_count=18,
            bearish_count=2,
            total_count=20,
        )

        spike = client.detect_spike("NEW", data)
        assert spike is None

    def test_detect_spike_extreme_bullish(self, client):
        # Seed neutral history (mean ~0, std ~0.08)
        client.seed_history("TEST", [0.0, 0.1, -0.1, 0.05, -0.05, 0.0])

        # Extremely bullish current
        data = SentimentData(
            symbol="TEST",
            timestamp=datetime.now(),
            bullish_count=19,
            bearish_count=1,
            total_count=20,
        )

        spike = client.detect_spike("TEST", data)

        assert spike is not None
        assert spike.z_score > 2.0
        assert spike.is_extreme is True
        assert spike.direction == "bullish"

    def test_detect_spike_extreme_bearish(self, client):
        client.seed_history("BEAR", [0.0, 0.1, -0.1, 0.05, -0.05, 0.0])

        data = SentimentData(
            symbol="BEAR",
            timestamp=datetime.now(),
            bullish_count=1,
            bearish_count=19,
            total_count=20,
        )

        spike = client.detect_spike("BEAR", data)

        assert spike is not None
        assert spike.z_score < -2.0
        assert spike.is_extreme is True
        assert spike.direction == "bearish"

    def test_detect_spike_not_extreme(self, client):
        client.seed_history("TEST", [0.3, 0.4, 0.2, 0.5, 0.3, 0.4])

        # Similar to history — not a spike
        data = SentimentData(
            symbol="TEST",
            timestamp=datetime.now(),
            bullish_count=14,
            bearish_count=6,
            total_count=20,
        )

        spike = client.detect_spike("TEST", data)

        assert spike is not None
        assert spike.is_extreme is False

    def test_min_messages_filter(self, client):
        client.seed_history("TEST", [0.0, 0.1, -0.1, 0.05, -0.05, 0.0])

        data = SentimentData(
            symbol="TEST",
            timestamp=datetime.now(),
            bullish_count=5,
            bearish_count=0,
            total_count=5,  # Below minimum
        )

        spike = client.detect_spike("TEST", data, min_messages=10)
        assert spike is None

    def test_detect_spike_low_std(self, client):
        """Should handle near-zero std (clamped to 0.1)."""
        # All identical scores → std = 0
        client.seed_history("FLAT", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        data = SentimentData(
            symbol="FLAT",
            timestamp=datetime.now(),
            bullish_count=18,
            bearish_count=2,
            total_count=20,
        )

        spike = client.detect_spike("FLAT", data)
        assert spike is not None
        # std clamped to 0.1, so z_score should be large
        assert spike.z_score > 2.0

    def test_history_maxlen(self, client):
        """History should not grow beyond maxlen."""
        scores = [i * 0.01 for i in range(50)]
        client.seed_history("LONG", scores)

        stats = client.get_historical_stats("LONG")
        assert stats is not None
        assert stats["count"] == client._history_maxlen


# ---------------------------------------------------------------------------
# SentimentScanner
# ---------------------------------------------------------------------------

class TestSentimentScanner:
    """Test SentimentScanner."""

    def test_edge_type(self):
        scanner = SentimentScanner()
        assert scanner.edge_type == EdgeType.SENTIMENT_SPIKE

    def test_is_active_weekday_market_hours(self):
        scanner = SentimentScanner()
        # Wednesday at 15:00 UTC (10:00 ET) — market hours
        ts = datetime(2025, 6, 4, 15, 0, 0)
        assert scanner.is_active(ts) is True

    def test_is_active_premarket(self):
        scanner = SentimentScanner()
        # Wednesday at 12:30 UTC (7:30 ET) — pre-market
        ts = datetime(2025, 6, 4, 12, 30, 0)
        assert scanner.is_active(ts) is True

    def test_is_not_active_weekend(self):
        scanner = SentimentScanner()
        # Saturday at 15:00 UTC
        ts = datetime(2025, 6, 7, 15, 0, 0)
        assert scanner.is_active(ts) is False

    def test_is_not_active_overnight(self):
        scanner = SentimentScanner()
        # Wednesday at 03:00 UTC (night)
        ts = datetime(2025, 6, 4, 3, 0, 0)
        assert scanner.is_active(ts) is False

    def test_check_volume_high_volume(self):
        """Should confirm when current volume > 1.5x average."""
        scanner = SentimentScanner()

        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=21, freq="D")
        avg_vol = 1_000_000

        # Last bar has 2x average volume
        volumes = [avg_vol] * 20 + [2_000_000]

        bars = pd.DataFrame(
            {
                "open": [100] * 21,
                "high": [102] * 21,
                "low": [98] * 21,
                "close": [101] * 21,
                "volume": volumes,
            },
            index=dates,
        )

        # Mock get_bars_safe to return our bars
        async def mock_get_bars_safe(symbol, timeframe, limit):
            return bars

        scanner.get_bars_safe = mock_get_bars_safe

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            scanner._check_volume("TEST")
        )

        assert result["confirmed"] is True
        assert result["ratio"] == pytest.approx(2.0, abs=0.01)

    def test_check_volume_low_volume(self):
        """Should reject when current volume < 1.5x average."""
        scanner = SentimentScanner()

        dates = pd.date_range(end=datetime.now(), periods=21, freq="D")
        avg_vol = 1_000_000

        # Last bar has only 1.2x average
        volumes = [avg_vol] * 20 + [1_200_000]

        bars = pd.DataFrame(
            {
                "open": [100] * 21,
                "high": [102] * 21,
                "low": [98] * 21,
                "close": [101] * 21,
                "volume": volumes,
            },
            index=dates,
        )

        async def mock_get_bars_safe(symbol, timeframe, limit):
            return bars

        scanner.get_bars_safe = mock_get_bars_safe

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            scanner._check_volume("TEST")
        )

        assert result["confirmed"] is False
        assert result["ratio"] == pytest.approx(1.2, abs=0.01)

    def test_check_volume_no_bars(self):
        """Should return unconfirmed when bars unavailable."""
        scanner = SentimentScanner()

        async def mock_get_bars_safe(symbol, timeframe, limit):
            return None

        scanner.get_bars_safe = mock_get_bars_safe

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            scanner._check_volume("TEST")
        )

        assert result["confirmed"] is False
        assert result["ratio"] == 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    """Test singleton pattern."""

    def test_singleton(self):
        import nexus.data.stocktwits as mod
        mod._st_client = None

        c1 = get_stocktwits_client()
        c2 = get_stocktwits_client()
        assert c1 is c2

        mod._st_client = None


# ---------------------------------------------------------------------------
# Orchestrator registration
# ---------------------------------------------------------------------------

class TestOrchestratorRegistration:
    """Test that SentimentScanner is registered in orchestrator."""

    def test_sentiment_in_edge_base_scores(self):
        from nexus.scanners.orchestrator import EDGE_BASE_SCORES
        assert EdgeType.SENTIMENT_SPIKE in EDGE_BASE_SCORES

    def test_sentiment_scanner_in_orchestrator(self):
        """SentimentScanner should be in the orchestrator's scanner list."""
        from nexus.scanners.orchestrator import ScannerOrchestrator

        # Create with a minimal mock provider
        mock_provider = type("MockProvider", (), {})()
        orch = ScannerOrchestrator(data_provider=mock_provider)

        scanner_names = [s.__class__.__name__ for s in orch.scanners]
        assert "SentimentScanner" in scanner_names


# ---------------------------------------------------------------------------
# Signal generator has SENTIMENT_SPIKE
# ---------------------------------------------------------------------------

class TestSignalGeneratorEdge:
    """Test that signal generator knows about SENTIMENT_SPIKE."""

    def test_expected_edges(self):
        from nexus.execution.signal_generator import SignalGenerator
        assert EdgeType.SENTIMENT_SPIKE in SignalGenerator.EXPECTED_EDGES

    def test_hold_times(self):
        from nexus.execution.signal_generator import SignalGenerator
        assert EdgeType.SENTIMENT_SPIKE in SignalGenerator.HOLD_TIMES

    def test_validity_hours(self):
        from nexus.execution.signal_generator import SignalGenerator
        assert EdgeType.SENTIMENT_SPIKE in SignalGenerator.VALIDITY_HOURS
