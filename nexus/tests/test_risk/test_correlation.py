"""
NEXUS Correlation Monitor tests.
"""

import pytest
from unittest.mock import MagicMock

from nexus.core.enums import Market, Direction
from nexus.core.models import CorrelationCheckResult
from nexus.risk.correlation import (
    HIGH_CORRELATION_PAIRS,
    SECTOR_MAPPING,
    CorrelationMonitor,
    PositionCorrelationInfo,
)


@pytest.fixture
def mock_settings():
    """Settings with correlation limits."""
    s = MagicMock()
    s.max_sector_positions = 3
    s.max_same_direction_per_market = 3
    s.correlation_warning_threshold = 0.7
    s.max_effective_risk_multiplier = 2.0
    return s


@pytest.fixture
def monitor(mock_settings):
    return CorrelationMonitor(settings=mock_settings)


def test_add_remove_position(monitor):
    """Basic add/remove operations."""
    monitor.add_position(
        "pos1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert len(monitor._positions) == 1
    assert monitor._positions["pos1"].symbol == "AAPL"

    monitor.remove_position("pos1")
    assert len(monitor._positions) == 0

    monitor.remove_position("nonexistent")  # no-op, no raise
    assert len(monitor._positions) == 0


def test_get_sector(monitor):
    """Returns correct sector from mapping."""
    assert monitor.get_sector("AAPL") == "Technology"
    assert monitor.get_sector("MSFT") == "Technology"
    assert monitor.get_sector("JPM") == "Financials"
    assert monitor.get_sector("XOM") == "Energy"
    assert monitor.get_sector("SPY") == "Index"
    assert monitor.get_sector("EUR/USD") == "EUR"


def test_get_sector_unknown(monitor):
    """Returns 'Unknown' for unmapped symbols."""
    assert monitor.get_sector("UNKNOWN_TICKER") == "Unknown"
    assert monitor.get_sector("XYZ123") == "Unknown"


def test_get_correlation(monitor):
    """Returns correlation for known pairs."""
    assert monitor.get_correlation("AAPL", "MSFT") == 0.85
    assert monitor.get_correlation("SPY", "QQQ") == 0.92
    assert monitor.get_correlation("XOM", "CVX") == 0.85


def test_get_correlation_reverse(monitor):
    """Works with reversed pair order."""
    assert monitor.get_correlation("MSFT", "AAPL") == 0.85
    assert monitor.get_correlation("QQQ", "SPY") == 0.92


def test_get_correlation_unknown(monitor):
    """Returns 0.0 for unknown pairs."""
    assert monitor.get_correlation("AAPL", "JNJ") == 0.0
    assert monitor.get_correlation("UNK1", "UNK2") == 0.0


def test_check_sector_limit_allowed(monitor):
    """Under sector limit passes."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "GOOGL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.allowed is True
    assert result.sector_count.get("Technology", 0) == 3
    assert not result.rejection_reasons


def test_check_sector_limit_exceeded(monitor):
    """Over sector limit rejects."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p3", "NVDA", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "GOOGL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.allowed is False
    assert any("Sector" in r and "positions" in r for r in result.rejection_reasons)


def test_check_direction_limit_allowed(monitor):
    """Under direction limit passes."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "JPM", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "BAC", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.allowed is True
    key = "us_stocks_long"
    assert result.direction_count.get(key, 0) == 3


def test_check_direction_limit_exceeded(monitor):
    """Over direction limit rejects."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p3", "JPM", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "GOOGL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.allowed is False
    assert any(
        "Same market+direction" in r or "positions" in r
        for r in result.rejection_reasons
    )


def test_check_correlation_warning(monitor):
    """Warns on high correlation."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "MSFT", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.allowed is True
    assert any("correlation" in w.lower() for w in result.warnings)
    assert any(
        "AAPL" in p[0] and "MSFT" in p[1] or "MSFT" in p[0] and "AAPL" in p[1]
        for p in result.high_correlation_pairs
    )


def test_effective_risk_single_position(monitor):
    """Single position has multiplier 1.0."""
    result = monitor.check_new_position(
        "AAPL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.nominal_risk_pct == 1.0
    assert result.effective_risk_pct == 1.0
    assert result.risk_multiplier == 1.0


def test_effective_risk_correlated_positions(monitor):
    """Correlated positions increase multiplier."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    result = monitor.check_new_position(
        "NVDA", Market.US_STOCKS, Direction.LONG, 1.0
    )
    assert result.nominal_risk_pct == 3.0
    assert result.effective_risk_pct > 3.0
    assert result.risk_multiplier > 1.0


def test_effective_risk_limit_exceeded(monitor):
    """Rejects when effective risk too high (multiplier > max)."""
    # Use a monitor with very low max multiplier so we can trigger rejection
    tight = MagicMock()
    tight.max_sector_positions = 5
    tight.max_same_direction_per_market = 5
    tight.correlation_warning_threshold = 0.7
    tight.max_effective_risk_multiplier = 1.2
    mon = CorrelationMonitor(settings=tight)
    # Add several tech names (same sector -> high effective risk)
    mon.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    mon.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    mon.add_position("p3", "NVDA", Market.US_STOCKS, Direction.LONG, 1.0)
    result = mon.check_new_position(
        "GOOGL", Market.US_STOCKS, Direction.LONG, 1.0
    )
    # With 4 positions in Technology, multiplier will exceed 1.2
    if result.risk_multiplier > 1.2:
        assert result.allowed is False
        assert any("multiplier" in r.lower() for r in result.rejection_reasons)


def test_concentration_summary(monitor):
    """Returns complete summary."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    summary = monitor.get_concentration_summary()
    assert summary["position_count"] == 2
    assert summary["nominal_risk"] == 2.0
    assert summary["effective_risk"] >= 2.0
    assert "Technology" in summary["by_sector"]
    assert summary["by_sector"]["Technology"]["count"] == 2
    assert summary["by_sector"]["Technology"]["risk"] == 2.0
    assert "us_stocks_long" in summary["by_market_direction"]
    assert isinstance(summary["high_correlation_pairs"], list)
    assert isinstance(summary["warnings"], list)


def test_clear_positions(monitor):
    """Clears all tracked positions."""
    monitor.add_position("p1", "AAPL", Market.US_STOCKS, Direction.LONG, 1.0)
    monitor.add_position("p2", "MSFT", Market.US_STOCKS, Direction.LONG, 1.0)
    assert len(monitor._positions) == 2
    monitor.clear_positions()
    assert len(monitor._positions) == 0
    summary = monitor.get_concentration_summary()
    assert summary["position_count"] == 0
    assert summary["nominal_risk"] == 0.0


def test_correlation_check_result_to_dict():
    """Serialization works."""
    r = CorrelationCheckResult(
        allowed=True,
        nominal_risk_pct=2.0,
        effective_risk_pct=2.5,
        risk_multiplier=1.25,
        sector_count={"Technology": 2},
        direction_count={"us_stocks_long": 2},
        high_correlation_pairs=[("AAPL", "MSFT", 0.85)],
        warnings=["High correlation between AAPL and MSFT (0.85)"],
        rejection_reasons=[],
    )
    d = r.to_dict()
    assert d["allowed"] is True
    assert d["nominal_risk_pct"] == 2.0
    assert d["effective_risk_pct"] == 2.5
    assert d["risk_multiplier"] == 1.25
    assert d["sector_count"] == {"Technology": 2}
    assert d["direction_count"] == {"us_stocks_long": 2}
    assert len(d["high_correlation_pairs"]) == 1
    assert d["high_correlation_pairs"][0]["symbol1"] == "AAPL"
    assert d["high_correlation_pairs"][0]["symbol2"] == "MSFT"
    assert d["high_correlation_pairs"][0]["correlation"] == 0.85
    assert "High correlation" in d["warnings"][0]
    assert d["rejection_reasons"] == []
