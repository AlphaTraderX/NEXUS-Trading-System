"""
NEXUS Dynamic Heat Manager tests.
"""

import pytest
from typing import Optional
from unittest.mock import MagicMock

from nexus.core.enums import Market, Direction
from nexus.core.models import TrackedPosition, HeatCheckResult, HeatSummary
from nexus.risk.heat_manager import DynamicHeatManager, BASE_HEAT_LIMIT, MAX_HEAT_LIMIT, MIN_HEAT_LIMIT, MAX_PER_MARKET


def _position(
    position_id: str = "pos_001",
    symbol: str = "SPY",
    market: Market = Market.US_STOCKS,
    direction: Direction = Direction.LONG,
    entry_price: float = 500.0,
    current_price: float = 500.0,
    stop_loss: float = 492.5,
    position_size: float = 20,
    risk_amount: float = 150.0,
    risk_pct: float = 1.5,
    sector: Optional[str] = None,
) -> TrackedPosition:
    return TrackedPosition(
        position_id=position_id,
        symbol=symbol,
        market=market,
        direction=direction,
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        position_size=position_size,
        risk_amount=risk_amount,
        risk_pct=risk_pct,
        sector=sector,
    )


@pytest.fixture
def mock_settings():
    """Settings with heat config."""
    s = MagicMock()
    s.base_heat_limit = 25.0
    s.max_heat_limit = 35.0
    s.min_heat_limit = 15.0
    s.max_per_market = 10.0
    s.max_correlated = 3
    return s


@pytest.fixture
def manager(mock_settings):
    return DynamicHeatManager(settings=mock_settings)


def test_add_remove_position(manager):
    """Basic add/remove operations."""
    pos = _position(position_id="p1", symbol="AAPL", risk_pct=1.0)
    manager.add_position(pos)
    assert manager.get_current_heat() == 1.0
    assert len(manager._positions) == 1

    removed = manager.remove_position("p1")
    assert removed is not None
    assert removed.symbol == "AAPL"
    assert manager.get_current_heat() == 0.0
    assert len(manager._positions) == 0

    assert manager.remove_position("nonexistent") is None


def test_current_heat_calculation(manager):
    """Sum of all position risks."""
    manager.add_position(_position(position_id="a", risk_pct=2.0))
    manager.add_position(_position(position_id="b", symbol="MSFT", risk_pct=1.5))
    manager.add_position(_position(position_id="c", symbol="GOOGL", risk_pct=1.0))
    assert manager.get_current_heat() == 2.0 + 1.5 + 1.0


def test_heat_by_market(manager):
    """Correct grouping by market."""
    manager.add_position(_position(position_id="1", symbol="SPY", market=Market.US_STOCKS, risk_pct=3.0))
    manager.add_position(_position(position_id="2", symbol="MSFT", market=Market.US_STOCKS, risk_pct=2.0))
    manager.add_position(_position(position_id="3", symbol="EUR/USD", market=Market.FOREX_MAJORS, risk_pct=1.5))

    by_market = manager.get_heat_by_market()
    assert by_market[Market.US_STOCKS] == 5.0
    assert by_market[Market.FOREX_MAJORS] == 1.5


def test_heat_by_sector(manager):
    """Correct grouping by sector."""
    manager.add_position(_position(position_id="1", symbol="AAPL", sector="Technology", risk_pct=2.0))
    manager.add_position(_position(position_id="2", symbol="MSFT", sector="Technology", risk_pct=1.5))
    manager.add_position(_position(position_id="3", symbol="XOM", sector="Energy", risk_pct=1.0))

    by_sector = manager.get_heat_by_sector()
    assert by_sector["Technology"] == 3.5
    assert by_sector["Energy"] == 1.0


def test_dynamic_heat_limit_profit(manager):
    """Limit increases when in profit."""
    assert manager.get_heat_limit(2.5) == MAX_HEAT_LIMIT  # 35%
    assert manager.get_heat_limit(1.5) == BASE_HEAT_LIMIT + 5.0  # 30%
    assert manager.get_heat_limit(0.5) == BASE_HEAT_LIMIT  # 25%


def test_dynamic_heat_limit_loss(manager):
    """Limit decreases when in loss."""
    assert manager.get_heat_limit(0.0) == BASE_HEAT_LIMIT  # 25%
    assert manager.get_heat_limit(-0.5) == BASE_HEAT_LIMIT - 5.0  # 20%
    assert manager.get_heat_limit(-1.5) == MIN_HEAT_LIMIT  # 15%


def test_can_add_position_allowed(manager):
    """Returns allowed when under limits."""
    manager.add_position(_position(position_id="1", risk_pct=5.0, market=Market.US_STOCKS))
    check = manager.can_add_position(
        new_risk_pct=2.0,
        market=Market.US_STOCKS,
        daily_pnl_pct=0.5,
        sector="Technology",
    )
    assert check.allowed is True
    assert check.current_heat == 5.0
    assert check.heat_after == 7.0
    assert check.heat_limit == BASE_HEAT_LIMIT
    assert check.rejection_reasons == []


def test_can_add_position_total_heat_exceeded(manager):
    """Rejects when total heat exceeded."""
    manager.add_position(_position(position_id="1", risk_pct=23.0, market=Market.US_STOCKS))
    check = manager.can_add_position(
        new_risk_pct=5.0,
        market=Market.US_STOCKS,
        daily_pnl_pct=0.0,
    )
    assert check.allowed is False
    assert 23.0 + 5.0 > check.heat_limit
    assert any("Total heat" in r for r in check.rejection_reasons)


def test_can_add_position_market_heat_exceeded(manager):
    """Rejects when market heat exceeded."""
    manager.add_position(_position(position_id="1", symbol="SPY", market=Market.US_STOCKS, risk_pct=8.0))
    check = manager.can_add_position(
        new_risk_pct=5.0,
        market=Market.US_STOCKS,
        daily_pnl_pct=0.0,
    )
    assert check.allowed is False
    assert check.market_heat_after == 13.0
    assert check.market_limit == MAX_PER_MARKET
    assert any("Market" in r for r in check.rejection_reasons)


def test_update_position_price(manager):
    """Price updates work correctly."""
    pos = _position(position_id="1", entry_price=100.0, current_price=100.0, position_size=10)
    manager.add_position(pos)
    manager.update_position_price("1", 105.0)
    with manager._lock:
        p = manager._positions["1"]
    assert p.current_price == 105.0
    assert p.current_pnl == (105.0 - 100.0) * 10
    assert p.current_pnl_pct == 5.0


def test_portfolio_summary(manager):
    """Summary includes all required fields."""
    manager.add_position(_position(position_id="1", risk_pct=2.0, sector="Tech"))
    summary = manager.get_portfolio_summary(daily_pnl_pct=0.0)

    assert summary.total_heat == 2.0
    assert summary.heat_limit == BASE_HEAT_LIMIT
    assert summary.headroom == BASE_HEAT_LIMIT - 2.0
    assert summary.position_count == 1
    assert isinstance(summary.heat_by_market, dict)
    assert isinstance(summary.heat_by_sector, dict)
    assert summary.heat_by_sector.get("Tech") == 2.0
    assert isinstance(summary.positions, list)
    assert len(summary.positions) == 1
    assert summary.positions[0]["symbol"] == "SPY"
    assert isinstance(summary.warnings, list)
    assert hasattr(summary, "to_dict")
    assert summary.to_dict()["total_heat"] == 2.0


def test_can_add_position_sector_correlated_exceeded(manager):
    """Rejects when sector would exceed max_correlated (3)."""
    for i in range(3):
        manager.add_position(
            _position(position_id=f"p{i}", symbol=f"S{i}", sector="Technology", risk_pct=0.5)
        )
    check = manager.can_add_position(
        new_risk_pct=0.5,
        market=Market.US_STOCKS,
        daily_pnl_pct=0.0,
        sector="Technology",
    )
    assert check.allowed is False
    assert any("Sector" in r or "correlated" in r for r in check.rejection_reasons)


def test_empty_portfolio(manager):
    """Handles zero positions gracefully."""
    assert manager.get_current_heat() == 0.0
    assert manager.get_heat_by_market() == {}
    assert manager.get_heat_by_sector() == {}

    summary = manager.get_portfolio_summary(daily_pnl_pct=0.0)
    assert summary.total_heat == 0.0
    assert summary.position_count == 0
    assert summary.positions == []
    assert summary.headroom == BASE_HEAT_LIMIT

    check = manager.can_add_position(1.0, Market.US_STOCKS, 0.0)
    assert check.allowed is True
    assert check.current_heat == 0.0
    assert check.heat_after == 1.0
