"""Tests for heat check integration.

Validates that DynamicHeatManager.can_add_position() correctly gates
new positions based on total heat, market heat, and daily P&L adjustments.
"""

import pytest
from unittest.mock import MagicMock

from nexus.core.enums import Market
from nexus.risk.heat_manager import DynamicHeatManager


@pytest.fixture
def heat_manager():
    """Heat manager with default limits: base=25, min=15, max=35."""
    settings = MagicMock()
    settings.base_heat_limit = 25.0
    settings.min_heat_limit = 15.0
    settings.max_heat_limit = 35.0
    settings.max_per_market = 10.0
    settings.max_correlated = 3
    return DynamicHeatManager(settings=settings)


class TestHeatManagerCanAddPosition:
    """Test can_add_position() gating logic."""

    def test_allows_within_limit(self, heat_manager):
        """Position that stays under limit should be allowed."""
        result = heat_manager.can_add_position(
            new_risk_pct=5.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        assert result.allowed is True
        assert result.heat_after == 5.0
        assert result.current_heat == 0.0

    def test_rejects_over_limit(self, heat_manager):
        """Position that exceeds heat limit should be rejected."""
        result = heat_manager.can_add_position(
            new_risk_pct=30.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        assert result.allowed is False
        assert len(result.rejection_reasons) > 0

    def test_expands_with_profits(self, heat_manager):
        """Profitable day expands heat limit — allows more risk."""
        # With daily_pnl +2.5%, limit should expand to max (35%)
        result = heat_manager.can_add_position(
            new_risk_pct=5.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=2.5,
        )
        assert result.heat_limit == 35.0
        assert result.allowed is True

    def test_contracts_with_losses(self, heat_manager):
        """Losing day contracts heat limit."""
        # With daily_pnl -1.5%, limit should contract to min (15%)
        result = heat_manager.can_add_position(
            new_risk_pct=5.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=-1.5,
        )
        assert result.heat_limit == 15.0

    def test_at_exactly_limit(self, heat_manager):
        """Position that exactly hits the per-market limit should be allowed."""
        result = heat_manager.can_add_position(
            new_risk_pct=10.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        assert result.allowed is True
        assert result.market_heat_after == 10.0

    def test_respects_max_limit(self, heat_manager):
        """Even with profits, cannot exceed max heat limit."""
        result = heat_manager.can_add_position(
            new_risk_pct=40.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=5.0,
        )
        assert result.allowed is False

    def test_zero_current_heat(self, heat_manager):
        """Empty portfolio should allow new positions."""
        result = heat_manager.can_add_position(
            new_risk_pct=10.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        assert result.allowed is True
        assert result.current_heat == 0.0

    def test_market_heat_limit(self, heat_manager):
        """Rejects when single market exceeds per-market limit (10%)."""
        result = heat_manager.can_add_position(
            new_risk_pct=12.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        assert result.allowed is False
        assert any("Market" in r for r in result.rejection_reasons)

    def test_sector_correlation_limit(self, heat_manager):
        """Rejects when sector has too many correlated positions."""
        # Add 3 positions in same sector to hit max_correlated=3
        from nexus.core.models import TrackedPosition
        from nexus.core.enums import Direction

        for i in range(3):
            pos = TrackedPosition(
                position_id=f"pos_{i}",
                symbol=f"STOCK_{i}",
                market=Market.US_STOCKS,
                direction=Direction.LONG,
                entry_price=100.0,
                current_price=100.0,
                stop_loss=95.0,
                position_size=10.0,
                risk_amount=50.0,
                risk_pct=1.0,
                sector="tech",
            )
            heat_manager.add_position(pos)

        result = heat_manager.can_add_position(
            new_risk_pct=1.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
            sector="tech",
        )
        assert result.allowed is False
        assert any("Sector" in r for r in result.rejection_reasons)

    def test_result_has_to_dict(self, heat_manager):
        """HeatCheckResult should be serializable."""
        result = heat_manager.can_add_position(
            new_risk_pct=5.0,
            market=Market.US_STOCKS,
            daily_pnl_pct=0.0,
        )
        d = result.to_dict()
        assert "allowed" in d
        assert "heat_limit" in d
        assert "current_heat" in d


class TestHeatLimitTiers:
    """Test dynamic heat limit tiers based on daily P&L."""

    def test_flat_day_base_limit(self, heat_manager):
        assert heat_manager.get_heat_limit(0.5) == 25.0

    def test_up_1_to_2_pct(self, heat_manager):
        assert heat_manager.get_heat_limit(1.5) == 30.0

    def test_up_2_plus(self, heat_manager):
        assert heat_manager.get_heat_limit(3.0) == 35.0

    def test_down_0_to_1(self, heat_manager):
        assert heat_manager.get_heat_limit(-0.5) == 20.0

    def test_down_1_plus(self, heat_manager):
        assert heat_manager.get_heat_limit(-2.0) == 15.0
