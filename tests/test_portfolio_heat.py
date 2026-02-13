"""Tests for portfolio heat manager."""

import pytest
from datetime import datetime
from nexus.risk.portfolio_heat import (
    PortfolioHeatManager,
    HeatPosition,
    HeatLevel,
)


class TestPortfolioHeat:
    def setup_method(self):
        self.manager = PortfolioHeatManager(current_equity=10000)

    def test_initial_heat_zero(self):
        """No positions = no heat."""
        assert self.manager.calculate_current_heat() == 0.0

    def test_heat_calculation(self):
        """Heat should equal total risk / equity."""
        self.manager.add_position(
            HeatPosition(
                symbol="AAPL",
                market="us_stocks",
                direction="long",
                entry_price=150,
                current_price=150,
                stop_loss=145,
                size=10,
                risk_amount=500,  # 5% of equity
                opened_at=datetime.utcnow(),
            )
        )

        heat = self.manager.calculate_current_heat()
        assert heat == 5.0  # 500/10000 = 5%

    def test_heat_limit_expands_with_profit(self):
        """Heat limit should expand when in profit."""
        self.manager.daily_pnl = 0
        assert self.manager.get_heat_limit() == 25.0  # Base

        self.manager.daily_pnl = 250  # +2.5%
        assert self.manager.get_heat_limit() == 30.0

        self.manager.daily_pnl = 600  # +6%
        assert self.manager.get_heat_limit() == 35.0  # Max

    def test_heat_limit_contracts_with_loss(self):
        """Heat limit should contract when in loss."""
        self.manager.daily_pnl = -50  # -0.5% (between 0 and -1%)
        assert self.manager.get_heat_limit() == 20.0

        self.manager.daily_pnl = -150  # -1.5% (below -1%)
        assert self.manager.get_heat_limit() == 15.0  # Min

    def test_position_blocked_at_limit(self):
        """New position should be blocked when at heat limit."""
        for i in range(5):
            self.manager.add_position(
                HeatPosition(
                    symbol=f"SYM{i}",
                    market="us_stocks",
                    direction="long",
                    entry_price=100,
                    current_price=100,
                    stop_loss=95,
                    size=10,
                    risk_amount=500,  # 5% each = 25% total
                    opened_at=datetime.utcnow(),
                )
            )

        result = self.manager.can_add_position(
            new_risk_amount=500,
            market="us_stocks",
            direction="long",
        )

        assert not result.allowed
        assert result.status == HeatLevel.RED

    def test_market_heat_limit(self):
        """Max 10% heat per market."""
        self.manager.add_position(
            HeatPosition(
                symbol="AAPL",
                market="us_stocks",
                direction="long",
                entry_price=100,
                current_price=100,
                stop_loss=90,
                size=10,
                risk_amount=1000,  # 10% - at market limit
                opened_at=datetime.utcnow(),
            )
        )

        result = self.manager.can_add_position(
            new_risk_amount=500,
            market="us_stocks",
            direction="long",
        )

        assert not result.allowed
        assert "Market heat limit" in result.reason

    def test_correlated_positions_limit(self):
        """Max 3 positions in same market+direction."""
        for i in range(3):
            self.manager.add_position(
                HeatPosition(
                    symbol=f"SYM{i}",
                    market="us_stocks",
                    direction="long",
                    entry_price=100,
                    current_price=100,
                    stop_loss=98,
                    size=10,
                    risk_amount=200,
                    opened_at=datetime.utcnow(),
                )
            )

        result = self.manager.can_add_position(
            new_risk_amount=200,
            market="us_stocks",
            direction="long",
        )

        assert not result.allowed
        assert "correlated" in result.reason.lower()

    def test_different_direction_allowed(self):
        """Different direction should not count as correlated."""
        for i in range(3):
            self.manager.add_position(
                HeatPosition(
                    symbol=f"SYM{i}",
                    market="us_stocks",
                    direction="long",
                    entry_price=100,
                    current_price=100,
                    stop_loss=98,
                    size=10,
                    risk_amount=200,
                    opened_at=datetime.utcnow(),
                )
            )

        # Short direction should be fine
        result = self.manager.can_add_position(
            new_risk_amount=200,
            market="us_stocks",
            direction="short",
        )
        assert result.allowed

    def test_remove_position_frees_heat(self):
        """Removing a position should free up heat."""
        self.manager.add_position(
            HeatPosition(
                symbol="AAPL",
                market="us_stocks",
                direction="long",
                entry_price=100,
                current_price=100,
                stop_loss=95,
                size=10,
                risk_amount=500,
                opened_at=datetime.utcnow(),
            )
        )
        assert self.manager.calculate_current_heat() == 5.0

        self.manager.remove_position("AAPL")
        assert self.manager.calculate_current_heat() == 0.0

    def test_status_report(self):
        """Status should report all heat metrics."""
        status = self.manager.get_status()
        assert "status" in status
        assert "current_heat_pct" in status
        assert "heat_limit_pct" in status
        assert "utilization_pct" in status
        assert status["status"] == "green"

    def test_reset_daily(self):
        """Daily reset should clear P&L and update equity."""
        self.manager.daily_pnl = 500
        self.manager.reset_daily(10500)

        assert self.manager.starting_equity == 10500
        assert self.manager.current_equity == 10500
        assert self.manager.daily_pnl == 0.0

    def test_green_status_allows_full_size(self):
        """Low utilization should suggest full size."""
        result = self.manager.can_add_position(
            new_risk_amount=200,  # 2% - very low
            market="us_stocks",
            direction="long",
        )
        assert result.allowed
        assert result.status == HeatLevel.GREEN
        assert result.suggested_size_multiplier == 1.0
