"""
Portfolio Heat Manager - Track and limit total portfolio risk.

Heat = Total risk across all open positions as % of equity.

Rules:
- Max heat: 25% base, expands to 35% when in profit
- Max per market: 10%
- Max correlated positions: 3
- No new positions when heat > limit
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
from enum import Enum


class HeatLevel(Enum):
    """Portfolio heat status."""

    GREEN = "green"  # < 50% of limit, all clear
    YELLOW = "yellow"  # 50-80% of limit, caution
    ORANGE = "orange"  # 80-100% of limit, reduce size
    RED = "red"  # At limit, no new positions


@dataclass
class HeatPosition:
    """Represents an open position for heat calculation."""

    symbol: str
    market: str
    direction: str  # "long" or "short"
    entry_price: float
    current_price: float
    stop_loss: float
    size: float
    risk_amount: float  # Pre-calculated risk to stop
    opened_at: datetime


@dataclass
class PortfolioHeatCheck:
    """Result of a heat check."""

    allowed: bool
    status: HeatLevel
    current_heat_pct: float
    heat_limit_pct: float
    headroom_pct: float
    market_heat: Dict[str, float]
    reason: Optional[str] = None
    suggested_size_multiplier: float = 1.0


class PortfolioHeatManager:
    """
    Manages portfolio-level risk (heat).

    Heat expands with profits:
    - Base limit: 25%
    - When daily P&L > +2%: 30%
    - When daily P&L > +5%: 35%

    Heat contracts with losses:
    - When daily P&L < -1%: 20%
    - When daily P&L < -2%: 15%
    """

    BASE_HEAT_LIMIT = 25.0
    MAX_HEAT_LIMIT = 35.0
    MIN_HEAT_LIMIT = 15.0

    MAX_MARKET_HEAT = 10.0  # Max 10% in any single market
    MAX_CORRELATED = 3  # Max 3 positions in same direction/sector

    def __init__(self, current_equity: float):
        self.current_equity = current_equity
        self.starting_equity = current_equity
        self.daily_pnl = 0.0

        self.positions: Dict[str, HeatPosition] = {}

        self._last_heat_calc: Optional[float] = None
        self._market_heat: Dict[str, float] = defaultdict(float)

    def update_equity(self, equity: float, daily_pnl: float) -> None:
        """Update current equity and daily P&L."""
        self.current_equity = equity
        self.daily_pnl = daily_pnl
        self._invalidate_cache()

    def add_position(self, position: HeatPosition) -> None:
        """Add a new position."""
        self.positions[position.symbol] = position
        self._invalidate_cache()

    def remove_position(self, symbol: str) -> Optional[HeatPosition]:
        """Remove a position."""
        pos = self.positions.pop(symbol, None)
        self._invalidate_cache()
        return pos

    def update_position(self, symbol: str, current_price: float) -> None:
        """Update position's current price."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price

    def get_heat_limit(self) -> float:
        """Get current heat limit based on daily P&L."""
        daily_pnl_pct = (
            (self.daily_pnl / self.starting_equity * 100)
            if self.starting_equity > 0
            else 0
        )

        if daily_pnl_pct >= 5.0:
            return self.MAX_HEAT_LIMIT
        elif daily_pnl_pct >= 2.0:
            return 30.0
        elif daily_pnl_pct >= 0:
            return self.BASE_HEAT_LIMIT
        elif daily_pnl_pct >= -1.0:
            return 20.0
        else:
            return self.MIN_HEAT_LIMIT

    def calculate_current_heat(self) -> float:
        """Calculate current portfolio heat as % of equity."""
        if self.current_equity <= 0:
            return 100.0

        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        return (total_risk / self.current_equity) * 100

    def calculate_market_heat(self) -> Dict[str, float]:
        """Calculate heat per market."""
        market_risk: Dict[str, float] = defaultdict(float)

        for pos in self.positions.values():
            market_risk[pos.market] += pos.risk_amount

        if self.current_equity <= 0:
            return dict(market_risk)

        return {
            market: (risk / self.current_equity) * 100
            for market, risk in market_risk.items()
        }

    def can_add_position(
        self,
        new_risk_amount: float,
        market: str,
        direction: str,
    ) -> PortfolioHeatCheck:
        """
        Check if a new position can be added.

        Args:
            new_risk_amount: Risk in currency for new position
            market: Market of new position
            direction: "long" or "short"

        Returns:
            PortfolioHeatCheck with allowed status and details
        """
        current_heat = self.calculate_current_heat()
        heat_limit = self.get_heat_limit()
        market_heat = self.calculate_market_heat()

        new_risk_pct = (
            (new_risk_amount / self.current_equity * 100)
            if self.current_equity > 0
            else 100
        )
        projected_heat = current_heat + new_risk_pct
        projected_market_heat = market_heat.get(market, 0) + new_risk_pct

        # Check 1: Total heat limit
        if projected_heat > heat_limit:
            headroom = max(0, heat_limit - current_heat)
            return PortfolioHeatCheck(
                allowed=False,
                status=HeatLevel.RED,
                current_heat_pct=current_heat,
                heat_limit_pct=heat_limit,
                headroom_pct=headroom,
                market_heat=market_heat,
                reason=f"Heat limit exceeded: {projected_heat:.1f}% > {heat_limit:.1f}%",
                suggested_size_multiplier=(
                    headroom / new_risk_pct if new_risk_pct > 0 else 0
                ),
            )

        # Check 2: Market heat limit
        if projected_market_heat > self.MAX_MARKET_HEAT:
            return PortfolioHeatCheck(
                allowed=False,
                status=HeatLevel.ORANGE,
                current_heat_pct=current_heat,
                heat_limit_pct=heat_limit,
                headroom_pct=heat_limit - current_heat,
                market_heat=market_heat,
                reason=f"Market heat limit: {market} at {projected_market_heat:.1f}% > {self.MAX_MARKET_HEAT}%",
                suggested_size_multiplier=0.5,
            )

        # Check 3: Correlated positions
        same_direction_count = sum(
            1
            for pos in self.positions.values()
            if pos.market == market and pos.direction == direction
        )
        if same_direction_count >= self.MAX_CORRELATED:
            return PortfolioHeatCheck(
                allowed=False,
                status=HeatLevel.YELLOW,
                current_heat_pct=current_heat,
                heat_limit_pct=heat_limit,
                headroom_pct=heat_limit - current_heat,
                market_heat=market_heat,
                reason=f"Max correlated: {same_direction_count} {direction} positions in {market}",
                suggested_size_multiplier=0.5,
            )

        # Determine status based on utilization
        utilization = projected_heat / heat_limit if heat_limit > 0 else 1.0
        if utilization < 0.5:
            status = HeatLevel.GREEN
            multiplier = 1.0
        elif utilization < 0.8:
            status = HeatLevel.YELLOW
            multiplier = 0.9
        else:
            status = HeatLevel.ORANGE
            multiplier = 0.7

        return PortfolioHeatCheck(
            allowed=True,
            status=status,
            current_heat_pct=current_heat,
            heat_limit_pct=heat_limit,
            headroom_pct=heat_limit - projected_heat,
            market_heat=market_heat,
            suggested_size_multiplier=multiplier,
        )

    def get_status(self) -> Dict:
        """Get full heat status."""
        current_heat = self.calculate_current_heat()
        heat_limit = self.get_heat_limit()
        market_heat = self.calculate_market_heat()

        utilization = current_heat / heat_limit if heat_limit > 0 else 1.0

        if utilization < 0.5:
            status = HeatLevel.GREEN
        elif utilization < 0.8:
            status = HeatLevel.YELLOW
        elif utilization < 1.0:
            status = HeatLevel.ORANGE
        else:
            status = HeatLevel.RED

        return {
            "status": status.value,
            "current_heat_pct": round(current_heat, 2),
            "heat_limit_pct": round(heat_limit, 2),
            "utilization_pct": round(utilization * 100, 1),
            "headroom_pct": round(heat_limit - current_heat, 2),
            "open_positions": len(self.positions),
            "market_heat": {k: round(v, 2) for k, v in market_heat.items()},
            "daily_pnl_pct": (
                round(self.daily_pnl / self.starting_equity * 100, 2)
                if self.starting_equity > 0
                else 0
            ),
        }

    def _invalidate_cache(self) -> None:
        """Invalidate cached calculations."""
        self._last_heat_calc = None
        self._market_heat.clear()

    def reset_daily(self, new_equity: float) -> None:
        """Reset for new trading day."""
        self.starting_equity = new_equity
        self.current_equity = new_equity
        self.daily_pnl = 0.0
        self._invalidate_cache()
