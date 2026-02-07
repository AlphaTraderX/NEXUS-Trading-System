"""
NEXUS Dynamic Heat Manager
Tracks portfolio-level risk and prevents overexposure.

HEAT = Total risk across all open positions as % of equity

KEY INSIGHT:
Heat limit should EXPAND when you're profitable (profits as buffer)
and CONTRACT when you're losing (protect capital).

LIMITS:
- Base heat limit: 25% of equity
- In profit: Can expand to 35%
- In loss: Contracts to 15-20%

RULES:
- Max heat per market: 10%
- Max correlated positions: 3
- Track by sector/market for concentration risk
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.core.enums import Market, Direction


@dataclass
class Position:
    """Represents an open position for heat tracking."""
    symbol: str
    market: Market
    direction: Direction
    entry_price: float
    current_price: float
    stop_loss: float
    size: float  # Number of units
    entry_time: Optional[datetime] = None
    sector: str = "unknown"
    correlation_group: Optional[str] = None  # e.g., "tech", "energy", "forex_usd"

    @property
    def risk_to_stop(self) -> float:
        """Dollar risk from current price to stop."""
        return abs(self.current_price - self.stop_loss) * self.size

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized P&L."""
        if self.direction == Direction.LONG or self.direction.value == "long":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of position value."""
        entry_value = self.entry_price * self.size
        if entry_value == 0:
            return 0
        return (self.unrealized_pnl / entry_value) * 100


@dataclass
class HeatAnalysis:
    """Analysis of current portfolio heat."""
    total_heat: float           # Total $ at risk
    heat_percent: float         # Heat as % of equity
    heat_limit: float           # Current heat limit %
    heat_remaining: float       # Remaining heat capacity %
    can_add_position: bool

    # Breakdowns
    by_market: Dict[str, float]       # Heat by market
    by_sector: Dict[str, float]       # Heat by sector
    by_direction: Dict[str, float]    # Heat by long/short
    by_correlation: Dict[str, int]    # Position count by correlation group

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_heat": round(self.total_heat, 2),
            "heat_percent": round(self.heat_percent, 2),
            "heat_limit": round(self.heat_limit, 2),
            "heat_remaining": round(self.heat_remaining, 2),
            "can_add_position": self.can_add_position,
            "by_market": {k: round(v, 2) for k, v in self.by_market.items()},
            "by_sector": {k: round(v, 2) for k, v in self.by_sector.items()},
            "by_direction": {k: round(v, 2) for k, v in self.by_direction.items()},
            "by_correlation": self.by_correlation,
            "warnings": self.warnings,
        }


class DynamicHeatManager:
    """
    Manage portfolio heat with dynamic limits.

    Heat limit expands when profitable, contracts when losing.
    This lets you be more aggressive when you have a buffer.
    """

    def __init__(
        self,
        base_heat_limit: float = 25.0,
        min_heat_limit: float = 15.0,
        max_heat_limit: float = 35.0,
        max_per_market: float = 10.0,
        max_per_sector: float = 12.0,
        max_correlated: int = 3,
        max_same_direction: int = 5,
    ):
        """
        Initialize heat manager.

        Args:
            base_heat_limit: Default heat limit as % of equity
            min_heat_limit: Minimum heat limit (when losing)
            max_heat_limit: Maximum heat limit (when profitable)
            max_per_market: Maximum heat per market
            max_per_sector: Maximum heat per sector
            max_correlated: Maximum positions in same correlation group
            max_same_direction: Maximum positions in same direction
        """
        self.base_heat_limit = base_heat_limit
        self.min_heat_limit = min_heat_limit
        self.max_heat_limit = max_heat_limit
        self.max_per_market = max_per_market
        self.max_per_sector = max_per_sector
        self.max_correlated = max_correlated
        self.max_same_direction = max_same_direction

        # Track positions
        self.positions: List[Position] = []

    def get_dynamic_heat_limit(self, daily_pnl_pct: float) -> float:
        """
        Calculate current heat limit based on daily P&L.

        When profitable, expand limit (profits as buffer).
        When losing, contract limit (protect capital).

        Args:
            daily_pnl_pct: Today's P&L as percentage

        Returns:
            Current heat limit as percentage
        """
        if daily_pnl_pct >= 2.0:
            # Up 2%+ - expand to max
            limit = self.max_heat_limit
        elif daily_pnl_pct >= 1.0:
            # Up 1-2% - expand moderately
            limit = self.base_heat_limit + 5.0
        elif daily_pnl_pct >= 0:
            # Flat to up 1% - use base
            limit = self.base_heat_limit
        elif daily_pnl_pct >= -1.0:
            # Down 0-1% - contract slightly
            limit = self.base_heat_limit - 5.0
        elif daily_pnl_pct >= -2.0:
            # Down 1-2% - contract more
            limit = self.base_heat_limit - 8.0
        else:
            # Down 2%+ - minimum limit
            limit = self.min_heat_limit

        return max(self.min_heat_limit, min(limit, self.max_heat_limit))

    def add_position(self, position: Position):
        """Add a position to track."""
        self.positions.append(position)

    def remove_position(self, symbol: str):
        """Remove a position by symbol."""
        self.positions = [p for p in self.positions if p.symbol != symbol]

    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        for pos in self.positions:
            if pos.symbol == symbol:
                pos.current_price = current_price
                break

    def clear_positions(self):
        """Clear all positions (e.g., end of day)."""
        self.positions = []

    def analyze(
        self,
        equity: float,
        daily_pnl_pct: float = 0.0,
        positions: Optional[List[Position]] = None
    ) -> HeatAnalysis:
        """
        Analyze current portfolio heat.

        Args:
            equity: Current account equity
            daily_pnl_pct: Today's P&L as percentage
            positions: Optional list of positions (uses internal if None)

        Returns:
            HeatAnalysis with full breakdown
        """
        if positions is not None:
            self.positions = positions

        # Get dynamic heat limit
        heat_limit = self.get_dynamic_heat_limit(daily_pnl_pct)

        # Calculate total heat
        total_heat = sum(p.risk_to_stop for p in self.positions)
        heat_percent = (total_heat / equity * 100) if equity > 0 else 0
        heat_remaining = heat_limit - heat_percent

        # Breakdowns
        by_market: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}
        by_direction: Dict[str, float] = {"long": 0.0, "short": 0.0}
        by_correlation: Dict[str, int] = {}

        for pos in self.positions:
            # By market
            market_key = pos.market.value if hasattr(pos.market, 'value') else str(pos.market)
            if market_key not in by_market:
                by_market[market_key] = 0
            by_market[market_key] += (pos.risk_to_stop / equity * 100) if equity > 0 else 0

            # By sector
            if pos.sector not in by_sector:
                by_sector[pos.sector] = 0
            by_sector[pos.sector] += (pos.risk_to_stop / equity * 100) if equity > 0 else 0

            # By direction
            dir_key = pos.direction.value if hasattr(pos.direction, 'value') else str(pos.direction)
            by_direction[dir_key] += (pos.risk_to_stop / equity * 100) if equity > 0 else 0

            # By correlation group
            if pos.correlation_group:
                if pos.correlation_group not in by_correlation:
                    by_correlation[pos.correlation_group] = 0
                by_correlation[pos.correlation_group] += 1

        # Generate warnings
        warnings: List[str] = []

        if heat_percent >= heat_limit * 0.9:
            warnings.append(f"Near heat limit: {heat_percent:.1f}% of {heat_limit:.1f}%")

        for market, pct in by_market.items():
            if pct > self.max_per_market:
                warnings.append(f"Market {market} over limit: {pct:.1f}% > {self.max_per_market}%")

        for sector, pct in by_sector.items():
            if pct > self.max_per_sector:
                warnings.append(f"Sector {sector} over limit: {pct:.1f}% > {self.max_per_sector}%")

        for group, count in by_correlation.items():
            if count > self.max_correlated:
                warnings.append(f"Correlation group {group}: {count} positions > max {self.max_correlated}")

        long_count = sum(1 for p in self.positions if (p.direction.value if hasattr(p.direction, 'value') else str(p.direction)) == "long")
        short_count = len(self.positions) - long_count

        if long_count > self.max_same_direction:
            warnings.append(f"Too many longs: {long_count} > max {self.max_same_direction}")
        if short_count > self.max_same_direction:
            warnings.append(f"Too many shorts: {short_count} > max {self.max_same_direction}")

        # Can we add more positions?
        can_add = heat_remaining > 0.5 and len(warnings) == 0

        return HeatAnalysis(
            total_heat=total_heat,
            heat_percent=heat_percent,
            heat_limit=heat_limit,
            heat_remaining=heat_remaining,
            can_add_position=can_add,
            by_market=by_market,
            by_sector=by_sector,
            by_direction=by_direction,
            by_correlation=by_correlation,
            warnings=warnings,
        )

    def can_add_position(
        self,
        new_position: Position,
        equity: float,
        daily_pnl_pct: float = 0.0
    ) -> Dict:
        """
        Check if a new position can be added.

        Args:
            new_position: The position to potentially add
            equity: Current equity
            daily_pnl_pct: Today's P&L percentage

        Returns:
            Dict with 'allowed', 'reason', and 'adjusted_size' if needed
        """
        # Get current analysis
        analysis = self.analyze(equity, daily_pnl_pct)

        # Calculate new position's heat
        new_heat_pct = (new_position.risk_to_stop / equity * 100) if equity > 0 else 0

        # Check total heat
        if analysis.heat_percent + new_heat_pct > analysis.heat_limit:
            max_allowed_heat = analysis.heat_remaining
            if max_allowed_heat <= 0:
                return {
                    "allowed": False,
                    "reason": f"Heat limit reached ({analysis.heat_percent:.1f}%/{analysis.heat_limit:.1f}%)",
                    "adjusted_size": 0
                }
            else:
                # Calculate reduced size to fit
                reduction_factor = max_allowed_heat / new_heat_pct
                return {
                    "allowed": True,
                    "reason": "Size reduced to fit heat limit",
                    "adjusted_size": new_position.size * reduction_factor,
                    "original_size": new_position.size,
                    "reduction_factor": reduction_factor
                }

        # Check market concentration
        market_key = new_position.market.value if hasattr(new_position.market, 'value') else str(new_position.market)
        current_market_heat = analysis.by_market.get(market_key, 0)
        if current_market_heat + new_heat_pct > self.max_per_market:
            return {
                "allowed": False,
                "reason": f"Market {market_key} would exceed limit ({current_market_heat + new_heat_pct:.1f}% > {self.max_per_market}%)"
            }

        # Check sector concentration
        current_sector_heat = analysis.by_sector.get(new_position.sector, 0)
        if current_sector_heat + new_heat_pct > self.max_per_sector:
            return {
                "allowed": False,
                "reason": f"Sector {new_position.sector} would exceed limit"
            }

        # Check correlation
        if new_position.correlation_group:
            current_corr_count = analysis.by_correlation.get(new_position.correlation_group, 0)
            if current_corr_count >= self.max_correlated:
                return {
                    "allowed": False,
                    "reason": f"Correlation group {new_position.correlation_group} at limit ({current_corr_count}/{self.max_correlated})"
                }

        # Check direction count
        dir_key = new_position.direction.value if hasattr(new_position.direction, 'value') else str(new_position.direction)
        current_dir_count = sum(1 for p in self.positions if (p.direction.value if hasattr(p.direction, 'value') else str(p.direction)) == dir_key)
        if current_dir_count >= self.max_same_direction:
            return {
                "allowed": False,
                "reason": f"Too many {dir_key} positions ({current_dir_count}/{self.max_same_direction})"
            }

        # All checks passed
        return {
            "allowed": True,
            "reason": "Position within all limits",
            "heat_after": analysis.heat_percent + new_heat_pct
        }

    def get_portfolio_summary(self, equity: float, daily_pnl_pct: float = 0.0) -> Dict:
        """Get a summary of portfolio state."""
        analysis = self.analyze(equity, daily_pnl_pct)

        total_unrealized = sum(p.unrealized_pnl for p in self.positions)

        return {
            "position_count": len(self.positions),
            "total_heat_pct": round(analysis.heat_percent, 2),
            "heat_limit_pct": round(analysis.heat_limit, 2),
            "heat_remaining_pct": round(analysis.heat_remaining, 2),
            "total_risk_amount": round(analysis.total_heat, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "can_add_more": analysis.can_add_position,
            "warnings": analysis.warnings,
        }


# Test the heat manager
if __name__ == "__main__":
    from datetime import datetime

    print("=" * 60)
    print("NEXUS DYNAMIC HEAT MANAGER TEST")
    print("=" * 60)

    manager = DynamicHeatManager(
        base_heat_limit=25.0,
        min_heat_limit=15.0,
        max_heat_limit=35.0,
        max_per_market=10.0,
        max_correlated=3,
    )

    # Test 1: Dynamic heat limits
    print("\n--- Test 1: Dynamic Heat Limits ---")
    test_pnls = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    for pnl in test_pnls:
        limit = manager.get_dynamic_heat_limit(pnl)
        print(f"Daily P&L {pnl:+.1f}% -> Heat limit: {limit:.1f}%")

    # Test 2: Add positions and analyze
    print("\n--- Test 2: Portfolio Heat Analysis ---")

    positions = [
        Position(
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=150.0,
            current_price=152.0,
            stop_loss=147.0,
            size=20,
            sector="tech",
            correlation_group="tech"
        ),
        Position(
            symbol="MSFT",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=300.0,
            current_price=305.0,
            stop_loss=290.0,
            size=10,
            sector="tech",
            correlation_group="tech"
        ),
        Position(
            symbol="EUR/USD",
            market=Market.FOREX_MAJORS,
            direction=Direction.SHORT,
            entry_price=1.0850,
            current_price=1.0830,
            stop_loss=1.0900,
            size=10000,
            sector="forex",
            correlation_group="usd"
        ),
    ]

    equity = 10000.0
    analysis = manager.analyze(equity, daily_pnl_pct=1.5, positions=positions)

    print(f"Total heat: ${analysis.total_heat:.2f} ({analysis.heat_percent:.2f}%)")
    print(f"Heat limit: {analysis.heat_limit:.2f}%")
    print(f"Heat remaining: {analysis.heat_remaining:.2f}%")
    print(f"Can add position: {analysis.can_add_position}")
    print(f"By market: {analysis.by_market}")
    print(f"By direction: {analysis.by_direction}")
    print(f"By correlation: {analysis.by_correlation}")

    # Test 3: Check if new position allowed
    print("\n--- Test 3: Can Add New Position ---")

    new_pos = Position(
        symbol="GOOGL",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=140.0,
        current_price=140.0,
        stop_loss=135.0,
        size=15,
        sector="tech",
        correlation_group="tech"
    )

    result = manager.can_add_position(new_pos, equity, daily_pnl_pct=1.5)
    print(f"GOOGL allowed: {result['allowed']}")
    print(f"Reason: {result['reason']}")

    # Test 4: Correlation limit
    print("\n--- Test 4: Correlation Limit ---")

    # Add another tech position to hit limit
    manager.add_position(Position(
        symbol="NVDA",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=400.0,
        current_price=410.0,
        stop_loss=390.0,
        size=5,
        sector="tech",
        correlation_group="tech"
    ))

    # Now try to add another tech stock
    new_tech = Position(
        symbol="AMD",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=100.0,
        current_price=100.0,
        stop_loss=95.0,
        size=20,
        sector="tech",
        correlation_group="tech"
    )

    result = manager.can_add_position(new_tech, equity, daily_pnl_pct=1.5)
    print(f"AMD (4th tech) allowed: {result['allowed']}")
    print(f"Reason: {result['reason']}")

    # Test 5: Portfolio summary
    print("\n--- Test 5: Portfolio Summary ---")
    summary = manager.get_portfolio_summary(equity, daily_pnl_pct=1.5)
    print(f"Positions: {summary['position_count']}")
    print(f"Total heat: {summary['total_heat_pct']}%")
    print(f"Unrealized P&L: ${summary['unrealized_pnl']:.2f}")
    print(f"Can add more: {summary['can_add_more']}")
    if summary['warnings']:
        print(f"Warnings: {summary['warnings']}")

    # Test 6: Heat limit contraction when losing
    print("\n--- Test 6: Heat Limit When Losing ---")
    manager.clear_positions()
    analysis_loss = manager.analyze(equity, daily_pnl_pct=-2.5, positions=[])
    print(f"Daily P&L: -2.5%")
    print(f"Heat limit contracted to: {analysis_loss.heat_limit:.1f}%")

    print("\n" + "=" * 60)
    print("HEAT MANAGER TEST COMPLETE [OK]")
    print("=" * 60)
