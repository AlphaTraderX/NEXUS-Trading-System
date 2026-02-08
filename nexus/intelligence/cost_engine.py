"""
NEXUS Cost Engine — Phase 5.1
Calculates true trading costs for each opportunity (spread, commission, slippage, overnight, FX).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from nexus.core.enums import EdgeType, Market
from nexus.core.models import Opportunity


@dataclass
class CostBreakdown:
    """Breakdown of all cost components for a trade (all values as decimals, e.g. 0.02 = 0.02%)."""

    spread: float = 0.0       # percentage, round trip
    commission: float = 0.0   # percentage of position value
    slippage: float = 0.0     # percentage, round trip
    overnight: float = 0.0    # percentage per day
    fx_conversion: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all cost components (as percentage)."""
        return (
            self.spread
            + self.commission
            + self.slippage
            + self.overnight
            + self.fx_conversion
            + self.other
        )

    def to_dict(self) -> dict:
        """Serialize to dict for storage/API."""
        return {
            "spread": self.spread,
            "commission": self.commission,
            "slippage": self.slippage,
            "overnight": self.overnight,
            "fx_conversion": self.fx_conversion,
            "other": self.other,
            "total": self.total,
        }


class CostEngine:
    """Calculate true trading costs for opportunities."""

    BROKER_COSTS: Dict[str, Dict[str, float]] = {
        "ibkr": {
            "commission_per_trade": 0.35,
            "commission_per_share": 0.005,
            "fx_conversion_pct": 0.00002,
        },
        "ig": {
            "commission_per_trade": 0,
            "spread_markup": 1.2,
            "fx_conversion_pct": 0,
        },
        "oanda": {
            "commission_per_trade": 0,
            "spread_markup": 1.0,
            "fx_conversion_pct": 0,
        },
        "polygon": {
            "commission_per_trade": 0.35,
            "commission_per_share": 0.005,
            "fx_conversion_pct": 0.00002,
        },
    }

    MARKET_COSTS: Dict[Market, Dict[str, float]] = {
        Market.US_STOCKS: {
            "spread_pct": 0.02,
            "slippage_pct": 0.02,
            "overnight_pct": 0.02,
        },
        Market.UK_STOCKS: {
            "spread_pct": 0.08,
            "slippage_pct": 0.03,
            "overnight_pct": 0.02,
        },
        Market.FOREX_MAJORS: {
            "spread_pct": 0.015,
            "slippage_pct": 0.01,
            "overnight_pct": 0.01,
        },
        Market.FOREX_CROSSES: {
            "spread_pct": 0.025,
            "slippage_pct": 0.015,
            "overnight_pct": 0.015,
        },
        Market.US_FUTURES: {
            "spread_pct": 0.01,
            "slippage_pct": 0.01,
            "overnight_pct": 0,
        },
        Market.COMMODITIES: {
            "spread_pct": 0.02,
            "slippage_pct": 0.015,
            "overnight_pct": 0,
        },
    }

    EDGE_ESTIMATES: Dict[EdgeType, float] = {
        EdgeType.INSIDER_CLUSTER: 0.35,
        EdgeType.VWAP_DEVIATION: 0.18,
        EdgeType.TURN_OF_MONTH: 0.30,
        EdgeType.MONTH_END: 0.20,
        EdgeType.GAP_FILL: 0.18,
        EdgeType.RSI_EXTREME: 0.15,
        EdgeType.POWER_HOUR: 0.12,
        EdgeType.ASIAN_RANGE: 0.12,
        EdgeType.ORB: 0.15,
        EdgeType.BOLLINGER_TOUCH: 0.12,
        EdgeType.LONDON_OPEN: 0.12,
        EdgeType.NY_OPEN: 0.12,
        EdgeType.EARNINGS_DRIFT: 0.20,
    }

    def calculate_costs(
        self,
        symbol: str,
        market: Market,
        broker: str,
        position_value: float,
        hold_days: float = 1.0,
    ) -> CostBreakdown:
        """
        Compute full cost breakdown for a trade.

        Uses MARKET_COSTS for base rates, BROKER_COSTS for broker markup.
        Spread and slippage are round trip (entry + exit). Overnight scales by hold_days.
        """
        base = self.MARKET_COSTS.get(market)
        if not base:
            base = {"spread_pct": 0.05, "slippage_pct": 0.03, "overnight_pct": 0.02}

        broker_cfg = self.BROKER_COSTS.get(broker, self.BROKER_COSTS["ibkr"])
        spread_markup = broker_cfg.get("spread_markup", 1.0)

        # Spread: round trip (×2)
        spread = base["spread_pct"] * spread_markup * 2

        # Commission as fraction of position value (then we store as decimal %)
        commission_per_trade = broker_cfg.get("commission_per_trade", 0) * 2  # round trip
        commission_pct = (commission_per_trade / position_value) if position_value > 0 else 0.0

        # Slippage: round trip (×2)
        slippage = base["slippage_pct"] * 2

        # Overnight: daily rate × hold_days
        overnight = base["overnight_pct"] * hold_days

        # FX conversion (e.g. non-GBP instruments)
        fx_pct = broker_cfg.get("fx_conversion_pct", 0)
        fx_conversion = fx_pct * 2  # round trip

        return CostBreakdown(
            spread=spread,
            commission=commission_pct,
            slippage=slippage,
            overnight=overnight,
            fx_conversion=fx_conversion,
            other=0.0,
        )

    def calculate_net_edge(self, gross_edge_pct: float, costs: CostBreakdown) -> Dict[str, Any]:
        """
        Compute net edge after costs and viability.

        Returns gross_edge, total_costs, net_edge, cost_ratio, viable flag, and warnings.
        """
        total_costs = costs.total
        net_edge = gross_edge_pct - total_costs
        cost_ratio = (total_costs / gross_edge_pct * 100) if gross_edge_pct > 0 else 100.0

        viable = net_edge >= 0.05 and cost_ratio < 70
        warnings: List[str] = []

        if cost_ratio > 50:
            warnings.append(f"Costs eat {cost_ratio:.0f}% of edge")
        if net_edge < 0.05:
            warnings.append(f"Net edge {net_edge:.2f}% below minimum 0.05%")
        if net_edge < 0:
            warnings.append("Trade is UNPROFITABLE after costs")

        return {
            "gross_edge": gross_edge_pct,
            "total_costs": total_costs,
            "net_edge": net_edge,
            "cost_ratio": cost_ratio,
            "viable": viable,
            "warnings": warnings,
        }

    def get_edge_estimate(self, edge_type: EdgeType) -> float:
        """Return expected gross edge (as decimal %) for the given edge type."""
        return self.EDGE_ESTIMATES.get(edge_type, 0.10)

    def analyze_opportunity(
        self,
        opportunity: Opportunity,
        broker: str,
        position_value: float,
        hold_days: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full analysis: costs, edge estimate, and net edge for an opportunity.
        """
        costs = self.calculate_costs(
            symbol=opportunity.symbol,
            market=opportunity.market,
            broker=broker,
            position_value=position_value,
            hold_days=hold_days,
        )
        gross_edge = self.get_edge_estimate(opportunity.primary_edge)
        net_analysis = self.calculate_net_edge(gross_edge, costs)

        return {
            "opportunity_id": opportunity.id,
            "symbol": opportunity.symbol,
            "market": opportunity.market.value if hasattr(opportunity.market, "value") else str(opportunity.market),
            "primary_edge": opportunity.primary_edge.value if hasattr(opportunity.primary_edge, "value") else str(opportunity.primary_edge),
            "costs": costs.to_dict(),
            "gross_edge": gross_edge,
            **net_analysis,
        }
