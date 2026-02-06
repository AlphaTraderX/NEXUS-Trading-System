"""
NEXUS Cost Engine
Calculates true trading costs including spread, commission, slippage, overnight, and FX.

This is what separates profitable from unprofitable traders.
Most retail traders IGNORE these costs and wonder why they lose.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import Market


@dataclass
class CostBreakdown:
    """Breakdown of all costs for a trade."""
    spread: float = 0.0          # Spread cost as % (round trip)
    commission: float = 0.0      # Commission as %
    slippage: float = 0.0        # Expected slippage as %
    overnight: float = 0.0       # Overnight financing cost as %
    fx_conversion: float = 0.0   # FX conversion cost as %
    other: float = 0.0           # Any other costs

    @property
    def total(self) -> float:
        """Total costs as percentage."""
        return self.spread + self.commission + self.slippage + self.overnight + self.fx_conversion + self.other

    def to_dict(self) -> dict:
        return {
            "spread": round(self.spread, 4),
            "commission": round(self.commission, 4),
            "slippage": round(self.slippage, 4),
            "overnight": round(self.overnight, 4),
            "fx_conversion": round(self.fx_conversion, 4),
            "other": round(self.other, 4),
            "total": round(self.total, 4)
        }


class CostEngine:
    """
    Calculate TRUE costs for each trade.

    BROKER COSTS vary by broker:
    - IBKR: Low commission, tight spreads
    - IG: Spread betting (wider spreads, no commission, TAX FREE)
    - OANDA: Forex specialist, tight forex spreads

    MARKET COSTS vary by asset class:
    - US Stocks: Tightest spreads, lowest costs
    - Forex Majors: Very tight, but overnight costs
    - UK Stocks: Higher spreads than US
    - Futures: Low costs, no overnight
    """

    # Broker-specific costs
    BROKER_COSTS = {
        "ibkr": {
            "commission_per_trade": 0.35,      # USD per trade
            "commission_per_share": 0.005,     # USD per share
            "min_commission": 0.35,
            "fx_conversion": 0.00002,          # 0.002% for FX
            "spread_markup": 1.0,              # No markup on spreads
        },
        "ig": {
            "commission_per_trade": 0,         # Spread betting = no commission
            "spread_markup": 1.2,              # 20% wider spreads (cost is in spread)
            "fx_conversion": 0,                # No FX cost (spread betting)
        },
        "oanda": {
            "commission_per_trade": 0,         # No commission
            "spread_markup": 1.0,              # Tight spreads
            "fx_conversion": 0,                # Native forex
        },
        "alpaca": {
            "commission_per_trade": 0,         # Commission free
            "spread_markup": 1.0,
            "fx_conversion": 0.001,            # 0.1% for non-USD accounts
        }
    }

    # Market-specific base costs (as percentages)
    MARKET_COSTS = {
        Market.US_STOCKS: {
            "spread": 0.02,        # 0.02% typical spread
            "slippage": 0.02,      # 0.02% expected slippage
            "overnight": 0.02,    # 0.02% per day financing
        },
        Market.UK_STOCKS: {
            "spread": 0.08,        # Wider spreads in UK
            "slippage": 0.03,
            "overnight": 0.02,
        },
        Market.EU_STOCKS: {
            "spread": 0.06,
            "slippage": 0.03,
            "overnight": 0.02,
        },
        Market.FOREX_MAJORS: {
            "spread": 0.015,       # Very tight
            "slippage": 0.01,
            "overnight": 0.01,    # Swap rates vary
        },
        Market.FOREX_CROSSES: {
            "spread": 0.025,
            "slippage": 0.015,
            "overnight": 0.015,
        },
        Market.US_FUTURES: {
            "spread": 0.01,
            "slippage": 0.01,
            "overnight": 0,        # No overnight cost
        },
        Market.COMMODITIES: {
            "spread": 0.02,
            "slippage": 0.015,
            "overnight": 0,
        },
    }

    # Default market costs for unknown markets
    DEFAULT_MARKET_COSTS = {
        "spread": 0.05,
        "slippage": 0.03,
        "overnight": 0.02,
    }

    def __init__(self, default_broker: str = "ibkr"):
        """Initialize with default broker."""
        self.default_broker = default_broker

    def calculate_costs(
        self,
        symbol: str,
        market: Market,
        broker: str = None,
        position_value: float = 1000.0,
        hold_days: float = 1.0,
        shares: int = None
    ) -> CostBreakdown:
        """
        Calculate all costs for a trade.

        Args:
            symbol: Trading symbol
            market: Market enum (US_STOCKS, FOREX_MAJORS, etc.)
            broker: Broker name (ibkr, ig, oanda, alpaca)
            position_value: Total position value in base currency
            hold_days: Expected holding period in days
            shares: Number of shares (for per-share commission)

        Returns:
            CostBreakdown with all costs as percentages
        """
        broker = broker or self.default_broker
        broker_config = self.BROKER_COSTS.get(broker, self.BROKER_COSTS["ibkr"])
        market_config = self.MARKET_COSTS.get(market, self.DEFAULT_MARKET_COSTS)

        # Spread (round trip = entry + exit)
        spread_markup = broker_config.get("spread_markup", 1.0)
        spread = market_config["spread"] * spread_markup * 2  # Round trip

        # Commission
        commission_fixed = broker_config.get("commission_per_trade", 0) * 2  # Round trip
        commission_per_share = broker_config.get("commission_per_share", 0)

        if shares and commission_per_share > 0:
            commission_total = (commission_per_share * shares * 2) + commission_fixed
        else:
            commission_total = commission_fixed

        # Convert to percentage
        commission_pct = (commission_total / position_value * 100) if position_value > 0 else 0

        # Slippage (round trip)
        slippage = market_config["slippage"] * 2

        # Overnight cost (scales with hold time)
        overnight = market_config["overnight"] * hold_days

        # FX conversion cost (if trading non-base currency)
        fx_conversion = broker_config.get("fx_conversion", 0) * 2

        return CostBreakdown(
            spread=spread,
            commission=commission_pct,
            slippage=slippage,
            overnight=overnight,
            fx_conversion=fx_conversion
        )

    def calculate_net_edge(
        self,
        gross_edge: float,
        costs: CostBreakdown
    ) -> dict:
        """
        Calculate net edge after all costs.

        This is THE critical calculation. If net edge is negative or tiny,
        DON'T TAKE THE TRADE.

        Args:
            gross_edge: Expected gross return as percentage (e.g., 0.35 for 0.35%)
            costs: CostBreakdown from calculate_costs()

        Returns:
            {
                "gross_edge": 0.35,
                "total_costs": 0.11,
                "net_edge": 0.24,
                "cost_ratio": 31.4,  # Costs as % of gross edge
                "viable": True,
                "min_edge_for_viability": 0.05,
                "warnings": []
            }
        """
        net_edge = gross_edge - costs.total
        cost_ratio = (costs.total / gross_edge * 100) if gross_edge > 0 else 100

        warnings = []

        # Warning thresholds
        if cost_ratio > 50:
            warnings.append(f"High cost ratio: costs eat {cost_ratio:.0f}% of gross edge")

        if net_edge < 0.05:
            warnings.append(f"Low net edge: {net_edge:.3f}% is below minimum 0.05%")

        if costs.spread > 0.10:
            warnings.append(f"High spread cost: {costs.spread:.2f}%")

        if costs.overnight > 0.05:
            warnings.append(f"Significant overnight cost: {costs.overnight:.2f}%")

        # Viability check: net edge must be positive and meaningful
        viable = net_edge >= 0.05 and cost_ratio < 70

        return {
            "gross_edge": round(gross_edge, 4),
            "total_costs": round(costs.total, 4),
            "net_edge": round(net_edge, 4),
            "cost_ratio": round(cost_ratio, 2),
            "viable": viable,
            "min_edge_for_viability": 0.05,
            "warnings": warnings,
            "cost_breakdown": costs.to_dict()
        }

    def get_minimum_edge_for_market(self, market: Market, broker: str = None) -> float:
        """
        Get minimum gross edge required for a market to be viable.

        This helps filter out opportunities that can't overcome costs.
        """
        # Calculate costs for a typical trade
        costs = self.calculate_costs(
            symbol="TEST",
            market=market,
            broker=broker,
            position_value=1000,
            hold_days=1
        )

        # Minimum viable net edge is 0.05%
        # So minimum gross = costs + 0.05%
        min_gross = costs.total + 0.05

        # Add buffer for safety
        return round(min_gross * 1.2, 4)  # 20% buffer

    def compare_brokers(self, market: Market, position_value: float = 1000) -> dict:
        """
        Compare costs across brokers for a given market.

        Useful for choosing optimal broker for each market.
        """
        comparison = {}

        for broker_name in self.BROKER_COSTS.keys():
            costs = self.calculate_costs(
                symbol="TEST",
                market=market,
                broker=broker_name,
                position_value=position_value,
                hold_days=1
            )
            comparison[broker_name] = {
                "total_cost": round(costs.total, 4),
                "breakdown": costs.to_dict()
            }

        # Sort by total cost
        sorted_brokers = sorted(comparison.items(), key=lambda x: x[1]["total_cost"])

        return {
            "market": market.value if hasattr(market, 'value') else str(market),
            "position_value": position_value,
            "brokers_ranked": [b[0] for b in sorted_brokers],
            "best_broker": sorted_brokers[0][0],
            "comparison": comparison
        }


# Test the cost engine
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS COST ENGINE TEST")
    print("=" * 60)

    engine = CostEngine(default_broker="ibkr")

    # Test 1: US Stocks with IBKR
    print("\n--- Test 1: US Stocks (IBKR) ---")
    costs = engine.calculate_costs(
        symbol="SPY",
        market=Market.US_STOCKS,
        broker="ibkr",
        position_value=5000,
        hold_days=1,
        shares=50
    )
    print(f"Cost breakdown: {costs.to_dict()}")
    print(f"Total costs: {costs.total:.4f}%")

    # Calculate net edge
    net = engine.calculate_net_edge(gross_edge=0.35, costs=costs)
    print(f"Gross edge: {net['gross_edge']}%")
    print(f"Net edge: {net['net_edge']}%")
    print(f"Cost ratio: {net['cost_ratio']}%")
    print(f"Viable: {net['viable']}")
    if net['warnings']:
        print(f"Warnings: {net['warnings']}")

    # Test 2: Forex with IG (spread betting)
    print("\n--- Test 2: Forex Majors (IG Spread Betting) ---")
    costs = engine.calculate_costs(
        symbol="EUR/USD",
        market=Market.FOREX_MAJORS,
        broker="ig",
        position_value=2000,
        hold_days=0.5  # Intraday
    )
    print(f"Cost breakdown: {costs.to_dict()}")

    net = engine.calculate_net_edge(gross_edge=0.20, costs=costs)
    print(f"Net edge: {net['net_edge']}%")
    print(f"Viable: {net['viable']}")

    # Test 3: UK Stocks (higher costs)
    print("\n--- Test 3: UK Stocks (IBKR) ---")
    costs = engine.calculate_costs(
        symbol="BP",
        market=Market.UK_STOCKS,
        broker="ibkr",
        position_value=3000,
        hold_days=2
    )
    print(f"Total costs: {costs.total:.4f}%")

    net = engine.calculate_net_edge(gross_edge=0.25, costs=costs)
    print(f"Net edge: {net['net_edge']}%")
    print(f"Viable: {net['viable']}")

    # Test 4: Minimum edge by market
    print("\n--- Test 4: Minimum Viable Edge by Market ---")
    for market in [Market.US_STOCKS, Market.UK_STOCKS, Market.FOREX_MAJORS, Market.US_FUTURES]:
        min_edge = engine.get_minimum_edge_for_market(market)
        print(f"{market.value}: {min_edge:.4f}% minimum gross edge required")

    # Test 5: Broker comparison
    print("\n--- Test 5: Broker Comparison for US Stocks ---")
    comparison = engine.compare_brokers(Market.US_STOCKS, position_value=5000)
    print(f"Best broker: {comparison['best_broker']}")
    print(f"Ranking: {comparison['brokers_ranked']}")
    for broker, data in comparison['comparison'].items():
        print(f"  {broker}: {data['total_cost']:.4f}% total")

    print("\n" + "=" * 60)
    print("COST ENGINE TEST COMPLETE [OK]")
    print("=" * 60)
