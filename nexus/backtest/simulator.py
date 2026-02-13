"""
Trade Simulator for Multi-Asset Backtesting

Simulates trade execution with realistic costs, slippage, and exit logic.
Uses the existing CostEngine for cost calculation.
"""

import logging
import uuid
from typing import Optional, Tuple

import pandas as pd

from nexus.core.enums import Direction, EdgeType, Market, Timeframe
from nexus.core.models import Opportunity
from nexus.intelligence.cost_engine import CostEngine
from .models import BacktestTrade, TradeOutcome

logger = logging.getLogger(__name__)


class BacktestSimulator:
    """
    Simulate trade execution in multi-asset backtesting.

    Features:
    - Realistic entry/exit at next bar open
    - Slippage simulation
    - Cost calculation via CostEngine
    - Stop loss and take profit execution
    - Time-based exits
    """

    def __init__(
        self,
        slippage_pct: float = 0.02,
        max_hold_bars: int = 20,
        use_next_bar_entry: bool = True,
        broker: str = "ibkr",
    ):
        """
        Initialize simulator.

        Args:
            slippage_pct: Slippage as % of price (applied to entry and exit)
            max_hold_bars: Maximum bars to hold before time exit
            use_next_bar_entry: Enter at next bar open (realistic) vs signal price
            broker: Broker for cost calculation
        """
        self.slippage_pct = slippage_pct / 100
        self.max_hold_bars = max_hold_bars
        self.use_next_bar_entry = use_next_bar_entry
        self.broker = broker
        self.cost_engine = CostEngine()

    def simulate_trade(
        self,
        opportunity: Opportunity,
        data: pd.DataFrame,
        entry_bar_idx: int,
        starting_equity: float,
        risk_pct: float = 1.0,
    ) -> Optional[BacktestTrade]:
        """
        Simulate a single trade from opportunity.

        Args:
            opportunity: The trading opportunity/signal
            data: OHLCV DataFrame for the instrument
            entry_bar_idx: Index of the signal bar
            starting_equity: Current account equity
            risk_pct: Risk % for position sizing

        Returns:
            BacktestTrade if trade executed, None if skipped
        """
        if entry_bar_idx >= len(data) - 1:
            return None

        # Get entry bar (next bar if realistic mode)
        actual_entry_idx = entry_bar_idx
        if self.use_next_bar_entry:
            actual_entry_idx += 1

        if actual_entry_idx >= len(data):
            return None

        entry_bar = data.iloc[actual_entry_idx]

        # Resolve direction as enum
        direction = opportunity.direction
        if isinstance(direction, str):
            direction = Direction(direction)

        # Calculate entry price with slippage
        entry_price = self._apply_slippage(
            entry_bar["open"],
            direction,
            is_entry=True,
        )

        # Stops from the opportunity
        stop_loss = opportunity.stop_loss
        take_profit = opportunity.take_profit

        # Simulate exit
        exit_result = self._find_exit(
            data=data,
            entry_idx=actual_entry_idx,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if exit_result is None:
            return None

        exit_idx, exit_price, exit_reason, outcome = exit_result
        exit_bar = data.iloc[exit_idx]

        # Apply slippage to exit
        exit_price = self._apply_slippage(
            exit_price,
            direction,
            is_entry=False,
        )

        # Calculate P&L
        if direction == Direction.LONG:
            gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Calculate position size and P&L in dollars
        risk_amount = starting_equity * (risk_pct / 100)
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100

        if stop_distance_pct == 0:
            stop_distance_pct = 1.0  # Prevent division by zero

        position_value = risk_amount / (stop_distance_pct / 100)
        gross_pnl = position_value * (gross_pnl_pct / 100)

        # Calculate costs using the real CostEngine
        # Resolve market value
        market_val = opportunity.market
        if isinstance(market_val, str):
            market_val = Market(market_val)

        hold_bars = exit_idx - actual_entry_idx
        tf_minutes = opportunity.edge_data.get("timeframe_minutes", 1440)
        hold_days = max((hold_bars * tf_minutes) / 1440, 0.1)

        cost_breakdown = self.cost_engine.calculate_costs(
            symbol=opportunity.symbol,
            market=market_val,
            broker=self.broker,
            position_value=position_value,
            hold_days=hold_days,
        )
        costs = cost_breakdown.total / 100 * position_value  # Convert pct to dollars

        net_pnl = gross_pnl - costs
        net_pnl_pct = gross_pnl_pct - (costs / position_value * 100) if position_value > 0 else gross_pnl_pct

        # Calculate R-multiple
        r_multiple = net_pnl / risk_amount if risk_amount > 0 else 0

        # Determine outcome
        if net_pnl > 0:
            outcome = TradeOutcome.WIN
        elif net_pnl < 0:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAKEVEN

        # Resolve edge_type
        edge_type_val = opportunity.primary_edge
        if isinstance(edge_type_val, str):
            edge_type_val = EdgeType(edge_type_val)

        # Resolve timeframe
        tf_str = opportunity.edge_data.get("timeframe", "1d")
        try:
            timeframe = Timeframe(tf_str)
        except ValueError:
            timeframe = Timeframe.D1

        # Build trade result
        return BacktestTrade(
            trade_id=str(uuid.uuid4())[:8],
            signal_id=opportunity.id[:8] if opportunity.id else str(uuid.uuid4())[:8],
            symbol=opportunity.symbol,
            market=market_val,
            direction=direction,
            edge_type=edge_type_val,
            timeframe=timeframe,
            score=opportunity.raw_score or 50,
            tier=self._score_to_tier(opportunity.raw_score or 50),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_price=exit_price,
            entry_time=pd.Timestamp(entry_bar["timestamp"]).to_pydatetime(),
            exit_time=pd.Timestamp(exit_bar["timestamp"]).to_pydatetime(),
            hold_duration_minutes=hold_bars * tf_minutes,
            gross_pnl=gross_pnl,
            gross_pnl_pct=gross_pnl_pct,
            costs=costs,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            r_multiple=r_multiple,
            outcome=outcome,
            exit_reason=exit_reason,
            edge_data=opportunity.edge_data,
        )

    def _apply_slippage(
        self,
        price: float,
        direction: Direction,
        is_entry: bool,
    ) -> float:
        """Apply slippage to price (always adverse)."""
        slippage = price * self.slippage_pct

        if direction == Direction.LONG:
            if is_entry:
                return price + slippage  # Pay more to enter long
            else:
                return price - slippage  # Get less when exiting long
        else:
            if is_entry:
                return price - slippage  # Get less to enter short
            else:
                return price + slippage  # Pay more when exiting short

    def _find_exit(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        direction: Direction,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Tuple[int, float, str, TradeOutcome]]:
        """
        Find exit point in data.

        Returns:
            Tuple of (exit_idx, exit_price, exit_reason, outcome)
        """
        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_bars + 1, len(data))):
            bar = data.iloc[i]

            if direction == Direction.LONG:
                # Check stop loss (uses low)
                if bar["low"] <= stop_loss:
                    return (i, stop_loss, "stop_loss", TradeOutcome.STOPPED)

                # Check take profit (uses high)
                if bar["high"] >= take_profit:
                    return (i, take_profit, "take_profit", TradeOutcome.TARGET)

            else:  # SHORT
                # Check stop loss (uses high)
                if bar["high"] >= stop_loss:
                    return (i, stop_loss, "stop_loss", TradeOutcome.STOPPED)

                # Check take profit (uses low)
                if bar["low"] <= take_profit:
                    return (i, take_profit, "take_profit", TradeOutcome.TARGET)

        # Time exit at max hold
        if entry_idx + self.max_hold_bars < len(data):
            exit_idx = entry_idx + self.max_hold_bars
            exit_price = data.iloc[exit_idx]["close"]
            return (exit_idx, exit_price, "time_exit", TradeOutcome.TIME_EXIT)

        # Exit at last available bar
        exit_idx = len(data) - 1
        exit_price = data.iloc[exit_idx]["close"]
        return (exit_idx, exit_price, "end_of_data", TradeOutcome.TIME_EXIT)

    def _score_to_tier(self, score: int) -> str:
        """Convert score to tier."""
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        else:
            return "F"
