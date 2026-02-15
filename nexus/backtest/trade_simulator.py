"""
Simulates trade execution for backtesting.

Takes Opportunity objects from scanners and simulates:
- Entry at signal price (with slippage)
- Exit at stop loss OR take profit OR time expiry
- Tracks P&L including costs
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, List, Optional

import pandas as pd

from nexus.core.enums import Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score-based position sizing tiers
# ---------------------------------------------------------------------------

TIER_THRESHOLDS = {"A": 80, "B": 65, "C": 50, "D": 40}
TIER_MULTIPLIERS = {"A": 1.5, "B": 1.25, "C": 1.0, "D": 0.5, "F": 0.0}


def score_to_tier(score: int) -> str:
    """Convert numeric score (0-100) to tier letter."""
    if score >= TIER_THRESHOLDS["A"]:
        return "A"
    elif score >= TIER_THRESHOLDS["B"]:
        return "B"
    elif score >= TIER_THRESHOLDS["C"]:
        return "C"
    elif score >= TIER_THRESHOLDS["D"]:
        return "D"
    return "F"


def tier_multiplier(tier: str) -> float:
    """Get position size multiplier for a tier."""
    return TIER_MULTIPLIERS.get(tier, 0.0)


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    INDICATOR_EXIT = "indicator_exit"
    TIME_EXPIRY = "time_expiry"
    END_OF_DATA = "end_of_data"


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop + breakeven logic.

    When passed to simulate_trade(), enables dynamic stop management:
    - Breakeven: move stop to entry price once profit >= breakeven_atr_mult * ATR
    - Trailing: trail stop at atr_trail_multiplier * ATR below (long) / above (short)
      the best price seen, once profit >= trailing_activation_atr * ATR

    ATR must be provided in the simulate_trade() call (atr parameter).
    """

    atr_trail_multiplier: float = 1.5    # Trail distance = ATR * this
    breakeven_atr_mult: float = 1.0      # Move SL to entry after this * ATR profit
    trailing_activation_atr: float = 1.5  # Start trailing after this * ATR profit
    enabled: bool = True


@dataclass
class SimulatedTrade:
    """Result of a simulated trade."""

    opportunity_id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: ExitReason
    position_size: float
    gross_pnl: float
    costs: float
    net_pnl: float
    net_pnl_pct: float
    hold_duration: timedelta

    # From opportunity
    primary_edge: str
    score: int
    score_tier: str  # "A", "B", "C", "D", "F"


@dataclass
class TradeSimulator:
    """
    Simulates trade execution using historical data.

    Assumptions:
    - Entry at next bar open (can't trade the signal bar's close)
    - Slippage applied to entry and exit
    - Stop/target checked against bar high/low
    - Costs applied at exit
    """

    # Cost assumptions (conservative)
    spread_pct: float = 0.02  # 0.02% spread
    slippage_pct: float = 0.02  # 0.02% slippage per side
    commission_per_trade: float = 0.0  # Most brokers = 0

    # Position sizing
    account_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0

    # Trade management
    max_hold_bars: int = 20  # Max bars before time exit

    def simulate_trade(
        self,
        opportunity: Opportunity,
        bars: pd.DataFrame,
        signal_bar_idx: int,
        exit_checker: Optional[Callable[[pd.Series, bool], bool]] = None,
        risk_multiplier: float = 1.0,
        trailing_config: Optional[TrailingStopConfig] = None,
        atr: Optional[float] = None,
    ) -> Optional[SimulatedTrade]:
        """
        Simulate a single trade from an opportunity.

        Args:
            opportunity: The detected opportunity
            bars: OHLCV DataFrame (must have open/high/low/close columns)
            signal_bar_idx: Integer position where signal was generated
            exit_checker: Optional callback(bar, is_long) -> bool for
                indicator-based exits.  Checked AFTER stop/target each bar.
            risk_multiplier: Score-based sizing multiplier (1.0 = baseline).
            trailing_config: Optional trailing stop configuration (None = v1 behavior).
            atr: Current ATR value (required if trailing_config is set).

        Returns:
            SimulatedTrade if trade could be executed, None otherwise
        """
        # Entry is NEXT bar after signal
        entry_bar_idx = signal_bar_idx + 1

        if entry_bar_idx >= len(bars):
            return None  # No more data for entry

        is_long = self._is_long(opportunity.direction)

        entry_bar = bars.iloc[entry_bar_idx]
        entry_time = bars.index[entry_bar_idx]

        # Entry price = open of next bar + slippage
        if is_long:
            entry_price = entry_bar["open"] * (1 + self.slippage_pct / 100)
        else:
            entry_price = entry_bar["open"] * (1 - self.slippage_pct / 100)

        # Calculate position size (scaled by risk_multiplier for score-based sizing)
        notional_pct = opportunity.edge_data.get("notional_pct")
        if notional_pct is not None:
            # Fixed capital allocation (for indicator-exit strategies
            # where stop is catastrophic safety net, not sizing input)
            position_size = (
                self.account_balance * notional_pct / 100 * risk_multiplier
            ) / entry_price
        else:
            # Standard risk-based sizing from stop distance
            stop_distance = abs(entry_price - opportunity.stop_loss)
            if stop_distance == 0:
                return None  # Invalid stop
            risk_amount = self.account_balance * (self.risk_per_trade_pct / 100) * risk_multiplier
            position_size = risk_amount / stop_distance

        # Simulate bar-by-bar until exit
        exit_price = None
        exit_reason = None
        exit_time = None

        last_possible = min(entry_bar_idx + self.max_hold_bars + 1, len(bars))

        # Trailing stop state (only active when trailing_config is provided)
        use_trailing = (
            trailing_config is not None
            and trailing_config.enabled
            and atr is not None
            and atr > 0
        )
        current_stop = opportunity.stop_loss
        high_since_entry = entry_price
        low_since_entry = entry_price
        at_breakeven = False

        for i in range(entry_bar_idx + 1, last_possible):
            bar = bars.iloc[i]
            bar_time = bars.index[i]

            # Update extreme prices for trailing logic
            if use_trailing:
                high_since_entry = max(high_since_entry, bar["high"])
                low_since_entry = min(low_since_entry, bar["low"])

            if is_long:
                # Check stop loss first (conservative: assume worst fills first)
                if bar["low"] <= current_stop:
                    exit_price = current_stop * (1 - self.slippage_pct / 100)
                    if use_trailing and at_breakeven:
                        exit_reason = ExitReason.BREAKEVEN_STOP
                    elif use_trailing and current_stop > opportunity.stop_loss:
                        exit_reason = ExitReason.TRAILING_STOP
                    else:
                        exit_reason = ExitReason.STOP_LOSS
                    exit_time = bar_time
                    break

                # Check take profit
                if bar["high"] >= opportunity.take_profit:
                    exit_price = opportunity.take_profit * (1 - self.slippage_pct / 100)
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_time = bar_time
                    break

                # Update trailing stop (after checking this bar's exit)
                if use_trailing:
                    profit_distance = high_since_entry - entry_price
                    # Breakeven: move stop to entry once profit >= breakeven threshold
                    if (
                        not at_breakeven
                        and profit_distance >= trailing_config.breakeven_atr_mult * atr
                    ):
                        current_stop = max(current_stop, entry_price)
                        at_breakeven = True
                    # Trailing: once profit >= activation, trail at ATR distance from high
                    if profit_distance >= trailing_config.trailing_activation_atr * atr:
                        trail_stop = high_since_entry - trailing_config.atr_trail_multiplier * atr
                        current_stop = max(current_stop, trail_stop)
            else:
                # SHORT: stop if high >= stop
                if bar["high"] >= current_stop:
                    exit_price = current_stop * (1 + self.slippage_pct / 100)
                    if use_trailing and at_breakeven:
                        exit_reason = ExitReason.BREAKEVEN_STOP
                    elif use_trailing and current_stop < opportunity.stop_loss:
                        exit_reason = ExitReason.TRAILING_STOP
                    else:
                        exit_reason = ExitReason.STOP_LOSS
                    exit_time = bar_time
                    break

                # SHORT: target if low <= target
                if bar["low"] <= opportunity.take_profit:
                    exit_price = opportunity.take_profit * (1 + self.slippage_pct / 100)
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_time = bar_time
                    break

                # Update trailing stop for short
                if use_trailing:
                    profit_distance = entry_price - low_since_entry
                    if (
                        not at_breakeven
                        and profit_distance >= trailing_config.breakeven_atr_mult * atr
                    ):
                        current_stop = min(current_stop, entry_price)
                        at_breakeven = True
                    if profit_distance >= trailing_config.trailing_activation_atr * atr:
                        trail_stop = low_since_entry + trailing_config.atr_trail_multiplier * atr
                        current_stop = min(current_stop, trail_stop)

            # Indicator-based exit (checked at bar close, after stop/target)
            if exit_checker is not None and exit_checker(bar, is_long):
                exit_price = bar["close"]
                exit_reason = ExitReason.INDICATOR_EXIT
                exit_time = bar_time
                break

        # If no exit yet, close at last bar
        if exit_price is None:
            last_idx = min(entry_bar_idx + self.max_hold_bars, len(bars) - 1)
            last_bar = bars.iloc[last_idx]
            exit_time = bars.index[last_idx]
            exit_price = last_bar["close"]
            exit_reason = (
                ExitReason.TIME_EXPIRY
                if entry_bar_idx + self.max_hold_bars < len(bars)
                else ExitReason.END_OF_DATA
            )

        # Calculate P&L
        if is_long:
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size

        # Calculate costs
        trade_value = entry_price * position_size
        spread_cost = trade_value * (self.spread_pct / 100)
        total_costs = spread_cost + self.commission_per_trade

        net_pnl = gross_pnl - total_costs
        net_pnl_pct = (net_pnl / self.account_balance) * 100

        # Resolve edge value (may already be a string due to use_enum_values)
        edge = opportunity.primary_edge
        edge_str = edge.value if hasattr(edge, "value") else str(edge)

        return SimulatedTrade(
            opportunity_id=opportunity.id,
            symbol=opportunity.symbol,
            direction="long" if is_long else "short",
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            position_size=position_size,
            gross_pnl=gross_pnl,
            costs=total_costs,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            hold_duration=exit_time - entry_time,
            primary_edge=edge_str,
            score=opportunity.raw_score,
            score_tier=score_to_tier(opportunity.raw_score),
        )

    def simulate_trades(
        self,
        opportunities: List[Opportunity],
        bars: pd.DataFrame,
        signal_bar_indices: List[int],
        exit_checker: Optional[Callable[[pd.Series, bool], bool]] = None,
    ) -> List[SimulatedTrade]:
        """
        Simulate multiple trades.

        Args:
            opportunities: List of opportunities to simulate
            bars: OHLCV DataFrame
            signal_bar_indices: Bar index for each opportunity
            exit_checker: Optional indicator-based exit callback

        Returns:
            List of SimulatedTrade results (skips failed entries)
        """
        if len(opportunities) != len(signal_bar_indices):
            raise ValueError(
                f"Length mismatch: {len(opportunities)} opportunities "
                f"vs {len(signal_bar_indices)} indices"
            )

        results = []
        for opp, idx in zip(opportunities, signal_bar_indices):
            trade = self.simulate_trade(opp, bars, idx, exit_checker=exit_checker)
            if trade:
                results.append(trade)
        return results

    @staticmethod
    def _is_long(direction) -> bool:
        """Check if direction is long, handling both enum and string values."""
        if isinstance(direction, Direction):
            return direction == Direction.LONG
        return str(direction).lower() == "long"

    @staticmethod
    def trades_to_dataframe(trades: List[SimulatedTrade]) -> pd.DataFrame:
        """Convert trade results to a DataFrame for analysis."""
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append(
                {
                    "opportunity_id": t.opportunity_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "primary_edge": t.primary_edge,
                    "score": t.score,
                    "score_tier": t.score_tier,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason.value,
                    "position_size": t.position_size,
                    "gross_pnl": round(t.gross_pnl, 2),
                    "costs": round(t.costs, 2),
                    "net_pnl": round(t.net_pnl, 2),
                    "net_pnl_pct": round(t.net_pnl_pct, 4),
                    "hold_bars": t.hold_duration,
                }
            )

        return pd.DataFrame(rows)
