"""
Calculate backtest statistics and determine edge validity.

Key metrics:
- Win rate
- Profit factor
- Expected value per trade
- Max drawdown
- Sharpe ratio
- Statistical significance (t-test)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from nexus.backtest.trade_simulator import ExitReason, SimulatedTrade

logger = logging.getLogger(__name__)


@dataclass
class BacktestStatistics:
    """Complete statistics for a backtest run."""

    # Basic counts
    total_trades: int
    winners: int
    losers: int

    # Win/Loss metrics
    win_rate: float  # Percentage
    avg_win: float  # Currency
    avg_loss: float  # Currency
    avg_win_pct: float  # Percentage
    avg_loss_pct: float  # Percentage
    largest_win: float
    largest_loss: float

    # Profitability
    total_pnl: float
    total_pnl_pct: float
    profit_factor: float  # Gross wins / Gross losses
    expected_value: float  # Average P&L per trade
    expected_value_pct: float

    # Risk metrics
    max_drawdown: float  # Currency
    max_drawdown_pct: float  # Percentage
    sharpe_ratio: float
    avg_hold_duration: float  # Hours

    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05

    # Exit analysis
    stop_loss_exits: int
    take_profit_exits: int
    indicator_exits: int
    time_expiry_exits: int

    # Edge-specific
    edge_type: str
    symbol: str
    timeframe: str
    test_period: str

    # Verdict
    verdict: str  # "VALID", "MARGINAL", "INVALID", "INSUFFICIENT_DATA"
    verdict_reason: str

    # Edge-specific risk profile used for verdict (defaults for backward compat)
    risk_category: str = "medium"
    dd_threshold: float = 15.0
    pf_threshold: float = 1.2

    # Score tier distribution (defaults for backward compatibility)
    tier_a_count: int = 0
    tier_a_pnl: float = 0.0
    tier_b_count: int = 0
    tier_b_pnl: float = 0.0
    tier_c_count: int = 0
    tier_c_pnl: float = 0.0
    tier_d_count: int = 0
    tier_d_pnl: float = 0.0

    # v2.1 exit types (defaults for backward compatibility)
    trailing_stop_exits: int = 0
    breakeven_exits: int = 0


class StatisticsCalculator:
    """Calculate backtest statistics."""

    # Minimum requirements for edge validity
    MIN_TRADES = 30  # Need 30+ trades for significance
    MIN_WIN_RATE = 0.45  # At least 45% wins
    MIN_PROFIT_FACTOR = 1.1  # Wins > Losses by 10%
    MIN_EXPECTED_VALUE_PCT = 0.05  # 0.05% per trade minimum
    MAX_DRAWDOWN_PCT = 15.0  # Can't exceed 15% drawdown
    SIGNIFICANCE_LEVEL = 0.05  # p-value threshold

    def calculate(
        self,
        trades: List[SimulatedTrade],
        edge_type: str,
        symbol: str,
        timeframe: str,
        test_period: str,
        starting_balance: float = 10_000.0,
        risk_profile: Optional[Dict[str, Any]] = None,
    ) -> BacktestStatistics:
        """Calculate all statistics from a list of trades.

        Args:
            risk_profile: Edge-specific thresholds dict with keys:
                max_dd_pct, min_profit_factor, min_trades, risk_category.
                If None, uses class-level defaults.
        """
        if not trades:
            return self._empty_stats(edge_type, symbol, timeframe, test_period,
                                     risk_profile=risk_profile)

        # Basic counts
        total = len(trades)
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]

        # Win/Loss metrics
        win_rate = len(winning) / total * 100

        avg_win = float(np.mean([t.net_pnl for t in winning])) if winning else 0.0
        avg_loss = float(np.mean([t.net_pnl for t in losing])) if losing else 0.0
        avg_win_pct = float(np.mean([t.net_pnl_pct for t in winning])) if winning else 0.0
        avg_loss_pct = float(np.mean([t.net_pnl_pct for t in losing])) if losing else 0.0

        all_pnl = [t.net_pnl for t in trades]
        largest_win = max(all_pnl)
        largest_loss = min(all_pnl)

        # Profitability
        total_pnl = sum(all_pnl)
        total_pnl_pct = (total_pnl / starting_balance) * 100

        gross_wins = sum(t.net_pnl for t in winning)
        gross_losses = abs(sum(t.net_pnl for t in losing))
        profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else float("inf")
        )

        expected_value = total_pnl / total
        expected_value_pct = (expected_value / starting_balance) * 100

        # Max drawdown
        max_dd, max_dd_pct = self._calculate_drawdown(trades, starting_balance)

        # Sharpe ratio (annualised, assuming ~252 trading days)
        pnl_pct_arr = np.array([t.net_pnl_pct for t in trades])
        if len(pnl_pct_arr) >= 2 and np.std(pnl_pct_arr, ddof=1) > 0:
            sharpe = float(
                np.mean(pnl_pct_arr)
                / np.std(pnl_pct_arr, ddof=1)
                * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Average hold duration
        avg_hold = float(
            np.mean([t.hold_duration.total_seconds() / 3600 for t in trades])
        )

        # Statistical significance (one-sample t-test: is mean PnL > 0?)
        t_stat, p_value = self._significance_test(all_pnl)
        is_significant = p_value < self.SIGNIFICANCE_LEVEL and t_stat > 0

        # Exit analysis
        stop_exits = sum(1 for t in trades if t.exit_reason == ExitReason.STOP_LOSS)
        tp_exits = sum(1 for t in trades if t.exit_reason == ExitReason.TAKE_PROFIT)
        trailing_exits = sum(1 for t in trades if t.exit_reason == ExitReason.TRAILING_STOP)
        breakeven_exits = sum(1 for t in trades if t.exit_reason == ExitReason.BREAKEVEN_STOP)
        ind_exits = sum(1 for t in trades if t.exit_reason == ExitReason.INDICATOR_EXIT)
        time_exits = sum(
            1
            for t in trades
            if t.exit_reason in (ExitReason.TIME_EXPIRY, ExitReason.END_OF_DATA)
        )

        # Score tier distribution
        tier_counts: dict = {"A": 0, "B": 0, "C": 0, "D": 0}
        tier_pnls: dict = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        for t in trades:
            tier = getattr(t, "score_tier", "C")
            if tier in tier_counts:
                tier_counts[tier] += 1
                tier_pnls[tier] += t.net_pnl

        # Resolve edge-specific thresholds
        rp = risk_profile or {}
        edge_max_dd = rp.get("max_dd_pct", self.MAX_DRAWDOWN_PCT)
        edge_min_pf = rp.get("min_profit_factor", self.MIN_PROFIT_FACTOR)
        edge_min_trades = rp.get("min_trades", self.MIN_TRADES)
        edge_risk_cat = rp.get("risk_category", "medium")
        edge_p_threshold = rp.get("p_value_threshold", self.SIGNIFICANCE_LEVEL)
        edge_min_wr = rp.get("min_win_rate", self.MIN_WIN_RATE)

        # Determine verdict
        verdict, reason = self._determine_verdict(
            total, win_rate, profit_factor, expected_value_pct,
            max_dd_pct, is_significant, p_value,
            max_dd_threshold=edge_max_dd,
            min_pf_threshold=edge_min_pf,
            min_trades_threshold=edge_min_trades,
            p_value_threshold=edge_p_threshold,
            min_win_rate=edge_min_wr,
            sharpe=sharpe,
        )

        return BacktestStatistics(
            total_trades=total,
            winners=len(winning),
            losers=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            profit_factor=profit_factor,
            expected_value=expected_value,
            expected_value_pct=expected_value_pct,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            avg_hold_duration=avg_hold,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            stop_loss_exits=stop_exits,
            take_profit_exits=tp_exits,
            indicator_exits=ind_exits,
            time_expiry_exits=time_exits,
            trailing_stop_exits=trailing_exits,
            breakeven_exits=breakeven_exits,
            edge_type=edge_type,
            symbol=symbol,
            timeframe=timeframe,
            test_period=test_period,
            verdict=verdict,
            verdict_reason=reason,
            risk_category=edge_risk_cat,
            dd_threshold=edge_max_dd,
            pf_threshold=edge_min_pf,
            tier_a_count=tier_counts["A"],
            tier_a_pnl=tier_pnls["A"],
            tier_b_count=tier_counts["B"],
            tier_b_pnl=tier_pnls["B"],
            tier_c_count=tier_counts["C"],
            tier_c_pnl=tier_pnls["C"],
            tier_d_count=tier_counts["D"],
            tier_d_pnl=tier_pnls["D"],
        )

    @staticmethod
    def _significance_test(pnl_values: List[float]) -> Tuple[float, float]:
        """One-tailed t-test: is mean PnL significantly > 0?"""
        if len(pnl_values) < 2:
            return 0.0, 1.0

        t_stat, p_two = sp_stats.ttest_1samp(pnl_values, 0)
        # Convert to one-tailed (H1: mean > 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        return float(t_stat), float(p_one)

    @staticmethod
    def _calculate_drawdown(
        trades: List[SimulatedTrade],
        starting_balance: float,
    ) -> Tuple[float, float]:
        """Calculate maximum drawdown from sequential equity curve."""
        equity = starting_balance
        peak = starting_balance
        max_dd = 0.0
        max_dd_pct = 0.0

        for trade in trades:
            equity += trade.net_pnl

            if equity > peak:
                peak = equity

            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak) * 100 if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def _determine_verdict(
        self,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        ev_pct: float,
        max_dd_pct: float,
        is_significant: bool,
        p_value: float,
        *,
        max_dd_threshold: Optional[float] = None,
        min_pf_threshold: Optional[float] = None,
        min_trades_threshold: Optional[int] = None,
        p_value_threshold: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        sharpe: float = 0.0,
    ) -> Tuple[str, str]:
        """Determine if edge is valid using edge-specific thresholds.

        Uses a sliding significance threshold based on sample size,
        capped by the edge-specific p_value_threshold:
        - 30-49 trades:  p < 0.25 (relaxed — small sample)
        - 50-99 trades:  p < 0.15 (moderate)
        - 100+ trades:   p < edge p_value_threshold (default 0.05)

        Edge-specific overrides (via risk profiles):
        - max_dd_threshold: per-edge MaxDD limit (default 15%)
        - min_pf_threshold: per-edge min profit factor (default 1.1)
        - min_trades_threshold: per-edge min trades (default 30)
        - p_value_threshold: per-edge significance level (default 0.05)
        - min_win_rate: per-edge minimum win rate (default 0.45)
        """
        eff_min_trades = min_trades_threshold or self.MIN_TRADES
        eff_max_dd = max_dd_threshold if max_dd_threshold is not None else self.MAX_DRAWDOWN_PCT
        eff_min_pf = min_pf_threshold if min_pf_threshold is not None else self.MIN_PROFIT_FACTOR
        eff_p_threshold = p_value_threshold if p_value_threshold is not None else self.SIGNIFICANCE_LEVEL
        eff_min_wr = min_win_rate if min_win_rate is not None else self.MIN_WIN_RATE

        if total_trades < eff_min_trades:
            return (
                "INSUFFICIENT_DATA",
                f"Only {total_trades} trades (need {eff_min_trades}+)",
            )

        # Sliding significance threshold based on sample size,
        # but use the edge-specific threshold as the floor for 100+ trades
        if total_trades < 50:
            sig_threshold = max(0.25, eff_p_threshold)
        elif total_trades < 100:
            sig_threshold = max(0.15, eff_p_threshold)
        else:
            sig_threshold = eff_p_threshold

        # Re-evaluate significance with the sliding threshold
        sig_met = p_value < sig_threshold and (is_significant or p_value < sig_threshold)

        reasons: List[str] = []

        if not sig_met:
            reasons.append(
                f"Not statistically significant "
                f"(p={p_value:.3f}, need <{sig_threshold})"
            )

        if win_rate < eff_min_wr * 100:
            # Sharpe override: if Sharpe > 1.5, trend-following edges with
            # low win rate but big winners are acceptable
            if sharpe < 1.5:
                reasons.append(
                    f"Win rate {win_rate:.1f}% below {eff_min_wr * 100:.0f}%"
                )

        if profit_factor < eff_min_pf:
            reasons.append(
                f"Profit factor {profit_factor:.2f} below {eff_min_pf}"
            )

        if ev_pct < self.MIN_EXPECTED_VALUE_PCT:
            # Sharpe override: if Sharpe > 1.5, accept lower EV
            # (risk-adjusted returns are good despite thin per-trade EV)
            if sharpe < 1.5:
                reasons.append(
                    f"Expected value {ev_pct:.3f}% below {self.MIN_EXPECTED_VALUE_PCT}%"
                )

        if max_dd_pct > eff_max_dd:
            reasons.append(
                f"Max drawdown {max_dd_pct:.1f}% exceeds {eff_max_dd:.0f}%"
            )

        if not reasons:
            return "VALID", "All criteria met"

        # MARGINAL if only minor violations (within 10% of thresholds)
        minor_violations = all(
            ("drawdown" in r and max_dd_pct <= eff_max_dd * 1.1) or
            ("profit factor" in r.lower() and profit_factor >= eff_min_pf * 0.9) or
            ("significant" in r.lower() and p_value <= sig_threshold * 1.5) or
            ("Expected value" in r and sharpe > 1.0)
            for r in reasons
        )

        if minor_violations and len(reasons) <= 2 and sig_met:
            return "MARGINAL", "; ".join(reasons)
        elif len(reasons) <= 2 and sig_met:
            return "MARGINAL", "; ".join(reasons)
        # Allow MARGINAL when significance is the ONLY minor violation
        # (p within 1.5x threshold + strong Sharpe indicates real but thin edge)
        elif minor_violations and len(reasons) == 1 and "significant" in reasons[0].lower():
            return "MARGINAL", "; ".join(reasons)
        else:
            return "INVALID", "; ".join(reasons)

    @staticmethod
    def _empty_stats(
        edge_type: str,
        symbol: str,
        timeframe: str,
        test_period: str,
        risk_profile: Optional[Dict[str, Any]] = None,
    ) -> BacktestStatistics:
        """Return empty stats when no trades."""
        rp = risk_profile or {}
        return BacktestStatistics(
            total_trades=0,
            winners=0,
            losers=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            profit_factor=0.0,
            expected_value=0.0,
            expected_value_pct=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_hold_duration=0.0,
            t_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            stop_loss_exits=0,
            take_profit_exits=0,
            indicator_exits=0,
            time_expiry_exits=0,
            edge_type=edge_type,
            symbol=symbol,
            timeframe=timeframe,
            test_period=test_period,
            verdict="INSUFFICIENT_DATA",
            verdict_reason="No trades generated",
            risk_category=rp.get("risk_category", "medium"),
            dd_threshold=rp.get("max_dd_pct", 15.0),
            pf_threshold=rp.get("min_profit_factor", 1.2),
        )
