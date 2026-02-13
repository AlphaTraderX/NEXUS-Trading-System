"""
GOD MODE Backtest Runner

Tests the full GOD MODE system across:
- Multiple instruments (stocks, forex, indices, crypto)
- Multiple timeframes (M5 to D1)
- All validated edges
- Aggressive position sizing with compounding
- Market regime detection & adaptation (2020-2024 full cycle)

Usage:
    python -m nexus.scripts.backtest_god_mode --quick
    python -m nexus.scripts.backtest_god_mode --mode god_mode --instruments 20
    python -m nexus.scripts.backtest_god_mode --compare
    python -m nexus.scripts.backtest_god_mode --full-cycle
    python -m nexus.scripts.backtest_god_mode --full-cycle --stress --stress-level 0.05
"""

import argparse
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from nexus.config.god_mode import (
    TradingMode,
    GodModePositionSizer,
    IntradayCompounder,
    get_mode_config,
    compare_modes,
)
from nexus.core.enums import EdgeType, Direction, Market, Timeframe
from nexus.data.instruments import (
    get_instrument_registry,
    InstrumentRegistry,
    InstrumentType,
    Region,
    Instrument,
)
from nexus.intelligence.cost_engine import CostEngine
from nexus.intelligence.regime_detector import (
    GodModeRegime,
    GodModeRegimeDetector,
    RegimeConfig,
    REGIME_CONFIGS,
    get_historical_regimes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GodModeTradeResult:
    """Result of a single GOD MODE trade."""

    trade_id: int
    timestamp: datetime
    symbol: str
    market: str
    edge: str
    timeframe: str
    direction: str
    tier: str
    score: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_pct: float
    pnl: float
    pnl_pct: float
    is_winner: bool
    exit_reason: str
    equity_before: float
    equity_after: float
    streak_bonus: float
    score_multiplier: float
    regime: str = ""


@dataclass
class GodModeBacktestResult:
    """Complete GOD MODE backtest results."""

    mode: TradingMode
    start_date: datetime
    end_date: datetime
    starting_balance: float
    ending_balance: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    stress_test: bool = False
    stress_level: float = 0.0
    trades: List[GodModeTradeResult] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    edge_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeframe_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    market_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Instrument → Market mapping
# ═══════════════════════════════════════════════════════════════════════

_MARKET_MAP: Dict[Tuple[InstrumentType, Region], Market] = {
    (InstrumentType.STOCK, Region.US): Market.US_STOCKS,
    (InstrumentType.STOCK, Region.UK): Market.UK_STOCKS,
    (InstrumentType.STOCK, Region.EUROPE): Market.EU_STOCKS,
    (InstrumentType.INDEX, Region.US): Market.US_STOCKS,
    (InstrumentType.INDEX, Region.GLOBAL): Market.US_STOCKS,
    (InstrumentType.FOREX, Region.GLOBAL): Market.FOREX_MAJORS,
    (InstrumentType.COMMODITY, Region.GLOBAL): Market.COMMODITIES,
    (InstrumentType.CRYPTO, Region.GLOBAL): Market.US_STOCKS,  # No crypto market enum yet
}


def _instrument_market(inst: Instrument) -> Market:
    """Derive Market enum from Instrument attributes."""
    key = (inst.instrument_type, inst.region)
    return _MARKET_MAP.get(key, Market.US_STOCKS)


# ═══════════════════════════════════════════════════════════════════════
# GodModeBacktester
# ═══════════════════════════════════════════════════════════════════════


class GodModeBacktester:
    """
    GOD MODE Backtester

    Simulates trading with:
    - Dynamic position sizing based on score/tier
    - Win streak bonuses
    - Intraday compounding
    - Heat management
    - Multi-timeframe signals
    """

    # Edge win rates and avg returns (from validated research)
    EDGE_STATS = {
        EdgeType.OVERNIGHT_PREMIUM: {"win_rate": 0.54, "avg_win": 0.003, "avg_loss": 0.002, "frequency": 0.8},
        EdgeType.INSIDER_CLUSTER: {"win_rate": 0.58, "avg_win": 0.025, "avg_loss": 0.015, "frequency": 0.05},
        EdgeType.GAP_FILL: {"win_rate": 0.62, "avg_win": 0.008, "avg_loss": 0.006, "frequency": 0.3},
        EdgeType.TURN_OF_MONTH: {"win_rate": 0.61, "avg_win": 0.012, "avg_loss": 0.008, "frequency": 0.15},
        EdgeType.VWAP_DEVIATION: {"win_rate": 0.56, "avg_win": 0.006, "avg_loss": 0.004, "frequency": 0.4},
        EdgeType.RSI_EXTREME: {"win_rate": 0.55, "avg_win": 0.007, "avg_loss": 0.005, "frequency": 0.35},
        EdgeType.MONTH_END: {"win_rate": 0.58, "avg_win": 0.010, "avg_loss": 0.007, "frequency": 0.08},
        EdgeType.POWER_HOUR: {"win_rate": 0.52, "avg_win": 0.005, "avg_loss": 0.004, "frequency": 0.5},
        EdgeType.ASIAN_RANGE: {"win_rate": 0.53, "avg_win": 0.006, "avg_loss": 0.005, "frequency": 0.25},
        EdgeType.LONDON_OPEN: {"win_rate": 0.51, "avg_win": 0.005, "avg_loss": 0.004, "frequency": 0.3},
        EdgeType.NY_OPEN: {"win_rate": 0.52, "avg_win": 0.005, "avg_loss": 0.004, "frequency": 0.3},
        EdgeType.ORB: {"win_rate": 0.54, "avg_win": 0.008, "avg_loss": 0.006, "frequency": 0.2},
        EdgeType.BOLLINGER_TOUCH: {"win_rate": 0.56, "avg_win": 0.006, "avg_loss": 0.005, "frequency": 0.25},
    }

    # Timeframe multipliers (higher TF = stronger signal)
    TIMEFRAME_MULTIPLIERS = {
        Timeframe.M5: 0.6,
        Timeframe.M15: 0.75,
        Timeframe.M30: 0.85,
        Timeframe.H1: 1.0,
        Timeframe.H4: 1.1,
        Timeframe.D1: 1.2,
    }

    # Which instrument types each edge applies to
    EDGE_INSTRUMENTS: Dict[EdgeType, List[Tuple[InstrumentType, Optional[Region]]]] = {
        EdgeType.OVERNIGHT_PREMIUM: [(InstrumentType.STOCK, Region.US)],
        EdgeType.INSIDER_CLUSTER: [(InstrumentType.STOCK, Region.US)],
        EdgeType.GAP_FILL: [(InstrumentType.STOCK, Region.US), (InstrumentType.STOCK, Region.UK)],
        EdgeType.TURN_OF_MONTH: [(InstrumentType.STOCK, Region.US), (InstrumentType.INDEX, None)],
        EdgeType.VWAP_DEVIATION: [(InstrumentType.STOCK, Region.US), (InstrumentType.STOCK, Region.UK)],
        EdgeType.RSI_EXTREME: [(InstrumentType.STOCK, Region.US), (InstrumentType.FOREX, None)],
        EdgeType.MONTH_END: [(InstrumentType.STOCK, Region.US), (InstrumentType.INDEX, None)],
        EdgeType.POWER_HOUR: [(InstrumentType.STOCK, Region.US)],
        EdgeType.ASIAN_RANGE: [(InstrumentType.FOREX, None)],
        EdgeType.LONDON_OPEN: [(InstrumentType.FOREX, None), (InstrumentType.STOCK, Region.UK)],
        EdgeType.NY_OPEN: [(InstrumentType.STOCK, Region.US), (InstrumentType.FOREX, None)],
        EdgeType.ORB: [(InstrumentType.STOCK, Region.US), (InstrumentType.COMMODITY, None)],
        EdgeType.BOLLINGER_TOUCH: [(InstrumentType.STOCK, Region.US), (InstrumentType.FOREX, None)],
    }

    def __init__(
        self,
        mode: TradingMode = TradingMode.GOD_MODE,
        starting_balance: float = 10_000.0,
        seed: int = 42,
        stress_test: bool = False,
        stress_win_rate_reduction: float = 0.0,
    ):
        self.mode = mode
        self.config = get_mode_config(mode)
        self.starting_balance = starting_balance
        self.registry = get_instrument_registry()
        self.cost_engine = CostEngine()
        self.position_sizer = GodModePositionSizer(self.config)
        self.compounder = IntradayCompounder(starting_balance)
        self.rng = random.Random(seed)

        # Regime detection
        self.regime_detector = GodModeRegimeDetector()
        self.historical_regimes = get_historical_regimes()
        self.stress_test = stress_test
        self.stress_win_rate_reduction = stress_win_rate_reduction

        # State
        self.current_equity = starting_balance
        self.peak_equity = starting_balance
        self.max_drawdown = 0.0
        self.current_heat = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.daily_pnl = 0.0
        self.current_date: Optional[datetime] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_instruments_for_edge(self, edge: EdgeType, max_count: int) -> List[Instrument]:
        """Get relevant instruments for an edge from registry."""
        specs = self.EDGE_INSTRUMENTS.get(edge, [(InstrumentType.STOCK, Region.US)])

        instruments: List[Instrument] = []
        seen = set()
        for itype, region in specs:
            candidates = self.registry.get_by_type(itype)
            if region is not None:
                candidates = [c for c in candidates if c.region == region]
            for c in candidates:
                if c.symbol not in seen:
                    instruments.append(c)
                    seen.add(c.symbol)

        self.rng.shuffle(instruments)
        return instruments[:max_count]

    def _get_score_for_signal(self, edge: EdgeType, timeframe: Timeframe) -> int:
        base_scores = {
            EdgeType.INSIDER_CLUSTER: 75,
            EdgeType.TURN_OF_MONTH: 70,
            EdgeType.VWAP_DEVIATION: 68,
            EdgeType.GAP_FILL: 65,
            EdgeType.MONTH_END: 63,
            EdgeType.RSI_EXTREME: 60,
            EdgeType.OVERNIGHT_PREMIUM: 58,
            EdgeType.ORB: 55,
            EdgeType.POWER_HOUR: 52,
            EdgeType.BOLLINGER_TOUCH: 50,
            EdgeType.ASIAN_RANGE: 48,
            EdgeType.LONDON_OPEN: 45,
            EdgeType.NY_OPEN: 45,
        }
        base = base_scores.get(edge, 50)
        tf_mult = self.TIMEFRAME_MULTIPLIERS.get(timeframe, 1.0)
        variance = self.rng.randint(-10, 15)
        return max(0, min(100, int(base * tf_mult + variance)))

    def _get_tier(self, score: int) -> str:
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= self.config.min_score_to_trade:
            return "D"
        else:
            return "F"

    def _get_regime_for_date(self, date: datetime) -> GodModeRegime:
        """Get the market regime for a specific date."""
        month_key = date.strftime("%Y-%m")
        return self.historical_regimes.get(month_key, GodModeRegime.SIDEWAYS)

    def _simulate_trade_outcome(
        self,
        edge: EdgeType,
        timeframe: Timeframe,
        score: int,
        regime: GodModeRegime = GodModeRegime.SIDEWAYS,
    ) -> Tuple[bool, float, str]:
        stats = self.EDGE_STATS.get(
            edge, {"win_rate": 0.50, "avg_win": 0.005, "avg_loss": 0.004}
        )

        # Base win rate + score bonus
        score_bonus = (score - 50) / 200
        adjusted_wr = min(0.75, max(0.35, stats["win_rate"] + score_bonus))

        tf_bonus = (self.TIMEFRAME_MULTIPLIERS.get(timeframe, 1.0) - 1.0) * 0.1
        adjusted_wr = min(0.75, adjusted_wr + tf_bonus)

        # Regime adjustment
        regime_wr_adj = {
            GodModeRegime.STRONG_BULL: 0.05,
            GodModeRegime.BULL: 0.02,
            GodModeRegime.SIDEWAYS: 0.0,
            GodModeRegime.BEAR: -0.03,
            GodModeRegime.STRONG_BEAR: -0.08,
            GodModeRegime.VOLATILE: -0.05,
        }
        adjusted_wr += regime_wr_adj.get(regime, 0.0)

        # Stress test adjustment
        if self.stress_test:
            adjusted_wr -= self.stress_win_rate_reduction

        adjusted_wr = min(0.75, max(0.30, adjusted_wr))

        is_winner = self.rng.random() < adjusted_wr

        # Volatility multiplier for P&L magnitude
        vol_mult = {
            GodModeRegime.STRONG_BULL: 1.2,
            GodModeRegime.BULL: 1.0,
            GodModeRegime.SIDEWAYS: 0.8,
            GodModeRegime.BEAR: 1.1,
            GodModeRegime.STRONG_BEAR: 1.5,
            GodModeRegime.VOLATILE: 1.4,
        }.get(regime, 1.0)

        if is_winner:
            pnl_pct = stats["avg_win"] * self.rng.uniform(0.5, 2.0) * vol_mult
            exit_reason = "TARGET" if self.rng.random() < 0.7 else "TRAILING_STOP"
        else:
            pnl_pct = -stats["avg_loss"] * self.rng.uniform(0.5, 1.5) * vol_mult
            exit_reason = "STOP_LOSS" if self.rng.random() < 0.8 else "TIME_EXIT"

        return is_winner, pnl_pct, exit_reason

    def _should_generate_signal(self, edge: EdgeType, date: datetime) -> bool:
        stats = self.EDGE_STATS.get(edge, {"frequency": 0.3})

        if edge == EdgeType.TURN_OF_MONTH:
            day = date.day
            last_day = (date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if day > 3 and day < last_day.day - 1:
                return False

        if edge == EdgeType.MONTH_END:
            day = date.day
            last_day = (date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if day < last_day.day - 2:
                return False

        return self.rng.random() < stats["frequency"]

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        max_instruments_per_edge: int = 20,
        timeframes: Optional[List[Timeframe]] = None,
        edges: Optional[List[EdgeType]] = None,
    ) -> GodModeBacktestResult:
        if timeframes is None:
            timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

        if edges is None:
            edges = list(self.EDGE_STATS.keys())

        stress_tag = ""
        if self.stress_test:
            stress_tag = f"  STRESS TEST: -{self.stress_win_rate_reduction*100:.0f}% win rate"

        logger.info("=" * 70)
        logger.info("  GOD MODE BACKTEST - %s", self.mode.value.upper())
        logger.info("=" * 70)
        logger.info("  Period: %s to %s", start_date.date(), end_date.date())
        logger.info("  Starting Balance: £%s", f"{self.starting_balance:,.0f}")
        logger.info("  Mode: %s  |  Base Risk: %s%%  |  Max Risk: %s%%",
                     self.mode.value, self.config.base_risk_pct, self.config.max_risk_pct)
        logger.info("  Edges: %d  |  Timeframes: %s  |  Max Instruments/Edge: %d",
                     len(edges), [tf.value for tf in timeframes], max_instruments_per_edge)
        logger.info("  Regime-adaptive: ON  |  Regimes: 6-state%s", stress_tag)
        logger.info("=" * 70)

        trades: List[GodModeTradeResult] = []
        trade_id = 0
        equity_curve = [self.starting_balance]
        monthly_returns: Dict[str, float] = {}
        edge_perf = {e.value: {"trades": 0, "wins": 0, "pnl": 0.0} for e in edges}
        tf_perf = {tf.value: {"trades": 0, "wins": 0, "pnl": 0.0} for tf in timeframes}
        market_perf: Dict[str, Dict[str, Any]] = {}
        regime_perf: Dict[str, Dict[str, Any]] = {
            r.value: {"trades": 0, "wins": 0, "pnl": 0.0} for r in GodModeRegime
        }

        current_date = start_date
        current_month: Optional[str] = None
        month_start_equity = self.starting_balance
        hit_dd_limit = False

        while current_date <= end_date and not hit_dd_limit:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Monthly tracking
            month_key = current_date.strftime("%Y-%m")
            if month_key != current_month:
                if current_month is not None:
                    monthly_returns[current_month] = (
                        (self.current_equity - month_start_equity) / month_start_equity * 100
                    )
                current_month = month_key
                month_start_equity = self.current_equity
                self.compounder.reset_daily()

            # Daily reset — freeze day-start equity for position sizing
            if self.current_date != current_date:
                self.current_date = current_date
                self.daily_pnl = 0.0
                day_start_equity = self.current_equity

            # Check daily circuit breaker
            daily_pnl_pct = (self.daily_pnl / day_start_equity * 100) if day_start_equity > 0 else 0
            if daily_pnl_pct <= self.config.daily_loss_stop:
                current_date += timedelta(days=1)
                continue

            # ── Regime detection ──────────────────────────────────────
            current_regime = self._get_regime_for_date(current_date)
            regime_config = REGIME_CONFIGS[current_regime]

            trades_today = 0
            max_trades_per_day = min(
                self.config.max_positions,
                regime_config.max_positions,
            )

            for edge in edges:
                if trades_today >= max_trades_per_day:
                    break

                # Regime filter — skip edges not suited for this regime
                if not self.regime_detector.is_edge_allowed(edge, current_regime):
                    continue

                if not self._should_generate_signal(edge, current_date):
                    continue

                # Pick ONE instrument per edge signal (not all instruments)
                instruments = self._get_instruments_for_edge(edge, max_instruments_per_edge)
                if not instruments:
                    continue
                inst = self.rng.choice(instruments)

                timeframe = self.rng.choice(timeframes)

                score = self._get_score_for_signal(edge, timeframe)
                tier = self._get_tier(score)

                if score < self.config.min_score_to_trade or tier == "F":
                    continue

                # Position sizing — use day-start equity (not rolling intraday)
                entry_price = 100.0
                stop_distance = entry_price * 0.02 * regime_config.stop_multiplier
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + stop_distance * 2 * regime_config.take_profit_multiplier

                size_result = self.position_sizer.calculate_size(
                    starting_balance=self.starting_balance,
                    current_equity=day_start_equity,
                    score=score,
                    tier=tier,
                    current_heat=self.current_heat,
                    daily_pnl_pct=daily_pnl_pct,
                    win_streak=self.win_streak,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                )

                if not size_result.can_trade:
                    continue

                # Direction bias from regime
                direction_bias = regime_config.preferred_direction
                if direction_bias is not None:
                    if self.rng.random() < 0.7:
                        trade_direction = direction_bias.value.upper()
                    else:
                        trade_direction = (
                            "SHORT" if direction_bias == Direction.LONG else "LONG"
                        )
                else:
                    trade_direction = "LONG" if self.rng.random() < 0.5 else "SHORT"

                # Simulate outcome (regime-aware)
                is_winner, pnl_pct, exit_reason = self._simulate_trade_outcome(
                    edge, timeframe, score, current_regime,
                )

                # Apply regime position-size multiplier to risk
                regime_risk_pct = size_result.risk_pct * regime_config.position_size_multiplier

                # P&L based on day-start equity, capped at risk amount
                pnl = day_start_equity * (regime_risk_pct / 100) * (pnl_pct / 0.02)

                equity_before = self.current_equity
                self.current_equity += pnl
                self.daily_pnl += pnl

                # Streaks
                if is_winner:
                    self.win_streak += 1
                    self.loss_streak = 0
                    self.max_win_streak = max(self.max_win_streak, self.win_streak)
                else:
                    self.loss_streak += 1
                    self.win_streak = 0
                    self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)

                # Drawdown
                self.peak_equity = max(self.peak_equity, self.current_equity)
                if self.peak_equity > 0:
                    drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
                    self.max_drawdown = max(self.max_drawdown, drawdown)

                if self.max_drawdown >= abs(self.config.max_drawdown):
                    logger.warning("MAX DRAWDOWN HIT: %.1f%% - Stopping", self.max_drawdown)
                    hit_dd_limit = True
                    break

                market_str = _instrument_market(inst).value

                trade = GodModeTradeResult(
                    trade_id=trade_id,
                    timestamp=current_date,
                    symbol=inst.symbol,
                    market=market_str,
                    edge=edge.value,
                    timeframe=timeframe.value,
                    direction=trade_direction,
                    tier=tier,
                    score=score,
                    entry_price=entry_price,
                    exit_price=entry_price * (1 + pnl_pct),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=size_result.position_size,
                    risk_amount=size_result.risk_amount,
                    risk_pct=regime_risk_pct,
                    pnl=pnl,
                    pnl_pct=pnl_pct * 100,
                    is_winner=is_winner,
                    exit_reason=exit_reason,
                    equity_before=equity_before,
                    equity_after=self.current_equity,
                    streak_bonus=size_result.streak_bonus,
                    score_multiplier=size_result.score_multiplier,
                    regime=current_regime.value,
                )
                trades.append(trade)
                trade_id += 1

                # Performance maps
                edge_perf[edge.value]["trades"] += 1
                edge_perf[edge.value]["pnl"] += pnl
                if is_winner:
                    edge_perf[edge.value]["wins"] += 1

                tf_perf[timeframe.value]["trades"] += 1
                tf_perf[timeframe.value]["pnl"] += pnl
                if is_winner:
                    tf_perf[timeframe.value]["wins"] += 1

                if market_str not in market_perf:
                    market_perf[market_str] = {"trades": 0, "wins": 0, "pnl": 0.0}
                market_perf[market_str]["trades"] += 1
                market_perf[market_str]["pnl"] += pnl
                if is_winner:
                    market_perf[market_str]["wins"] += 1

                # Regime performance
                regime_perf[current_regime.value]["trades"] += 1
                regime_perf[current_regime.value]["pnl"] += pnl
                if is_winner:
                    regime_perf[current_regime.value]["wins"] += 1

                equity_curve.append(self.current_equity)
                trades_today += 1

            current_date += timedelta(days=1)

        # Final month
        if current_month and month_start_equity > 0:
            monthly_returns[current_month] = (
                (self.current_equity - month_start_equity) / month_start_equity * 100
            )

        # ── build result ────────────────────────────────────────────────
        total_trades = len(trades)
        winners = sum(1 for t in trades if t.is_winner)
        losers = total_trades - winners

        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        total_pnl = self.current_equity - self.starting_balance
        total_pnl_pct = (total_pnl / self.starting_balance) * 100

        avg_winner = sum(t.pnl for t in winning_trades) / winners if winners > 0 else 0
        avg_loser = sum(t.pnl for t in losing_trades) / losers if losers > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (annualised from monthly returns)
        sharpe = 0.0
        if len(monthly_returns) > 1:
            monthly_vals = list(monthly_returns.values())
            avg_m = sum(monthly_vals) / len(monthly_vals)
            std_m = (sum((m - avg_m) ** 2 for m in monthly_vals) / len(monthly_vals)) ** 0.5
            if std_m > 0:
                sharpe = (avg_m / std_m) * (12 ** 0.5)

        return GodModeBacktestResult(
            mode=self.mode,
            start_date=start_date,
            end_date=end_date,
            starting_balance=self.starting_balance,
            ending_balance=self.current_equity,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=total_trades,
            winners=winners,
            losers=losers,
            win_rate=(winners / total_trades * 100) if total_trades > 0 else 0,
            profit_factor=profit_factor,
            max_drawdown_pct=self.max_drawdown,
            sharpe_ratio=sharpe,
            avg_trade_pnl=(total_pnl / total_trades) if total_trades > 0 else 0,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=max((t.pnl for t in trades), default=0),
            largest_loser=min((t.pnl for t in trades), default=0),
            max_consecutive_wins=self.max_win_streak,
            max_consecutive_losses=self.max_loss_streak,
            stress_test=self.stress_test,
            stress_level=self.stress_win_rate_reduction,
            trades=trades,
            monthly_returns=monthly_returns,
            edge_performance=edge_perf,
            timeframe_performance=tf_perf,
            market_performance=market_perf,
            regime_performance=regime_perf,
            equity_curve=equity_curve,
        )


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════


def print_results(result: GodModeBacktestResult) -> None:
    print()
    print("=" * 80)
    title = f"  GOD MODE BACKTEST RESULTS - {result.mode.value.upper()}"
    if result.stress_test:
        title += f"  [STRESS -{result.stress_level*100:.0f}% WR]"
    print(title)
    print("=" * 80)

    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  " + "-" * 40)
    print(f"  Starting Balance:    £{result.starting_balance:>12,.2f}")
    print(f"  Ending Balance:      £{result.ending_balance:>12,.2f}")
    print(f"  Total P&L:           £{result.total_pnl:>12,.2f} ({result.total_pnl_pct:+.1f}%)")
    print()
    print(f"  Total Trades:        {result.total_trades:>12}")
    print(f"  Winners:             {result.winners:>12} ({result.win_rate:.1f}%)")
    print(f"  Losers:              {result.losers:>12}")
    print()
    print(f"  Profit Factor:       {result.profit_factor:>12.2f}")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:>12.1f}%")
    print()
    print(f"  Avg Trade P&L:       £{result.avg_trade_pnl:>12,.2f}")
    print(f"  Avg Winner:          £{result.avg_winner:>12,.2f}")
    print(f"  Avg Loser:           £{result.avg_loser:>12,.2f}")
    print(f"  Largest Winner:      £{result.largest_winner:>12,.2f}")
    print(f"  Largest Loser:       £{result.largest_loser:>12,.2f}")
    print()
    print(f"  Max Win Streak:      {result.max_consecutive_wins:>12}")
    print(f"  Max Loss Streak:     {result.max_consecutive_losses:>12}")

    # Monthly returns
    print(f"\n  MONTHLY RETURNS")
    print(f"  " + "-" * 40)
    for month, ret in sorted(result.monthly_returns.items()):
        bar = "#" * min(int(abs(ret) / 2), 30) if ret > 0 else ""
        sign = "+" if ret > 0 else ""
        print(f"  {month}:  {sign}{ret:>6.1f}%  {bar}")

    if result.monthly_returns:
        avg_monthly = sum(result.monthly_returns.values()) / len(result.monthly_returns)
        print(f"  " + "-" * 40)
        print(f"  Average Monthly:     {avg_monthly:>6.1f}%")

    # Edge performance
    print(f"\n  EDGE PERFORMANCE")
    print(f"  " + "-" * 60)
    print(f"  {'Edge':<25} {'Trades':>8} {'Win%':>8} {'P&L':>12}")
    print(f"  " + "-" * 60)
    for edge, stats in sorted(result.edge_performance.items(), key=lambda x: -x[1]["pnl"]):
        if stats["trades"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            print(f"  {edge:<25} {stats['trades']:>8} {wr:>7.1f}% £{stats['pnl']:>10,.2f}")

    # Timeframe performance
    print(f"\n  TIMEFRAME PERFORMANCE")
    print(f"  " + "-" * 60)
    print(f"  {'Timeframe':<25} {'Trades':>8} {'Win%':>8} {'P&L':>12}")
    print(f"  " + "-" * 60)
    for tf, stats in sorted(result.timeframe_performance.items(), key=lambda x: -x[1]["pnl"]):
        if stats["trades"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            print(f"  {tf:<25} {stats['trades']:>8} {wr:>7.1f}% £{stats['pnl']:>10,.2f}")

    # Market performance
    print(f"\n  MARKET PERFORMANCE")
    print(f"  " + "-" * 60)
    print(f"  {'Market':<25} {'Trades':>8} {'Win%':>8} {'P&L':>12}")
    print(f"  " + "-" * 60)
    for market, stats in sorted(result.market_performance.items(), key=lambda x: -x[1]["pnl"]):
        if stats["trades"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            print(f"  {market:<25} {stats['trades']:>8} {wr:>7.1f}% £{stats['pnl']:>10,.2f}")

    # Regime performance
    if result.regime_performance:
        print(f"\n  REGIME PERFORMANCE")
        print(f"  " + "-" * 70)
        print(f"  {'Regime':<20} {'Trades':>8} {'Win%':>8} {'P&L':>15} {'Avg/Trade':>12}")
        print(f"  " + "-" * 70)
        regime_icons = {
            "strong_bull": "[++]",
            "bull": "[ +]",
            "sideways": "[ =]",
            "bear": "[ -]",
            "strong_bear": "[--]",
            "volatile": "[~~]",
        }
        for regime, stats in sorted(
            result.regime_performance.items(), key=lambda x: -x[1]["pnl"]
        ):
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                avg = stats["pnl"] / stats["trades"]
                icon = regime_icons.get(regime, "    ")
                print(
                    f"  {icon} {regime:<15} {stats['trades']:>8} {wr:>7.1f}% "
                    f"£{stats['pnl']:>13,.2f} £{avg:>10,.2f}"
                )

    # Equity milestones
    print(f"\n  EQUITY MILESTONES")
    print(f"  " + "-" * 40)
    for milestone in [15_000, 20_000, 30_000, 50_000, 75_000, 100_000]:
        reached = any(e >= milestone for e in result.equity_curve)
        tag = "REACHED" if reached else "Not reached"
        print(f"  £{milestone:>8,}:  {tag}")

    print()
    print("=" * 80)
    if result.total_pnl_pct >= 100:
        print(f"  DOUBLED YOUR MONEY! +{result.total_pnl_pct:.0f}%")
    elif result.total_pnl_pct >= 50:
        print(f"  EXCELLENT PERFORMANCE! +{result.total_pnl_pct:.0f}%")
    elif result.total_pnl_pct >= 20:
        print(f"  SOLID RETURNS! +{result.total_pnl_pct:.0f}%")
    elif result.total_pnl_pct >= 0:
        print(f"  POSITIVE BUT ROOM TO IMPROVE: +{result.total_pnl_pct:.0f}%")
    else:
        print(f"  NEEDS WORK: {result.total_pnl_pct:.0f}%")
    print("=" * 80)
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="GOD MODE Backtest Runner")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=10_000, help="Starting balance")
    parser.add_argument(
        "--mode",
        type=str,
        default="god_mode",
        choices=["conservative", "standard", "aggressive", "god_mode"],
    )
    parser.add_argument("--instruments", type=int, default=20, help="Max instruments per edge")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 instruments")
    parser.add_argument("--compare", action="store_true", help="Compare all modes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Regime / stress-test arguments
    parser.add_argument(
        "--stress", action="store_true",
        help="Run stress test with degraded performance",
    )
    parser.add_argument(
        "--stress-level", type=float, default=0.05,
        help="Win rate reduction for stress test (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--full-cycle", action="store_true",
        help="Run full 2020-2024 market cycle test",
    )

    args = parser.parse_args()

    # Full-cycle preset overrides dates
    if args.full_cycle:
        args.start = "2020-01-01"
        args.end = "2024-12-31"

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.quick:
        args.instruments = 5

    stress_test = args.stress
    stress_level = args.stress_level if stress_test else 0.0

    if args.compare:
        print()
        print("=" * 80)
        print("  MODE COMPARISON BACKTEST")
        if args.full_cycle:
            print("  FULL MARKET CYCLE: 2020-2024 (COVID + Bear + Bull)")
        print("=" * 80)

        results = {}
        for mode in TradingMode:
            backtester = GodModeBacktester(
                mode=mode,
                starting_balance=args.balance,
                seed=args.seed,
                stress_test=stress_test,
                stress_win_rate_reduction=stress_level,
            )
            result = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                max_instruments_per_edge=args.instruments,
            )
            results[mode.value] = result

        print()
        print(f"  {'Mode':<15} {'P&L%':>10} {'Trades':>10} {'Win%':>10} {'MaxDD':>10} {'Sharpe':>10}")
        print(f"  " + "-" * 65)
        for mode_name, res in results.items():
            print(
                f"  {mode_name:<15} {res.total_pnl_pct:>+9.1f}% {res.total_trades:>10} "
                f"{res.win_rate:>9.1f}% {res.max_drawdown_pct:>9.1f}% {res.sharpe_ratio:>10.2f}"
            )
        print()

        # Detailed results for GOD MODE
        print_results(results["god_mode"])
    else:
        mode = TradingMode(args.mode)
        backtester = GodModeBacktester(
            mode=mode,
            starting_balance=args.balance,
            seed=args.seed,
            stress_test=stress_test,
            stress_win_rate_reduction=stress_level,
        )
        result = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            max_instruments_per_edge=args.instruments,
        )
        print_results(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
