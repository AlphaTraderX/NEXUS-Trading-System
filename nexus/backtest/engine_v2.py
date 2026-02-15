"""
Backtest Engine v2.1 — Enhanced with regime filtering, trailing stops, and heat management.

Wraps BacktestEngine v1 via composition, reusing its signal generation, indicator
computation, and data loading.  Adds:

- Regime detection per-bar (GodModeRegimeDetector)
- Edge filtering by regime (REGIME_CONFIGS.allowed_edges)
- Dynamic position sizing with regime + momentum multipliers
- Portfolio heat tracking with dynamic limits
- Trailing stop + breakeven exits (TrailingStopConfig)
- Win streak / momentum scaling

Usage::

    engine = BacktestEngineV2(starting_balance=10_000)
    result = await engine.run_edge_backtest(EdgeType.GAP_FILL, start_date="2022-01-01")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nexus.backtest.engine import BacktestEngine, BacktestResult
from nexus.backtest.statistics import StatisticsCalculator
from nexus.backtest.trade_simulator import (
    ExitReason,
    SimulatedTrade,
    TradeSimulator,
    TrailingStopConfig,
    score_to_tier,
    tier_multiplier,
)
from nexus.core.enums import EdgeType
from nexus.intelligence.regime_detector import (
    REGIME_CONFIGS,
    GodModeRegime,
    GodModeRegimeDetector,
    RegimeConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-edge trailing stop configuration
# ---------------------------------------------------------------------------

EDGE_TRAILING_CONFIG: Dict[EdgeType, Optional[TrailingStopConfig]] = {
    EdgeType.GAP_FILL: TrailingStopConfig(
        atr_trail_multiplier=1.5,
        breakeven_atr_mult=1.0,
        trailing_activation_atr=1.5,
    ),
    EdgeType.INSIDER_CLUSTER: TrailingStopConfig(
        atr_trail_multiplier=2.0,
        breakeven_atr_mult=1.5,
        trailing_activation_atr=2.0,
    ),
    EdgeType.VWAP_DEVIATION: TrailingStopConfig(
        atr_trail_multiplier=1.0,
        breakeven_atr_mult=0.75,
        trailing_activation_atr=1.0,
    ),
    EdgeType.RSI_EXTREME: None,           # Uses indicator exit, not trailing
    EdgeType.OVERNIGHT_PREMIUM: None,     # 1-bar hold, no trailing
    EdgeType.TURN_OF_MONTH: None,         # Calendar edge, fixed stop/target
    EdgeType.MONTH_END: None,             # Calendar edge, fixed stop/target
}


# ---------------------------------------------------------------------------
# Lightweight heat tracker for backtesting (no threading, no settings deps)
# ---------------------------------------------------------------------------

class BacktestHeatTracker:
    """Track portfolio heat during backtesting.

    Heat = sum of all open position risks as % of equity.
    Limits expand when profitable, contract when losing (same logic as
    nexus.risk.heat_manager.DynamicHeatManager.get_heat_limit).
    """

    BASE_LIMIT = 25.0
    MAX_LIMIT = 35.0
    MIN_LIMIT = 15.0

    def __init__(self) -> None:
        self.current_heat: float = 0.0
        self._positions: Dict[str, float] = {}  # id -> risk_pct

    def get_heat_limit(self, daily_pnl_pct: float) -> float:
        """Dynamic heat limit based on daily P&L."""
        if daily_pnl_pct >= 2.0:
            limit = self.MAX_LIMIT
        elif daily_pnl_pct >= 1.0:
            limit = self.BASE_LIMIT + 5.0
        elif daily_pnl_pct >= 0:
            limit = self.BASE_LIMIT
        elif daily_pnl_pct >= -1.0:
            limit = self.BASE_LIMIT - 5.0
        else:
            limit = self.MIN_LIMIT
        return max(self.MIN_LIMIT, min(limit, self.MAX_LIMIT))

    def can_add(self, risk_pct: float, daily_pnl_pct: float = 0.0) -> bool:
        """Check if adding a position would exceed heat limit."""
        limit = self.get_heat_limit(daily_pnl_pct)
        return (self.current_heat + risk_pct) <= limit

    def add_position(self, position_id: str, risk_pct: float) -> None:
        self._positions[position_id] = risk_pct
        self.current_heat = sum(self._positions.values())

    def remove_position(self, position_id: str) -> None:
        self._positions.pop(position_id, None)
        self.current_heat = sum(self._positions.values())

    def reset(self) -> None:
        self._positions.clear()
        self.current_heat = 0.0


# ---------------------------------------------------------------------------
# Engine v2.1
# ---------------------------------------------------------------------------

class BacktestEngineV2:
    """Enhanced backtest engine with regime, heat, and trailing stop integration.

    Delegates signal generation and data loading to BacktestEngine v1.
    Adds regime-aware filtering, dynamic sizing, heat tracking, and trailing stops.
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        risk_per_trade: float = 1.0,
        use_score_sizing: bool = True,
        use_registry: bool = False,
        # v2.1 feature toggles (for A/B testing)
        use_regime_filter: bool = True,
        use_trailing_stops: bool = True,
        use_heat_management: bool = True,
        use_momentum_scaling: bool = True,
    ):
        self.starting_balance = starting_balance
        self.risk_per_trade = risk_per_trade
        self.use_score_sizing = use_score_sizing

        # Feature toggles
        self.use_regime_filter = use_regime_filter
        self.use_trailing_stops = use_trailing_stops
        self.use_heat_management = use_heat_management
        self.use_momentum_scaling = use_momentum_scaling

        # Delegate to v1 for signal generation and data loading
        self.v1 = BacktestEngine(
            starting_balance=starting_balance,
            risk_per_trade=risk_per_trade,
            use_score_sizing=use_score_sizing,
            use_registry=use_registry,
        )

        # v2.1 components
        self.regime_detector = GodModeRegimeDetector()
        self.heat_tracker = BacktestHeatTracker()
        self.stats_calc = StatisticsCalculator()

        # Momentum tracking
        self._win_streak: int = 0
        self._daily_pnl: float = 0.0
        self._daily_start_balance: float = starting_balance

        # Regime stats for reporting
        self._regime_signal_counts: Dict[str, int] = {}
        self._regime_filtered_counts: Dict[str, int] = {}

    # Expose v1 constants for backtest_v2 script
    @property
    def SCANNER_MAP(self):
        return self.v1.SCANNER_MAP

    @property
    def DISABLED_EDGES(self):
        return self.v1.DISABLED_EDGES

    def _reset_for_edge(self) -> None:
        """Reset state for a new edge backtest."""
        self.v1._reset_balance()
        self.heat_tracker.reset()
        self._win_streak = 0
        self._daily_pnl = 0.0
        self._daily_start_balance = self.starting_balance
        self._regime_signal_counts = {}
        self._regime_filtered_counts = {}

    def _get_momentum_multiplier(self) -> float:
        """Scale position size based on consecutive wins."""
        if not self.use_momentum_scaling or self._win_streak < 2:
            return 1.0
        return min(1.0 + (self._win_streak * 0.1), 1.3)

    def _detect_regime(self, bars: pd.DataFrame) -> GodModeRegime:
        """Detect regime from price bars (needs 200+ bars)."""
        if len(bars) < 200:
            return GodModeRegime.SIDEWAYS
        prices = bars["close"].values
        return self.regime_detector.detect_regime(prices)

    def _get_regime_config(self, regime: GodModeRegime) -> RegimeConfig:
        """Get configuration for current regime."""
        return REGIME_CONFIGS.get(regime, REGIME_CONFIGS[GodModeRegime.SIDEWAYS])

    def _compound(self, net_pnl: float) -> None:
        """Update balance after a trade (same logic as v1)."""
        if self.use_score_sizing:
            self.v1.simulator.account_balance = max(
                0, self.v1.simulator.account_balance + net_pnl
            )

    def _update_streak(self, net_pnl: float) -> None:
        """Update win streak tracker."""
        if net_pnl > 0:
            self._win_streak += 1
        else:
            self._win_streak = 0

    def _update_daily_pnl(self, net_pnl: float) -> None:
        """Track daily P&L for heat limit calculation."""
        self._daily_pnl += net_pnl

    @property
    def _daily_pnl_pct(self) -> float:
        if self._daily_start_balance <= 0:
            return 0.0
        return (self._daily_pnl / self._daily_start_balance) * 100

    # ------------------------------------------------------------------
    # Main backtest method
    # ------------------------------------------------------------------

    async def run_edge_backtest(
        self,
        edge_type: EdgeType,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
    ) -> BacktestResult:
        """Run v2.1 backtest for a single edge (with regime/heat/trailing)."""
        self._reset_for_edge()

        timeframe = timeframe or self.v1.EDGE_TIMEFRAMES.get(edge_type, "1d")

        # Skip disabled edges (delegate to v1)
        if edge_type in self.v1.DISABLED_EDGES:
            return await self.v1.run_edge_backtest(
                edge_type, symbol, timeframe, start_date, end_date,
            )

        # Resolve instruments
        edge_instruments = self.v1.get_instruments_for_edge(
            edge_type, timeframe, start_date, end_date,
        )
        use_multi = (
            symbol is None
            and edge_type in self.v1.MULTI_SYMBOL_EDGES
            and len(edge_instruments) > 1
        )
        symbol_resolved = symbol or edge_instruments[0]

        if use_multi:
            return await self._run_multi_symbol(
                edge_type, timeframe, start_date, end_date,
            )
        return await self._run_single_symbol(
            edge_type, symbol_resolved, timeframe, start_date, end_date,
        )

    async def _run_single_symbol(
        self,
        edge_type: EdgeType,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """v2.1 single-symbol backtest with regime + heat + trailing."""
        self.v1._current_timeframe = timeframe
        self.v1.simulator.max_hold_bars = self.v1.EDGE_MAX_HOLD_BARS.get(edge_type, 20)
        self.v1._last_signal_bar = {}
        self.v1._signals_today = {}

        # Load and prepare data (delegate to v1)
        bars = await self.v1.data_loader.load_bars(
            symbol=symbol, timeframe=timeframe,
            start_date=start_date, end_date=end_date,
        )
        if bars.empty:
            raise ValueError(f"No data for {symbol} {timeframe} {start_date}-{end_date}")

        bars = self._prepare_indicators(edge_type, bars)

        # Scan for signals (reuse v1's signal generation)
        opportunities = self.v1._scan_historical(edge_type, symbol, bars)

        # Apply forex cost profile if needed
        _saved_costs = self._apply_forex_costs(edge_type)

        # Get trailing config for this edge
        trailing_config = EDGE_TRAILING_CONFIG.get(edge_type) if self.use_trailing_stops else None

        # Get exit checker from v1
        exit_checker = self._get_exit_checker(edge_type)

        # Execute trades with v2.1 enhancements
        if edge_type == EdgeType.OVERNIGHT_PREMIUM:
            trades = self.v1._simulate_overnight_trades(opportunities, bars)
            for t in trades:
                self._compound(t.net_pnl)
                self._update_streak(t.net_pnl)
                self._update_daily_pnl(t.net_pnl)
        else:
            trades = self._execute_trades_v2(
                opportunities, bars, edge_type, trailing_config, exit_checker,
            )

        # Restore costs
        self._restore_costs(_saved_costs)

        # Calculate stats
        stats = self.stats_calc.calculate(
            trades=trades,
            edge_type=edge_type.value,
            symbol=symbol,
            timeframe=timeframe,
            test_period=f"{start_date} to {end_date}",
            starting_balance=self.starting_balance,
            risk_profile=self.v1._get_risk_profile(edge_type),
        )

        return BacktestResult(
            statistics=stats,
            trades=trades,
            equity_curve=self.v1._build_equity_curve(trades),
            parameters={
                "edge_type": edge_type.value,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "starting_balance": self.starting_balance,
                "risk_per_trade": self.risk_per_trade,
                "v2_features": {
                    "regime_filter": self.use_regime_filter,
                    "trailing_stops": self.use_trailing_stops,
                    "heat_management": self.use_heat_management,
                    "momentum_scaling": self.use_momentum_scaling,
                },
                "regime_signals": dict(self._regime_signal_counts),
                "regime_filtered": dict(self._regime_filtered_counts),
            },
        )

    async def _run_multi_symbol(
        self,
        edge_type: EdgeType,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """v2.1 multi-symbol backtest with regime + heat + trailing."""
        symbols = self.v1.get_instruments_for_edge(edge_type, timeframe, start_date, end_date)
        self.v1._current_timeframe = timeframe
        self.v1.simulator.max_hold_bars = self.v1.EDGE_MAX_HOLD_BARS.get(edge_type, 20)

        exit_checker = self._get_exit_checker(edge_type)
        trailing_config = EDGE_TRAILING_CONFIG.get(edge_type) if self.use_trailing_stops else None

        # Collect all signals across symbols
        all_candidates: List[Tuple] = []  # (opp, bar_idx, bars_df)

        for sym in symbols:
            self.v1._last_signal_bar = {}
            self.v1._signals_today = {}

            try:
                bars = await self.v1.data_loader.load_bars(
                    symbol=sym, timeframe=timeframe,
                    start_date=start_date, end_date=end_date,
                )
            except Exception as e:
                logger.warning("Failed to load %s: %s", sym, e)
                continue

            min_data = self._get_min_bars(edge_type)
            if bars.empty or len(bars) < min_data:
                continue

            bars = self._prepare_indicators(edge_type, bars)
            opps = self.v1._scan_historical(edge_type, sym, bars)
            for opp, idx in opps:
                all_candidates.append((opp, idx, bars))
            logger.info("%s %s: %d signals found", edge_type.value, sym, len(opps))

        if not all_candidates:
            symbol_str = "+".join(symbols)
            stats = self.stats_calc.calculate(
                trades=[], edge_type=edge_type.value, symbol=symbol_str,
                timeframe=timeframe, test_period=f"{start_date} to {end_date}",
                starting_balance=self.starting_balance,
                risk_profile=self.v1._get_risk_profile(edge_type),
            )
            return BacktestResult(
                statistics=stats, trades=[], equity_curve=pd.DataFrame(),
                parameters={"edge_type": edge_type.value, "symbol": symbol_str},
            )

        # Sort by signal time for correct compounding order
        all_candidates.sort(key=lambda c: c[2].index[c[1]])

        _saved_costs = self._apply_forex_costs(edge_type)

        # Execute trades
        if edge_type == EdgeType.OVERNIGHT_PREMIUM:
            trades: List[SimulatedTrade] = []
            for opp, bar_idx, sym_bars in all_candidates:
                overnight = self.v1._simulate_overnight_trades(
                    [(opp, bar_idx)], sym_bars,
                )
                trades.extend(overnight)
                for t in overnight:
                    self._compound(t.net_pnl)
                    self._update_streak(t.net_pnl)
                    self._update_daily_pnl(t.net_pnl)
        else:
            trades = self._execute_trades_v2(
                all_candidates, None, edge_type, trailing_config, exit_checker,
                multi_symbol=True,
            )

        self._restore_costs(_saved_costs)

        trades.sort(key=lambda t: t.entry_time)
        symbol_str = "+".join(symbols)

        stats = self.stats_calc.calculate(
            trades=trades,
            edge_type=edge_type.value,
            symbol=symbol_str,
            timeframe=timeframe,
            test_period=f"{start_date} to {end_date}",
            starting_balance=self.starting_balance,
            risk_profile=self.v1._get_risk_profile(edge_type),
        )

        return BacktestResult(
            statistics=stats,
            trades=trades,
            equity_curve=self.v1._build_equity_curve(trades),
            parameters={
                "edge_type": edge_type.value,
                "symbol": symbol_str,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "starting_balance": self.starting_balance,
                "v2_features": {
                    "regime_filter": self.use_regime_filter,
                    "trailing_stops": self.use_trailing_stops,
                    "heat_management": self.use_heat_management,
                    "momentum_scaling": self.use_momentum_scaling,
                },
                "regime_signals": dict(self._regime_signal_counts),
                "regime_filtered": dict(self._regime_filtered_counts),
            },
        )

    # ------------------------------------------------------------------
    # Core v2.1 trade execution
    # ------------------------------------------------------------------

    def _execute_trades_v2(
        self,
        opportunities,
        bars: Optional[pd.DataFrame],
        edge_type: EdgeType,
        trailing_config: Optional[TrailingStopConfig],
        exit_checker,
        multi_symbol: bool = False,
    ) -> List[SimulatedTrade]:
        """Execute trades with v2.1 enhancements (regime, heat, trailing, momentum)."""
        trades: List[SimulatedTrade] = []

        for item in opportunities:
            if multi_symbol:
                opp, bar_idx, sym_bars = item
            else:
                opp, bar_idx = item
                sym_bars = bars

            # --- Score-based multiplier (v1 logic) ---
            score_mult = self.v1._get_risk_multiplier(opp.raw_score)
            if score_mult == 0.0:
                continue  # F-tier: skip

            # --- Regime filter ---
            if self.use_regime_filter:
                regime = self._detect_regime(sym_bars.iloc[:bar_idx + 1])
                regime_config = self._get_regime_config(regime)
                regime_name = regime.value

                self._regime_signal_counts[regime_name] = (
                    self._regime_signal_counts.get(regime_name, 0) + 1
                )

                if edge_type not in regime_config.allowed_edges:
                    self._regime_filtered_counts[regime_name] = (
                        self._regime_filtered_counts.get(regime_name, 0) + 1
                    )
                    continue

                regime_size_mult = regime_config.position_size_multiplier
            else:
                regime_size_mult = 1.0

            # --- Momentum multiplier ---
            momentum_mult = self._get_momentum_multiplier()

            # --- Combined risk multiplier ---
            combined_mult = score_mult * regime_size_mult * momentum_mult

            # --- Heat check ---
            risk_pct = self.risk_per_trade * combined_mult
            if self.use_heat_management:
                if not self.heat_tracker.can_add(risk_pct, self._daily_pnl_pct):
                    continue  # Over heat limit

            # --- Calculate ATR for trailing stops ---
            atr = None
            if trailing_config is not None:
                atr = self.v1._calculate_atr(sym_bars.iloc[:bar_idx + 1])

            # --- Simulate trade ---
            trade = self.v1.simulator.simulate_trade(
                opp, sym_bars, bar_idx,
                exit_checker=exit_checker,
                risk_multiplier=combined_mult,
                trailing_config=trailing_config,
                atr=atr,
            )

            if trade:
                trades.append(trade)
                self._compound(trade.net_pnl)
                self._update_streak(trade.net_pnl)
                self._update_daily_pnl(trade.net_pnl)

                # Heat: add then remove (single-bar resolution for backtest)
                if self.use_heat_management:
                    self.heat_tracker.add_position(trade.opportunity_id, risk_pct)
                    self.heat_tracker.remove_position(trade.opportunity_id)

        return trades

    # ------------------------------------------------------------------
    # Helpers (delegate to v1 where possible)
    # ------------------------------------------------------------------

    def _prepare_indicators(self, edge_type: EdgeType, bars: pd.DataFrame) -> pd.DataFrame:
        """Prepare indicators (delegate to v1's methods)."""
        if edge_type == EdgeType.RSI_EXTREME:
            return self.v1._prepare_rsi_indicators(bars)
        elif edge_type == EdgeType.ORB:
            return self.v1._prepare_orb_indicators(bars)
        elif edge_type in self.v1.FOREX_EDGES:
            return self.v1._prepare_forex_indicators(bars)
        return bars

    def _get_exit_checker(self, edge_type: EdgeType):
        """Get edge-specific exit checker."""
        if edge_type == EdgeType.RSI_EXTREME:
            return self.v1._rsi_exit_checker
        elif edge_type == EdgeType.ORB:
            return self.v1._orb_exit_checker
        return None

    def _get_min_bars(self, edge_type: EdgeType) -> int:
        if edge_type == EdgeType.RSI_EXTREME:
            return 200
        elif edge_type == EdgeType.OVERNIGHT_PREMIUM:
            return 201
        elif edge_type == EdgeType.ORB:
            return 400
        return 50

    def _apply_forex_costs(self, edge_type: EdgeType):
        """Apply forex cost profile, return saved costs or None."""
        if edge_type in self.v1.FOREX_EDGES:
            saved = (
                self.v1.simulator.spread_pct,
                self.v1.simulator.slippage_pct,
                self.v1.simulator.commission_per_trade,
            )
            self.v1.simulator.spread_pct = self.v1.FOREX_COST_PROFILE["spread_pct"]
            self.v1.simulator.slippage_pct = self.v1.FOREX_COST_PROFILE["slippage_pct"]
            self.v1.simulator.commission_per_trade = self.v1.FOREX_COST_PROFILE["commission_per_trade"]
            return saved
        return None

    def _restore_costs(self, saved):
        if saved is not None:
            (
                self.v1.simulator.spread_pct,
                self.v1.simulator.slippage_pct,
                self.v1.simulator.commission_per_trade,
            ) = saved
