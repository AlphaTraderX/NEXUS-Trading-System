"""
Main backtest engine that ties everything together.

Usage::

    engine = BacktestEngine()
    result = await engine.run_edge_backtest(
        edge_type=EdgeType.VWAP_DEVIATION,
        symbol="SPY",
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(result.statistics.verdict)
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nexus.core.enums import Direction, EdgeType, Market
from nexus.core.models import Opportunity

from nexus.backtest.data_loader import BacktestDataLoader
from nexus.backtest.trade_simulator import (
    ExitReason, SimulatedTrade, TradeSimulator, score_to_tier, tier_multiplier,
)
from nexus.backtest.statistics import BacktestStatistics, StatisticsCalculator
from nexus.data.instruments import (
    DataProvider, InstrumentType, get_instrument_registry,
)
from nexus.data.cache import get_cache_manager

# Scanner classes – imported for the SCANNER_MAP reference only;
# actual backtesting uses inline signal logic in _check_bar_for_signal.
from nexus.scanners.vwap import VWAPScanner
from nexus.scanners.rsi import RSIScanner
from nexus.scanners.calendar import TurnOfMonthScanner, MonthEndScanner
from nexus.scanners.gap import GapScanner
from nexus.scanners.session import LondonOpenScanner, NYOpenScanner, PowerHourScanner
from nexus.scanners.orb import ORBScanner
from nexus.scanners.bollinger import BollingerScanner

logger = logging.getLogger(__name__)

# Forex major pairs for market detection
_FOREX_MAJORS = {"EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"}
_FOREX_CROSSES = {"EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/NZD"}


def _market_for_symbol(symbol: str) -> Market:
    """Infer market enum from symbol string."""
    if symbol in _FOREX_MAJORS:
        return Market.FOREX_MAJORS
    if symbol in _FOREX_CROSSES:
        return Market.FOREX_CROSSES
    return Market.US_STOCKS


@dataclass
class BacktestResult:
    """Complete backtest result."""

    statistics: BacktestStatistics
    trades: List[SimulatedTrade]
    equity_curve: pd.DataFrame
    parameters: Dict[str, Any]


class BacktestEngine:
    """
    Main engine for running backtests.

    Connects:
    - Data loader (historical prices)
    - Scanners (signal generation)
    - Trade simulator (execution simulation)
    - Statistics calculator (analysis)
    """

    # Edges validated as unprofitable — do not trade
    DISABLED_EDGES: set = {
        EdgeType.POWER_HOUR,      # v15: daily close-position proxy -2% P&L, PF 0.94 — no edge
        EdgeType.BOLLINGER_TOUCH, # -9.4% unfiltered; +2.4% with ADX<25 but PF 1.03 — still no edge
        EdgeType.EARNINGS_DRIFT,  # No data available
        # 5m intraday edges — failing
        # GAP_FILL removed v13 — reversed to Gap and Go (momentum continuation)
        EdgeType.ORB,             # INVALID on SPY — needs individual stock testing
        # VWAP_DEVIATION removed v12 — reversed to trend-following per Zarattini & Aziz 2023
        # FOREX EDGES PARKED v26:
        # - london_open: EUR/USD fade PF 1.30, but only 66 trades, +$15 total
        # - asian_range: AUD/USD fade PF 1.65, but only 16 trades
        # - ny_open: PF 0.62-0.84 — no edge either direction
        # Focus on stock edges that generate 99.8% of returns
        EdgeType.LONDON_OPEN,     # v26: Parked — MARGINAL, only +$15 contribution
        EdgeType.NY_OPEN,         # v26: No edge in either direction
        EdgeType.ASIAN_RANGE,     # v26: Parked — MARGINAL, insufficient trades
    }

    # ATR multipliers for stop/target by timeframe
    # Tighter on fast timeframes to avoid constant time-expiry exits
    TIMEFRAME_ATR_MULTIPLIERS: Dict[str, Tuple[float, float]] = {
        "1m":  (0.5, 1.0),
        "5m":  (0.75, 1.5),
        "15m": (1.0, 2.0),
        "1h":  (1.25, 2.5),
        "4h":  (1.5, 3.0),
        "1d":  (1.5, 2.5),
    }

    # Max hold bars per edge (reduces time-expiry exits)
    EDGE_MAX_HOLD_BARS: Dict[EdgeType, int] = {
        EdgeType.VWAP_DEVIATION: 10,   # 10 days on daily (trend-following)
        EdgeType.RSI_EXTREME: 10,      # 10 days on daily
        EdgeType.TURN_OF_MONTH: 4,     # 4 days (TOM window)
        EdgeType.MONTH_END: 3,         # 3 days
        EdgeType.GAP_FILL: 5,          # 5 days for momentum continuation
        EdgeType.POWER_HOUR: 3,        # 3 days momentum continuation
        EdgeType.ORB: 100,             # Full day on 5m (EOD exit handles it)
        EdgeType.ASIAN_RANGE: 8,       # 8 hours on 1h
        EdgeType.LONDON_OPEN: 16,     # 4 hours on 15m (full London session)
        EdgeType.NY_OPEN: 16,         # 4 hours on 15m
        EdgeType.INSIDER_CLUSTER: 20,  # Swing trade
        EdgeType.OVERNIGHT_PREMIUM: 1, # 1 bar: close to next open
    }

    # Scanner mapping (edge -> live scanner class)
    SCANNER_MAP = {
        EdgeType.VWAP_DEVIATION: VWAPScanner,
        EdgeType.RSI_EXTREME: RSIScanner,
        EdgeType.TURN_OF_MONTH: TurnOfMonthScanner,
        EdgeType.MONTH_END: MonthEndScanner,
        EdgeType.GAP_FILL: GapScanner,
        EdgeType.LONDON_OPEN: LondonOpenScanner,
        EdgeType.NY_OPEN: NYOpenScanner,
        EdgeType.POWER_HOUR: PowerHourScanner,
        EdgeType.ORB: ORBScanner,
        EdgeType.BOLLINGER_TOUCH: BollingerScanner,
        EdgeType.INSIDER_CLUSTER: None,  # Simulated in backtest
        EdgeType.ASIAN_RANGE: None,      # Implemented inline
        EdgeType.EARNINGS_DRIFT: None,   # Requires external data
        EdgeType.OVERNIGHT_PREMIUM: None,  # Implemented inline
    }

    # Default instruments per edge
    EDGE_INSTRUMENTS = {
        EdgeType.VWAP_DEVIATION: ["SPY", "QQQ", "IWM"],
        EdgeType.RSI_EXTREME: ["SPY", "QQQ"],  # Only index ETFs validated (IWM/DIA INVALID)
        EdgeType.TURN_OF_MONTH: ["SPY", "QQQ", "IWM"],  # TOM effect is market-wide
        EdgeType.MONTH_END: ["SPY", "QQQ"],  # +QQQ to reach 30+ trade threshold
        EdgeType.GAP_FILL: ["SPY", "NVDA", "TSLA", "AAPL", "AMD", "COIN", "ROKU", "SHOP", "SQ", "MARA"],  # 10 validated
        EdgeType.LONDON_OPEN: ["EUR/USD"],  # Only EUR/USD fades (others trend or neutral)
        EdgeType.NY_OPEN: ["GBP/USD", "USD/JPY"],
        EdgeType.POWER_HOUR: ["SPY", "QQQ"],
        EdgeType.ORB: ["SPY"],
        EdgeType.BOLLINGER_TOUCH: ["SPY", "EUR/USD"],
        EdgeType.INSIDER_CLUSTER: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"],  # More signals
        EdgeType.ASIAN_RANGE: ["USD/JPY", "AUD/USD"],  # Both fade; AUD/USD flipped 0.26→1.65
        EdgeType.EARNINGS_DRIFT: ["SPY"],
        EdgeType.OVERNIGHT_PREMIUM: ["SPY", "QQQ", "TSLA", "NVDA", "AMD", "AAPL", "GOOGL", "META", "NFLX", "CRM"],
        # v26 expansion tested: SMCI(PF 1.35), PLTR(PF 1.34), XLK(PF 1.30) pass individually
        # but combined MaxDD 22-23% exceeds 16% limit (correlated tech drawdowns with compounding)
    }

    # Registry filters: map each edge to (provider, instrument_type, max_instruments)
    # Used when use_registry=True to query InstrumentRegistry instead of EDGE_INSTRUMENTS.
    # Only Polygon (US stocks) and OANDA (forex) have BacktestDataLoader support.
    EDGE_REGISTRY_FILTERS = {
        # Stock edges — Polygon US
        EdgeType.GAP_FILL: (DataProvider.POLYGON, InstrumentType.STOCK, 30),
        EdgeType.OVERNIGHT_PREMIUM: (DataProvider.POLYGON, InstrumentType.STOCK, 30),
        EdgeType.INSIDER_CLUSTER: (DataProvider.POLYGON, InstrumentType.STOCK, 20),
        EdgeType.VWAP_DEVIATION: (DataProvider.POLYGON, InstrumentType.STOCK, 20),
        EdgeType.RSI_EXTREME: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        EdgeType.TURN_OF_MONTH: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        EdgeType.MONTH_END: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        EdgeType.POWER_HOUR: (DataProvider.POLYGON, InstrumentType.STOCK, 20),
        EdgeType.ORB: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        EdgeType.BOLLINGER_TOUCH: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        EdgeType.EARNINGS_DRIFT: (DataProvider.POLYGON, InstrumentType.STOCK, 10),
        # Forex edges — OANDA
        EdgeType.LONDON_OPEN: (DataProvider.OANDA, InstrumentType.FOREX, 10),
        EdgeType.NY_OPEN: (DataProvider.OANDA, InstrumentType.FOREX, 10),
        EdgeType.ASIAN_RANGE: (DataProvider.OANDA, InstrumentType.FOREX, 10),
    }

    # Edge-specific risk profiles — replace single 15% MaxDD with per-edge thresholds
    # Each profile can include p_value_threshold and min_win_rate overrides.
    EDGE_RISK_PROFILES = {
        # High-risk edges — relaxed p-value (0.10) and higher MaxDD tolerance
        EdgeType.GAP_FILL: {
            "max_dd_pct": 22.0,
            "min_profit_factor": 1.25,
            "min_trades": 30,
            "risk_category": "high",
            "p_value_threshold": 0.10,
        },
        EdgeType.INSIDER_CLUSTER: {
            "max_dd_pct": 28.0,
            "min_profit_factor": 1.2,
            "min_trades": 30,
            "risk_category": "high",
            "p_value_threshold": 0.10,
        },
        EdgeType.EARNINGS_DRIFT: {
            "max_dd_pct": 22.0,
            "min_profit_factor": 1.25,
            "min_trades": 30,
            "risk_category": "high",
            "p_value_threshold": 0.10,
        },
        # Medium-risk edges — slight MaxDD relaxation
        EdgeType.OVERNIGHT_PREMIUM: {
            "max_dd_pct": 16.0,
            "min_profit_factor": 1.2,
            "min_trades": 100,
            "risk_category": "medium",
            "p_value_threshold": 0.05,
        },
        EdgeType.VWAP_DEVIATION: {
            "max_dd_pct": 16.0,
            "min_profit_factor": 1.15,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.15,
            "min_win_rate": 0.38,
        },
        EdgeType.TURN_OF_MONTH: {
            "max_dd_pct": 15.0,
            "min_profit_factor": 1.05,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.15,
        },
        EdgeType.MONTH_END: {
            "max_dd_pct": 15.0,
            "min_profit_factor": 1.05,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.15,
        },
        # Mean reversion edges — should be stable
        EdgeType.RSI_EXTREME: {
            "max_dd_pct": 12.0,
            "min_profit_factor": 1.5,
            "min_trades": 30,
            "risk_category": "low",
            "p_value_threshold": 0.05,
        },
        EdgeType.BOLLINGER_TOUCH: {
            "max_dd_pct": 12.0,
            "min_profit_factor": 1.3,
            "min_trades": 30,
            "risk_category": "low",
            "p_value_threshold": 0.05,
        },
        # Forex session edges — relaxed p-value (0.10), higher MaxDD tolerance
        EdgeType.LONDON_OPEN: {
            "max_dd_pct": 18.0,
            "min_profit_factor": 1.15,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.10,
            "min_win_rate": 0.45,
        },
        EdgeType.NY_OPEN: {
            "max_dd_pct": 18.0,
            "min_profit_factor": 1.15,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.10,
            "min_win_rate": 0.45,
        },
        EdgeType.POWER_HOUR: {
            "max_dd_pct": 15.0,
            "min_profit_factor": 1.1,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.05,
        },
        EdgeType.ASIAN_RANGE: {
            "max_dd_pct": 18.0,
            "min_profit_factor": 1.15,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.10,
            "min_win_rate": 0.45,
        },
        EdgeType.ORB: {
            "max_dd_pct": 18.0,
            "min_profit_factor": 1.2,
            "min_trades": 30,
            "risk_category": "medium",
            "p_value_threshold": 0.05,
        },
    }

    DEFAULT_RISK_PROFILE = {
        "max_dd_pct": 15.0,
        "min_profit_factor": 1.2,
        "min_trades": 30,
        "risk_category": "medium",
        "p_value_threshold": 0.05,
        "min_win_rate": 0.45,
    }

    # Portfolio-level max drawdown across all edges combined
    PORTFOLIO_MAX_DD_PCT = 30.0

    # Edges that run across multiple symbols (all signals taken for diversification)
    MULTI_SYMBOL_EDGES: set = {
        EdgeType.RSI_EXTREME,
        EdgeType.OVERNIGHT_PREMIUM,
        EdgeType.GAP_FILL,
        EdgeType.POWER_HOUR,
        EdgeType.TURN_OF_MONTH,
        EdgeType.INSIDER_CLUSTER,
        EdgeType.VWAP_DEVIATION,
        EdgeType.MONTH_END,
        EdgeType.LONDON_OPEN,
        EdgeType.ASIAN_RANGE,
        EdgeType.NY_OPEN,
    }

    # Per-pair forex strategy: EUR/USD fades breakouts, others follow with retest
    FOREX_STRATEGY = {
        "EUR/USD": "fade",      # v25: PF 1.30 at London open
        "GBP/USD": "retest",    # fade PF 0.62 at NY, retest PF 0.91 — retest wins
        "USD/JPY": "fade",      # fade PF 0.97 at Asian (vs retest 0.91)
        "AUD/USD": "fade",      # fade PF 1.65 at Asian (vs retest 0.26!) — big flip
        "EUR/GBP": "retest",    # EUR/GBP trends at London (fade PF 0.49)
        "USD/CHF": "fade",      # Inverse EUR/USD — likely mean-reverts
    }

    # Forex cost profile (tighter than stocks: no commissions, tighter spreads)
    FOREX_COST_PROFILE = {
        "spread_pct": 0.015,       # 1.5 bps
        "slippage_pct": 0.01,      # 1 bp per side
        "commission_per_trade": 0.0,
    }

    # Edges that trade forex instruments (use tighter cost profile)
    FOREX_EDGES: set = {
        EdgeType.LONDON_OPEN,
        EdgeType.ASIAN_RANGE,
        EdgeType.NY_OPEN,
    }

    # Natural timeframe for each edge
    EDGE_TIMEFRAMES = {
        EdgeType.VWAP_DEVIATION: "1d",
        EdgeType.RSI_EXTREME: "1d",
        EdgeType.TURN_OF_MONTH: "1d",
        EdgeType.MONTH_END: "1d",
        EdgeType.GAP_FILL: "1d",
        EdgeType.LONDON_OPEN: "15m",
        EdgeType.NY_OPEN: "15m",
        EdgeType.POWER_HOUR: "1d",
        EdgeType.ORB: "5m",
        EdgeType.BOLLINGER_TOUCH: "1h",
        EdgeType.INSIDER_CLUSTER: "1d",
        EdgeType.ASIAN_RANGE: "1h",
        EdgeType.EARNINGS_DRIFT: "1d",
        EdgeType.OVERNIGHT_PREMIUM: "1d",
    }

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        risk_per_trade: float = 1.0,
        use_score_sizing: bool = False,
        use_registry: bool = False,
    ):
        self.starting_balance = starting_balance
        self.risk_per_trade = risk_per_trade
        self.use_score_sizing = use_score_sizing
        self.use_registry = use_registry

        self.data_loader = BacktestDataLoader()
        self.simulator = TradeSimulator(
            account_balance=starting_balance,
            risk_per_trade_pct=risk_per_trade,
        )
        self.stats_calc = StatisticsCalculator()

        # Registry + cache for dynamic instrument discovery
        if use_registry:
            self._registry = get_instrument_registry()
            self._cache_mgr = get_cache_manager()
        else:
            self._registry = None
            self._cache_mgr = None

        # Current timeframe — updated per backtest for stop/target sizing
        self._current_timeframe: str = "1d"

        # Signal deduplication / cooldown tracking
        # {date_str: {symbol_edge_direction}}
        self._signals_today: Dict[str, set] = {}
        # {symbol_edge_direction: last_bar_index}
        self._last_signal_bar: Dict[str, int] = {}
        # Cooldown in bars between signals for a given setup
        self.SIGNAL_COOLDOWN_BARS = {
            EdgeType.VWAP_DEVIATION: 5,    # 5 days cooldown on daily bars
            EdgeType.RSI_EXTREME: 5,   # RSI(2) trades last 3-7 days
            EdgeType.BOLLINGER_TOUCH: 12,
            EdgeType.POWER_HOUR: 3,        # 3 day cooldown on daily bars
            EdgeType.ORB: 78,              # One per day (~78 5m bars)
            EdgeType.GAP_FILL: 5,          # 5 days cooldown on daily bars
            EdgeType.LONDON_OPEN: 8,       # 2 hour cooldown on 15m
            EdgeType.NY_OPEN: 8,
            EdgeType.MONTH_END: 15,        # One signal per month-end window
            EdgeType.ASIAN_RANGE: 999,     # Effectively 1 per day
            EdgeType.OVERNIGHT_PREMIUM: 1, # 1 signal per day
        }

    # ------------------------------------------------------------------
    # Risk profile helpers
    # ------------------------------------------------------------------

    def _get_risk_profile(self, edge_type: EdgeType) -> dict:
        """Get edge-specific risk profile, falling back to defaults."""
        return self.EDGE_RISK_PROFILES.get(edge_type, self.DEFAULT_RISK_PROFILE)

    # ------------------------------------------------------------------
    # Instrument discovery
    # ------------------------------------------------------------------

    def get_instruments_for_edge(
        self,
        edge_type: EdgeType,
        timeframe: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """Get instrument list for an edge.

        When ``use_registry=True``, queries the InstrumentRegistry filtered
        by provider/type, then narrows to instruments with cached data.
        Hardcoded (validated) symbols are always placed first in the list.

        When ``use_registry=False`` (default), returns EDGE_INSTRUMENTS.
        """
        hardcoded = self.EDGE_INSTRUMENTS.get(edge_type, ["SPY"])

        if not self.use_registry or self._registry is None:
            return hardcoded

        filt = self.EDGE_REGISTRY_FILTERS.get(edge_type)
        if filt is None:
            return hardcoded

        provider, inst_type, max_inst = filt

        # Query registry for all matching instruments
        candidates = [
            inst.symbol
            for inst in self._registry.get_by_provider(provider)
            if inst.instrument_type == inst_type
        ]

        # Narrow to instruments with cached data when dates are known
        if self._cache_mgr and timeframe and start_date and end_date:
            cached = [
                s for s in candidates
                if self._cache_mgr.has_data(s, timeframe, start_date, end_date)
            ]
            if cached:
                candidates = cached
            else:
                logger.warning(
                    "No cached registry data for %s — falling back to defaults",
                    edge_type.value,
                )
                return hardcoded

        # Validated symbols first, then registry additions
        ordered = list(hardcoded)
        for s in candidates:
            if s not in ordered:
                ordered.append(s)

        return ordered[:max_inst]

    # ------------------------------------------------------------------
    # Score-based sizing helpers
    # ------------------------------------------------------------------

    def _get_risk_multiplier(self, score: int) -> float:
        """Get risk multiplier from score, or 1.0 if score sizing disabled."""
        if not self.use_score_sizing:
            return 1.0
        return tier_multiplier(score_to_tier(score))

    def _reset_balance(self) -> None:
        """Reset simulator balance to starting_balance for new edge backtest."""
        self.simulator.account_balance = self.starting_balance

    def _compound(self, net_pnl: float) -> None:
        """Update simulator balance after a trade (compounding)."""
        if self.use_score_sizing:
            self.simulator.account_balance = max(0, self.simulator.account_balance + net_pnl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_edge_backtest(
        self,
        edge_type: EdgeType,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
    ) -> BacktestResult:
        """
        Run backtest for a single edge.

        Args:
            edge_type: Which edge to test
            symbol: Instrument (defaults to edge's primary default).
                     For multi-symbol edges (e.g. RSI_EXTREME), pass None
                     to test all configured symbols with best-per-day selection.
            timeframe: Bar size (defaults to edge's natural timeframe)
            start_date: ISO date, inclusive
            end_date: ISO date, inclusive

        Returns:
            BacktestResult with statistics, trades, equity curve
        """
        # Reset balance for independent per-edge testing
        self._reset_balance()

        timeframe = timeframe or self.EDGE_TIMEFRAMES.get(edge_type, "1d")

        # Resolve instrument list (registry-aware when use_registry=True)
        edge_instruments = self.get_instruments_for_edge(
            edge_type, timeframe, start_date, end_date,
        )

        # Multi-symbol path: when no explicit symbol and edge supports it
        use_multi = (
            symbol is None
            and edge_type in self.MULTI_SYMBOL_EDGES
            and len(edge_instruments) > 1
        )

        symbol = symbol or edge_instruments[0]

        # Skip disabled edges with empty result
        if edge_type in self.DISABLED_EDGES:
            logger.warning("Edge %s is DISABLED (validated as unprofitable)", edge_type.value)
            stats = self.stats_calc.calculate(
                trades=[],
                edge_type=edge_type.value,
                symbol=symbol,
                timeframe=timeframe,
                test_period=f"{start_date} to {end_date}",
                starting_balance=self.starting_balance,
                risk_profile=self._get_risk_profile(edge_type),
            )
            return BacktestResult(
                statistics=stats,
                trades=[],
                equity_curve=pd.DataFrame(),
                parameters={
                    "edge_type": edge_type.value,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "starting_balance": self.starting_balance,
                    "risk_per_trade": self.risk_per_trade,
                    "disabled": True,
                },
            )

        if use_multi:
            return await self._run_multi_symbol_backtest(
                edge_type=edge_type,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        return await self._run_single_symbol_backtest(
            edge_type=edge_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

    async def _run_single_symbol_backtest(
        self,
        edge_type: EdgeType,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Run backtest for a single edge on a single symbol."""
        # Store current timeframe for stop/target sizing
        self._current_timeframe = timeframe

        # Set max hold bars for this edge
        self.simulator.max_hold_bars = self.EDGE_MAX_HOLD_BARS.get(edge_type, 20)

        # Reset signal tracking for this backtest run
        self._last_signal_bar = {}
        self._signals_today = {}

        # Load historical data
        bars = await self.data_loader.load_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if bars.empty:
            raise ValueError(
                f"No data for {symbol} {timeframe} {start_date}–{end_date}"
            )

        # Validate edge type has a scanner
        if edge_type not in self.SCANNER_MAP:
            raise ValueError(f"No scanner for edge type: {edge_type}")

        # Pre-compute indicators for edges that need them
        if edge_type == EdgeType.RSI_EXTREME:
            bars = self._prepare_rsi_indicators(bars)
        elif edge_type == EdgeType.ORB:
            bars = self._prepare_orb_indicators(bars)
        elif edge_type in self.FOREX_EDGES:
            bars = self._prepare_forex_indicators(bars)

        # Run scanner logic on historical bars
        opportunities = self._scan_historical(edge_type, symbol, bars)

        # Forex edges: use tighter cost profile
        _saved_costs = None
        if edge_type in self.FOREX_EDGES:
            _saved_costs = (
                self.simulator.spread_pct,
                self.simulator.slippage_pct,
                self.simulator.commission_per_trade,
            )
            self.simulator.spread_pct = self.FOREX_COST_PROFILE["spread_pct"]
            self.simulator.slippage_pct = self.FOREX_COST_PROFILE["slippage_pct"]
            self.simulator.commission_per_trade = self.FOREX_COST_PROFILE["commission_per_trade"]

        # Simulate trades
        if edge_type == EdgeType.OVERNIGHT_PREMIUM:
            # Overnight bypasses TradeSimulator (entry at close, exit at open)
            trades = self._simulate_overnight_trades(opportunities, bars)
        else:
            if edge_type == EdgeType.RSI_EXTREME:
                exit_checker = self._rsi_exit_checker
            elif edge_type == EdgeType.ORB:
                exit_checker = self._orb_exit_checker
            else:
                exit_checker = None
            trades: List[SimulatedTrade] = []
            for opp, bar_idx in opportunities:
                multiplier = self._get_risk_multiplier(opp.raw_score)
                if multiplier == 0.0:
                    continue  # F-tier: skip trade
                trade = self.simulator.simulate_trade(
                    opp, bars, bar_idx, exit_checker=exit_checker,
                    risk_multiplier=multiplier,
                )
                if trade:
                    trades.append(trade)
                    self._compound(trade.net_pnl)

        # Restore cost profile
        if _saved_costs is not None:
            (self.simulator.spread_pct,
             self.simulator.slippage_pct,
             self.simulator.commission_per_trade) = _saved_costs

        # Calculate statistics
        stats = self.stats_calc.calculate(
            trades=trades,
            edge_type=edge_type.value,
            symbol=symbol,
            timeframe=timeframe,
            test_period=f"{start_date} to {end_date}",
            starting_balance=self.starting_balance,
            risk_profile=self._get_risk_profile(edge_type),
        )

        # Build equity curve
        equity_curve = self._build_equity_curve(trades)

        return BacktestResult(
            statistics=stats,
            trades=trades,
            equity_curve=equity_curve,
            parameters={
                "edge_type": edge_type.value,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "starting_balance": self.starting_balance,
                "risk_per_trade": self.risk_per_trade,
            },
        )

    async def _run_multi_symbol_backtest(
        self,
        edge_type: EdgeType,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Run backtest across multiple symbols with best-per-day selection.

        For each symbol: load data, compute indicators, scan for signals.
        When multiple symbols signal on the same day, pick the best one
        (lowest RSI for mean reversion = most extreme = highest probability).
        """
        symbols = self.get_instruments_for_edge(edge_type, timeframe, start_date, end_date)
        self._current_timeframe = timeframe
        self.simulator.max_hold_bars = self.EDGE_MAX_HOLD_BARS.get(edge_type, 20)

        if edge_type == EdgeType.RSI_EXTREME:
            exit_checker = self._rsi_exit_checker
        else:
            exit_checker = None

        # Collect all opportunities across symbols, each with its bars
        # (opp, bar_idx, symbol_bars_df)
        all_candidates: List[Tuple[Opportunity, int, pd.DataFrame]] = []

        for sym in symbols:
            # Reset signal tracking per symbol
            self._last_signal_bar = {}
            self._signals_today = {}

            try:
                bars = await self.data_loader.load_bars(
                    symbol=sym,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                logger.warning(f"Failed to load {sym}: {e}")
                continue

            if edge_type == EdgeType.RSI_EXTREME:
                min_data = 200
            elif edge_type == EdgeType.OVERNIGHT_PREMIUM:
                min_data = 201
            else:
                min_data = 50  # Match _scan_historical min_bars
            if bars.empty or len(bars) < min_data:
                logger.warning(f"Insufficient data for {sym}: {len(bars)} bars")
                continue

            # Pre-compute indicators
            if edge_type == EdgeType.RSI_EXTREME:
                bars = self._prepare_rsi_indicators(bars)
            elif edge_type in self.FOREX_EDGES:
                bars = self._prepare_forex_indicators(bars)

            # Scan for signals
            opps = self._scan_historical(edge_type, sym, bars)
            for opp, idx in opps:
                all_candidates.append((opp, idx, bars))
            logger.info(f"{edge_type.value} {sym}: {len(opps)} signals found")

        if not all_candidates:
            symbol_str = "+".join(symbols)
            stats = self.stats_calc.calculate(
                trades=[],
                edge_type=edge_type.value,
                symbol=symbol_str,
                timeframe=timeframe,
                test_period=f"{start_date} to {end_date}",
                starting_balance=self.starting_balance,
                risk_profile=self._get_risk_profile(edge_type),
            )
            return BacktestResult(
                statistics=stats, trades=[], equity_curve=pd.DataFrame(),
                parameters={"edge_type": edge_type.value, "symbol": symbol_str},
            )

        # Sort all candidates by signal time for correct compounding order
        all_candidates.sort(key=lambda c: c[2].index[c[1]])

        # Forex edges: use tighter cost profile
        _saved_costs = None
        if edge_type in self.FOREX_EDGES:
            _saved_costs = (
                self.simulator.spread_pct,
                self.simulator.slippage_pct,
                self.simulator.commission_per_trade,
            )
            self.simulator.spread_pct = self.FOREX_COST_PROFILE["spread_pct"]
            self.simulator.slippage_pct = self.FOREX_COST_PROFILE["slippage_pct"]
            self.simulator.commission_per_trade = self.FOREX_COST_PROFILE["commission_per_trade"]

        # Take all signals — per-symbol dedup already limits to 1/day/symbol.
        # Multiple positions across different ETFs provide diversification.
        if edge_type == EdgeType.OVERNIGHT_PREMIUM:
            # Overnight bypasses TradeSimulator (entry at close, exit at open)
            trades: List[SimulatedTrade] = []
            for opp, bar_idx, sym_bars in all_candidates:
                overnight = self._simulate_overnight_trades([(opp, bar_idx)], sym_bars)
                trades.extend(overnight)
        else:
            trades: List[SimulatedTrade] = []
            for opp, bar_idx, bars in all_candidates:
                multiplier = self._get_risk_multiplier(opp.raw_score)
                if multiplier == 0.0:
                    continue  # F-tier: skip trade
                trade = self.simulator.simulate_trade(
                    opp, bars, bar_idx, exit_checker=exit_checker,
                    risk_multiplier=multiplier,
                )
                if trade:
                    trades.append(trade)
                    self._compound(trade.net_pnl)

        # Restore cost profile
        if _saved_costs is not None:
            (self.simulator.spread_pct,
             self.simulator.slippage_pct,
             self.simulator.commission_per_trade) = _saved_costs

        # Sort trades by entry time for consistent reporting
        trades.sort(key=lambda t: t.entry_time)

        symbol_str = "+".join(symbols)
        stats = self.stats_calc.calculate(
            trades=trades,
            edge_type=edge_type.value,
            symbol=symbol_str,
            timeframe=timeframe,
            test_period=f"{start_date} to {end_date}",
            starting_balance=self.starting_balance,
            risk_profile=self._get_risk_profile(edge_type),
        )

        equity_curve = self._build_equity_curve(trades)

        return BacktestResult(
            statistics=stats,
            trades=trades,
            equity_curve=equity_curve,
            parameters={
                "edge_type": edge_type.value,
                "symbol": symbol_str,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "starting_balance": self.starting_balance,
                "risk_per_trade": self.risk_per_trade,
            },
        )

    async def run_all_edges(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
    ) -> Dict[str, BacktestResult]:
        """Run backtests for all configured edges."""
        results: Dict[str, BacktestResult] = {}

        for edge_type in self.SCANNER_MAP:
            try:
                result = await self.run_edge_backtest(
                    edge_type=edge_type,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[edge_type.value] = result
            except Exception as e:
                logger.warning(f"Failed to backtest {edge_type.value}: {e}")

        return results

    # ------------------------------------------------------------------
    # Historical scanning
    # ------------------------------------------------------------------

    def _scan_historical(
        self,
        edge_type: EdgeType,
        symbol: str,
        bars: pd.DataFrame,
    ) -> List[Tuple[Opportunity, int]]:
        """
        Replay scanner logic on historical bars.

        Returns list of (Opportunity, bar_index) tuples.

        Applies:
        - Per-setup-per-day deduplication
        - Cooldown in bars between repeated setups
        """
        opportunities: List[Tuple[Opportunity, int]] = []
        # Edge-specific minimum history for indicator warmup
        if edge_type == EdgeType.RSI_EXTREME:
            min_bars = 200   # SMA(200) warmup
        elif edge_type == EdgeType.ORB:
            min_bars = 400   # ~5 sessions warmup (min_periods=5 for vol avg)
        elif edge_type == EdgeType.OVERNIGHT_PREMIUM:
            min_bars = 201   # Need 200 SMA warmup
        else:
            min_bars = 50

        # Track signals per day to avoid duplicates
        signals_by_date: Dict[str, set] = {}

        for i in range(min_bars, len(bars)):
            # Present the scanner with bars *up to and including* bar i
            bars_view = bars.iloc[: i + 1]
            current_ts = bars.index[i]
            current_date = current_ts.strftime("%Y-%m-%d")

            if current_date not in signals_by_date:
                signals_by_date[current_date] = set()

            opp = self._check_bar_for_signal(edge_type, symbol, bars_view)
            if opp is not None:
                # Create dedup key: symbol_edge_direction
                direction_str = (
                    opp.direction.value if hasattr(opp.direction, "value") else str(opp.direction)
                )
                dedup_key = f"{symbol}_{edge_type.value}_{direction_str}"

                # One signal per setup per day
                if dedup_key in signals_by_date[current_date]:
                    continue

                # Cooldown-based deduplication (for intraday edges)
                cooldown_bars = self.SIGNAL_COOLDOWN_BARS.get(edge_type, 12)
                if dedup_key in self._last_signal_bar:
                    bars_since = i - self._last_signal_bar[dedup_key]
                    if bars_since < cooldown_bars:
                        continue

                # Record this signal
                signals_by_date[current_date].add(dedup_key)
                self._last_signal_bar[dedup_key] = i
                opportunities.append((opp, i))

        # Store for introspection / debugging
        self._signals_today = signals_by_date

        return opportunities

    def _check_bar_for_signal(
        self,
        edge_type: EdgeType,
        symbol: str,
        bars: pd.DataFrame,
    ) -> Optional[Opportunity]:
        """
        Check if the last bar generates a signal.

        Each block replicates the core detection logic of its scanner.
        """
        if edge_type == EdgeType.VWAP_DEVIATION:
            return self._signal_vwap(symbol, bars)
        if edge_type == EdgeType.RSI_EXTREME:
            return self._signal_rsi(symbol, bars)
        if edge_type == EdgeType.TURN_OF_MONTH:
            return self._signal_turn_of_month(symbol, bars)
        if edge_type == EdgeType.MONTH_END:
            return self._signal_month_end(symbol, bars)
        if edge_type == EdgeType.GAP_FILL:
            return self._signal_gap_fill(symbol, bars)
        if edge_type == EdgeType.LONDON_OPEN:
            return self._signal_london_open(symbol, bars)
        if edge_type == EdgeType.NY_OPEN:
            return self._signal_ny_open(symbol, bars)
        if edge_type == EdgeType.POWER_HOUR:
            return self._signal_power_hour(symbol, bars)
        if edge_type == EdgeType.ORB:
            return self._signal_orb(symbol, bars)
        if edge_type == EdgeType.BOLLINGER_TOUCH:
            return self._signal_bollinger_touch(symbol, bars)
        if edge_type == EdgeType.INSIDER_CLUSTER:
            return self._signal_insider_cluster(symbol, bars)
        if edge_type == EdgeType.ASIAN_RANGE:
            return self._signal_asian_range(symbol, bars)
        if edge_type == EdgeType.EARNINGS_DRIFT:
            return self._signal_earnings_drift(symbol, bars)
        if edge_type == EdgeType.OVERNIGHT_PREMIUM:
            return self._signal_overnight(symbol, bars)
        return None

    # ---- individual edge detectors ------------------------------------

    def _signal_vwap(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """VWAP Trend-Following: trade WITH the deviation, not against it.

        Research basis: Zarattini & Aziz (2023) SSRN #4631351
        - Buy when price is ABOVE VWAP (bullish momentum)
        - Sell/short when price is BELOW VWAP (bearish momentum)
        - VWAP acts as dynamic support/resistance

        Reversed from original mean-reversion implementation (which was -12% P&L).
        """
        if len(bars) < 20:
            return None

        # Rolling VWAP over 10 days (approximation for daily timeframe)
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3

        if "volume" in bars.columns:
            vwap_window = 10
            cum_vol = bars["volume"].rolling(vwap_window).sum()
            cum_tp_vol = (typical_price * bars["volume"]).rolling(vwap_window).sum()
            vwap = cum_tp_vol / cum_vol
        else:
            # Fallback: use typical price SMA
            vwap = typical_price.rolling(10).mean()

        cur_price = bars["close"].iloc[-1]
        prev_price = bars["close"].iloc[-2]
        cur_vwap = vwap.iloc[-1]
        prev_vwap = vwap.iloc[-2]

        if pd.isna(cur_vwap) or pd.isna(prev_vwap):
            return None

        # Calculate deviation from VWAP
        deviation_pct = ((cur_price - cur_vwap) / cur_vwap) * 100

        # Need meaningful deviation (at least 0.3% from VWAP)
        min_deviation = 0.3

        # Volume confirmation (if available)
        vol_ratio = 1.0
        if "volume" in bars.columns:
            avg_vol = bars["volume"].rolling(20).mean().iloc[-1]
            cur_vol = bars["volume"].iloc[-1]
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        # TREND-FOLLOWING LOGIC (opposite of mean reversion)

        # LONG: Price crosses ABOVE VWAP (bullish breakout)
        if prev_price < prev_vwap and cur_price > cur_vwap:
            if deviation_pct >= min_deviation and vol_ratio >= 1.0:
                entry = cur_price
                stop = cur_vwap * 0.995  # Stop just below VWAP
                risk = entry - stop
                target = entry + (risk * 2.5)  # 2.5:1 R:R

                return self._make_opportunity(
                    symbol=symbol,
                    bars=bars,
                    edge_type=EdgeType.VWAP_DEVIATION,
                    scanner_name="VWAPTrendScanner",
                    direction=Direction.LONG,
                    entry=entry,
                    stop=stop,
                    target=target,
                    edge_data={
                        "strategy": "trend_following",
                        "vwap": float(cur_vwap),
                        "deviation_pct": float(deviation_pct),
                        "volume_ratio": float(vol_ratio),
                        "cross_direction": "bullish",
                        "notional_pct": 16,
                    },
                    score=65,
                )

        # SHORT: Price crosses BELOW VWAP (bearish breakdown)
        elif prev_price > prev_vwap and cur_price < cur_vwap:
            if deviation_pct <= -min_deviation and vol_ratio >= 1.0:
                entry = cur_price
                stop = cur_vwap * 1.005  # Stop just above VWAP
                risk = stop - entry
                target = entry - (risk * 2.5)  # 2.5:1 R:R

                return self._make_opportunity(
                    symbol=symbol,
                    bars=bars,
                    edge_type=EdgeType.VWAP_DEVIATION,
                    scanner_name="VWAPTrendScanner",
                    direction=Direction.SHORT,
                    entry=entry,
                    stop=stop,
                    target=target,
                    edge_data={
                        "strategy": "trend_following",
                        "vwap": float(cur_vwap),
                        "deviation_pct": float(deviation_pct),
                        "volume_ratio": float(vol_ratio),
                        "cross_direction": "bearish",
                        "notional_pct": 16,
                    },
                    score=65,
                )

        return None

    def _signal_rsi(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Connors RSI(2) mean-reversion signal.

        Entry: RSI(2) < 10 (long) or RSI(2) > 90 (short)
        Filter: 200 SMA trend alignment (CRITICAL)
        Stop: 10% catastrophic only — tight stops destroy mean reversion
        Target: Unreachable placeholder — exit via indicator checker
        """
        cur_rsi = bars["rsi_2"].iloc[-1]
        cur_sma_200 = bars["sma_200"].iloc[-1]
        cur_price = bars["close"].iloc[-1]

        if pd.isna(cur_rsi) or pd.isna(cur_sma_200):
            return None

        # Regime filter: Skip if ADX > 40 (extreme trend = mean reversion fails)
        cur_adx = bars["adx_14"].iloc[-1]
        if pd.isna(cur_adx) or cur_adx > 40:
            return None

        # Long: RSI(2) < 10 AND price > 200 SMA (uptrend only)
        if cur_rsi < 10 and cur_price > cur_sma_200:
            direction = Direction.LONG
            entry_type = "rsi2_oversold"
            stop = cur_price * 0.90   # 10% catastrophic stop
            target = cur_price * 1.20  # Unreachable — exit via indicator

        # Short: RSI(2) > 90 AND price < 200 SMA (downtrend only)
        elif cur_rsi > 90 and cur_price < cur_sma_200:
            direction = Direction.SHORT
            entry_type = "rsi2_overbought"
            stop = cur_price * 1.10
            target = cur_price * 0.80

        else:
            return None

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.RSI_EXTREME,
            scanner_name="RSIScanner",
            direction=direction,
            entry=cur_price,
            stop=stop,
            target=target,
            edge_data={
                "rsi_value": float(cur_rsi),
                "sma_200": float(cur_sma_200),
                "trend_aligned": True,
                "entry_type": entry_type,
                "notional_pct": 16,  # 16% of capital (~40% Kelly for 75% WR, PF 2.2)
            },
            score=75,
        )

    # ---- calendar edge detectors ---------------------------------------

    def _signal_turn_of_month(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Turn of month long bias: last trading day through day 3."""
        ts = bars.index[-1]
        tom_day = self._get_tom_day(ts)
        if tom_day == 0:
            return None
        if not self._is_trading_day(ts):
            return None

        cur_price = bars["close"].iloc[-1]
        atr = self._calculate_atr(bars)
        if pd.isna(atr) or atr == 0:
            return None

        stop_mult, target_mult = self._get_stop_target_multipliers(
            self._current_timeframe
        )

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.TURN_OF_MONTH,
            scanner_name="TurnOfMonthScanner",
            direction=Direction.LONG,
            entry=cur_price,
            stop=cur_price - atr * stop_mult,
            target=cur_price + atr * target_mult,
            edge_data={"tom_day": tom_day},
            score=75,
        )

    def _signal_month_end(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Month-End Rebalancing: trade contrarian to month's move.

        Research: $7.5T pension funds rebalance to target allocations at month-end.
        Strong month → funds overweight stocks → SELL pressure → SHORT.
        Weak month → funds underweight stocks → BUY pressure → LONG.
        """
        if len(bars) < 25:
            return None

        ts = bars.index[-1]
        if not self._is_trading_day(ts):
            return None

        # Count trading days remaining in the month after today
        trading_days_after = 0
        for d in range(1, 8):
            future = ts + timedelta(days=d)
            if future.weekday() < 5:
                if future.month != ts.month:
                    break
                trading_days_after += 1
                if trading_days_after >= 2:
                    break

        if trading_days_after >= 2:
            return None  # Not in last-2-day window

        # Month performance: compare first bar of month to current
        month_mask = (bars.index.month == ts.month) & (bars.index.year == ts.year)
        month_bars = bars[month_mask]
        if len(month_bars) < 5:
            return None

        month_open = month_bars["open"].iloc[0]
        cur_price = bars["close"].iloc[-1]
        if month_open == 0:
            return None

        mtd_return = ((cur_price - month_open) / month_open) * 100
        if abs(mtd_return) < 1.5:
            return None  # Not enough move to trigger significant rebalancing

        atr = self._calculate_atr(bars)
        if pd.isna(atr) or atr == 0:
            return None

        # CONTRARIAN: trade OPPOSITE of month's trend
        if mtd_return > 1.5:
            # Strong month → pension funds SELL to rebalance → SHORT
            direction = Direction.SHORT
            stop = cur_price + atr * 1.5
            target = cur_price - atr * 2.0
            rebalance_bias = "selling_pressure"
        else:
            # Weak month → pension funds BUY to rebalance → LONG
            direction = Direction.LONG
            stop = cur_price - atr * 1.5
            target = cur_price + atr * 2.0
            rebalance_bias = "buying_pressure"

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.MONTH_END,
            scanner_name="MonthEndScanner",
            direction=direction,
            entry=cur_price,
            stop=stop,
            target=target,
            edge_data={
                "mtd_return_pct": float(mtd_return),
                "day_of_month": int(ts.day),
                "rebalance_bias": rebalance_bias,
                "month_bars_count": int(len(month_bars)),
                "notional_pct": 16,
            },
            score=65,
        )

    # ---- gap / price-action detectors ----------------------------------

    @staticmethod
    def _calculate_gap_score(
        gap_pct: float,
        volume_ratio: float,
        trend_aligned: bool,
    ) -> int:
        """Dynamic scoring for gap signals based on signal quality.

        Factors:
        - Gap size (sweet spot 2-3.5%)
        - Volume ratio (higher = stronger confirmation)
        - Trend alignment (20-day SMA direction matches gap)

        Returns score 0-100 mapping to tiers:
          A(80+)=1.5x, B(65-79)=1.25x, C(50-64)=1.0x, D(40-49)=0.5x, F(<40)=skip

        Base=35 ensures weakest valid signals (1% gap, 1.5x vol) land in C-tier
        (no amplification), preventing drawdown amplification on marginal signals.
        """
        base_score = 35

        # Gap size scoring (sweet spot is 2-3.5%)
        abs_gap = abs(gap_pct)
        if 2.0 <= abs_gap <= 3.5:
            base_score += 25
        elif 1.0 <= abs_gap < 2.0:
            base_score += 15
        elif 3.5 < abs_gap <= 5.0:
            base_score += 5
        else:
            base_score -= 10

        # Volume confirmation
        if volume_ratio >= 2.0:
            base_score += 15
        elif volume_ratio >= 1.5:
            base_score += 10
        elif volume_ratio >= 1.2:
            base_score += 5
        elif volume_ratio < 0.8:
            base_score -= 10

        # Trend alignment bonus
        if trend_aligned:
            base_score += 10

        return max(0, min(100, base_score))

    def _signal_gap_fill(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Gap and Go: trade WITH gap momentum on high-volume breakaway gaps.

        Reversed from original gap fill (fade) which lost -1.1%.

        OLD: Gap up → SHORT (fade). NEW: Gap up + volume → LONG (momentum).
        Key filters: gap 1-5%, volume 150%+, day confirms gap direction.
        Dynamic scoring: strong gaps get A-tier sizing, weak gaps get D/F.
        """
        if len(bars) < 21:  # Need 20 bars for volume average
            return None

        cur = bars.iloc[-1]
        prev = bars.iloc[-2]

        # Calculate gap percentage
        if prev["close"] == 0:
            return None
        gap_pct = ((cur["open"] - prev["close"]) / prev["close"]) * 100

        # FILTER 1: Minimum gap size (skip tiny gaps that fill)
        min_gap_pct = 1.0
        if abs(gap_pct) < min_gap_pct:
            return None

        # FILTER 2: Maximum gap size (avoid crazy gaps that reverse)
        max_gap_pct = 5.0
        if abs(gap_pct) > max_gap_pct:
            return None

        # FILTER 3: Volume confirmation (breakaway gaps have high volume)
        if "volume" not in bars.columns:
            return None

        avg_volume = bars["volume"].iloc[-21:-1].mean()  # 20-day avg excluding today
        cur_volume = cur["volume"]
        volume_ratio = cur_volume / avg_volume if avg_volume > 0 else 0

        min_volume_ratio = 1.5  # Need 150%+ of average volume
        if volume_ratio < min_volume_ratio:
            return None

        # FILTER 4: Day must close in gap direction (confirmation)
        day_move = cur["close"] - cur["open"]

        # Trend alignment: 20-day SMA direction matches gap direction
        sma_20 = bars["close"].rolling(20).mean().iloc[-1]
        trend_aligned = False
        if not pd.isna(sma_20):
            if gap_pct > 0 and cur["close"] > sma_20:
                trend_aligned = True
            elif gap_pct < 0 and cur["close"] < sma_20:
                trend_aligned = True

        # Dynamic score based on signal quality
        gap_score = self._calculate_gap_score(gap_pct, volume_ratio, trend_aligned)

        # GAP UP + VOLUME → LONG (momentum continuation)
        if gap_pct >= min_gap_pct:
            if day_move < 0:  # Gap up but closed red = filling, skip
                return None

            entry = cur["close"]
            stop = prev["close"] * 0.995  # Just below prior close
            risk = entry - stop
            target = entry + (risk * 2.0)  # 2:1 R:R

            return self._make_opportunity(
                symbol=symbol,
                bars=bars,
                edge_type=EdgeType.GAP_FILL,
                scanner_name="GapAndGoScanner",
                direction=Direction.LONG,
                entry=entry,
                stop=stop,
                target=target,
                edge_data={
                    "strategy": "gap_and_go",
                    "gap_pct": float(gap_pct),
                    "volume_ratio": float(volume_ratio),
                    "day_move_pct": float((day_move / cur["open"]) * 100),
                    "gap_direction": "up",
                    "trend_aligned": trend_aligned,
                    "notional_pct": 16,
                },
                score=gap_score,
            )

        # GAP DOWN + VOLUME → SHORT (momentum continuation)
        elif gap_pct <= -min_gap_pct:
            if day_move > 0:  # Gap down but closed green = filling, skip
                return None

            entry = cur["close"]
            stop = prev["close"] * 1.005  # Just above prior close
            risk = stop - entry
            target = entry - (risk * 2.0)  # 2:1 R:R

            return self._make_opportunity(
                symbol=symbol,
                bars=bars,
                edge_type=EdgeType.GAP_FILL,
                scanner_name="GapAndGoScanner",
                direction=Direction.SHORT,
                entry=entry,
                stop=stop,
                target=target,
                edge_data={
                    "strategy": "gap_and_go",
                    "gap_pct": float(gap_pct),
                    "volume_ratio": float(volume_ratio),
                    "day_move_pct": float((day_move / cur["open"]) * 100),
                    "gap_direction": "down",
                    "trend_aligned": trend_aligned,
                    "notional_pct": 16,
                },
                score=gap_score,
            )

        return None

    def _signal_bollinger_touch(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Bollinger Band touch with RSI confirmation for mean reversion.

        ADX regime filter: only trade in ranging markets (ADX < 25).
        In trending markets, price touches band and keeps going.
        """
        if len(bars) < 30:  # Need enough bars for ADX(14) + BB(20)
            return None

        # REGIME FILTER: Only trade when ADX < 25 (ranging market)
        adx = self._calculate_adx(bars, 14)
        cur_adx = adx.iloc[-1]
        if pd.isna(cur_adx) or cur_adx >= 25:
            return None  # Trending market — Bollinger mean reversion fails

        # Bollinger Bands (20-period, 2 std dev)
        sma = bars["close"].rolling(20).mean()
        std = bars["close"].rolling(20).std(ddof=1)
        upper = sma + 2 * std
        lower = sma - 2 * std

        cur_close = bars["close"].iloc[-1]
        cur_upper = upper.iloc[-1]
        cur_lower = lower.iloc[-1]
        cur_sma = sma.iloc[-1]

        if pd.isna(cur_upper) or pd.isna(cur_lower) or pd.isna(cur_sma):
            return None

        half_width = (cur_upper - cur_lower) / 2
        if half_width == 0:
            return None

        # RSI(14) for confirmation
        delta = bars["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        cur_rsi = rsi.iloc[-1]

        if pd.isna(cur_rsi):
            return None

        # Touch lower band + RSI < 40 -> LONG (mean reversion)
        if cur_close <= cur_lower and cur_rsi < 40:
            direction = Direction.LONG
            stop = cur_lower - half_width * 1.5
            target = cur_sma
        # Touch upper band + RSI > 60 -> SHORT (mean reversion)
        elif cur_close >= cur_upper and cur_rsi > 60:
            direction = Direction.SHORT
            stop = cur_upper + half_width * 1.5
            target = cur_sma
        else:
            return None

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.BOLLINGER_TOUCH,
            scanner_name="BollingerScanner",
            direction=direction,
            entry=cur_close,
            stop=stop,
            target=target,
            edge_data={
                "upper": float(cur_upper),
                "lower": float(cur_lower),
                "sma": float(cur_sma),
                "rsi": float(cur_rsi),
                "adx": float(cur_adx),
            },
            score=55,
        )

    # ---- session / time-based detectors --------------------------------

    def _signal_london_open(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """London open: breakout of Asian session range (forex only).

        Enhanced with:
        - Retest confirmation (GBP/USD) or fade (EUR/USD) strategy
        - ADX < 25 filter (consolidation phase only)
        - Range quality filter (3-8x ATR — tight consolidation)
        - Candle close requirement (no wick triggers)
        """
        if "/" not in symbol:
            return None

        ts = bars.index[-1]
        # 07:00-10:00 UTC for fade pattern to develop (3 hours)
        if not (7 <= ts.hour < 10):
            return None

        # Find Asian session bars (00:00-07:00 UTC, same day)
        day_start = ts.floor("D")
        asian_end = day_start + pd.Timedelta(hours=7)
        asian_bars = bars[(bars.index >= day_start) & (bars.index < asian_end)]

        if len(asian_bars) < 3:
            return None

        asian_high = asian_bars["high"].max()
        asian_low = asian_bars["low"].min()
        asian_range = asian_high - asian_low

        if asian_range == 0:
            return None

        # Asian session = 28 bars on 15m → range typically 3-8x single-bar ATR
        return self._forex_session_signal(
            symbol=symbol,
            bars=bars,
            range_high=asian_high,
            range_low=asian_low,
            range_size=asian_range,
            edge_type=EdgeType.LONDON_OPEN,
            session_start_hour=7,
            adx_threshold=30.0,
            range_atr_min=3.0,
            range_atr_max=8.0,
        )

    def _signal_ny_open(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """NY open: breakout of pre-NY range (forex only).

        Enhanced with:
        - Retest confirmation (GBP/USD, USD/JPY) or fade (EUR/USD)
        - ADX < 25 filter
        - Range quality filter
        - Candle close requirement
        """
        if "/" not in symbol:
            return None

        ts = bars.index[-1]
        time_mins = ts.hour * 60 + ts.minute

        # Widened to 13:45-16:00 UTC for retest/fade pattern
        if not (13 * 60 + 45 <= time_mins < 16 * 60):
            return None

        # Find opening range bars (13:30-13:45 UTC, same day)
        day_start = ts.floor("D")
        or_start = day_start + pd.Timedelta(hours=13, minutes=30)
        or_end = day_start + pd.Timedelta(hours=13, minutes=45)
        or_bars = bars[(bars.index >= or_start) & (bars.index < or_end)]

        if len(or_bars) < 1:
            return None

        range_high = or_bars["high"].max()
        range_low = or_bars["low"].min()
        opening_range = range_high - range_low

        if opening_range == 0:
            return None

        # Opening range = 1 bar on 15m → ratio ≈ 0.5-1.5x ATR
        # ADX disabled (100) — hurts NY open (trending days are better)
        return self._forex_session_signal(
            symbol=symbol,
            bars=bars,
            range_high=range_high,
            range_low=range_low,
            range_size=opening_range,
            edge_type=EdgeType.NY_OPEN,
            session_start_hour=13,
            adx_threshold=100.0,
            range_atr_min=0.5,
            range_atr_max=2.0,
        )

    def _signal_power_hour(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Power Hour momentum: trade continuation when close shows strong end-of-day bias.

        Reimplemented for daily bars (old 5m version was -36% P&L).

        Research: U-shaped volume pattern means institutions active at close.
        Close near high = buyers won power hour → momentum continues next day.
        Close near low = sellers won power hour → weakness continues.
        """
        if len(bars) < 21:  # Need history for volume average
            return None

        cur = bars.iloc[-1]

        # Close position within day's range: 0 = at low, 1 = at high
        day_range = cur["high"] - cur["low"]
        if day_range == 0:
            return None
        close_position = (cur["close"] - cur["low"]) / day_range

        # FILTER 1: Decisive close (top 25% or bottom 25% of range)
        bullish_threshold = 0.75
        bearish_threshold = 0.25
        if bearish_threshold < close_position < bullish_threshold:
            return None  # Indecisive close, skip

        # FILTER 2: Volume confirmation
        if "volume" not in bars.columns:
            return None
        avg_volume = bars["volume"].iloc[-21:-1].mean()
        cur_volume = cur["volume"]
        volume_ratio = cur_volume / avg_volume if avg_volume > 0 else 0
        if volume_ratio < 1.0:
            return None

        # FILTER 3: Meaningful range (not a doji/inside day)
        avg_range = (bars["high"] - bars["low"]).iloc[-21:-1].mean()
        if day_range < avg_range * 0.5:
            return None

        if close_position >= bullish_threshold:
            direction = Direction.LONG
            entry = cur["close"]
            stop = cur["low"] * 0.998  # Below day's low
            risk = entry - stop
            target = entry + (risk * 2.0)

            return self._make_opportunity(
                symbol=symbol,
                bars=bars,
                edge_type=EdgeType.POWER_HOUR,
                scanner_name="PowerHourScanner",
                direction=direction,
                entry=entry,
                stop=stop,
                target=target,
                edge_data={
                    "close_position": float(close_position),
                    "volume_ratio": float(volume_ratio),
                    "day_range_pct": float((day_range / cur["low"]) * 100),
                    "power_hour_bias": "bullish",
                    "notional_pct": 16,
                },
                score=65,
            )

        elif close_position <= bearish_threshold:
            direction = Direction.SHORT
            entry = cur["close"]
            stop = cur["high"] * 1.002  # Above day's high
            risk = stop - entry
            target = entry - (risk * 2.0)

            return self._make_opportunity(
                symbol=symbol,
                bars=bars,
                edge_type=EdgeType.POWER_HOUR,
                scanner_name="PowerHourScanner",
                direction=direction,
                entry=entry,
                stop=stop,
                target=target,
                edge_data={
                    "close_position": float(close_position),
                    "volume_ratio": float(volume_ratio),
                    "day_range_pct": float((day_range / cur["low"]) * 100),
                    "power_hour_bias": "bearish",
                    "notional_pct": 16,
                },
                score=65,
            )

        return None

    def _signal_orb(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """ORB with volume filter (Zarattini et al. 2024).

        Entry: Breakout of first 5m bar + direction bias + volume >= 100%
        Filter: Relative volume of first bar vs 20-day avg (CRITICAL)
        Stop: 10% of daily ATR (tight — moves fast or fails fast)
        Target: Unreachable placeholder — exit at EOD via exit checker
        """
        cur = bars.iloc[-1]

        # Must be in entry window (bars 1-5 of session)
        if not cur.get("in_entry_window", False):
            return None

        # Volume filter (CRITICAL — this IS the edge)
        rel_vol = cur.get("relative_volume")
        if pd.isna(rel_vol) or rel_vol < 1.0:
            return None

        # Opening range from pre-computed columns
        or_high = cur.get("or_high")
        or_low = cur.get("or_low")
        is_bullish = cur.get("or_bullish", False)

        if pd.isna(or_high) or pd.isna(or_low):
            return None

        or_size = or_high - or_low
        if or_size <= 0:
            return None

        cur_price = cur["close"]

        # Direction bias + breakout with close confirmation
        if is_bullish and cur_price > or_high:
            direction = Direction.LONG
        elif not is_bullish and cur_price < or_low:
            direction = Direction.SHORT
        else:
            return None

        # Tight ATR stop (10% of daily ATR — moves fast or fails fast)
        daily_atr = cur.get("daily_atr")
        if pd.isna(daily_atr) or daily_atr <= 0:
            return None
        stop_distance = daily_atr * 0.10

        if direction == Direction.LONG:
            stop = cur_price - stop_distance
            target = cur_price * 1.10  # Placeholder — EOD exit
        else:
            stop = cur_price + stop_distance
            target = cur_price * 0.90

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.ORB,
            scanner_name="ORBScanner",
            direction=direction,
            entry=cur_price,
            stop=stop,
            target=target,
            edge_data={
                "relative_volume": float(rel_vol),
                "range_high": float(or_high),
                "range_low": float(or_low),
                "range_size": float(or_size),
                "first_bar_bullish": bool(is_bullish),
                "daily_atr": float(daily_atr),
                "stop_distance": float(stop_distance),
                "exit_method": "end_of_day",
                "notional_pct": 20,
            },
            score=70,
        )

    # ---- simulated / external data edges -------------------------------

    def _signal_insider_cluster(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Insider cluster: simulated signals for backtesting.

        Real insider data requires SEC EDGAR filings.  This simulation
        generates deterministic pseudo-random signals (~2-3 per month)
        to estimate edge characteristics with wider stops/targets
        appropriate for swing-trade hold periods.
        """
        if not self._simulate_insider_signal(bars, symbol):
            return None

        cur_price = bars["close"].iloc[-1]
        atr = self._calculate_atr(bars)
        if pd.isna(atr) or atr == 0:
            return None

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.INSIDER_CLUSTER,
            scanner_name="InsiderClusterSimulated",
            direction=Direction.LONG,
            entry=cur_price,
            stop=cur_price - atr * 2.0,
            target=cur_price + atr * 4.0,
            edge_data={"simulated": True},
            score=80,
        )

    def _signal_asian_range(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Asian range breakout with retest confirmation (forex only).

        Enhanced from midpoint bias to proper breakout detection with:
        - Retest confirmation (all pairs: USD/JPY, AUD/USD)
        - ADX < 25 filter
        - Range quality filter
        - Candle close requirement
        """
        if "/" not in symbol:
            return None

        ts = bars.index[-1]
        # Widened to 07:00-11:00 UTC for retest pattern on 1h bars
        if not (7 <= ts.hour < 11):
            return None

        # Asian session range (00:00-07:00 UTC, same day)
        day_start = ts.floor("D")
        asian_end = day_start + pd.Timedelta(hours=7)
        asian_bars = bars[(bars.index >= day_start) & (bars.index < asian_end)]

        if len(asian_bars) < 3:
            return None

        asian_high = asian_bars["high"].max()
        asian_low = asian_bars["low"].min()
        asian_range = asian_high - asian_low

        if asian_range == 0:
            return None

        # Asian session = 7 bars on 1h → range typically 1.5-6x single-bar ATR
        return self._forex_session_signal(
            symbol=symbol,
            bars=bars,
            range_high=asian_high,
            range_low=asian_low,
            range_size=asian_range,
            edge_type=EdgeType.ASIAN_RANGE,
            session_start_hour=7,
            adx_threshold=30.0,
            range_atr_min=1.5,
            range_atr_max=8.0,
        )

    # ---- forex session helpers (retest / fade / filters) -----------------

    def _forex_session_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        range_high: float,
        range_low: float,
        range_size: float,
        edge_type: EdgeType,
        session_start_hour: int,
        adx_threshold: float = 30.0,
        range_atr_min: float = 3.0,
        range_atr_max: float = 8.0,
    ) -> Optional[Opportunity]:
        """Common forex session signal with ADX filter, range filter, and
        strategy dispatch (retest vs fade).

        Shared by london_open, ny_open, and asian_range detectors.

        Args:
            adx_threshold: Max ADX for entry (consolidation filter)
            range_atr_min: Min session-range / ATR ratio (skip noise)
            range_atr_max: Max session-range / ATR ratio (skip trending)
        """
        # Day-of-week filter: skip Monday (0, slow start) and Friday (4, risk-off)
        ts = bars.index[-1]
        if ts.dayofweek in (0, 4):
            return None

        # ADX filter: only trade in consolidation (skip if threshold >= 100)
        if adx_threshold < 100 and "adx_14" in bars.columns:
            adx = bars["adx_14"].iloc[-1]
            if pd.isna(adx) or adx >= adx_threshold:
                return None

        # Range quality filter: session range vs single-bar ATR
        # Too small = noise, too large = already trending
        if "atr_14" in bars.columns:
            atr = bars["atr_14"].iloc[-1]
            if pd.isna(atr) or atr == 0:
                return None
            range_ratio = range_size / atr
            if range_ratio < range_atr_min or range_ratio > range_atr_max:
                return None
        else:
            return None

        # Dispatch to pair-specific strategy
        strategy = self.FOREX_STRATEGY.get(symbol, "retest")
        if strategy == "fade":
            return self._forex_fade_signal(
                symbol, bars, range_high, range_low, range_size,
                edge_type, session_start_hour,
            )
        else:
            return self._forex_retest_signal(
                symbol, bars, range_high, range_low, range_size,
                edge_type, session_start_hour,
            )

    def _forex_retest_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        range_high: float,
        range_low: float,
        range_size: float,
        edge_type: EdgeType,
        session_start_hour: int,
    ) -> Optional[Opportunity]:
        """Retest confirmation: breakout -> pullback to level -> bounce.

        Pattern (3-step over 3+ bars):
        1. A bar CLOSES beyond range (breakout — candle close, not wick)
        2. A subsequent bar's wick nearly touches the broken level (true retest)
        3. Current bar CLOSES beyond the level AND higher/lower than pullback bar

        Tight stop just below the broken level (if retest fails, the breakout
        was false). Target = 1x range extension from the level.

        R:R ≈ 1:3 to 1:5 with tight stop.
        """
        ts = bars.index[-1]
        day_start = ts.floor("D")
        session_start = day_start + pd.Timedelta(hours=session_start_hour)

        session_bars = bars[(bars.index >= session_start) & (bars.index <= ts)]
        if len(session_bars) < 3:
            return None

        cur = session_bars.iloc[-1]
        cur_close = cur["close"]

        # Step 1: Find a breakout bar (candle CLOSE beyond range, not just wick)
        breakout_dir = None
        breakout_idx = None

        # Only look at bars before the last 1 (need room for pullback + current)
        for j in range(len(session_bars) - 2):
            bar = session_bars.iloc[j]
            if bar["close"] > range_high:
                breakout_dir = "long"
                breakout_idx = j
                break
            elif bar["close"] < range_low:
                breakout_dir = "short"
                breakout_idx = j
                break

        if breakout_dir is None:
            return None

        # Step 2: True retest — price must pull back CLOSE to the broken level
        # Long: a bar's low must come within 15% of range_size from range_high
        # Short: a bar's high must come within 15% of range_size from range_low
        pullback_found = False
        pullback_close = None
        for j in range(breakout_idx + 1, len(session_bars) - 1):
            bar = session_bars.iloc[j]
            if breakout_dir == "long":
                if bar["low"] <= range_high + range_size * 0.15:
                    pullback_found = True
                    pullback_close = bar["close"]
                    break
            else:
                if bar["high"] >= range_low - range_size * 0.15:
                    pullback_found = True
                    pullback_close = bar["close"]
                    break

        if not pullback_found:
            return None

        # Step 3: Confirmation — current bar closes beyond level AND shows
        # momentum returning (close beyond pullback bar's close)
        if breakout_dir == "long":
            if cur_close <= range_high:
                return None
            if pullback_close is not None and cur_close <= pullback_close:
                return None  # No bounce — still weak
            direction = Direction.LONG
            entry = cur_close
        else:
            if cur_close >= range_low:
                return None
            if pullback_close is not None and cur_close >= pullback_close:
                return None  # No bounce
            direction = Direction.SHORT
            entry = cur_close

        # ATR-based stop, session-close exit (let winners run to time_expiry)
        atr = bars["atr_14"].iloc[-1] if "atr_14" in bars.columns else range_size
        if pd.isna(atr) or atr == 0:
            atr = range_size

        if direction == Direction.LONG:
            stop = entry - 1.5 * atr
            # Far target — effectively unreachable; session time_expiry handles exit
            target = entry + 10.0 * atr
        else:
            stop = entry + 1.5 * atr
            target = entry - 10.0 * atr

        adx_val = bars["adx_14"].iloc[-1] if "adx_14" in bars.columns else 0

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=edge_type,
            scanner_name=f"{edge_type.value}_retest",
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            edge_data={
                "strategy": "retest",
                "range_high": float(range_high),
                "range_low": float(range_low),
                "range_size": float(range_size),
                "breakout_direction": breakout_dir,
                "adx": float(adx_val),
                "notional_pct": 10,
            },
            score=60,
        )

    def _forex_fade_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        range_high: float,
        range_low: float,
        range_size: float,
        edge_type: EdgeType,
        session_start_hour: int,
    ) -> Optional[Opportunity]:
        """Fade: failed breakout -> enter opposite direction.

        Pattern:
        1. A bar CLOSES beyond range (breakout)
        2. Within 2-6 bars, current bar CLOSES back inside range
        3. Entry must be near the failed breakout side (upper/lower 40% of range)
        4. Enter OPPOSITE direction, target opposite side minus buffer

        EUR/USD breakouts fail more often than they succeed.
        Tight stop just beyond the breakout extreme.
        """
        ts = bars.index[-1]
        day_start = ts.floor("D")
        session_start = day_start + pd.Timedelta(hours=session_start_hour)

        session_bars = bars[(bars.index >= session_start) & (bars.index <= ts)]
        if len(session_bars) < 3:
            return None

        cur = session_bars.iloc[-1]
        cur_close = cur["close"]

        # Current bar must close INSIDE range (confirming failed breakout)
        if cur_close > range_high or cur_close < range_low:
            return None

        # Find a breakout in earlier session bars (at least 2 bars ago)
        breakout_dir = None
        breakout_extreme = None

        # Look 2-10 bars back from current (wider window for late fades)
        for j in range(max(0, len(session_bars) - 10), len(session_bars) - 2):
            bar = session_bars.iloc[j]
            if bar["close"] > range_high:
                breakout_dir = "long"
                if breakout_extreme is None or bar["high"] > breakout_extreme:
                    breakout_extreme = bar["high"]
            elif bar["close"] < range_low:
                breakout_dir = "short"
                if breakout_extreme is None or bar["low"] < breakout_extreme:
                    breakout_extreme = bar["low"]

        if breakout_dir is None or breakout_extreme is None:
            return None

        # ATR-based stop, session-close exit
        atr = bars["atr_14"].iloc[-1] if "atr_14" in bars.columns else range_size
        if pd.isna(atr) or atr == 0:
            atr = range_size

        # FADE: enter opposite of the failed breakout
        if breakout_dir == "long":
            # Failed breakout up → SHORT
            direction = Direction.SHORT
            entry = cur_close
            stop = entry + 1.5 * atr
            target = entry - 10.0 * atr  # Far target; time_expiry exits
        else:
            # Failed breakout down → LONG
            direction = Direction.LONG
            entry = cur_close
            stop = entry - 1.5 * atr
            target = entry + 10.0 * atr

        adx_val = bars["adx_14"].iloc[-1] if "adx_14" in bars.columns else 0

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=edge_type,
            scanner_name=f"{edge_type.value}_fade",
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            edge_data={
                "strategy": "fade",
                "range_high": float(range_high),
                "range_low": float(range_low),
                "range_size": float(range_size),
                "failed_breakout": breakout_dir,
                "breakout_extreme": float(breakout_extreme),
                "adx": float(adx_val),
                "notional_pct": 10,
            },
            score=60,
        )

    def _signal_earnings_drift(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Post-earnings announcement drift.

        TODO: Implement with real earnings calendar data.
        This edge cannot be backtested with price data alone — it needs
        earnings announcement dates, EPS surprise %, and consensus
        estimates.  Will be validated separately with Estimize or
        SEC EDGAR data.
        """
        return None

    # ---- overnight premium ------------------------------------------------

    def _signal_overnight(
        self, symbol: str, bars: pd.DataFrame
    ) -> Optional[Opportunity]:
        """Overnight Return Premium: buy at close, sell at next open.

        Academic basis: Nearly 100% of equity premium earned overnight.

        Filters:
        - Skip Friday nights (weekend returns are negative)
        - Only trade when price > 200 SMA (bull market regime)
        """
        if len(bars) < 201:  # Need 200 bars for SMA
            return None

        ts = bars.index[-1]

        # Skip Fridays (weekday 4) — weekend overnight returns are negative
        if ts.weekday() == 4:
            return None

        # Skip weekends (defensive)
        if ts.weekday() >= 5:
            return None

        cur_price = bars["close"].iloc[-1]

        # REGIME FILTER: Only trade in bull markets (price > 200 SMA)
        sma_200 = bars["close"].rolling(200).mean().iloc[-1]
        if cur_price < sma_200:
            return None

        return self._make_opportunity(
            symbol=symbol,
            bars=bars,
            edge_type=EdgeType.OVERNIGHT_PREMIUM,
            scanner_name="OvernightPremiumScanner",
            direction=Direction.LONG,
            entry=cur_price,
            stop=cur_price * 0.95,   # 5% catastrophic stop (never hit overnight)
            target=cur_price * 1.10,  # Unreachable placeholder
            edge_data={
                "entry_type": "close_to_open",
                "skip_friday": True,
                "regime_filter": "sma_200",
                "sma_200": sma_200,
                "price_vs_sma": cur_price / sma_200,
                "notional_pct": 20,
            },
            score=60,
        )

    def _simulate_overnight_trades(
        self,
        opportunities: List[Tuple[Opportunity, int]],
        bars: pd.DataFrame,
    ) -> List[SimulatedTrade]:
        """Simulate overnight trades directly (bypass TradeSimulator).

        Overnight trades are fundamentally different:
        - Entry at signal bar's CLOSE (not next bar's open)
        - Exit at next bar's OPEN (not bar's close)
        - No stop management (single overnight hold)
        - Execution via MOC/MOO auction orders — near-zero slippage

        The standard simulator can't handle this, so we build
        SimulatedTrade objects directly.
        """
        # MOC/MOO auction execution: slippage is effectively zero.
        # Use a minimal spread to account for any micro-structure cost.
        AUCTION_SPREAD_PCT = 0.005  # 0.5 bps round-trip for SPY auctions

        trades: List[SimulatedTrade] = []

        for opp, signal_idx in opportunities:
            next_idx = signal_idx + 1
            if next_idx >= len(bars):
                continue

            # Entry at today's close (MOC order — auction price)
            entry_price = bars.iloc[signal_idx]["close"]
            entry_time = bars.index[signal_idx]

            # Exit at tomorrow's open (MOO order — auction price)
            exit_price = bars.iloc[next_idx]["open"]
            exit_time = bars.index[next_idx]

            # Score-based sizing multiplier
            multiplier = self._get_risk_multiplier(opp.raw_score)
            if multiplier == 0.0:
                continue  # F-tier: skip trade

            # Position sizing: 20% notional * score multiplier
            notional_pct = opp.edge_data.get("notional_pct", 20)
            position_size = (
                self.simulator.account_balance * notional_pct / 100 * multiplier
            ) / entry_price

            # P&L (always long)
            gross_pnl = (exit_price - entry_price) * position_size

            # Costs: minimal for auction orders
            trade_value = entry_price * position_size
            spread_cost = trade_value * (AUCTION_SPREAD_PCT / 100)
            total_costs = spread_cost + self.simulator.commission_per_trade

            net_pnl = gross_pnl - total_costs
            net_pnl_pct = (net_pnl / self.simulator.account_balance) * 100

            edge = opp.primary_edge
            edge_str = edge.value if hasattr(edge, "value") else str(edge)

            trade = SimulatedTrade(
                opportunity_id=opp.id,
                symbol=opp.symbol,
                direction="long",
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=ExitReason.INDICATOR_EXIT,  # Close-to-open counts as indicator
                position_size=position_size,
                gross_pnl=gross_pnl,
                costs=total_costs,
                net_pnl=net_pnl,
                net_pnl_pct=net_pnl_pct,
                hold_duration=exit_time - entry_time,
                primary_edge=edge_str,
                score=opp.raw_score,
                score_tier=score_to_tier(opp.raw_score),
            )
            trades.append(trade)
            self._compound(trade.net_pnl)

        return trades

    # ---- ORB indicator helpers --------------------------------------------

    @staticmethod
    def _prepare_orb_indicators(bars: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute ORB indicators on 5m bars.

        Adds columns: session_date, bar_num, or_high, or_low, or_bullish,
        in_entry_window, is_eod, relative_volume, daily_atr.
        """
        # Filter to regular US market hours only (14:30-21:00 UTC = 09:30-16:00 ET).
        # Polygon returns pre-market and after-hours bars that corrupt session
        # analysis (bar_num, opening range, is_eod would all be wrong).
        minutes = bars.index.hour * 60 + bars.index.minute
        market_mask = (minutes >= 14 * 60 + 30) & (minutes < 21 * 60)
        bars = bars[market_mask].copy()

        # Session date (UTC date — US market 14:30-21:00 UTC, same day)
        bars["session_date"] = bars.index.floor("D")

        # Bar number within each session (0 = first bar)
        bars["bar_num"] = bars.groupby("session_date").cumcount()

        # Opening range from first bar of each session
        first_bars = bars.groupby("session_date").first()
        bars["or_high"] = bars["session_date"].map(first_bars["high"])
        bars["or_low"] = bars["session_date"].map(first_bars["low"])
        bars["or_bullish"] = bars["session_date"].map(
            first_bars["close"] > first_bars["open"]
        )

        # Entry window: bars 1-5 (within 30 min after opening bar)
        bars["in_entry_window"] = bars["bar_num"].between(1, 5)

        # EOD marker: last bar of each session
        bars["is_eod"] = False
        last_idx = bars.groupby("session_date").tail(1).index
        bars.loc[last_idx, "is_eod"] = True

        # Relative volume: first bar vol / 20-session rolling avg (excluding today)
        first_vol = first_bars["volume"]
        avg_vol = first_vol.shift(1).rolling(20, min_periods=5).mean()
        rel_vol = first_vol / avg_vol
        bars["relative_volume"] = bars["session_date"].map(rel_vol)

        # Daily ATR from per-session OHLC
        daily = bars.groupby("session_date").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        tr = pd.concat(
            [
                daily["high"] - daily["low"],
                (daily["high"] - daily["close"].shift(1)).abs(),
                (daily["low"] - daily["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        daily_atr = tr.rolling(14, min_periods=5).mean()
        bars["daily_atr"] = bars["session_date"].map(daily_atr)

        return bars

    @staticmethod
    def _orb_exit_checker(bar: pd.Series, is_long: bool) -> bool:
        """ORB end-of-day exit — close all positions at session end."""
        return bool(bar.get("is_eod", False))

    # ---- RSI(2) indicator helpers ----------------------------------------

    @staticmethod
    def _calculate_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        ADX measures trend strength:
        - ADX < 20: Weak/no trend (mean reversion optimal)
        - ADX 20-25: Developing trend (caution)
        - ADX > 25: Strong trend (skip mean reversion)
        - ADX > 40: Very strong trend (definitely skip)
        """
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    @staticmethod
    def _prepare_forex_indicators(bars: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute ADX(14) and ATR(14) for forex session edge filtering."""
        bars["adx_14"] = BacktestEngine._calculate_adx(bars, 14)

        # ATR for range quality filtering
        tr = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - bars["close"].shift(1)).abs(),
                (bars["low"] - bars["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        bars["atr_14"] = tr.rolling(14).mean()

        return bars

    @staticmethod
    def _prepare_rsi_indicators(bars: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute RSI(2), 200 SMA, 5 SMA, and ADX for Connors strategy."""
        delta = bars["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=2, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=2, adjust=False).mean()
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        bars["rsi_2"] = 100 - (100 / (1 + rs))
        bars["sma_200"] = bars["close"].rolling(window=200).mean()
        bars["sma_5"] = bars["close"].rolling(window=5).mean()

        # ADX for regime filtering (mean reversion only works in low-trend environments)
        bars["adx_14"] = BacktestEngine._calculate_adx(bars, 14)

        return bars

    @staticmethod
    def _rsi_exit_checker(bar: pd.Series, is_long: bool) -> bool:
        """Connors RSI(2) indicator-based exit.

        Long exit:  RSI(2) > 50  OR  close > 5 SMA
        Short exit: RSI(2) < 50  OR  close < 5 SMA
        """
        rsi = bar.get("rsi_2")
        sma_5 = bar.get("sma_5")
        close = bar["close"]

        if pd.isna(rsi) or pd.isna(sma_5):
            return False

        if is_long:
            return rsi > 50 or close > sma_5
        else:
            return rsi < 50 or close < sma_5

    # ---- helpers -------------------------------------------------------

    def _get_stop_target_multipliers(self, timeframe: str) -> Tuple[float, float]:
        """Get ATR multipliers for stop/target based on timeframe."""
        return self.TIMEFRAME_ATR_MULTIPLIERS.get(timeframe, (1.5, 2.5))

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        tr = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - bars["close"].shift(1)).abs(),
                (bars["low"] - bars["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _get_market(self, symbol: str) -> Market:
        """Determine market type from symbol."""
        if "/" in symbol or symbol in {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD"}:
            return Market.FOREX_MAJORS
        return Market.US_STOCKS

    @staticmethod
    def _is_trading_day(ts) -> bool:
        """Check if date is a trading day (Mon-Fri)."""
        return ts.weekday() < 5

    def _get_tom_day(self, ts) -> int:
        """
        Get Turn of Month day number.

        Returns:
            -1: Last trading day of month
            1, 2, 3: First three days of new month
            0: Not in TOM window
        """
        if ts.day <= 3:
            return ts.day
        for d in range(1, 4):
            future = ts + timedelta(days=d)
            if future.weekday() < 5:
                return -1 if future.month != ts.month else 0
        return 0

    def _simulate_insider_signal(self, bars: pd.DataFrame, symbol: str) -> bool:
        """Deterministic pseudo-random insider signal.

        Generates ~2-3 signals per month per symbol using a date-seeded
        hash so results are reproducible across runs.
        """
        ts = bars.index[-1]
        seed_str = f"{symbol}_{ts.strftime('%Y-%m-%d')}"
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
        # ~10% chance per trading day ≈ 2-3 signals per month
        return (hash_val % 100) < 10

    @staticmethod
    def _make_opportunity(
        *,
        symbol: str,
        bars: pd.DataFrame,
        edge_type: EdgeType,
        scanner_name: str,
        direction: Direction,
        entry: float,
        stop: float,
        target: float,
        edge_data: dict,
        score: int,
    ) -> Opportunity:
        """Build an Opportunity from signal parameters."""
        ts = bars.index[-1]
        detected_at = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

        return Opportunity(
            id=str(uuid.uuid4()),
            detected_at=detected_at,
            scanner=scanner_name,
            symbol=symbol,
            market=_market_for_symbol(symbol),
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            primary_edge=edge_type,
            secondary_edges=[],
            edge_data=edge_data,
            raw_score=score,
        )

    def _build_equity_curve(
        self, trades: List[SimulatedTrade]
    ) -> pd.DataFrame:
        """Build equity curve from chronologically-sorted trades."""
        if not trades:
            return pd.DataFrame()

        equity = self.starting_balance
        rows = []

        for trade in sorted(trades, key=lambda t: t.exit_time):
            equity += trade.net_pnl
            rows.append(
                {
                    "timestamp": trade.exit_time,
                    "equity": equity,
                    "trade_pnl": trade.net_pnl,
                    "trade_pnl_pct": trade.net_pnl_pct,
                }
            )

        return pd.DataFrame(rows).set_index("timestamp")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

async def main() -> None:
    """Run backtest from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="NEXUS Backtest Engine")
    parser.add_argument("--edge", type=str, help="Edge type to test")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--all", action="store_true", help="Test all edges")

    args = parser.parse_args()
    engine = BacktestEngine()

    if args.all:
        results = await engine.run_all_edges(args.start, args.end)

        print("\n" + "=" * 70)
        print("NEXUS EDGE VALIDATION REPORT")
        print("=" * 70)

        for edge_name, result in results.items():
            s = result.statistics
            print(f"\n{edge_name}:")
            print(f"  Verdict: {s.verdict}")
            print(f"  Trades: {s.total_trades}")
            print(f"  Win Rate: {s.win_rate:.1f}%")
            print(f"  Profit Factor: {s.profit_factor:.2f}")
            print(f"  Total P&L: {s.total_pnl:.2f} ({s.total_pnl_pct:.1f}%)")
            print(f"  Max Drawdown: {s.max_drawdown_pct:.1f}%")
            print(f"  Reason: {s.verdict_reason}")
    else:
        edge_type = EdgeType(args.edge) if args.edge else EdgeType.VWAP_DEVIATION

        result = await engine.run_edge_backtest(
            edge_type=edge_type,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )

        s = result.statistics

        print("\n" + "=" * 70)
        print(f"BACKTEST RESULT: {edge_type.value}")
        print("=" * 70)
        print(f"Symbol: {s.symbol}")
        print(f"Timeframe: {s.timeframe}")
        print(f"Period: {s.test_period}")
        print("-" * 70)
        print(f"Total Trades: {s.total_trades}")
        print(f"Winners: {s.winners} | Losers: {s.losers}")
        print(f"Win Rate: {s.win_rate:.1f}%")
        print("-" * 70)
        print(f"Total P&L: {s.total_pnl:.2f} ({s.total_pnl_pct:.1f}%)")
        print(f"Profit Factor: {s.profit_factor:.2f}")
        print(f"Expected Value: {s.expected_value:.2f} ({s.expected_value_pct:.3f}%)")
        print("-" * 70)
        print(f"Max Drawdown: {s.max_drawdown:.2f} ({s.max_drawdown_pct:.1f}%)")
        print(f"Sharpe Ratio: {s.sharpe_ratio:.2f}")
        print(f"Avg Hold: {s.avg_hold_duration:.1f} hours")
        print("-" * 70)
        print(f"T-Statistic: {s.t_statistic:.2f}")
        print(f"P-Value: {s.p_value:.4f}")
        print(f"Statistically Significant: {s.is_significant}")
        print("=" * 70)
        print(f"VERDICT: {s.verdict}")
        print(f"REASON: {s.verdict_reason}")
        print("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
