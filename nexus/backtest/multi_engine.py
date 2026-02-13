"""
Multi-Asset Backtest Engine

Orchestrates the complete multi-asset, multi-timeframe backtesting process:
1. Load historical data for multiple instruments
2. Scan data for signals using inline edge logic
3. Simulate trades via BacktestSimulator
4. Aggregate results by edge, timeframe, and market
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from nexus.core.enums import EdgeType, Direction, Market, Timeframe
from nexus.core.models import Opportunity
from nexus.data.instruments import get_instrument_registry, InstrumentType

from .historical_loader import HistoricalDataLoader
from .simulator import BacktestSimulator
from .models import BacktestTrade, MultiBacktestResult, EdgePerformance

logger = logging.getLogger(__name__)


class MultiBacktestEngine:
    """
    Multi-asset backtesting engine.

    Runs edge scanners against historical data across multiple instruments
    and timeframes, then simulates trading.
    """

    def __init__(
        self,
        starting_balance: float = 100_000,
        risk_per_trade: float = 1.0,
        max_positions: int = 10,
        slippage_pct: float = 0.02,
        polygon_api_key: Optional[str] = None,
    ):
        self.starting_balance = starting_balance
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

        self.data_loader = HistoricalDataLoader(polygon_api_key)
        self.simulator = BacktestSimulator(slippage_pct=slippage_pct)
        self.registry = get_instrument_registry()

        # Results storage
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []

    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[Timeframe]] = None,
        edge_types: Optional[List[EdgeType]] = None,
        progress_callback: Optional[callable] = None,
    ) -> MultiBacktestResult:
        """
        Run complete multi-asset backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: Symbols to test (None = top 50 US stocks)
            timeframes: Timeframes to test (None = H1, H4, D1)
            edge_types: Edge types to test (None = all supported)
            progress_callback: Optional callback(current, total, symbol, timeframe)

        Returns:
            MultiBacktestResult with all trades and performance metrics
        """
        logger.info(f"Starting multi-asset backtest from {start_date} to {end_date}")

        if symbols is None:
            us_instruments = self.registry.get_by_type(InstrumentType.STOCK)
            symbols = [i.symbol for i in us_instruments[:50]]

        if timeframes is None:
            timeframes = [Timeframe.H1, Timeframe.H4, Timeframe.D1]

        # Reset state
        self.trades = []
        self.equity_curve = []
        current_equity = self.starting_balance

        total_iterations = len(symbols) * len(timeframes)
        current_iteration = 0

        for symbol in symbols:
            for timeframe in timeframes:
                current_iteration += 1

                if progress_callback:
                    progress_callback(current_iteration, total_iterations, symbol, timeframe)

                try:
                    # Load data
                    data = await self.data_loader.load_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    if data is None or len(data) < 50:
                        continue

                    # Generate signals
                    signals = self._generate_signals(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=data,
                        edge_types=edge_types,
                    )

                    # Simulate each signal
                    for signal_idx, signal in signals:
                        trade = self.simulator.simulate_trade(
                            opportunity=signal,
                            data=data,
                            entry_bar_idx=signal_idx,
                            starting_equity=current_equity,
                            risk_pct=self.risk_per_trade,
                        )

                        if trade:
                            self.trades.append(trade)
                            current_equity += trade.net_pnl

                            self.equity_curve.append({
                                "timestamp": trade.exit_time,
                                "equity": current_equity,
                                "trade_pnl": trade.net_pnl,
                                "symbol": symbol,
                                "edge": trade.edge_type.value if hasattr(trade.edge_type, "value") else str(trade.edge_type),
                            })

                except Exception as e:
                    logger.error(f"Error processing {symbol}/{timeframe.value}: {e}")
                    continue

        result = self._build_result(start_date, end_date, current_equity)

        logger.info(
            f"Multi-asset backtest complete: {result.total_trades} trades, "
            f"net P&L: ${result.net_pnl:,.2f} ({result.net_return_pct:.2f}%)"
        )

        return result

    def _generate_signals(
        self,
        symbol: str,
        timeframe: Timeframe,
        data: pd.DataFrame,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> List[tuple]:
        """Generate trading signals by scanning historical data bar-by-bar."""
        signals = []

        instrument = self.registry.get(symbol)
        market = Market.US_STOCKS
        if instrument:
            market_str = getattr(instrument, "market", None)
            if market_str:
                try:
                    market = Market(market_str) if isinstance(market_str, str) else market_str
                except ValueError:
                    pass

        edges_to_scan = edge_types or [
            EdgeType.RSI_EXTREME,
            EdgeType.VWAP_DEVIATION,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.GAP_FILL,
            EdgeType.TURN_OF_MONTH,
        ]

        for edge_type in edges_to_scan:
            try:
                edge_signals = self._scan_data_for_edge(
                    data=data,
                    symbol=symbol,
                    market=market,
                    timeframe=timeframe,
                    edge_type=edge_type,
                )
                signals.extend(edge_signals)
            except Exception as e:
                logger.debug(f"Scanner {edge_type.value} error on {symbol}: {e}")

        return signals

    def _scan_data_for_edge(
        self,
        data: pd.DataFrame,
        symbol: str,
        market: Market,
        timeframe: Timeframe,
        edge_type: EdgeType,
    ) -> List[tuple]:
        """Scan historical data for a specific edge."""
        signals = []

        if len(data) < 50:
            return signals

        # Calculate indicators once
        df = data.copy()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["rsi_2"] = self._calculate_rsi(df["close"], 2)
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)
        df["atr"] = self._calculate_atr(df, 14)
        df["vwap"] = self._calculate_vwap(df)
        df["bb_upper"], df["bb_lower"] = self._calculate_bollinger(df)

        # Scan bars (skip first 50 for indicator warmup)
        for i in range(50, len(df) - 1):
            bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]

            signal = None

            if edge_type == EdgeType.RSI_EXTREME:
                signal = self._check_rsi_extreme(bar, prev_bar, symbol, market, timeframe)

            elif edge_type == EdgeType.VWAP_DEVIATION:
                signal = self._check_vwap_deviation(bar, df, i, symbol, market, timeframe)

            elif edge_type == EdgeType.BOLLINGER_TOUCH:
                signal = self._check_bollinger_touch(bar, prev_bar, symbol, market, timeframe)

            elif edge_type == EdgeType.GAP_FILL:
                signal = self._check_gap(df, i, symbol, market, timeframe)

            elif edge_type == EdgeType.TURN_OF_MONTH:
                signal = self._check_turn_of_month(bar, symbol, market, timeframe)

            if signal:
                signals.append((i, signal))

        return signals

    # ------------------------------------------------------------------
    # Signal checks
    # ------------------------------------------------------------------

    def _check_rsi_extreme(
        self, bar, prev_bar, symbol: str, market: Market, timeframe: Timeframe
    ) -> Optional[Opportunity]:
        rsi = bar["rsi_2"]
        if pd.isna(rsi):
            return None

        atr = bar["atr"] if not pd.isna(bar["atr"]) else bar["close"] * 0.02

        if rsi < 20:
            return Opportunity(
                id=f"rsi_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="RSIBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.LONG,
                entry_price=bar["close"],
                stop_loss=bar["close"] - (atr * 1.5),
                take_profit=bar["close"] + (atr * 2.0),
                primary_edge=EdgeType.RSI_EXTREME,
                edge_data={
                    "rsi": rsi,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=65,
            )

        elif rsi > 80:
            return Opportunity(
                id=f"rsi_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="RSIBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.SHORT,
                entry_price=bar["close"],
                stop_loss=bar["close"] + (atr * 1.5),
                take_profit=bar["close"] - (atr * 2.0),
                primary_edge=EdgeType.RSI_EXTREME,
                edge_data={
                    "rsi": rsi,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=65,
            )

        return None

    def _check_vwap_deviation(
        self, bar, data: pd.DataFrame, idx: int, symbol: str, market: Market, timeframe: Timeframe
    ) -> Optional[Opportunity]:
        vwap = bar["vwap"]
        close = bar["close"]

        if pd.isna(vwap) or vwap == 0:
            return None

        recent_data = data.iloc[max(0, idx - 20):idx + 1]
        vwap_std = (recent_data["close"] - recent_data["vwap"]).std()

        if vwap_std == 0:
            return None

        deviation = (close - vwap) / vwap_std
        atr = bar["atr"] if not pd.isna(bar["atr"]) else close * 0.02

        if deviation < -2.0:
            return Opportunity(
                id=f"vwap_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="VWAPBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.LONG,
                entry_price=close,
                stop_loss=close - (atr * 1.5),
                take_profit=vwap,
                primary_edge=EdgeType.VWAP_DEVIATION,
                edge_data={
                    "vwap": vwap,
                    "deviation": deviation,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=70,
            )

        elif deviation > 2.0:
            return Opportunity(
                id=f"vwap_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="VWAPBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.SHORT,
                entry_price=close,
                stop_loss=close + (atr * 1.5),
                take_profit=vwap,
                primary_edge=EdgeType.VWAP_DEVIATION,
                edge_data={
                    "vwap": vwap,
                    "deviation": deviation,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=70,
            )

        return None

    def _check_bollinger_touch(
        self, bar, prev_bar, symbol: str, market: Market, timeframe: Timeframe
    ) -> Optional[Opportunity]:
        close = bar["close"]
        bb_upper = bar["bb_upper"]
        bb_lower = bar["bb_lower"]

        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None

        atr = bar["atr"] if not pd.isna(bar["atr"]) else close * 0.02
        mid = (bb_upper + bb_lower) / 2

        prev_bb_lower = prev_bar["bb_lower"] if not pd.isna(prev_bar["bb_lower"]) else bb_lower
        prev_bb_upper = prev_bar["bb_upper"] if not pd.isna(prev_bar["bb_upper"]) else bb_upper

        if close <= bb_lower and prev_bar["close"] > prev_bb_lower:
            return Opportunity(
                id=f"bb_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="BollingerBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.LONG,
                entry_price=close,
                stop_loss=close - (atr * 1.5),
                take_profit=mid,
                primary_edge=EdgeType.BOLLINGER_TOUCH,
                edge_data={
                    "bb_lower": bb_lower,
                    "bb_upper": bb_upper,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=60,
            )

        if close >= bb_upper and prev_bar["close"] < prev_bb_upper:
            return Opportunity(
                id=f"bb_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="BollingerBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.SHORT,
                entry_price=close,
                stop_loss=close + (atr * 1.5),
                take_profit=mid,
                primary_edge=EdgeType.BOLLINGER_TOUCH,
                edge_data={
                    "bb_lower": bb_lower,
                    "bb_upper": bb_upper,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=60,
            )

        return None

    def _check_gap(
        self, data: pd.DataFrame, idx: int, symbol: str, market: Market, timeframe: Timeframe
    ) -> Optional[Opportunity]:
        if idx < 1:
            return None

        bar = data.iloc[idx]
        prev_bar = data.iloc[idx - 1]

        prev_close = prev_bar["close"]
        current_open = bar["open"]

        gap_pct = (current_open - prev_close) / prev_close * 100
        atr = bar["atr"] if not pd.isna(bar["atr"]) else bar["close"] * 0.02

        if gap_pct < -0.5 and gap_pct > -3.0:
            return Opportunity(
                id=f"gap_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="GapBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.LONG,
                entry_price=current_open,
                stop_loss=current_open - (atr * 2.0),
                take_profit=prev_close,
                primary_edge=EdgeType.GAP_FILL,
                edge_data={
                    "gap_pct": gap_pct,
                    "prev_close": prev_close,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=65,
            )

        if gap_pct > 0.5 and gap_pct < 3.0:
            return Opportunity(
                id=f"gap_{symbol}_{bar['timestamp']}",
                detected_at=bar["timestamp"],
                scanner="GapBacktest",
                symbol=symbol,
                market=market,
                direction=Direction.SHORT,
                entry_price=current_open,
                stop_loss=current_open + (atr * 2.0),
                take_profit=prev_close,
                primary_edge=EdgeType.GAP_FILL,
                edge_data={
                    "gap_pct": gap_pct,
                    "prev_close": prev_close,
                    "timeframe": timeframe.value,
                    "timeframe_minutes": timeframe.minutes,
                },
                raw_score=65,
            )

        return None

    def _check_turn_of_month(
        self, bar, symbol: str, market: Market, timeframe: Timeframe
    ) -> Optional[Opportunity]:
        timestamp = pd.Timestamp(bar["timestamp"])
        day = timestamp.day

        is_tom_window = day >= 28 or day <= 3
        if not is_tom_window:
            return None

        if timeframe != Timeframe.D1:
            return None

        atr = bar["atr"] if not pd.isna(bar["atr"]) else bar["close"] * 0.02

        return Opportunity(
            id=f"tom_{symbol}_{bar['timestamp']}",
            detected_at=bar["timestamp"],
            scanner="TOMBacktest",
            symbol=symbol,
            market=market,
            direction=Direction.LONG,
            entry_price=bar["close"],
            stop_loss=bar["close"] - (atr * 2.0),
            take_profit=bar["close"] + (atr * 3.0),
            primary_edge=EdgeType.TURN_OF_MONTH,
            edge_data={
                "day": day,
                "timeframe": timeframe.value,
                "timeframe_minutes": timeframe.minutes,
            },
            raw_score=75,
        )

    # ------------------------------------------------------------------
    # Technical indicator calculations
    # ------------------------------------------------------------------

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
        return vwap

    def _calculate_bollinger(
        self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> tuple:
        sma = data["close"].rolling(period).mean()
        std = data["close"].rolling(period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    def _build_result(
        self,
        start_date: datetime,
        end_date: datetime,
        ending_equity: float,
    ) -> MultiBacktestResult:
        result = MultiBacktestResult(
            start_date=start_date,
            end_date=end_date,
            starting_balance=self.starting_balance,
            ending_balance=ending_equity,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

        result.total_trades = len(self.trades)
        result.total_winners = sum(1 for t in self.trades if t.is_winner)
        result.total_losers = sum(1 for t in self.trades if t.is_loser)
        result.gross_pnl = sum(t.gross_pnl for t in self.trades)
        result.total_costs = sum(t.costs for t in self.trades)
        result.net_pnl = sum(t.net_pnl for t in self.trades)

        result.max_drawdown, result.max_drawdown_pct = self._calculate_max_drawdown()
        result.edge_performance = self._aggregate_by_edge()
        result.timeframe_performance = self._aggregate_by_timeframe()
        result.market_performance = self._aggregate_by_market()

        return result

    def _calculate_max_drawdown(self) -> tuple:
        if not self.equity_curve:
            return 0.0, 0.0

        equities = [self.starting_balance] + [e["equity"] for e in self.equity_curve]

        peak = equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for equity in equities:
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            drawdown_pct = drawdown / peak * 100 if peak > 0 else 0.0

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct

        return max_dd, max_dd_pct

    def _aggregate_by_edge(self) -> Dict[str, EdgePerformance]:
        performance = {}

        for edge_type in EdgeType:
            edge_val = edge_type.value
            edge_trades = [
                t for t in self.trades
                if (t.edge_type.value if hasattr(t.edge_type, "value") else str(t.edge_type)) == edge_val
            ]

            if not edge_trades:
                continue

            perf = EdgePerformance(edge_type=edge_type)
            perf.total_trades = len(edge_trades)
            perf.winners = sum(1 for t in edge_trades if t.is_winner)
            perf.losers = sum(1 for t in edge_trades if t.is_loser)
            perf.breakeven = perf.total_trades - perf.winners - perf.losers
            perf.gross_pnl = sum(t.gross_pnl for t in edge_trades)
            perf.total_costs = sum(t.costs for t in edge_trades)
            perf.net_pnl = sum(t.net_pnl for t in edge_trades)
            perf.total_gross_pct = sum(t.gross_pnl_pct for t in edge_trades)
            perf.total_net_pct = sum(t.net_pnl_pct for t in edge_trades)
            perf.total_risk = sum(t.risk_amount for t in edge_trades)
            perf.total_r_multiple = sum(t.r_multiple for t in edge_trades)
            perf.total_hold_minutes = sum(t.hold_duration_minutes for t in edge_trades)
            perf.avg_hold_minutes = perf.total_hold_minutes / perf.total_trades

            performance[edge_val] = perf

        return performance

    def _aggregate_by_timeframe(self) -> Dict[str, EdgePerformance]:
        performance = {}

        for tf in Timeframe:
            tf_val = tf.value
            tf_trades = [
                t for t in self.trades
                if (t.timeframe.value if hasattr(t.timeframe, "value") else str(t.timeframe)) == tf_val
            ]

            if not tf_trades:
                continue

            perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME, timeframe=tf)
            perf.total_trades = len(tf_trades)
            perf.winners = sum(1 for t in tf_trades if t.is_winner)
            perf.losers = sum(1 for t in tf_trades if t.is_loser)
            perf.net_pnl = sum(t.net_pnl for t in tf_trades)
            perf.total_net_pct = sum(t.net_pnl_pct for t in tf_trades)
            perf.total_r_multiple = sum(t.r_multiple for t in tf_trades)

            performance[tf_val] = perf

        return performance

    def _aggregate_by_market(self) -> Dict[str, EdgePerformance]:
        performance = {}

        for market in Market:
            market_val = market.value
            market_trades = [
                t for t in self.trades
                if (t.market.value if hasattr(t.market, "value") else str(t.market)) == market_val
            ]

            if not market_trades:
                continue

            perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME, market=market)
            perf.total_trades = len(market_trades)
            perf.winners = sum(1 for t in market_trades if t.is_winner)
            perf.losers = sum(1 for t in market_trades if t.is_loser)
            perf.net_pnl = sum(t.net_pnl for t in market_trades)
            perf.total_net_pct = sum(t.net_pnl_pct for t in market_trades)
            perf.total_r_multiple = sum(t.r_multiple for t in market_trades)

            performance[market_val] = perf

        return performance
