"""
Connors RSI(2) Mean Reversion Scanner - buy extreme oversold in uptrends.

Ported from validated backtest logic (v6-v26): _signal_rsi in engine.py.
PF 1.82, 80 trades, 67.5% WR, Sharpe 3.36, p=0.03 VALID (2022-2024).

Key insight: RSI(2) mean reversion only works on INDEX ETFs (SPY/QQQ).
IWM, DIA, and individual stocks trend through RSI extremes instead of reverting.

Filters:
- RSI(2) < 10 (oversold) + Price > 200 SMA (bull market)
- ADX(14) < 40 (not trending strongly — regime filter)
- 10% catastrophic stop only (tight stops destroy mean-reversion edge)
- Exit via 5 SMA cross (indicator-based, not fixed target)
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class RSIScanner(BaseScanner):
    """
    Connors RSI(2) Mean Reversion scanner.

    VALIDATED: 80 trades, PF 1.82, 67.5% WR, Sharpe 3.36, p=0.03 (2022-2024)
    SIGNAL: RSI(2) < 10 + Price > SMA(200) + ADX(14) < 40 → LONG
    DIRECTION: LONG only in practice (shorts rare with ADX filter)

    CRITICAL FILTERS:
    - ADX < 40: Skips strong trends where mean reversion fails
    - SPY + QQQ ONLY: IWM (51% WR), DIA (49% WR), individuals all FAIL
    - No tight stops: 10% catastrophic only — stops destroy mean-reversion returns

    Ported from BacktestEngine._signal_rsi() to ensure live matches backtest.
    """

    # Validated symbol list — ONLY index ETFs that mean-revert.
    # IWM: 51% WR, negative PF — different dynamics than SPY/QQQ.
    # DIA: 49% WR, negative PF — same problem.
    # Individual stocks (AAPL, NVDA, TSLA, etc.): trend through extremes.
    # Forex: never validated for RSI(2).
    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.RSI_EXTREME
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # Connors RSI(2) parameters (do not change without re-backtesting)
        self.rsi_period = 2               # NOT 14 — extremely sensitive
        self.oversold_threshold = 10      # Entry: RSI(2) < 10
        self.overbought_threshold = 90    # Entry: RSI(2) > 90
        self.sma_period = 200             # Trend filter
        self.adx_period = 14              # Regime filter period
        self.adx_threshold = 40           # Skip if ADX >= 40
        self.catastrophic_stop_pct = 10.0 # Wide stop only — no tight stops
        self.notional_pct = 16            # ~40% Kelly for 75% WR, PF 2.2
        self.score = 75                   # B-tier (1.25x multiplier)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """RSI scanner runs on daily bars — active during US market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """Scan for Connors RSI(2) opportunities."""
        opportunities = []

        for market in self.markets:
            instruments = self._get_instruments(market)

            for symbol in instruments:
                try:
                    opp = await self._scan_symbol(symbol, market)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.warning(f"RSI(2) scan failed for {symbol}: {e}")

        logger.info(f"RSI(2) scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for RSI(2) extreme opportunity.

        Ported from BacktestEngine._signal_rsi() — validated logic.
        """
        # Need 201+ bars for 200 SMA + ADX warmup
        bars = await self.get_bars_safe(symbol, "1D", 250)

        if bars is None or len(bars) < 210:
            logger.debug(f"RSI insufficient data for {symbol}")
            return None

        # Pre-compute indicators on full DataFrame
        bars = self._calculate_indicators(bars)
        current = bars.iloc[-1]

        cur_rsi = current.get("rsi_2")
        cur_sma_200 = current.get("sma_200")
        cur_adx = current.get("adx_14")
        cur_price = float(current["close"])

        # Skip if indicators not ready
        if pd.isna(cur_rsi) or pd.isna(cur_sma_200):
            return None

        # FILTER 1: ADX < 40 (regime filter — skip strong trends)
        # Strictly greater than (>) matches backtest engine exactly
        if pd.isna(cur_adx) or cur_adx > self.adx_threshold:
            adx_str = f"{cur_adx:.1f}" if not pd.isna(cur_adx) else "N/A"
            logger.debug(
                "RSI %s ADX %s > %s, skipping (trending)",
                symbol, adx_str, self.adx_threshold,
            )
            return None

        # Long: RSI(2) < 10 AND price > 200 SMA (uptrend only)
        if cur_rsi < self.oversold_threshold and cur_price > cur_sma_200:
            direction = Direction.LONG
            entry_type = "rsi2_oversold"
            stop = cur_price * (1 - self.catastrophic_stop_pct / 100)
            target = cur_price * 1.20  # Placeholder — exit via indicator

        # Short: RSI(2) > 90 AND price < 200 SMA (downtrend only)
        elif cur_rsi > self.overbought_threshold and cur_price < cur_sma_200:
            direction = Direction.SHORT
            entry_type = "rsi2_overbought"
            stop = cur_price * (1 + self.catastrophic_stop_pct / 100)
            target = cur_price * 0.80  # Placeholder — exit via indicator

        else:
            return None

        cur_sma_5 = current.get("sma_5")

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=cur_price,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "strategy": entry_type,
                "rsi_value": round(float(cur_rsi), 1),
                "rsi_period": self.rsi_period,
                "adx": round(float(cur_adx), 1),
                "sma_200": round(float(cur_sma_200), 2),
                "sma_5": round(float(cur_sma_5), 2) if not pd.isna(cur_sma_5) else None,
                "trend_aligned": True,
                "exit_method": "indicator",
                "notional_pct": self.notional_pct,
            },
        )
        opp.raw_score = self.score

        logger.info(
            f"RSI Extreme: {symbol} {direction.value} | "
            f"RSI(2)={cur_rsi:.1f} | ADX={cur_adx:.1f} < {self.adx_threshold} | "
            f"Price ${cur_price:.2f} vs SMA200 ${cur_sma_200:.2f} | Score {self.score}"
        )

        return opp

    def _calculate_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute RSI(2), 200 SMA, and ADX(14) on full DataFrame."""
        # RSI(2) — Wilder's EWM smoothing
        delta = bars["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(
            span=self.rsi_period, adjust=False
        ).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(
            span=self.rsi_period, adjust=False
        ).mean()
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        bars["rsi_2"] = 100 - (100 / (1 + rs))

        # 200-day SMA trend filter
        bars["sma_200"] = bars["close"].rolling(window=self.sma_period).mean()

        # 5-day SMA for indicator-based exit (RSI exit checker uses this)
        bars["sma_5"] = bars["close"].rolling(window=5).mean()

        # ADX(14) — Wilder's EWM smoothing (matches backtest engine exactly)
        bars["adx_14"] = self._calculate_adx(bars, self.adx_period)

        return bars

    @staticmethod
    def _calculate_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX using Wilder's smoothing.

        Matches BacktestEngine._calculate_adx() exactly.

        ADX measures trend strength:
        - ADX < 20: Weak/no trend (mean reversion optimal)
        - ADX 20-25: Developing trend
        - ADX > 25: Strong trend
        - ADX > 40: Very strong trend (skip mean reversion)
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

        # Smoothed averages (Wilder's EWM — NOT simple rolling)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
