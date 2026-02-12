"""
VWAP Trend-Following Scanner - trade WITH VWAP crossover momentum.

Ported from validated backtest logic (v12-v26): _signal_vwap in engine.py.
PF 1.30 (scored), +$828 on $10k (2022-2024), 142 trades, 40.1% WR.

Key insight: The original mean reversion approach (fade 2σ deviation on 5m bars)
produced -12% P&L in v11. Reversing to trend-following (trade WITH crossover on
daily bars) flipped it to profitable.

Strategy: Price crossing rolling VWAP with volume confirmation = momentum entry.
Symbols: SPY, QQQ, IWM only (individual stocks fail — too noisy).
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class VWAPScanner(BaseScanner):
    """
    VWAP Trend-Following scanner - trade WITH VWAP crossover.

    VALIDATED: 142 trades, PF 1.30, +$828, Sharpe 1.87 (2022-2024)
    SIGNAL: Price crosses rolling VWAP + volume confirmation
    DIRECTION: LONG on cross above, SHORT on cross below (trend-following)

    WARNING: Do NOT revert to mean reversion (fade deviation) — that lost -12%.
    Ported from BacktestEngine._signal_vwap() to ensure live matches backtest.
    """

    # Validated symbol list — only broad indices work.
    # Individual stocks (AAPL, MSFT, NVDA, TSLA, AMD) FAIL: too noisy.
    # UK stocks were never validated.
    INSTRUMENTS = {
        Market.US_STOCKS: ["SPY", "QQQ", "IWM"],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.VWAP_DEVIATION
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # Validated parameters (do not change without re-backtesting)
        self.vwap_period = 10         # Rolling 10-day VWAP
        self.min_deviation_pct = 0.3  # Minimum 0.3% from VWAP
        self.notional_pct = 16        # Position size (% of capital)
        self.score = 65               # B-tier (1.25x multiplier with score-sizing)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """VWAP scanner runs on daily bars — active during US market hours."""
        return True

    async def scan(self) -> List[Opportunity]:
        """Scan for VWAP trend-following opportunities."""
        opportunities = []

        for market in self.markets:
            instruments = self._get_instruments(market)

            for symbol in instruments:
                try:
                    opp = await self._scan_symbol(symbol, market)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.warning(f"VWAP scan failed for {symbol}: {e}")

        logger.info(f"VWAP scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for VWAP crossover opportunity.

        Ported from BacktestEngine._signal_vwap() — validated logic.

        CRITICAL: This is TREND-FOLLOWING, not mean reversion.
        Price crosses ABOVE VWAP → LONG, crosses BELOW → SHORT.
        """
        # Need enough bars for rolling VWAP (10) + volume average (20) + buffer
        bars = await self.get_bars_safe(symbol, "1D", 30)

        if bars is None or len(bars) < 20:
            logger.debug(f"VWAP insufficient data for {symbol}")
            return None

        # Calculate rolling VWAP (volume-weighted, 10-day window)
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3

        if "volume" in bars.columns:
            cum_vol = bars["volume"].rolling(self.vwap_period).sum()
            cum_tp_vol = (typical_price * bars["volume"]).rolling(self.vwap_period).sum()
            vwap = cum_tp_vol / cum_vol
        else:
            # Fallback: use typical price SMA
            vwap = typical_price.rolling(self.vwap_period).mean()

        cur_price = float(bars["close"].iloc[-1])
        prev_price = float(bars["close"].iloc[-2])
        cur_vwap = vwap.iloc[-1]
        prev_vwap = vwap.iloc[-2]

        if pd.isna(cur_vwap) or pd.isna(prev_vwap):
            return None

        cur_vwap = float(cur_vwap)
        prev_vwap = float(prev_vwap)

        # Calculate deviation from VWAP
        deviation_pct = ((cur_price - cur_vwap) / cur_vwap) * 100

        # Volume confirmation
        vol_ratio = 1.0
        if "volume" in bars.columns:
            avg_vol = bars["volume"].rolling(20).mean().iloc[-1]
            cur_vol = bars["volume"].iloc[-1]
            vol_ratio = float(cur_vol / avg_vol) if avg_vol > 0 else 1.0

        # TREND-FOLLOWING CROSSOVER LOGIC (ported from backtest engine)

        # LONG: Price crosses ABOVE VWAP (bullish breakout)
        if prev_price < prev_vwap and cur_price > cur_vwap:
            if deviation_pct >= self.min_deviation_pct and vol_ratio >= 1.0:
                entry = cur_price
                stop = cur_vwap * 0.995   # 0.5% below VWAP
                risk = entry - stop
                target = entry + (risk * 2.5)  # 2.5:1 R:R
                direction = Direction.LONG
                cross_direction = "bullish"
            else:
                return None

        # SHORT: Price crosses BELOW VWAP (bearish breakdown)
        elif prev_price > prev_vwap and cur_price < cur_vwap:
            if deviation_pct <= -self.min_deviation_pct and vol_ratio >= 1.0:
                entry = cur_price
                stop = cur_vwap * 1.005   # 0.5% above VWAP
                risk = stop - entry
                target = entry - (risk * 2.5)  # 2.5:1 R:R
                direction = Direction.SHORT
                cross_direction = "bearish"
            else:
                return None

        else:
            return None

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "strategy": "trend_following",
                "vwap": round(cur_vwap, 2),
                "deviation_pct": round(deviation_pct, 2),
                "volume_ratio": round(vol_ratio, 2),
                "cross_direction": cross_direction,
                "notional_pct": self.notional_pct,
            },
        )
        opp.raw_score = self.score

        logger.info(
            f"VWAP Crossover: {symbol} {direction.value} ({cross_direction}) | "
            f"Price ${entry:.2f} vs VWAP ${cur_vwap:.2f} ({deviation_pct:+.2f}%) | "
            f"Vol {vol_ratio:.1f}x | Score {self.score}"
        )

        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
