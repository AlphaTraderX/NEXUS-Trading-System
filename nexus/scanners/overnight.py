"""
Overnight Premium Scanner - capture the overnight equity premium.

Ported from validated backtest logic (v10-v26): _signal_overnight in engine.py.
PF 1.39 (scored), +$12,787 on $10k (2022-2024), 3,471 trades, 53.7% WR.

Academic basis: Nearly 100% of equity premium is earned overnight.
Strategy: Buy MOC (market on close), sell MOO (market on open).

Key filters:
- Skip Fridays (weekend gap risk — negative overnight returns)
- Price > 200 SMA (bull market regime only)
- 10 validated symbols: SPY, QQQ + high-beta tech

This is the highest P&L edge (44% of total returns).
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity

logger = logging.getLogger(__name__)


class OvernightPremiumScanner(BaseScanner):
    """
    Overnight Premium scanner - buy at close, sell at next open.

    VALIDATED: 3,471 trades, PF 1.39, +$12,787, Sharpe 1.28 (2022-2024)
    SIGNAL: Price > 200 SMA + not Friday → buy MOC, sell MOO
    DIRECTION: Always LONG (overnight premium is a long-only phenomenon)

    Ported from BacktestEngine._signal_overnight() to ensure live matches backtest.
    """

    # Validated symbol list — index ETFs + high-beta momentum stocks.
    # v26 tested SMCI/PLTR/XLK — pass individually but combined MaxDD too high.
    INSTRUMENTS = {
        Market.US_STOCKS: [
            "SPY", "QQQ", "TSLA", "NVDA", "AMD",
            "AAPL", "GOOGL", "META", "NFLX", "CRM",
        ],
    }

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.OVERNIGHT_PREMIUM
        self.markets = [Market.US_STOCKS]
        self.instruments = []

        # Validated parameters (do not change without re-backtesting)
        self.sma_period = 200     # Bull market filter
        self.notional_pct = 20    # Position size (% of capital)
        self.score = 60           # C-tier (1.0x multiplier with score-sizing)

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Overnight scanner runs Mon-Thu (skip Friday = weekend risk).

        Should ideally run near market close (3:30-4:00 PM ET) to generate
        MOC signals. For paper trading, can run anytime during market hours.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Skip Fridays (weekday 4) — weekend overnight returns are negative
        if timestamp.weekday() == 4:
            return False

        # Skip weekends (defensive)
        if timestamp.weekday() >= 5:
            return False

        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for overnight premium opportunities.

        Logic (validated v10-v26):
        1. Skip Fridays and weekends
        2. Price must be above 200 SMA (bull market filter)
        3. Generate MOC buy signal — exit at next day's MOO
        """
        opportunities = []

        for market in self.markets:
            instruments = self._get_instruments(market)

            for symbol in instruments:
                try:
                    opp = await self._scan_symbol(symbol, market)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.warning(f"Overnight scan failed for {symbol}: {e}")

        logger.info(f"Overnight scan complete: {len(opportunities)} opportunities found")
        return opportunities

    async def _scan_symbol(self, symbol: str, market: Market) -> Optional[Opportunity]:
        """Scan a single symbol for overnight premium opportunity.

        Ported from BacktestEngine._signal_overnight() — validated logic.
        """
        # Need 201+ bars for 200 SMA
        bars = await self.get_bars_safe(symbol, "1D", 210)

        if bars is None or len(bars) < 201:
            logger.debug(f"Overnight insufficient data for {symbol}")
            return None

        ts = bars.index[-1]

        # Skip Fridays (weekday 4) — weekend overnight returns are negative
        if hasattr(ts, "weekday") and ts.weekday() == 4:
            return None

        # Skip weekends (defensive)
        if hasattr(ts, "weekday") and ts.weekday() >= 5:
            return None

        cur_price = float(bars["close"].iloc[-1])

        # REGIME FILTER: Only trade in bull markets (price > 200 SMA)
        sma_200 = bars["close"].rolling(self.sma_period).mean().iloc[-1]
        if pd.isna(sma_200) or cur_price < sma_200:
            return None

        sma_200 = float(sma_200)

        # Entry at close (MOC), exit at next open (MOO)
        entry = cur_price
        stop = cur_price * 0.95    # 5% catastrophic stop (never hit overnight)
        target = cur_price * 1.10  # Unreachable placeholder — exit at open

        opp = self.create_opportunity(
            symbol=symbol,
            market=market,
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "entry_type": "close_to_open",
                "skip_friday": True,
                "regime_filter": "sma_200",
                "sma_200": round(sma_200, 2),
                "price_vs_sma": round(cur_price / sma_200, 4),
                "notional_pct": self.notional_pct,
            },
        )
        opp.raw_score = self.score

        logger.info(
            f"Overnight: {symbol} LONG MOC | "
            f"Price ${entry:.2f} > SMA200 ${sma_200:.2f} "
            f"({(cur_price / sma_200 - 1) * 100:+.1f}%) | Score {self.score}"
        )

        return opp

    def _get_instruments(self, market: Market) -> List[str]:
        """Get instruments to scan for a market."""
        if self.instruments:
            return self.instruments
        return self.INSTRUMENTS.get(market, [])
