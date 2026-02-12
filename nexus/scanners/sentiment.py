"""
Sentiment Scanner - SENTIMENT_SPIKE edge detection.

Academic evidence: Sentiment + Volume = Sharpe ratio 3.17
Rule: z-score >= 2.0 AND volume >= 150% average = SIGNAL

This scanner requires BOTH conditions:
1. Extreme sentiment (>= 2 std dev from mean)
2. Volume confirmation (>= 150% of 20-day average)
"""

import logging
from datetime import datetime
from typing import List, Optional

from nexus.core.enums import Direction, EdgeType, Market
from nexus.core.models import Opportunity
from nexus.scanners.base import BaseScanner

logger = logging.getLogger(__name__)


class SentimentScanner(BaseScanner):
    """
    Scans for sentiment spikes with volume confirmation.

    Edge: SENTIMENT_SPIKE
    Academic backing: Sharpe 3.17 when combined with volume

    Rules:
    - Sentiment z-score >= 2.0 (extreme)
    - Volume >= 150% of 20-day average
    """

    def __init__(self, data_provider=None, settings=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.SENTIMENT_SPIKE
        self.markets = [Market.US_STOCKS]
        self.instruments: List[str] = []

        # Configuration
        self.min_z_score = 2.0
        self.min_volume_ratio = 1.5
        self.min_messages = 10

        # Symbols to scan (high-activity stocks on StockTwits)
        self.symbols = [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD",
            "META", "GOOGL", "AMZN", "NFLX", "COIN", "PLTR", "SOFI",
        ]

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Sentiment scanner runs during US extended hours (Mon-Fri)."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Skip weekends
        if timestamp.weekday() >= 5:
            return False

        # US pre-market through close: 12:00-21:00 UTC (7:00-16:00 ET)
        return 12 <= timestamp.hour <= 21

    async def scan(self) -> List[Opportunity]:
        """Scan for sentiment spikes with volume confirmation."""
        opportunities: List[Opportunity] = []

        try:
            from nexus.data.stocktwits import get_stocktwits_client

            st = get_stocktwits_client()

            # Get sentiment spikes
            spikes = await st.scan_symbols(
                self.symbols,
                min_z_score=self.min_z_score,
                min_messages=self.min_messages,
            )

            for spike in spikes:
                try:
                    opp = await self._process_spike(spike)
                    if opp is not None:
                        opportunities.append(opp)
                except Exception as e:
                    logger.debug(f"Error processing spike {spike.symbol}: {e}")
                    continue

        except ImportError:
            logger.warning("StockTwits client not available")
        except Exception as e:
            logger.error(f"Sentiment scan failed: {e}")

        logger.info(f"Sentiment scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _process_spike(self, spike) -> Optional[Opportunity]:
        """Process a single sentiment spike into an opportunity."""
        # Volume confirmation
        volume_check = await self._check_volume(spike.symbol)

        if not volume_check["confirmed"]:
            logger.debug(
                f"Sentiment spike {spike.symbol} rejected: "
                f"volume ratio {volume_check['ratio']:.2f} < {self.min_volume_ratio}"
            )
            return None

        # Get bars for ATR calculation
        bars = await self.get_bars_safe(spike.symbol, "1D", 30)
        if bars is None or len(bars) < 5:
            return None

        current_price = float(bars["close"].iloc[-1])
        atr = self.calculate_atr(bars, 14)
        if atr <= 0:
            atr = current_price * 0.02

        # Direction based on sentiment
        if spike.direction == "bullish":
            direction = Direction.LONG
        else:
            direction = Direction.SHORT

        entry, stop, target = self.calculate_entry_stop_target(
            current_price, direction, atr,
            stop_multiplier=1.5,
            target_multiplier=2.5,
        )

        opp = self.create_opportunity(
            symbol=spike.symbol,
            market=Market.US_STOCKS,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "sentiment_z_score": round(spike.z_score, 2),
                "sentiment_direction": spike.direction,
                "message_count": spike.message_count,
                "volume_ratio": round(volume_check["ratio"], 2),
                "current_volume": volume_check.get("current", 0),
                "avg_volume": round(volume_check.get("average", 0), 0),
                "historical_mean": round(spike.historical_mean, 4),
                "historical_std": round(spike.historical_std, 4),
            },
        )

        logger.info(
            f"Sentiment opportunity: {spike.symbol} {direction.value} "
            f"z={spike.z_score:.2f} vol={volume_check['ratio']:.2f}x"
        )

        return opp

    async def _check_volume(self, symbol: str) -> dict:
        """Check if volume confirms the sentiment spike."""
        try:
            bars = await self.get_bars_safe(symbol, "1D", 21)

            if bars is None or len(bars) < 5:
                return {"confirmed": False, "ratio": 0.0}

            current_volume = float(bars["volume"].iloc[-1])
            avg_volume = float(bars["volume"].iloc[:-1].mean())

            ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

            return {
                "confirmed": ratio >= self.min_volume_ratio,
                "ratio": ratio,
                "current": current_volume,
                "average": avg_volume,
            }

        except Exception as e:
            logger.debug(f"Volume check failed for {symbol}: {e}")
            return {"confirmed": False, "ratio": 0.0}
