"""
StockTwits Sentiment Client

Fetches social sentiment data for SENTIMENT_SPIKE edge detection.
Academic evidence: Sentiment + Volume = Sharpe 3.17

API: https://api.stocktwits.com/api/2/
Rate Limit: 200 requests/hour (free tier)
"""

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment data for a symbol."""
    symbol: str
    timestamp: datetime
    bullish_count: int
    bearish_count: int
    total_count: int

    @property
    def bullish_ratio(self) -> float:
        """Ratio of bullish messages (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.5
        return self.bullish_count / self.total_count

    @property
    def sentiment_score(self) -> float:
        """Sentiment score (-1.0 bearish to +1.0 bullish)."""
        return (self.bullish_ratio - 0.5) * 2


@dataclass
class SentimentSpike:
    """Detected sentiment spike."""
    symbol: str
    timestamp: datetime
    current_sentiment: float
    historical_mean: float
    historical_std: float
    z_score: float
    message_count: int
    direction: str  # "bullish" or "bearish"

    @property
    def is_extreme(self) -> bool:
        """Z-score >= 2.0 is considered extreme."""
        return abs(self.z_score) >= 2.0


class StockTwitsClient:
    """
    Client for StockTwits sentiment API.

    Usage:
        client = StockTwitsClient()
        sentiment = await client.get_sentiment("AAPL")
        spike = client.detect_spike("AAPL", sentiment)
    """

    BASE_URL = "https://api.stocktwits.com/api/2"

    # Rate limit: 200/hour = ~3.3/min, use 3/min to be safe
    REQUEST_DELAY = 20.0  # 20 seconds between requests

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "NEXUS Trading System",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        self._last_request_time = 0.0

        # Historical sentiment for z-score calculation
        # Store last 20 readings per symbol (~1 week if checked 3x/day)
        self._history: Dict[str, deque] = {}
        self._history_maxlen = 20

        # Cache current readings (5 min TTL)
        self._cache: Dict[str, Tuple[datetime, SentimentData]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    async def get_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        Get current sentiment for a symbol.

        Returns:
            SentimentData with bullish/bearish counts, or None if failed
        """
        symbol_upper = symbol.upper().replace("/", "")

        # Check cache
        if symbol_upper in self._cache:
            cached_time, cached_data = self._cache[symbol_upper]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        try:
            await self._rate_limit()

            response = await self._client.get(
                f"{self.BASE_URL}/streams/symbol/{symbol_upper}.json",
                params={"filter": "top", "limit": 30},
            )

            if response.status_code == 200:
                data = response.json()
                messages = data.get("messages", [])

                bullish = 0
                bearish = 0

                for msg in messages:
                    entities = msg.get("entities", {})
                    sentiment = entities.get("sentiment", {})

                    if sentiment:
                        if sentiment.get("basic") == "Bullish":
                            bullish += 1
                        elif sentiment.get("basic") == "Bearish":
                            bearish += 1

                result = SentimentData(
                    symbol=symbol_upper,
                    timestamp=datetime.now(),
                    bullish_count=bullish,
                    bearish_count=bearish,
                    total_count=bullish + bearish,
                )

                # Cache result
                self._cache[symbol_upper] = (datetime.now(), result)

                # Add to history
                self._add_to_history(symbol_upper, result)

                return result

            elif response.status_code == 404:
                logger.debug(f"Symbol {symbol_upper} not found on StockTwits")
                return None

            elif response.status_code == 429:
                logger.warning("StockTwits rate limit hit")
                return None

            else:
                logger.warning(f"StockTwits returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch StockTwits sentiment: {e}")
            return None

    def _add_to_history(self, symbol: str, data: SentimentData):
        """Add sentiment reading to historical data."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_maxlen)
        self._history[symbol].append(data)

    def get_historical_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get historical sentiment statistics for a symbol.

        Returns dict with mean, std, count, min, max or None if insufficient data.
        """
        symbol_upper = symbol.upper().replace("/", "")

        if symbol_upper not in self._history:
            return None

        history = self._history[symbol_upper]

        if len(history) < 5:
            return None  # Not enough data

        scores = [d.sentiment_score for d in history]

        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.1,
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
        }

    def detect_spike(
        self,
        symbol: str,
        current: SentimentData,
        min_messages: int = 10,
    ) -> Optional[SentimentSpike]:
        """
        Detect if current sentiment is a significant spike.

        Args:
            symbol: Stock symbol
            current: Current sentiment data
            min_messages: Minimum messages required

        Returns:
            SentimentSpike if enough data to compute z-score, None otherwise
        """
        if current.total_count < min_messages:
            return None

        stats = self.get_historical_stats(symbol)

        if stats is None:
            return None

        mean = stats["mean"]
        std = stats["std"]

        # Avoid division by zero
        if std < 0.01:
            std = 0.1

        z_score = (current.sentiment_score - mean) / std

        return SentimentSpike(
            symbol=symbol.upper(),
            timestamp=current.timestamp,
            current_sentiment=current.sentiment_score,
            historical_mean=mean,
            historical_std=std,
            z_score=z_score,
            message_count=current.total_count,
            direction="bullish" if z_score > 0 else "bearish",
        )

    async def scan_symbols(
        self,
        symbols: List[str],
        min_z_score: float = 2.0,
        min_messages: int = 10,
    ) -> List[SentimentSpike]:
        """
        Scan multiple symbols for sentiment spikes.

        Args:
            symbols: List of symbols to scan
            min_z_score: Minimum absolute z-score for spike
            min_messages: Minimum messages required

        Returns:
            List of detected spikes with |z_score| >= min_z_score
        """
        spikes: List[SentimentSpike] = []

        for symbol in symbols:
            try:
                sentiment = await self.get_sentiment(symbol)

                if sentiment is None:
                    continue

                spike = self.detect_spike(symbol, sentiment, min_messages)

                if spike and abs(spike.z_score) >= min_z_score:
                    spikes.append(spike)
                    logger.info(
                        f"Sentiment spike: {symbol} z={spike.z_score:.2f} "
                        f"({spike.direction}, {spike.message_count} msgs)"
                    )

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        return spikes

    def seed_history(self, symbol: str, historical_scores: List[float]):
        """
        Seed historical data for a symbol (for testing or initialization).

        Args:
            symbol: Stock symbol
            historical_scores: List of sentiment scores (-1 to 1)
        """
        symbol_upper = symbol.upper()

        if symbol_upper not in self._history:
            self._history[symbol_upper] = deque(maxlen=self._history_maxlen)

        for score in historical_scores[-self._history_maxlen:]:
            bullish = int((score + 1) / 2 * 20)  # Map -1..1 to 0..20
            bearish = 20 - bullish

            self._history[symbol_upper].append(SentimentData(
                symbol=symbol_upper,
                timestamp=datetime.now(),
                bullish_count=bullish,
                bearish_count=bearish,
                total_count=20,
            ))

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Singleton
_st_client: Optional[StockTwitsClient] = None


def get_stocktwits_client() -> StockTwitsClient:
    """Get global StockTwits client."""
    global _st_client
    if _st_client is None:
        _st_client = StockTwitsClient()
    return _st_client
