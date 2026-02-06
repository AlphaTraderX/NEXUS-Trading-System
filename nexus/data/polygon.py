"""
NEXUS Polygon.io Data Provider

Market data provider for US stocks and forex.
Provides historical bars, quotes, and news.

This is DATA ONLY - extends BaseDataProvider, not BaseBroker.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd

from config.settings import settings
from data.base import BaseDataProvider, Quote

logger = logging.getLogger(__name__)


class PolygonProvider(BaseDataProvider):
    """
    Polygon.io market data provider.
    
    Features:
    - Historical OHLCV bars (minute to weekly)
    - Real-time quotes (with subscription)
    - Previous close data
    - Market gainers/losers
    - News headlines
    
    Rate Limits (Free tier):
    - 5 API calls per minute
    - Delayed data (15 min)
    
    Paid tier removes limits and provides real-time.
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        super().__init__()
        self._api_key = settings.polygon_api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_limit_remaining = 5
        self._last_request_time: Optional[datetime] = None
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    async def connect(self) -> bool:
        """Verify API key works."""
        try:
            # Test with a simple request
            response = await self._client.get(
                f"{self.BASE_URL}/v2/aggs/ticker/SPY/prev",
                params={"apiKey": self._api_key},
            )
            
            if response.status_code == 200:
                self._connected = True
                logger.info("Connected to Polygon.io")
                return True
            elif response.status_code == 401:
                logger.error("Polygon API key invalid")
                return False
            else:
                logger.error(f"Polygon connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Polygon connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect (cleanup)."""
        self._connected = False
        logger.info("Disconnected from Polygon.io")
    
    async def _rate_limit_wait(self) -> None:
        """Wait if rate limited (free tier: 5/min)."""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < 12:  # 5 per minute = 1 per 12 seconds
                await asyncio.sleep(12 - elapsed)
        self._last_request_time = datetime.now()
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get current/last quote for symbol."""
        await self._rate_limit_wait()
        
        # Use snapshot endpoint for current data
        response = await self._client.get(
            f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
            params={"apiKey": self._api_key},
        )
        
        if response.status_code != 200:
            # Fallback to previous close
            return await self._get_prev_close_quote(symbol)
        
        data = response.json()
        ticker = data.get("ticker", {})
        
        day = ticker.get("day", {})
        last_quote = ticker.get("lastQuote", {})
        last_trade = ticker.get("lastTrade", {})
        
        return Quote(
            symbol=symbol,
            bid=float(last_quote.get("p", 0)) if last_quote else float(day.get("c", 0)),
            ask=float(last_quote.get("P", 0)) if last_quote else float(day.get("c", 0)),
            last=float(last_trade.get("p", 0)) if last_trade else float(day.get("c", 0)),
            volume=int(day.get("v", 0)),
            timestamp=datetime.now(),
        )
    
    async def _get_prev_close_quote(self, symbol: str) -> Quote:
        """Get previous close as fallback."""
        response = await self._client.get(
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/prev",
            params={"apiKey": self._api_key},
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get quote for {symbol}")
        
        data = response.json()
        results = data.get("results", [{}])
        
        if not results:
            raise Exception(f"No data for {symbol}")
        
        bar = results[0]
        close = float(bar.get("c", 0))
        
        return Quote(
            symbol=symbol,
            bid=close,
            ask=close,
            last=close,
            volume=int(bar.get("v", 0)),
            timestamp=datetime.now(),
        )
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV bars.
        
        Args:
            symbol: Stock symbol (e.g., "SPY", "AAPL")
            timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
            limit: Number of bars
            end_date: End date for historical data
        """
        await self._rate_limit_wait()
        
        # Convert timeframe
        multiplier, timespan = self._convert_timeframe(timeframe)
        
        # Calculate date range
        end = end_date or datetime.now()
        start = self._calculate_start_date(timeframe, limit, end)
        
        response = await self._client.get(
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            params={
                "apiKey": self._api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
            },
        )
        
        if response.status_code != 200:
            logger.warning(f"Polygon bars request failed: {response.status_code}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(r.get("t", 0) / 1000),
            'open': float(r.get("o", 0)),
            'high': float(r.get("h", 0)),
            'low': float(r.get("l", 0)),
            'close': float(r.get("c", 0)),
            'volume': int(r.get("v", 0)),
            'vwap': float(r.get("vw", 0)),
            'trades': int(r.get("n", 0)),
        } for r in results])
        
        return df.tail(limit)
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> bool:
        """Subscribe to real-time quotes (polling implementation)."""
        for symbol in symbols:
            self._subscriptions[symbol] = callback
        logger.info(f"Subscribed to {len(symbols)} symbols (polling mode)")
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            self._subscriptions.pop(symbol, None)
    
    # =========================================================================
    # ADDITIONAL DATA
    # =========================================================================
    
    async def get_previous_close(self, symbol: str) -> Dict[str, Any]:
        """Get previous day's OHLCV."""
        await self._rate_limit_wait()
        
        response = await self._client.get(
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/prev",
            params={"apiKey": self._api_key},
        )
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        results = data.get("results", [{}])
        
        if not results:
            return {}
        
        bar = results[0]
        return {
            "symbol": symbol,
            "open": float(bar.get("o", 0)),
            "high": float(bar.get("h", 0)),
            "low": float(bar.get("l", 0)),
            "close": float(bar.get("c", 0)),
            "volume": int(bar.get("v", 0)),
            "vwap": float(bar.get("vw", 0)),
        }
    
    async def get_gainers_losers(self, direction: str = "gainers") -> List[Dict[str, Any]]:
        """
        Get market gainers or losers.
        
        Args:
            direction: "gainers" or "losers"
        """
        await self._rate_limit_wait()
        
        response = await self._client.get(
            f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/{direction}",
            params={"apiKey": self._api_key},
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        tickers = data.get("tickers", [])
        
        return [{
            "symbol": t.get("ticker"),
            "change_pct": float(t.get("todaysChangePerc", 0)),
            "change": float(t.get("todaysChange", 0)),
            "price": float(t.get("day", {}).get("c", 0)),
            "volume": int(t.get("day", {}).get("v", 0)),
        } for t in tickers[:20]]
    
    async def get_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news headlines."""
        await self._rate_limit_wait()
        
        params = {
            "apiKey": self._api_key,
            "limit": limit,
            "sort": "published_utc",
        }
        
        if symbol:
            params["ticker"] = symbol
        
        response = await self._client.get(
            f"{self.BASE_URL}/v2/reference/news",
            params=params,
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        results = data.get("results", [])
        
        return [{
            "title": n.get("title"),
            "author": n.get("author"),
            "published": n.get("published_utc"),
            "url": n.get("article_url"),
            "tickers": n.get("tickers", []),
            "description": n.get("description", "")[:200],
        } for n in results]
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    def calculate_vwap(self, bars: pd.DataFrame) -> float:
        """Calculate VWAP from bars."""
        if bars.empty:
            return 0.0
        
        typical_price = (bars['high'] + bars['low'] + bars['close']) / 3
        vwap = (typical_price * bars['volume']).sum() / bars['volume'].sum()
        return float(vwap)
    
    def calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from bars."""
        if len(bars) < period:
            return 0.0
        
        high = bars['high']
        low = bars['low']
        close = bars['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return float(atr)
    
    def calculate_rsi(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI from bars."""
        if len(bars) < period + 1:
            return 50.0  # Neutral
        
        delta = bars['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    
    def calculate_bollinger_bands(
        self, 
        bars: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(bars) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "width": 0}
        
        close = bars['close']
        middle = close.rolling(window=period).mean().iloc[-1]
        std = close.rolling(window=period).std().iloc[-1]
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle * 100  # As percentage
        
        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "width": float(width),
        }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _convert_timeframe(self, timeframe: str) -> tuple:
        """Convert NEXUS timeframe to Polygon format."""
        mapping = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "4h": (4, "hour"),
            "1d": (1, "day"),
            "1w": (1, "week"),
        }
        return mapping.get(timeframe, (1, "hour"))
    
    def _calculate_start_date(self, timeframe: str, limit: int, end: datetime) -> datetime:
        """Calculate start date based on timeframe and limit."""
        if timeframe in ["1m", "5m"]:
            # Minutes - need fewer days
            days = max(1, limit // 100)
        elif timeframe in ["15m", "30m"]:
            days = max(2, limit // 30)
        elif timeframe in ["1h", "4h"]:
            days = max(5, limit // 6)
        elif timeframe == "1d":
            days = limit + 10  # Account for weekends
        elif timeframe == "1w":
            days = limit * 7 + 10
        else:
            days = 30
        
        return end - timedelta(days=days)
