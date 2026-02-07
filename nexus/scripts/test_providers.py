"""
NEXUS Massive Data Provider

Connects to Polygon.io (now branded as Massive) for US stock market data.
Provides historical bars, real-time quotes, and market snapshots.

API Docs: https://polygon.io/docs
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from .base import BaseDataProvider, Quote, normalize_timeframe

logger = logging.getLogger(__name__)

UTC = timezone.utc


class MassiveProvider(BaseDataProvider):
    """
    Polygon.io (branded as Massive) data provider.
    
    Provides US stock market data including:
    - Historical OHLCV bars
    - Real-time and delayed quotes
    - Market snapshots
    - Gainers/losers
    
    Rate Limits (Free tier):
    - 5 API calls per minute
    - 15-minute delayed data
    
    Rate Limits (Paid):
    - Unlimited calls
    - Real-time data
    """
    
    # IMPORTANT: Use polygon.io URL - the API hasn't migrated to massive.com yet
    BASE_URL = "https://api.polygon.io"
    
    # Timeframe mapping to Massive API format
    TIMEFRAME_MAP = {
        "1m": ("1", "minute"),
        "5m": ("5", "minute"),
        "15m": ("15", "minute"),
        "30m": ("30", "minute"),
        "1h": ("1", "hour"),
        "4h": ("4", "hour"),
        "1D": ("1", "day"),
        "1W": ("1", "week"),
    }
    
    def __init__(self, api_key: str):
        """
        Initialize Massive provider.
        
        Args:
            api_key: Polygon/Massive API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = self.BASE_URL
        self.client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting
        self._last_call_time: Optional[datetime] = None
        self._calls_this_minute = 0
        self._rate_limit = 5  # Free tier
    
    async def connect(self) -> bool:
        """Initialize HTTP client."""
        try:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # Test connection with a simple call
            response = await self._request("GET", "/v2/aggs/ticker/SPY/prev")
            
            if response.get("status") == "OK" or response.get("status") == "DELAYED":
                self._connected = True
                logger.info("Connected to Massive API")
                return True
            else:
                logger.error(f"Massive API test failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Massive: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._connected = False
        logger.info("Disconnected from Massive API")
    
    async def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """
        Make rate-limited API request.
        
        Handles:
        - Rate limiting (5 calls/min on free tier)
        - Error handling
        - Response parsing
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Rate limiting
        await self._rate_limit_wait()
        
        # Add API key to params
        params = params or {}
        params["apiKey"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self.client.request(method, url, params=params)
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limited by Massive API, waiting...")
                await asyncio.sleep(60)
                return await self._request(method, endpoint, params)
            raise
    
    async def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        now = datetime.now(UTC)
        
        if self._last_call_time:
            elapsed = (now - self._last_call_time).total_seconds()
            
            if elapsed < 60:
                self._calls_this_minute += 1
                if self._calls_this_minute >= self._rate_limit:
                    wait_time = 60 - elapsed
                    logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self._calls_this_minute = 0
            else:
                self._calls_this_minute = 0
        
        self._last_call_time = now
    
    async def get_quote(self, symbol: str) -> Quote:
        """
        Get quote for a symbol using previous-close data (free-tier compatible).
        
        Uses /v2/aggs/ticker/{symbol}/prev instead of snapshot endpoint.
        Snapshot requires paid tier (403 on free). Returns previous day's close
        as last; bid/ask are estimated with a small spread.
        """
        endpoint = f"/v2/aggs/ticker/{symbol.upper()}/prev"
        data = await self._request("GET", endpoint)
        
        status = data.get("status")
        if status not in ("OK", "DELAYED") or not data.get("results"):
            raise ValueError(f"Failed to get quote for {symbol}: {data}")
        
        r = data["results"][0]
        close = float(r.get("c", 0))
        volume = int(r.get("v", 0))
        ts_ms = r.get("t")
        timestamp = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            if ts_ms is not None
            else datetime.now(UTC)
        )
        
        # Estimate bid/ask from close with a small spread (free tier has no NBBO)
        spread = max(0.01, close * 0.0001)
        bid = close - spread
        ask = close + spread
        
        return Quote(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            last=close,
            volume=volume,
            timestamp=timestamp,
        )
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.
        
        Args:
            symbol: Stock symbol (e.g., "SPY", "AAPL")
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 4h, 1D)
            limit: Number of bars to fetch (max 50000)
            end_date: End date for bars (default: now)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Normalize timeframe
        tf = normalize_timeframe(timeframe)
        
        if tf not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        multiplier, timespan = self.TIMEFRAME_MAP[tf]
        
        # Calculate date range
        end = end_date or datetime.now(UTC)
        
        # Calculate start based on timeframe and limit
        if timespan == "minute":
            start = end - timedelta(minutes=int(multiplier) * limit)
        elif timespan == "hour":
            start = end - timedelta(hours=int(multiplier) * limit)
        elif timespan == "day":
            start = end - timedelta(days=int(multiplier) * limit)
        elif timespan == "week":
            start = end - timedelta(weeks=int(multiplier) * limit)
        
        # Format dates
        from_date = start.strftime("%Y-%m-%d")
        to_date = end.strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": min(limit, 50000)
        }
        
        data = await self._request("GET", endpoint, params)
        
        status = data.get("status")
        if status not in ("OK", "DELAYED") or "results" not in data:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Parse results into DataFrame
        bars = data["results"]
        
        df = pd.DataFrame(bars)
        
        # Rename columns to standard format
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions"
        })
        
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        # Select and order columns
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if "vwap" in df.columns:
            columns.append("vwap")
        
        df = df[columns]
        
        # Sort by timestamp and limit
        df = df.sort_values("timestamp").tail(limit).reset_index(drop=True)
        
        return df
    
    async def get_previous_close(self, symbol: str) -> Dict[str, Any]:
        """Get previous day's OHLCV."""
        endpoint = f"/v2/aggs/ticker/{symbol.upper()}/prev"
        
        data = await self._request("GET", endpoint)
        
        status = data.get("status")
        if status in ("OK", "DELAYED") and data.get("results"):
            result = data["results"][0]
            return {
                "open": result.get("o"),
                "high": result.get("h"),
                "low": result.get("l"),
                "close": result.get("c"),
                "volume": result.get("v"),
                "vwap": result.get("vw"),
            }
        
        return {}
    
    async def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get full market snapshot including today's data."""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}"
        
        data = await self._request("GET", endpoint)
        
        status = data.get("status")
        if status in ("OK", "DELAYED"):
            return data.get("ticker", {})
        
        return {}
    
    async def get_gainers_losers(self, direction: str = "gainers") -> List[Dict]:
        """
        Get top gainers or losers.
        
        Args:
            direction: "gainers" or "losers"
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
        
        data = await self._request("GET", endpoint)
        
        status = data.get("status")
        if status in ("OK", "DELAYED"):
            return data.get("tickers", [])
        
        return []
    
    # Indicator helpers (calculate from bars)
    
    @staticmethod
    def calculate_vwap(bars: pd.DataFrame) -> float:
        """Calculate VWAP from bars."""
        if bars.empty:
            return 0.0
        
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        vwap = (typical_price * bars["volume"]).sum() / bars["volume"].sum()
        return float(vwap)
    
    @staticmethod
    def calculate_atr(bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(bars) < period:
            return 0.0
        
        high = bars["high"]
        low = bars["low"]
        close = bars["close"].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return float(atr)
    
    @staticmethod
    def calculate_rsi(bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        if len(bars) < period + 1:
            return 50.0
        
        delta = bars["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])