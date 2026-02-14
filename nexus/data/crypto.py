"""
Binance & Kraken Crypto Data Providers

Provides 24/7 cryptocurrency market data for weekend trading.
Uses public REST APIs (no authentication required for market data).

Binance rate limits: 1200 requests/minute
Kraken rate limits: ~15 requests/second (public)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import httpx
import pandas as pd

from .base import BaseDataProvider, Quote

logger = logging.getLogger(__name__)


# =============================================================================
# Binance Provider
# =============================================================================


class BinanceProvider(BaseDataProvider):
    """
    Binance cryptocurrency data provider.

    Provides market data for crypto pairs. 24/7 availability.
    Uses public REST API -- no authentication needed for market data.

    Supported pairs: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, MATIC, LINK,
                     DOGE, SHIB, LTC, UNI, ATOM, ARB, OP, APT, etc.
    """

    BASE_URL = "https://api.binance.com"

    # Timeframe mapping: NEXUS -> Binance
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1D": "1d",
        "1w": "1w",
        "1W": "1w",
    }

    def __init__(self):
        """Initialize Binance provider.

        Reads API key from settings if available. Authenticated requests
        get higher rate limits (1200/min vs default).
        """
        super().__init__()
        from nexus.config.settings import get_settings
        settings = get_settings()
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
        self._last_ping: Optional[datetime] = None

    async def connect(self) -> bool:
        """Test connection to Binance API."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/api/v3/ping")
            if response.status_code == 200:
                self._connected = True
                self._last_ping = datetime.now(timezone.utc)
                logger.info("Connected to Binance API")
                return True
            else:
                logger.error(f"Binance ping failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        self._connected = False
        logger.info("Disconnected from Binance API")

    def _convert_symbol(self, nexus_symbol: str) -> str:
        """
        Convert NEXUS symbol format to Binance format.

        Examples:
            BTC_USD -> BTCUSDT
            ETH_USD -> ETHUSDT
        """
        if "_" in nexus_symbol:
            base, quote = nexus_symbol.split("_", 1)
            if quote == "USD":
                quote = "USDT"
            return f"{base}{quote}"
        return nexus_symbol

    def _convert_symbol_to_nexus(self, binance_symbol: str) -> str:
        """
        Convert Binance symbol to NEXUS format.

        Examples:
            BTCUSDT -> BTC_USD
            ETHUSDT -> ETH_USD
        """
        for quote in ["USDT", "BUSD", "USD", "BTC", "ETH"]:
            if binance_symbol.endswith(quote):
                base = binance_symbol[: -len(quote)]
                if quote == "USDT":
                    quote = "USD"
                return f"{base}_{quote}"
        return binance_symbol

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote for a crypto pair."""
        binance_symbol = self._convert_symbol(symbol)

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/api/v3/ticker/bookTicker",
                params={"symbol": binance_symbol},
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get quote for {symbol}: {response.status_code}")
                return None

            data = response.json()

            bid = float(data["bidPrice"])
            ask = float(data["askPrice"])
            last = (bid + ask) / 2

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=0,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV bars for a crypto pair.

        Args:
            symbol: Symbol in NEXUS format (e.g., BTC_USD)
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of bars to fetch (max 1000)
            end_date: End date for bars (default: now)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        binance_symbol = self._convert_symbol(symbol)
        binance_tf = self.TIMEFRAME_MAP.get(timeframe, "1h")

        params: Dict[str, Any] = {
            "symbol": binance_symbol,
            "interval": binance_tf,
            "limit": min(limit, 1000),
        }

        if end_date is not None:
            params["endTime"] = int(end_date.timestamp() * 1000)

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/api/v3/klines",
                params=params,
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get bars for {symbol}: {response.status_code}")
                return None

            data = response.json()
            if not data:
                return None

            # Binance kline format:
            # [open_time, open, high, low, close, volume, close_time, ...]
            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore",
                ],
            )

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

            return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return None

    async def get_24h_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24-hour statistics for a crypto pair."""
        binance_symbol = self._convert_symbol(symbol)

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/api/v3/ticker/24hr",
                params={"symbol": binance_symbol},
            )

            if response.status_code != 200:
                return None

            data = response.json()

            return {
                "symbol": symbol,
                "price_change": float(data["priceChange"]),
                "price_change_pct": float(data["priceChangePercent"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "volume_24h": float(data["volume"]),
                "quote_volume_24h": float(data["quoteVolume"]),
                "trades_24h": int(data["count"]),
            }

        except Exception as e:
            logger.error(f"Error getting 24h stats for {symbol}: {e}")
            return None

    async def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Get prices for all USDT trading pairs."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/api/v3/ticker/price")

            if response.status_code != 200:
                return []

            data = response.json()

            return [
                {
                    "symbol": self._convert_symbol_to_nexus(t["symbol"]),
                    "price": float(t["price"]),
                }
                for t in data
                if t["symbol"].endswith("USDT")
            ]

        except Exception as e:
            logger.error(f"Error getting all tickers: {e}")
            return []


# =============================================================================
# Kraken Provider (UK FCA-friendly alternative)
# =============================================================================


class KrakenProvider(BaseDataProvider):
    """
    Kraken cryptocurrency data provider.

    Alternative to Binance -- Kraken is more UK/EU regulatory friendly.
    Uses public REST API.
    """

    BASE_URL = "https://api.kraken.com/0/public"

    SYMBOL_MAP = {
        "BTC_USD": "XXBTZUSD",
        "ETH_USD": "XETHZUSD",
        "SOL_USD": "SOLUSD",
        "XRP_USD": "XXRPZUSD",
        "ADA_USD": "ADAUSD",
        "DOT_USD": "DOTUSD",
        "LINK_USD": "LINKUSD",
        "MATIC_USD": "MATICUSD",
        "AVAX_USD": "AVAXUSD",
        "ATOM_USD": "ATOMUSD",
        "LTC_USD": "XLTCZUSD",
        "DOGE_USD": "XDGUSD",
    }

    TIMEFRAME_MAP = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1D": 1440,
    }

    def __init__(self):
        """Initialize Kraken provider."""
        super().__init__()
        self.client = httpx.AsyncClient(timeout=30.0)

    async def connect(self) -> bool:
        """Test connection to Kraken API."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/Time")
            data = response.json()

            if data.get("error") == []:
                self._connected = True
                logger.info("Connected to Kraken API")
                return True
            else:
                logger.error(f"Kraken connection error: {data.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
            return False

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        self._connected = False

    def _convert_symbol(self, nexus_symbol: str) -> str:
        """Convert NEXUS symbol to Kraken format."""
        return self.SYMBOL_MAP.get(nexus_symbol, nexus_symbol)

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote for a crypto pair."""
        kraken_symbol = self._convert_symbol(symbol)

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/Ticker",
                params={"pair": kraken_symbol},
            )

            data = response.json()

            if data.get("error"):
                logger.warning(f"Kraken error for {symbol}: {data['error']}")
                return None

            result = data.get("result", {})
            if not result:
                return None

            ticker = list(result.values())[0]

            bid = float(ticker["b"][0])
            ask = float(ticker["a"][0])
            last = float(ticker["c"][0])
            volume = float(ticker["v"][1])

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=int(volume),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error getting Kraken quote for {symbol}: {e}")
            return None

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV bars for a crypto pair."""
        kraken_symbol = self._convert_symbol(symbol)
        kraken_interval = self.TIMEFRAME_MAP.get(timeframe, 60)

        params: Dict[str, Any] = {
            "pair": kraken_symbol,
            "interval": kraken_interval,
        }

        if end_date is not None:
            # Kraken uses 'since' as Unix timestamp
            params["since"] = int(end_date.timestamp())

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/OHLC",
                params=params,
            )

            data = response.json()

            if data.get("error"):
                logger.warning(f"Kraken OHLC error: {data['error']}")
                return None

            result = data.get("result", {})
            result.pop("last", None)

            if not result:
                return None

            ohlc = list(result.values())[0]

            # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
            df = pd.DataFrame(
                ohlc,
                columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
            )

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

            return (
                df[["timestamp", "open", "high", "low", "close", "volume"]]
                .tail(limit)
                .copy()
            )

        except Exception as e:
            logger.error(f"Error getting Kraken bars for {symbol}: {e}")
            return None
