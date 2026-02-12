"""
NEXUS Base Data Provider

Abstract base classes and data models for all data providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime

    @property
    def spread(self) -> float:
        """Spread in absolute terms."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = (self.bid + self.ask) / 2
        return (self.spread / mid) * 100 if mid > 0 else 0

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class AccountInfo:
    """Broker account information."""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: str = "USD"
    unrealized_pnl: float = 0.0


@dataclass
class Position:
    """Open position information."""
    symbol: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    margin_used: float = 0.0
    open_time: Optional[datetime] = None


@dataclass
class Order:
    """Order request."""
    symbol: str
    direction: str  # "long" or "short"
    size: float
    order_type: str  # "market", "limit", "stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    status: str  # "filled", "pending", "rejected", "cancelled"
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    filled_size: float = 0.0
    message: str = ""


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.

    Provides market data (quotes, bars) but not execution.
    """

    def __init__(self):
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data provider."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol.

        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 4h, 1D, etc.)
            limit: Number of bars to fetch
            end_date: End date for bars (default: now)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    async def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """Subscribe to real-time updates."""
        for symbol in symbols:
            self._subscriptions[symbol] = callback

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time updates."""
        for symbol in symbols:
            self._subscriptions.pop(symbol, None)


class BaseBroker(BaseDataProvider):
    """
    Abstract base class for brokers.

    Extends data provider with execution capabilities.
    """

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass

    @abstractmethod
    async def close_position(self, symbol: str, size: Optional[float] = None) -> OrderResult:
        """Close a position (full or partial)."""
        pass

    @abstractmethod
    async def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions (emergency)."""
        pass


class ReconnectionMixin:
    """
    Mixin for automatic broker reconnection.

    Add to any BaseBroker subclass to get:
    - ensure_connected() with automatic reconnect
    - Exponential backoff (5s, 10s, 20s, 40s, 80s)
    - Heartbeat tracking
    """

    _reconnect_attempts: int
    _max_reconnect_attempts: int
    _reconnect_delay: int
    _last_heartbeat: datetime
    _heartbeat_interval: int
    _reconnecting: bool

    def _init_reconnection(self) -> None:
        """Initialize reconnection state. Call from subclass __init__."""
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds, doubles each attempt
        self._last_heartbeat = datetime.now(timezone.utc)
        self._heartbeat_interval = 30
        self._reconnecting = False

    async def ensure_connected(self) -> bool:
        """Ensure broker is connected, reconnect if needed."""
        if self.is_connected:
            self._last_heartbeat = datetime.now(timezone.utc)
            return True

        if self._reconnecting:
            return False

        return await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect to broker with exponential backoff."""
        self._reconnecting = True

        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))

            logger.warning(
                f"Reconnecting to broker "
                f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})..."
            )

            try:
                if await self.connect():
                    logger.info("Reconnected to broker successfully")
                    self._reconnect_attempts = 0
                    self._reconnecting = False
                    return True
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

            await asyncio.sleep(delay)

        self._reconnecting = False
        logger.critical(
            f"Failed to reconnect after {self._max_reconnect_attempts} attempts"
        )
        return False

    def seconds_since_heartbeat(self) -> float:
        """Get seconds since last successful heartbeat."""
        return (datetime.now(timezone.utc) - self._last_heartbeat).total_seconds()

    async def heartbeat(self) -> bool:
        """Send heartbeat / check connection."""
        try:
            if self.is_connected:
                self._last_heartbeat = datetime.now(timezone.utc)
                return True
            return False
        except Exception:
            return False


# Timeframe mapping utilities
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1D": 1440,
    "1W": 10080,
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe string to standard format."""
    timeframe = timeframe.upper().replace("MIN", "M").replace("HOUR", "H").replace("DAY", "D")

    mappings = {
        "1MIN": "1m", "5MIN": "5m", "15MIN": "15m", "30MIN": "30m",
        "1M": "1m", "5M": "5m", "15M": "15m", "30M": "30m",
        "1H": "1h", "4H": "4h",
        "1D": "1D", "D": "1D", "DAILY": "1D",
        "1W": "1W", "W": "1W", "WEEKLY": "1W",
    }

    return mappings.get(timeframe, timeframe.lower())
