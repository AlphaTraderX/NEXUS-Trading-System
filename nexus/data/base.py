"""
NEXUS Base Data Provider

Abstract base classes and data models for all data providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
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
