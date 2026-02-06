"""
NEXUS Data Provider Base Classes

Abstract interfaces that all data providers and brokers must implement.
This ensures consistent behavior across different data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


# =============================================================================
# DATA CLASSES
# =============================================================================

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
    def mid(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Spread in price."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        if self.mid == 0:
            return 0
        return (self.spread / self.mid) * 100


@dataclass
class AccountInfo:
    """Trading account information."""
    account_id: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: str
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    
    @property
    def margin_usage_pct(self) -> float:
        """Margin usage as percentage."""
        if self.equity == 0:
            return 0
        return (self.margin_used / self.equity) * 100


@dataclass
class Position:
    """Open trading position."""
    symbol: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    opened_at: Optional[datetime] = None
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order to submit."""
    symbol: str
    direction: str  # "long" or "short" (for entry) or "close"
    size: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc, fok
    
    # Metadata
    signal_id: Optional[str] = None
    notes: str = ""


@dataclass
class OrderResult:
    """Result of order submission."""
    order_id: str
    status: OrderStatus
    symbol: str
    direction: str
    requested_size: float
    filled_size: float = 0.0
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    message: str = ""
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def slippage(self) -> Optional[float]:
        """Slippage if limit price was set."""
        return None  # Calculated by caller with reference price


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# =============================================================================
# BASE DATA PROVIDER
# =============================================================================

class BaseDataProvider(ABC):
    """
    Base class for data-only providers (e.g., Polygon).
    
    Provides market data but no execution capability.
    """
    
    def __init__(self):
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to data source."""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to data source.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Current quote data
        """
        pass
    
    @abstractmethod
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
            symbol: Instrument symbol
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            limit: Number of bars to fetch
            end_date: End date for historical data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> bool:
        """
        Subscribe to real-time quotes.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Function to call with new quotes
            
        Returns:
            True if subscription successful
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {
            "connected": self._connected,
            "subscriptions": len(self._subscriptions),
        }


# =============================================================================
# BASE BROKER
# =============================================================================

class BaseBroker(BaseDataProvider):
    """
    Base class for brokers (data + execution).
    
    Extends BaseDataProvider with execution capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self._account_id: Optional[str] = None
    
    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================
    
    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            Current account info
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of current positions
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for specific symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Position if exists, None otherwise
        """
        pass
    
    # =========================================================================
    # ORDER METHODS
    # =========================================================================
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """
        Place an order.
        
        Args:
            order: Order to place
            
        Returns:
            Order result with fill details
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order result or None if not found
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self) -> List[OrderResult]:
        """Get all open orders."""
        pass
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close
            size: Size to close (None = close all)
            
        Returns:
            Order result
        """
        pass
    
    @abstractmethod
    async def close_all_positions(self) -> List[OrderResult]:
        """
        Close all positions (emergency).
        
        Returns:
            List of order results
        """
        pass
    
    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        pass
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def convert_symbol(self, nexus_symbol: str) -> str:
        """
        Convert NEXUS symbol to broker format.
        
        Override in subclass if broker uses different format.
        """
        return nexus_symbol
    
    def convert_symbol_from_broker(self, broker_symbol: str) -> str:
        """
        Convert broker symbol to NEXUS format.
        
        Override in subclass if broker uses different format.
        """
        return broker_symbol
    
    async def health_check(self) -> Dict[str, Any]:
        """Check broker health."""
        base_health = await super().health_check()
        
        try:
            account = await self.get_account()
            base_health["account_connected"] = True
            base_health["equity"] = account.equity
        except Exception:
            base_health["account_connected"] = False
        
        return base_health
