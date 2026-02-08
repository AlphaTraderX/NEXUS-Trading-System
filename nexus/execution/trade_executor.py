"""
Trade Executor - The Field Crew

Actually submits orders to brokers and handles responses.
Broker-agnostic interface with specific implementations.

Think of this as the field crew that actually does the physical work -
they take the work orders and execute them on site, reporting back
on progress and any issues encountered.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid

from nexus.core.enums import Market, Direction
from nexus.execution.order_manager import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderPurpose,
    OrderManager,
)
from nexus.core.models import NexusSignal
from nexus.storage.service import get_storage_service


logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    OANDA = "oanda"
    ALPACA = "alpaca"
    IBKR = "ibkr"
    IG = "ig"
    PAPER = "paper"  # Paper trading / simulation


@dataclass
class BrokerConfig:
    """Configuration for a broker connection."""
    broker_type: BrokerType
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""
    environment: str = "practice"  # practice or live
    base_url: Optional[str] = None

    # Connection settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    success: bool
    order_id: str
    broker_order_id: Optional[str] = None
    status: str = ""
    message: str = ""
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "status": self.status,
            "message": self.message,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BrokerPosition:
    """Position as reported by broker."""
    symbol: str
    side: str  # "long" or "short"
    size: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    market_value: float = 0.0


@dataclass
class BrokerAccount:
    """Account info as reported by broker."""
    account_id: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: str = "USD"


class BaseBrokerExecutor(ABC):
    """
    Abstract base class for broker executors.

    Each broker implementation must provide these methods.
    Think of this as the contract that all field crews must follow.
    """

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connected = False
        self._last_heartbeat: Optional[datetime] = None

    @property
    def broker_name(self) -> str:
        """Return broker name."""
        return self.config.broker_type.value

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit an order to the broker.

        Args:
            order: The order to submit

        Returns:
            ExecutionResult with success/failure and fill info
        """
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> ExecutionResult:
        """
        Cancel an order at the broker.

        Args:
            broker_order_id: The broker's order ID

        Returns:
            ExecutionResult indicating success/failure
        """
        pass

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> ExecutionResult:
        """
        Get current status of an order.

        Args:
            broker_order_id: The broker's order ID

        Returns:
            ExecutionResult with current status
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get all open positions from broker.

        Returns:
            List of current positions
        """
        pass

    @abstractmethod
    async def get_account(self) -> BrokerAccount:
        """
        Get account information from broker.

        Returns:
            Account info including balance, equity, margin
        """
        pass

    @abstractmethod
    async def close_position(self, symbol: str, size: Optional[float] = None) -> ExecutionResult:
        """
        Close a position (fully or partially).

        Args:
            symbol: Symbol to close
            size: Size to close (None = close all)

        Returns:
            ExecutionResult
        """
        pass

    async def heartbeat(self) -> bool:
        """
        Check if connection is still alive.

        Returns:
            True if connection is healthy
        """
        self._last_heartbeat = datetime.utcnow()
        return self.connected

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds since last heartbeat."""
        if not self._last_heartbeat:
            return float('inf')
        return (datetime.utcnow() - self._last_heartbeat).total_seconds()


class PaperBrokerExecutor(BaseBrokerExecutor):
    """
    Paper trading executor for testing and simulation.

    Simulates order execution with configurable slippage and delays.
    Perfect for testing the full pipeline without risking real money.
    """

    def __init__(
        self,
        config: BrokerConfig,
        simulated_slippage_pct: float = 0.01,
        simulated_commission: float = 1.0,
        fill_delay_seconds: float = 0.1,
        partial_fill_probability: float = 0.0,
    ):
        super().__init__(config)
        self.simulated_slippage_pct = simulated_slippage_pct
        self.simulated_commission = simulated_commission
        self.fill_delay_seconds = fill_delay_seconds
        self.partial_fill_probability = partial_fill_probability

        # Simulated state
        self._positions: Dict[str, BrokerPosition] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._account = BrokerAccount(
            account_id=config.account_id or "PAPER-001",
            balance=100000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=100000.0,
            currency="GBP",
        )
        self._order_counter = 0

    async def connect(self) -> bool:
        """Simulate connection."""
        logger.info(f"Paper broker connected (account: {self._account.account_id})")
        self.connected = True
        self._last_heartbeat = datetime.utcnow()
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        logger.info("Paper broker disconnected")
        self.connected = False

    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Simulate order execution.

        For paper trading, we immediately fill market orders
        and queue limit orders.
        """
        if not self.connected:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                status="rejected",
                message="Not connected to broker",
            )

        # Simulate network delay
        if self.fill_delay_seconds > 0:
            await asyncio.sleep(self.fill_delay_seconds)

        # Generate broker order ID
        self._order_counter += 1
        broker_order_id = f"PAPER-{self._order_counter:06d}"

        # Store order
        self._orders[broker_order_id] = {
            "order": order,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            return await self._simulate_fill(order, broker_order_id)

        # For limit orders, check if price is already favorable
        elif order.order_type == OrderType.LIMIT:
            # In paper trading, assume limit orders fill at limit price
            # (In reality, you'd check market price)
            return await self._simulate_fill(order, broker_order_id, order.limit_price)

        # For stop orders, they become pending until triggered
        elif order.order_type == OrderType.STOP:
            self._orders[broker_order_id]["status"] = "pending_trigger"
            return ExecutionResult(
                success=True,
                order_id=order.order_id,
                broker_order_id=broker_order_id,
                status="pending",
                message="Stop order placed, waiting for trigger",
            )

        return ExecutionResult(
            success=False,
            order_id=order.order_id,
            status="rejected",
            message=f"Unsupported order type: {order.order_type}",
        )

    async def _simulate_fill(
        self,
        order: Order,
        broker_order_id: str,
        base_price: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate order fill with slippage."""

        # Determine fill price
        if base_price is None:
            base_price = order.expected_price or order.limit_price or 100.0

        # Apply slippage (unfavorable for trader)
        slippage = base_price * (self.simulated_slippage_pct / 100)
        if order.side == OrderSide.BUY:
            fill_price = base_price + slippage  # Pay more
        else:
            fill_price = base_price - slippage  # Receive less

        fill_price = round(fill_price, 4)
        fill_size = order.quantity

        # Update simulated positions
        self._update_simulated_position(order, fill_price, fill_size)

        # Update order status
        self._orders[broker_order_id]["status"] = "filled"
        self._orders[broker_order_id]["fill_price"] = fill_price
        self._orders[broker_order_id]["fill_size"] = fill_size

        logger.info(
            f"Paper fill: {order.symbol} {order.side.value} {fill_size} @ {fill_price} "
            f"(slippage: {slippage:.4f})"
        )

        return ExecutionResult(
            success=True,
            order_id=order.order_id,
            broker_order_id=broker_order_id,
            status="filled",
            message="Order filled (paper)",
            fill_price=fill_price,
            fill_size=fill_size,
            commission=self.simulated_commission,
        )

    def _update_simulated_position(self, order: Order, fill_price: float, fill_size: float):
        """Update simulated position tracking."""
        symbol = order.symbol

        if symbol not in self._positions:
            # New position
            side = "long" if order.side == OrderSide.BUY else "short"
            self._positions[symbol] = BrokerPosition(
                symbol=symbol,
                side=side,
                size=fill_size,
                avg_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
            )
        else:
            pos = self._positions[symbol]

            # Same direction = add to position
            if (order.side == OrderSide.BUY and pos.side == "long") or \
               (order.side == OrderSide.SELL and pos.side == "short"):
                # Average in
                total_cost = (pos.avg_entry_price * pos.size) + (fill_price * fill_size)
                pos.size += fill_size
                pos.avg_entry_price = total_cost / pos.size

            # Opposite direction = reduce/close position
            else:
                if fill_size >= pos.size:
                    # Close position
                    del self._positions[symbol]
                else:
                    # Reduce position
                    pos.size -= fill_size

    async def cancel_order(self, broker_order_id: str) -> ExecutionResult:
        """Cancel a paper order."""
        if broker_order_id not in self._orders:
            return ExecutionResult(
                success=False,
                order_id="",
                broker_order_id=broker_order_id,
                status="error",
                message="Order not found",
            )

        order_info = self._orders[broker_order_id]
        if order_info["status"] == "filled":
            return ExecutionResult(
                success=False,
                order_id=order_info["order"].order_id,
                broker_order_id=broker_order_id,
                status="error",
                message="Cannot cancel filled order",
            )

        order_info["status"] = "cancelled"

        return ExecutionResult(
            success=True,
            order_id=order_info["order"].order_id,
            broker_order_id=broker_order_id,
            status="cancelled",
            message="Order cancelled (paper)",
        )

    async def get_order_status(self, broker_order_id: str) -> ExecutionResult:
        """Get paper order status."""
        if broker_order_id not in self._orders:
            return ExecutionResult(
                success=False,
                order_id="",
                broker_order_id=broker_order_id,
                status="error",
                message="Order not found",
            )

        order_info = self._orders[broker_order_id]

        return ExecutionResult(
            success=True,
            order_id=order_info["order"].order_id,
            broker_order_id=broker_order_id,
            status=order_info["status"],
            fill_price=order_info.get("fill_price"),
            fill_size=order_info.get("fill_size"),
        )

    async def get_positions(self) -> List[BrokerPosition]:
        """Get paper positions."""
        return list(self._positions.values())

    async def get_account(self) -> BrokerAccount:
        """Get paper account info."""
        # Update equity based on positions
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        self._account.equity = self._account.balance + unrealized
        return self._account

    async def close_position(self, symbol: str, size: Optional[float] = None) -> ExecutionResult:
        """Close a paper position."""
        if symbol not in self._positions:
            return ExecutionResult(
                success=False,
                order_id="",
                status="error",
                message=f"No position for {symbol}",
            )

        pos = self._positions[symbol]
        close_size = size or pos.size

        # Create closing order
        side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY

        order = Order(
            order_id=str(uuid.uuid4()),
            signal_id=None,
            position_id=None,
            symbol=symbol,
            market=Market.US_STOCKS,  # Default
            side=side,
            order_type=OrderType.MARKET,
            purpose=OrderPurpose.EXIT_MANUAL,
            quantity=close_size,
            expected_price=pos.current_price,
        )

        return await self.submit_order(order)

    def set_account_balance(self, balance: float):
        """Set paper account balance (for testing)."""
        self._account.balance = balance
        self._account.equity = balance
        self._account.margin_available = balance

    def update_position_price(self, symbol: str, price: float):
        """Update position's current price (for P&L simulation)."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.avg_entry_price - price) * pos.size


class TradeExecutor:
    """
    Main trade executor that coordinates broker interactions.

    Routes orders to the appropriate broker based on market,
    handles retries, and maintains execution state.
    """

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self._brokers: Dict[str, BaseBrokerExecutor] = {}
        self._market_broker_map: Dict[Market, str] = {}
        self._default_broker: Optional[str] = None

        # Execution callbacks
        self._on_fill_callbacks: List[Callable[[Order, ExecutionResult], None]] = []
        self._on_reject_callbacks: List[Callable[[Order, ExecutionResult], None]] = []

        # Stats
        self._execution_stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "orders_cancelled": 0,
            "total_retries": 0,
        }

    def register_broker(
        self,
        name: str,
        executor: BaseBrokerExecutor,
        markets: Optional[List[Market]] = None,
        set_default: bool = False,
    ):
        """
        Register a broker executor.

        Args:
            name: Broker name (for routing)
            executor: The broker executor instance
            markets: Markets this broker handles
            set_default: Make this the default broker
        """
        self._brokers[name] = executor

        if markets:
            for market in markets:
                self._market_broker_map[market] = name

        if set_default or self._default_broker is None:
            self._default_broker = name

        logger.info(f"Registered broker: {name} (markets: {markets or 'default'})")

    def get_broker_for_market(self, market: Market) -> Optional[BaseBrokerExecutor]:
        """Get the appropriate broker for a market."""
        broker_name = self._market_broker_map.get(market, self._default_broker)
        if broker_name:
            return self._brokers.get(broker_name)
        return None

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered brokers."""
        results = {}
        for name, broker in self._brokers.items():
            try:
                success = await broker.connect()
                results[name] = success
                logger.info(f"Broker {name} connection: {'OK' if success else 'FAILED'}")
            except Exception as e:
                results[name] = False
                logger.error(f"Broker {name} connection error: {e}")
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all brokers."""
        for name, broker in self._brokers.items():
            try:
                await broker.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")

    async def execute_order(
        self,
        order: Order,
        broker_name: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute an order through the appropriate broker.

        Args:
            order: Order to execute
            broker_name: Specific broker to use (optional)

        Returns:
            ExecutionResult
        """
        # Get broker
        if broker_name:
            broker = self._brokers.get(broker_name)
        else:
            broker = self.get_broker_for_market(order.market)

        if not broker:
            result = ExecutionResult(
                success=False,
                order_id=order.order_id,
                status="rejected",
                message=f"No broker available for market {order.market}",
            )
            self._handle_rejection(order, result)
            return result

        if not broker.connected:
            result = ExecutionResult(
                success=False,
                order_id=order.order_id,
                status="rejected",
                message=f"Broker {broker.broker_name} not connected",
            )
            self._handle_rejection(order, result)
            return result

        # Mark order as submitted in OrderManager
        self.order_manager.mark_submitted(order.order_id, broker.broker_name)
        self._execution_stats["orders_submitted"] += 1

        # Submit to broker with retry logic
        result = await self._submit_with_retry(broker, order)

        # Store broker order ID on the order when we get a successful response
        if result.success and result.broker_order_id:
            order.broker_order_id = result.broker_order_id
            order.broker = broker.broker_name

        # Process result
        if result.success and result.status == "filled":
            await self._handle_fill(order, result)
        elif not result.success:
            self._handle_rejection(order, result)

        return result

    async def execute_signal(self, signal: NexusSignal) -> dict:
        """Execute a trading signal (create entry order and submit to broker)."""
        try:
            order = self.order_manager.create_entry_order(
                signal,
                order_type=OrderType.MARKET,
            )
            result = await self.execute_order(order)
            return {
                "success": result.success,
                "order_id": order.order_id,
                "error": None if result.success else result.message,
            }
        except Exception as e:
            logger.exception("execute_signal failed: %s", e)
            return {
                "success": False,
                "order_id": None,
                "error": str(e),
            }

    async def _submit_with_retry(
        self,
        broker: BaseBrokerExecutor,
        order: Order,
        max_retries: int = 3,
    ) -> ExecutionResult:
        """Submit order with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await broker.submit_order(order)

                if result.success:
                    return result

                # Don't retry on explicit rejections
                if result.status == "rejected":
                    return result

                last_error = result.message

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order submission attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                self._execution_stats["total_retries"] += 1
                await asyncio.sleep(broker.config.retry_delay_seconds * (attempt + 1))

        return ExecutionResult(
            success=False,
            order_id=order.order_id,
            status="error",
            message=f"Failed after {max_retries} attempts: {last_error}",
        )

    async def _handle_fill(self, order: Order, result: ExecutionResult):
        """Handle a successful fill."""
        # Record fill in OrderManager
        self.order_manager.record_fill(
            order.order_id,
            fill_price=result.fill_price,
            fill_size=result.fill_size,
            commission=result.commission,
            fill_time=result.timestamp,
        )

        self._execution_stats["orders_filled"] += 1

        # Log trade to database
        try:
            storage = get_storage_service()
            if storage._initialized and order.signal_id:
                fill_price = result.fill_price or 0.0
                fill_size = result.fill_size or order.quantity
                direction = "long" if order.side == OrderSide.BUY else "short"
                trade_data = {
                    "entry_price": fill_price,
                    "entry_time": result.timestamp,
                    "position_size": fill_size,
                    "direction": direction,
                    "symbol": order.symbol,
                    "market": order.market.value if hasattr(order.market, "value") else str(order.market),
                }
                await storage.save_trade(trade_data, order.signal_id)
        except Exception as e:
            logger.warning("Failed to save trade to database: %s", e)

        # Notify callbacks
        for callback in self._on_fill_callbacks:
            try:
                callback(order, result)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _handle_rejection(self, order: Order, result: ExecutionResult):
        """Handle an order rejection."""
        self.order_manager.reject_order(order.order_id, result.message)
        self._execution_stats["orders_rejected"] += 1

        # Notify callbacks
        for callback in self._on_reject_callbacks:
            try:
                callback(order, result)
            except Exception as e:
                logger.error(f"Reject callback error: {e}")

    async def cancel_order(self, order: Order) -> ExecutionResult:
        """Cancel an order at the broker."""
        if not order.broker or not order.broker_order_id:
            # Order never submitted, just cancel locally
            self.order_manager.cancel_order(order.order_id, "cancelled_before_submission")
            return ExecutionResult(
                success=True,
                order_id=order.order_id,
                status="cancelled",
                message="Cancelled before submission",
            )

        broker = self._brokers.get(order.broker)
        if not broker:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                status="error",
                message=f"Broker {order.broker} not found",
            )

        result = await broker.cancel_order(order.broker_order_id)

        if result.success:
            self.order_manager.cancel_order(order.order_id, "cancelled_at_broker")
            self._execution_stats["orders_cancelled"] += 1

        return result

    async def cancel_all_orders(self) -> List[ExecutionResult]:
        """Cancel all active orders."""
        results = []

        for order in self.order_manager.active_orders:
            result = await self.cancel_order(order)
            results.append(result)

        return results

    async def close_all_positions(self, broker_name: Optional[str] = None) -> List[ExecutionResult]:
        """Close all positions (emergency)."""
        results = []

        brokers = [self._brokers[broker_name]] if broker_name else list(self._brokers.values())

        for broker in brokers:
            try:
                positions = await broker.get_positions()
                for pos in positions:
                    result = await broker.close_position(pos.symbol)
                    results.append(result)
            except Exception as e:
                logger.error(f"Error closing positions: {e}")

        return results

    async def sync_positions(self) -> Dict[str, List[BrokerPosition]]:
        """Get positions from all brokers."""
        positions = {}

        for name, broker in self._brokers.items():
            try:
                broker_positions = await broker.get_positions()
                positions[name] = broker_positions
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {e}")
                positions[name] = []

        return positions

    async def get_account_status(self) -> Dict[str, BrokerAccount]:
        """Get account status from all brokers."""
        accounts = {}

        for name, broker in self._brokers.items():
            try:
                account = await broker.get_account()
                accounts[name] = account
            except Exception as e:
                logger.error(f"Error getting account from {name}: {e}")

        return accounts

    def on_fill(self, callback: Callable[[Order, ExecutionResult], None]):
        """Register callback for order fills."""
        self._on_fill_callbacks.append(callback)

    def on_reject(self, callback: Callable[[Order, ExecutionResult], None]):
        """Register callback for order rejections."""
        self._on_reject_callbacks.append(callback)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._execution_stats,
            "brokers_connected": sum(1 for b in self._brokers.values() if b.connected),
            "brokers_registered": len(self._brokers),
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self._execution_stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "orders_cancelled": 0,
            "total_retries": 0,
        }


# Factory function for creating executors
def create_paper_executor(
    account_balance: float = 100000.0,
    slippage_pct: float = 0.01,
    commission: float = 1.0,
) -> PaperBrokerExecutor:
    """Create a paper trading executor."""
    config = BrokerConfig(
        broker_type=BrokerType.PAPER,
        account_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
    )

    executor = PaperBrokerExecutor(
        config=config,
        simulated_slippage_pct=slippage_pct,
        simulated_commission=commission,
    )

    executor.set_account_balance(account_balance)

    return executor
