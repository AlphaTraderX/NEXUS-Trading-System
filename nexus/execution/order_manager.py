"""
Order Manager - The Procurement Specialist

Handles the order lifecycle:
- Create orders from signals (place material orders)
- Track order status (pending -> submitted -> filled)
- Handle partial fills (partial deliveries)
- Calculate slippage (expected vs actual price)
- Order timeout handling (cancel stale orders)

Think of this as the procurement specialist who places orders
for materials, tracks their delivery, and flags any price changes.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from nexus.core.enums import Market, Direction
from nexus.core.models import NexusSignal


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Type of order."""
    MARKET = "market"           # Execute at current market price
    LIMIT = "limit"             # Execute at specified price or better
    STOP = "stop"               # Trigger market order when price hits stop
    STOP_LIMIT = "stop_limit"   # Trigger limit order when price hits stop


class OrderSide(Enum):
    """Side of the order (maps to Direction but broker-friendly)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle status."""
    CREATED = "created"         # Order object created
    PENDING = "pending"         # Waiting to be submitted
    SUBMITTED = "submitted"     # Sent to broker
    PARTIAL = "partial"         # Partially filled
    FILLED = "filled"           # Completely filled
    CANCELLED = "cancelled"     # Cancelled before fill
    REJECTED = "rejected"       # Rejected by broker
    EXPIRED = "expired"         # Timed out


class OrderPurpose(Enum):
    """What the order is for."""
    ENTRY = "entry"             # Opening a position
    EXIT_STOP = "exit_stop"     # Stop loss exit
    EXIT_TARGET = "exit_target" # Take profit exit
    EXIT_MANUAL = "exit_manual" # Manual close
    EXIT_SIGNAL = "exit_signal" # Signal-based exit


@dataclass
class OrderFill:
    """Represents a fill (execution) of an order."""
    fill_id: str
    order_id: str
    filled_at: datetime
    fill_price: float
    fill_size: float
    commission: float = 0.0

    @property
    def fill_value(self) -> float:
        """Total value of this fill."""
        return self.fill_price * self.fill_size


@dataclass
class Order:
    """
    Represents an order to be sent to a broker.

    This is our internal order representation - will be converted
    to broker-specific format when submitted.
    """
    # Identifiers
    order_id: str
    signal_id: Optional[str]    # Link back to signal (None for manual orders)
    position_id: Optional[str]  # Link to position (for exit orders)

    # Order details
    symbol: str
    market: Market
    side: OrderSide
    order_type: OrderType
    purpose: OrderPurpose

    # Sizing
    quantity: float             # Requested quantity
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0

    # Pricing
    limit_price: Optional[float] = None   # For limit orders
    stop_price: Optional[float] = None    # For stop orders
    expected_price: float = 0.0           # Price when signal was generated

    # Fill tracking
    avg_fill_price: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)

    # Status and timing
    status: OrderStatus = OrderStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Broker info
    broker_order_id: Optional[str] = None
    broker: Optional[str] = None

    # Slippage tracking
    slippage: float = 0.0           # In price terms
    slippage_pct: float = 0.0       # As percentage

    # Costs
    total_commission: float = 0.0

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        """Initialize calculated fields."""
        self.remaining_quantity = self.quantity - self.filled_quantity

    def add_fill(self, fill: OrderFill):
        """Add a fill to this order."""
        self.fills.append(fill)
        self.filled_quantity += fill.fill_size
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_commission += fill.commission

        # Update average fill price
        total_value = sum(f.fill_price * f.fill_size for f in self.fills)
        self.avg_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0

        # Calculate slippage
        if self.expected_price > 0:
            self.slippage = self.avg_fill_price - self.expected_price
            self.slippage_pct = (self.slippage / self.expected_price) * 100

            # For sell orders, positive slippage is good
            if self.side == OrderSide.SELL:
                self.slippage = -self.slippage
                self.slippage_pct = -self.slippage_pct

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.filled_at
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled or cancelled)."""
        return self.status in (
            OrderStatus.CREATED,
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        )

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def fill_rate(self) -> float:
        """Percentage of order that has been filled."""
        if self.quantity <= 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def total_fill_value(self) -> float:
        """Total value of all fills."""
        return sum(f.fill_value for f in self.fills)

    @property
    def time_to_fill_seconds(self) -> Optional[float]:
        """Time from submission to complete fill."""
        if self.submitted_at and self.filled_at:
            return (self.filled_at - self.submitted_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "market": self.market.value if isinstance(self.market, Market) else self.market,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "purpose": self.purpose.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "expected_price": self.expected_price,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "slippage": self.slippage,
            "slippage_pct": self.slippage_pct,
            "total_commission": self.total_commission,
            "fill_rate": self.fill_rate,
            "broker_order_id": self.broker_order_id,
        }


@dataclass
class SlippageStats:
    """Aggregated slippage statistics."""
    total_orders: int = 0
    filled_orders: int = 0
    total_slippage: float = 0.0
    avg_slippage: float = 0.0
    avg_slippage_pct: float = 0.0
    max_slippage: float = 0.0
    max_slippage_pct: float = 0.0
    positive_slippage_count: int = 0   # Orders with favorable slippage
    negative_slippage_count: int = 0   # Orders with unfavorable slippage
    avg_fill_time_seconds: float = 0.0


class OrderManager:
    """
    Manages order lifecycle from creation to fill.

    Responsibilities:
    - Create orders from signals
    - Track order status
    - Handle fills (including partial fills)
    - Calculate and track slippage
    - Handle order timeouts
    - Provide order queries and statistics
    """

    DEFAULT_TIMEOUT_MINUTES = 5
    MAX_RETRY_COUNT = 3

    def __init__(
        self,
        default_order_type: OrderType = OrderType.LIMIT,
        default_timeout_minutes: int = 5,
        max_slippage_pct: float = 0.5,
    ):
        """
        Initialize the Order Manager.

        Args:
            default_order_type: Default order type for new orders
            default_timeout_minutes: Default order timeout
            max_slippage_pct: Maximum acceptable slippage percentage
        """
        self.default_order_type = default_order_type
        self.default_timeout_minutes = default_timeout_minutes
        self.max_slippage_pct = max_slippage_pct

        # Order storage
        self._orders: Dict[str, Order] = {}  # order_id -> Order
        self._orders_by_signal: Dict[str, List[str]] = {}  # signal_id -> [order_ids]
        self._orders_by_position: Dict[str, List[str]] = {}  # position_id -> [order_ids]
        self._orders_by_symbol: Dict[str, List[str]] = {}  # symbol -> [order_ids]

        # Completed orders history
        self._completed_orders: List[Order] = []

        # Statistics
        self._session_stats = {
            "orders_created": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0,
        }

    def create_entry_order(
        self,
        signal: NexusSignal,
        order_type: Optional[OrderType] = None,
        limit_price: Optional[float] = None,
        timeout_minutes: Optional[int] = None,
    ) -> Order:
        """
        Create an entry order from a signal.

        Args:
            signal: The signal to create an order for
            order_type: Order type (defaults to manager's default)
            limit_price: Limit price (uses signal entry if not specified)
            timeout_minutes: Order timeout (uses default if not specified)

        Returns:
            The created Order object
        """
        # Determine order type
        if order_type is None:
            order_type = self.default_order_type

        # Determine side from direction
        direction = signal.direction if isinstance(signal.direction, Direction) else Direction(signal.direction)
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL

        # Determine market
        market = signal.market if isinstance(signal.market, Market) else Market(signal.market)

        # Set limit price for limit orders
        if order_type == OrderType.LIMIT and limit_price is None:
            limit_price = signal.entry_price

        # Calculate expiry
        timeout = timeout_minutes or self.default_timeout_minutes
        expires_at = datetime.utcnow() + timedelta(minutes=timeout)

        order = Order(
            order_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            position_id=None,  # Will be set when position is created
            symbol=signal.symbol,
            market=market,
            side=side,
            order_type=order_type,
            purpose=OrderPurpose.ENTRY,
            quantity=signal.position_size,
            limit_price=limit_price,
            expected_price=signal.entry_price,
            expires_at=expires_at,
            status=OrderStatus.CREATED,
        )

        # Store order
        self._store_order(order)
        self._session_stats["orders_created"] += 1

        logger.info(
            f"Entry order created: {order.order_id[:8]} for {signal.symbol} "
            f"({side.value} {order.quantity} @ {limit_price or 'market'})"
        )

        return order

    def create_exit_order(
        self,
        position_id: str,
        symbol: str,
        market: Market,
        side: OrderSide,
        quantity: float,
        purpose: OrderPurpose,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        expected_price: Optional[float] = None,
        signal_id: Optional[str] = None,
    ) -> Order:
        """
        Create an exit order for a position.

        Args:
            position_id: ID of the position to exit
            symbol: Symbol to trade
            market: Market
            side: Order side (opposite of position direction)
            quantity: Quantity to close
            purpose: Purpose of the exit
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            expected_price: Expected fill price
            signal_id: Optional link to signal

        Returns:
            The created Order object
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            signal_id=signal_id,
            position_id=position_id,
            symbol=symbol,
            market=market,
            side=side,
            order_type=order_type,
            purpose=purpose,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            expected_price=expected_price or 0.0,
            status=OrderStatus.CREATED,
        )

        # Store order
        self._store_order(order)
        self._session_stats["orders_created"] += 1

        logger.info(
            f"Exit order created: {order.order_id[:8]} for position {position_id[:8]} "
            f"({side.value} {quantity} {symbol}, purpose: {purpose.value})"
        )

        return order

    def create_stop_loss_order(
        self,
        position_id: str,
        symbol: str,
        market: Market,
        direction: Direction,
        quantity: float,
        stop_price: float,
        signal_id: Optional[str] = None,
    ) -> Order:
        """
        Create a stop loss order for a position.

        Args:
            position_id: Position to protect
            symbol: Symbol
            market: Market
            direction: Position direction (stop will be opposite side)
            quantity: Quantity to close
            stop_price: Stop trigger price
            signal_id: Optional signal ID

        Returns:
            The created stop order
        """
        # Stop loss sells for long, buys for short
        side = OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY

        return self.create_exit_order(
            position_id=position_id,
            symbol=symbol,
            market=market,
            side=side,
            quantity=quantity,
            purpose=OrderPurpose.EXIT_STOP,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            expected_price=stop_price,
            signal_id=signal_id,
        )

    def create_take_profit_order(
        self,
        position_id: str,
        symbol: str,
        market: Market,
        direction: Direction,
        quantity: float,
        limit_price: float,
        signal_id: Optional[str] = None,
    ) -> Order:
        """
        Create a take profit order for a position.

        Args:
            position_id: Position to take profit on
            symbol: Symbol
            market: Market
            direction: Position direction
            quantity: Quantity to close
            limit_price: Target price
            signal_id: Optional signal ID

        Returns:
            The created limit order
        """
        # Take profit sells for long, buys for short
        side = OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY

        return self.create_exit_order(
            position_id=position_id,
            symbol=symbol,
            market=market,
            side=side,
            quantity=quantity,
            purpose=OrderPurpose.EXIT_TARGET,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            expected_price=limit_price,
            signal_id=signal_id,
        )

    def mark_submitted(
        self,
        order_id: str,
        broker: str,
        broker_order_id: Optional[str] = None,
    ) -> Order:
        """
        Mark an order as submitted to broker.

        Args:
            order_id: Order ID
            broker: Broker name
            broker_order_id: Broker's order ID

        Returns:
            Updated order
        """
        order = self._get_order(order_id)

        if order.status not in (OrderStatus.CREATED, OrderStatus.PENDING):
            raise ValueError(f"Order {order_id} cannot be submitted (status: {order.status})")

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        order.broker = broker
        order.broker_order_id = broker_order_id

        logger.info(f"Order {order_id[:8]} submitted to {broker}")

        return order

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_size: float,
        commission: float = 0.0,
        fill_time: Optional[datetime] = None,
    ) -> Order:
        """
        Record a fill for an order.

        Args:
            order_id: Order ID
            fill_price: Price at which order was filled
            fill_size: Quantity filled
            commission: Commission for this fill
            fill_time: Time of fill

        Returns:
            Updated order
        """
        order = self._get_order(order_id)

        if not order.is_active:
            raise ValueError(f"Order {order_id} is not active (status: {order.status})")

        if fill_size > order.remaining_quantity:
            raise ValueError(
                f"Fill size {fill_size} exceeds remaining quantity {order.remaining_quantity}"
            )

        fill = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            filled_at=fill_time or datetime.utcnow(),
            fill_price=fill_price,
            fill_size=fill_size,
            commission=commission,
        )

        order.add_fill(fill)

        # Update session stats
        self._session_stats["total_slippage"] += abs(order.slippage) * fill_size
        self._session_stats["total_commission"] += commission

        if order.status == OrderStatus.FILLED:
            self._session_stats["orders_filled"] += 1
            self._move_to_completed(order)
            logger.info(
                f"Order {order_id[:8]} FILLED @ {order.avg_fill_price:.4f} "
                f"(slippage: {order.slippage_pct:.3f}%)"
            )
        else:
            logger.info(
                f"Order {order_id[:8]} partial fill: {fill_size} @ {fill_price} "
                f"({order.fill_rate:.1f}% complete)"
            )

        return order

    def cancel_order(
        self,
        order_id: str,
        reason: str = "cancelled",
    ) -> Order:
        """
        Cancel an order.

        Args:
            order_id: Order ID
            reason: Cancellation reason

        Returns:
            Updated order
        """
        order = self._get_order(order_id)

        if not order.is_active:
            raise ValueError(f"Order {order_id} cannot be cancelled (status: {order.status})")

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.utcnow()
        order.error_message = reason

        self._session_stats["orders_cancelled"] += 1
        self._move_to_completed(order)

        logger.info(f"Order {order_id[:8]} cancelled: {reason}")

        return order

    def reject_order(
        self,
        order_id: str,
        reason: str,
    ) -> Order:
        """
        Mark an order as rejected by broker.

        Args:
            order_id: Order ID
            reason: Rejection reason

        Returns:
            Updated order
        """
        order = self._get_order(order_id)

        order.status = OrderStatus.REJECTED
        order.error_message = reason

        self._session_stats["orders_rejected"] += 1
        self._move_to_completed(order)

        logger.info(f"Order {order_id[:8]} rejected: {reason}")

        return order

    def check_expired_orders(self) -> List[Order]:
        """
        Check for and expire timed-out orders.

        Returns:
            List of expired orders
        """
        now = datetime.utcnow()
        expired = []

        for order in list(self._orders.values()):
            if order.is_active and order.expires_at and now > order.expires_at:
                order.status = OrderStatus.EXPIRED
                order.error_message = "Order timed out"
                self._move_to_completed(order)
                expired.append(order)
                logger.info(f"Order {order.order_id[:8]} expired")

        return expired

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def get_orders_for_signal(self, signal_id: str) -> List[Order]:
        """Get all orders for a signal."""
        order_ids = self._orders_by_signal.get(signal_id, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_orders_for_position(self, position_id: str) -> List[Order]:
        """Get all orders for a position."""
        order_ids = self._orders_by_position.get(position_id, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    @property
    def active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [o for o in self._orders.values() if o.is_active]

    @property
    def pending_orders(self) -> List[Order]:
        """Get orders waiting to be submitted."""
        return [o for o in self._orders.values() if o.status in (OrderStatus.CREATED, OrderStatus.PENDING)]

    @property
    def submitted_orders(self) -> List[Order]:
        """Get orders that have been submitted."""
        return [o for o in self._orders.values() if o.status == OrderStatus.SUBMITTED]

    def get_slippage_stats(self) -> SlippageStats:
        """Calculate slippage statistics."""
        stats = SlippageStats()

        filled_orders = [o for o in self._completed_orders if o.status == OrderStatus.FILLED]
        stats.total_orders = len(self._completed_orders)
        stats.filled_orders = len(filled_orders)

        if not filled_orders:
            return stats

        slippages = [o.slippage_pct for o in filled_orders if o.expected_price > 0]
        fill_times = [o.time_to_fill_seconds for o in filled_orders if o.time_to_fill_seconds is not None]

        if slippages:
            stats.total_slippage = sum(abs(s) for s in slippages)
            stats.avg_slippage_pct = sum(slippages) / len(slippages)
            stats.max_slippage_pct = max(abs(s) for s in slippages)
            stats.positive_slippage_count = sum(1 for s in slippages if s < 0)  # Better than expected
            stats.negative_slippage_count = sum(1 for s in slippages if s > 0)  # Worse than expected

        if fill_times:
            stats.avg_fill_time_seconds = sum(fill_times) / len(fill_times)

        return stats

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            **self._session_stats,
            "active_orders": len(self.active_orders),
            "pending_orders": len(self.pending_orders),
            "submitted_orders": len(self.submitted_orders),
        }

    def reset_session_stats(self):
        """Reset session statistics."""
        self._session_stats = {
            "orders_created": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0,
        }

    def link_order_to_position(self, order_id: str, position_id: str):
        """Link an order to a position (typically after entry fill)."""
        order = self._get_order(order_id)
        order.position_id = position_id

        if position_id not in self._orders_by_position:
            self._orders_by_position[position_id] = []
        if order_id not in self._orders_by_position[position_id]:
            self._orders_by_position[position_id].append(order_id)

    def _store_order(self, order: Order):
        """Store an order in all indexes."""
        self._orders[order.order_id] = order

        if order.signal_id:
            if order.signal_id not in self._orders_by_signal:
                self._orders_by_signal[order.signal_id] = []
            self._orders_by_signal[order.signal_id].append(order.order_id)

        if order.position_id:
            if order.position_id not in self._orders_by_position:
                self._orders_by_position[order.position_id] = []
            self._orders_by_position[order.position_id].append(order.order_id)

        if order.symbol not in self._orders_by_symbol:
            self._orders_by_symbol[order.symbol] = []
        self._orders_by_symbol[order.symbol].append(order.order_id)

    def _move_to_completed(self, order: Order):
        """Move a completed order to history."""
        self._completed_orders.append(order)

    def _get_order(self, order_id: str) -> Order:
        """Get an order or raise error."""
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        return order

    def get_orders_summary(self) -> str:
        """Get a human-readable summary of active orders."""
        active = self.active_orders
        if not active:
            return "No active orders"

        lines = [f"Active Orders ({len(active)}):"]
        for order in active:
            price_str = f"@ {order.limit_price}" if order.limit_price else "@ market"
            lines.append(
                f"  {order.symbol} {order.side.value.upper()} {order.quantity} {price_str} "
                f"[{order.status.value}] ({order.purpose.value})"
            )

        return "\n".join(lines)
