"""
NEXUS Hardware Stop Protection

Places stop orders directly with broker as hardware protection.
These execute even if NEXUS crashes or disconnects.

Philosophy:
- Software stops = first line of defense (fast, flexible)
- Hardware stops = second line of defense (guaranteed execution)
- Hardware stops should be WIDER than software stops (emergency only)
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class HardwareStop:
    """Tracks a hardware stop order placed with broker."""

    position_id: str
    symbol: str
    broker_order_id: str
    stop_price: float
    quantity: float
    direction: str  # LONG or SHORT - determines if stop is sell or buy
    placed_at: datetime
    status: str = "ACTIVE"  # ACTIVE, TRIGGERED, CANCELLED

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "broker_order_id": self.broker_order_id,
            "stop_price": self.stop_price,
            "quantity": self.quantity,
            "direction": self.direction,
            "placed_at": self.placed_at.isoformat(),
            "status": self.status,
        }


class HardwareStopManager:
    """
    Manages hardware (broker-side) stop orders.

    For each position, places a stop order with the broker that will
    execute even if NEXUS disconnects.

    Hardware stop is placed WIDER than software stop:
    - Software stop: 1.5 ATR (normal exit)
    - Hardware stop: 2.5 ATR (emergency only, bigger loss but guaranteed)
    """

    HARDWARE_STOP_MULTIPLIER = 1.5  # Hardware stop is 1.5x further than software stop

    def __init__(self, broker: Any):
        """
        Args:
            broker: Broker instance with place_order, cancel_order methods
        """
        self.broker = broker
        self._stops: Dict[str, HardwareStop] = {}  # position_id -> HardwareStop

    async def place_hardware_stop(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        software_stop: float,
        quantity: float,
    ) -> Optional[HardwareStop]:
        """
        Place a hardware stop order with broker.

        Hardware stop is placed further out than software stop
        as a catastrophic protection layer.
        """
        # Calculate hardware stop price (wider than software)
        stop_distance = abs(entry_price - software_stop)
        hardware_distance = stop_distance * self.HARDWARE_STOP_MULTIPLIER

        if direction.upper() == "LONG":
            hardware_stop_price = entry_price - hardware_distance
            order_side = "SELL"
        else:
            hardware_stop_price = entry_price + hardware_distance
            order_side = "BUY"

        # Round to appropriate precision
        hardware_stop_price = round(hardware_stop_price, 2)

        try:
            # Place stop order with broker
            order_result = await self.broker.place_order(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "quantity": quantity,
                    "order_type": "STOP",
                    "stop_price": hardware_stop_price,
                    "time_in_force": "GTC",  # Good til cancelled
                }
            )

            if not order_result or not order_result.get("order_id"):
                logger.error(f"Failed to place hardware stop for {symbol}")
                return None

            stop = HardwareStop(
                position_id=position_id,
                symbol=symbol,
                broker_order_id=order_result["order_id"],
                stop_price=hardware_stop_price,
                quantity=quantity,
                direction=direction.upper(),
                placed_at=datetime.now(timezone.utc),
            )

            self._stops[position_id] = stop

            logger.info(
                f"Hardware stop placed: {symbol} @ {hardware_stop_price} "
                f"(software: {software_stop}, hardware {self.HARDWARE_STOP_MULTIPLIER}x wider)"
            )

            return stop

        except Exception as e:
            logger.error(f"Error placing hardware stop for {symbol}: {e}")
            return None

    async def cancel_hardware_stop(self, position_id: str) -> bool:
        """
        Cancel hardware stop when position is closed normally.

        Call this when:
        - Position hits software stop (don't need hardware)
        - Position hits take profit
        - Position manually closed
        """
        if position_id not in self._stops:
            return True  # No stop to cancel

        stop = self._stops[position_id]

        try:
            result = await self.broker.cancel_order(stop.broker_order_id)

            if result:
                stop.status = "CANCELLED"
                del self._stops[position_id]
                logger.info(f"Hardware stop cancelled for {stop.symbol}")
                return True
            else:
                logger.warning(f"Failed to cancel hardware stop for {stop.symbol}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling hardware stop: {e}")
            return False

    async def update_hardware_stop(
        self,
        position_id: str,
        new_software_stop: float,
        current_price: float,
    ) -> bool:
        """
        Update hardware stop when trailing or moving stops.

        Hardware stop moves with software stop but stays wider.
        """
        if position_id not in self._stops:
            return False

        stop = self._stops[position_id]

        # Calculate new hardware stop
        stop_distance = abs(current_price - new_software_stop)
        hardware_distance = stop_distance * self.HARDWARE_STOP_MULTIPLIER

        if stop.direction == "LONG":
            new_hardware_stop = current_price - hardware_distance
            # Only move stop up (trailing)
            if new_hardware_stop <= stop.stop_price:
                return True  # No update needed
        else:
            new_hardware_stop = current_price + hardware_distance
            # Only move stop down (trailing)
            if new_hardware_stop >= stop.stop_price:
                return True

        new_hardware_stop = round(new_hardware_stop, 2)

        try:
            # Cancel old, place new (some brokers don't support modify)
            await self.broker.cancel_order(stop.broker_order_id)

            order_result = await self.broker.place_order(
                {
                    "symbol": stop.symbol,
                    "side": "SELL" if stop.direction == "LONG" else "BUY",
                    "quantity": stop.quantity,
                    "order_type": "STOP",
                    "stop_price": new_hardware_stop,
                    "time_in_force": "GTC",
                }
            )

            if order_result and order_result.get("order_id"):
                stop.broker_order_id = order_result["order_id"]
                stop.stop_price = new_hardware_stop
                logger.info(
                    f"Hardware stop updated: {stop.symbol} @ {new_hardware_stop}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating hardware stop: {e}")
            return False

    async def sync_with_broker(self) -> Dict[str, str]:
        """
        Sync local state with broker's actual orders.

        Call this on startup to recover state after restart.
        """
        results: Dict[str, str] = {}

        try:
            # Get all open orders from broker
            broker_orders = await self.broker.get_open_orders()
            broker_order_ids = {
                o["order_id"]
                for o in broker_orders
                if o.get("order_type") == "STOP"
            }

            # Check our stops against broker
            for position_id, stop in list(self._stops.items()):
                if stop.broker_order_id not in broker_order_ids:
                    # Order no longer exists - might have triggered
                    stop.status = "TRIGGERED"
                    results[position_id] = "TRIGGERED"
                    logger.warning(
                        f"Hardware stop for {stop.symbol} may have triggered"
                    )
                else:
                    results[position_id] = "ACTIVE"

            return results

        except Exception as e:
            logger.error(f"Error syncing hardware stops: {e}")
            return {}

    def get_all_stops(self) -> List[dict]:
        """Get all active hardware stops."""
        return [s.to_dict() for s in self._stops.values() if s.status == "ACTIVE"]

    async def cancel_all_stops(self) -> int:
        """Cancel all hardware stops (emergency or shutdown)."""
        cancelled = 0
        for position_id in list(self._stops.keys()):
            if await self.cancel_hardware_stop(position_id):
                cancelled += 1
        return cancelled
