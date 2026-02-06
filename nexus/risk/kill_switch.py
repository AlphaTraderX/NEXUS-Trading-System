"""
NEXUS Kill Switch
Emergency shutdown system for catastrophic scenarios.

THIS IS THE NUCLEAR OPTION.

When triggered:
1. Cancel ALL pending orders across all brokers
2. Close ALL open positions
3. Disable ALL new trading
4. Send emergency alerts via ALL channels
5. Log everything for post-mortem

TRIGGERS:
- Manual activation (panic button)
- Daily loss exceeds limit
- Drawdown exceeds limit
- Connection loss > 5 minutes
- Data feed stale > 30 seconds
- Multiple broker errors
- Unexpected position discrepancy

KNIGHT CAPITAL WARNING:
They lost $440 million in 45 minutes without a kill switch.
This module prevents that.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KillReason(Enum):
    """Reasons for kill switch activation."""
    MANUAL = "manual"                    # Human triggered
    DAILY_LOSS = "daily_loss"            # Daily loss limit
    WEEKLY_LOSS = "weekly_loss"          # Weekly loss limit
    MAX_DRAWDOWN = "max_drawdown"        # Max drawdown hit
    CONNECTION_LOSS = "connection_loss"  # Lost broker connection
    DATA_STALE = "data_stale"            # Data feed not updating
    BROKER_ERROR = "broker_error"        # Multiple broker errors
    POSITION_MISMATCH = "position_mismatch"  # Position reconciliation failed
    SYSTEM_ERROR = "system_error"        # Unexpected system error


class KillAction(Enum):
    """Actions to take on kill switch activation."""
    CANCEL_ORDERS = "cancel_orders"      # Cancel all pending orders
    CLOSE_POSITIONS = "close_positions"  # Close all positions
    DISABLE_TRADING = "disable_trading"  # Prevent new trades
    ALERT = "alert"                      # Send emergency alerts
    LOG = "log"                          # Log the event


@dataclass
class KillEvent:
    """Record of a kill switch event."""
    timestamp: datetime
    reason: KillReason
    trigger_value: float
    threshold_value: float
    message: str
    actions_taken: List[str]
    positions_closed: int
    orders_cancelled: int
    alerts_sent: List[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "trigger_value": self.trigger_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "actions_taken": self.actions_taken,
            "positions_closed": self.positions_closed,
            "orders_cancelled": self.orders_cancelled,
            "alerts_sent": self.alerts_sent,
        }


@dataclass
class SystemState:
    """Current system state for kill switch monitoring."""
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    seconds_since_heartbeat: float = 0.0
    data_age_seconds: float = 0.0
    broker_error_count: int = 0
    position_count: int = 0
    expected_position_count: int = 0
    last_update: Optional[datetime] = None


@dataclass
class KillSwitchStatus:
    """Current status of the kill switch."""
    is_active: bool
    is_armed: bool  # Ready to trigger
    reason: Optional[KillReason]
    activated_at: Optional[datetime]
    message: str
    can_trade: bool
    requires_manual_reset: bool
    conditions_checked: Dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "is_active": self.is_active,
            "is_armed": self.is_armed,
            "reason": self.reason.value if self.reason else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "message": self.message,
            "can_trade": self.can_trade,
            "requires_manual_reset": self.requires_manual_reset,
            "conditions_checked": self.conditions_checked,
        }


class KillSwitch:
    """
    Emergency shutdown system.

    The last line of defense against catastrophic losses.
    """

    def __init__(
        self,
        # Loss thresholds (use circuit breaker values)
        daily_loss_threshold: float = -3.0,
        weekly_loss_threshold: float = -6.0,
        max_drawdown_threshold: float = -10.0,

        # Connection thresholds
        connection_timeout_seconds: float = 300.0,  # 5 minutes
        data_stale_seconds: float = 30.0,
        max_broker_errors: int = 5,

        # Position reconciliation
        position_mismatch_tolerance: int = 0,  # Must match exactly

        # Alert callbacks (set these to actual functions)
        discord_callback: Optional[Callable] = None,
        telegram_callback: Optional[Callable] = None,
        email_callback: Optional[Callable] = None,
        sms_callback: Optional[Callable] = None,
    ):
        """
        Initialize kill switch.

        Args:
            daily_loss_threshold: Daily loss % that triggers kill
            weekly_loss_threshold: Weekly loss % that triggers kill
            max_drawdown_threshold: Drawdown % that triggers kill
            connection_timeout_seconds: Seconds without heartbeat before kill
            data_stale_seconds: Seconds of stale data before kill
            max_broker_errors: Number of broker errors before kill
            position_mismatch_tolerance: Allowed position count difference
            discord_callback: Function to send Discord alert
            telegram_callback: Function to send Telegram alert
            email_callback: Function to send email alert
            sms_callback: Function to send SMS alert
        """
        # Thresholds
        self.daily_loss_threshold = daily_loss_threshold
        self.weekly_loss_threshold = weekly_loss_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.connection_timeout_seconds = connection_timeout_seconds
        self.data_stale_seconds = data_stale_seconds
        self.max_broker_errors = max_broker_errors
        self.position_mismatch_tolerance = position_mismatch_tolerance

        # Alert callbacks
        self.discord_callback = discord_callback
        self.telegram_callback = telegram_callback
        self.email_callback = email_callback
        self.sms_callback = sms_callback

        # State
        self.is_active = False
        self.is_armed = True  # Ready to trigger by default
        self.activation_reason: Optional[KillReason] = None
        self.activated_at: Optional[datetime] = None
        self.activation_message: str = ""

        # History
        self.kill_events: List[KillEvent] = []

        # Broker interfaces (set these when initializing brokers)
        self.brokers: Dict[str, Any] = {}

    def register_broker(self, name: str, broker: Any):
        """Register a broker for emergency operations."""
        self.brokers[name] = broker

    def check_conditions(self, state: SystemState) -> KillSwitchStatus:
        """
        Check all kill switch conditions.

        Args:
            state: Current system state

        Returns:
            KillSwitchStatus with current status and which conditions triggered
        """
        conditions = {}
        should_kill = False
        kill_reason = None
        kill_message = ""
        trigger_value = 0.0
        threshold_value = 0.0

        # If already active, return current status
        if self.is_active:
            return KillSwitchStatus(
                is_active=True,
                is_armed=self.is_armed,
                reason=self.activation_reason,
                activated_at=self.activated_at,
                message=self.activation_message,
                can_trade=False,
                requires_manual_reset=True,
                conditions_checked={"already_active": True},
            )

        # Check daily loss
        conditions["daily_loss"] = state.daily_pnl_pct <= self.daily_loss_threshold
        if conditions["daily_loss"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.DAILY_LOSS
            kill_message = f"Daily loss {state.daily_pnl_pct:.2f}% exceeded threshold {self.daily_loss_threshold}%"
            trigger_value = state.daily_pnl_pct
            threshold_value = self.daily_loss_threshold

        # Check weekly loss
        conditions["weekly_loss"] = state.weekly_pnl_pct <= self.weekly_loss_threshold
        if conditions["weekly_loss"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.WEEKLY_LOSS
            kill_message = f"Weekly loss {state.weekly_pnl_pct:.2f}% exceeded threshold {self.weekly_loss_threshold}%"
            trigger_value = state.weekly_pnl_pct
            threshold_value = self.weekly_loss_threshold

        # Check drawdown
        conditions["max_drawdown"] = state.drawdown_pct <= self.max_drawdown_threshold
        if conditions["max_drawdown"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.MAX_DRAWDOWN
            kill_message = f"Drawdown {state.drawdown_pct:.2f}% exceeded threshold {self.max_drawdown_threshold}%"
            trigger_value = state.drawdown_pct
            threshold_value = self.max_drawdown_threshold

        # Check connection
        conditions["connection_loss"] = state.seconds_since_heartbeat > self.connection_timeout_seconds
        if conditions["connection_loss"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.CONNECTION_LOSS
            kill_message = f"No heartbeat for {state.seconds_since_heartbeat:.0f}s (threshold: {self.connection_timeout_seconds}s)"
            trigger_value = state.seconds_since_heartbeat
            threshold_value = self.connection_timeout_seconds

        # Check data freshness
        conditions["data_stale"] = state.data_age_seconds > self.data_stale_seconds
        if conditions["data_stale"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.DATA_STALE
            kill_message = f"Data is {state.data_age_seconds:.0f}s old (threshold: {self.data_stale_seconds}s)"
            trigger_value = state.data_age_seconds
            threshold_value = self.data_stale_seconds

        # Check broker errors
        conditions["broker_errors"] = state.broker_error_count >= self.max_broker_errors
        if conditions["broker_errors"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.BROKER_ERROR
            kill_message = f"Too many broker errors: {state.broker_error_count} (threshold: {self.max_broker_errors})"
            trigger_value = state.broker_error_count
            threshold_value = self.max_broker_errors

        # Check position mismatch
        position_diff = abs(state.position_count - state.expected_position_count)
        conditions["position_mismatch"] = position_diff > self.position_mismatch_tolerance
        if conditions["position_mismatch"] and not should_kill:
            should_kill = True
            kill_reason = KillReason.POSITION_MISMATCH
            kill_message = f"Position mismatch: have {state.position_count}, expected {state.expected_position_count}"
            trigger_value = position_diff
            threshold_value = self.position_mismatch_tolerance

        # If should kill and armed, activate
        if should_kill and self.is_armed:
            self._activate(kill_reason, kill_message, trigger_value, threshold_value)

        return KillSwitchStatus(
            is_active=self.is_active,
            is_armed=self.is_armed,
            reason=self.activation_reason,
            activated_at=self.activated_at,
            message=self.activation_message if self.is_active else "System operational",
            can_trade=not self.is_active,
            requires_manual_reset=self.is_active,
            conditions_checked=conditions,
        )

    def activate_manual(self, reason: str = "Manual activation"):
        """
        Manually activate the kill switch.

        This is the panic button.
        """
        self._activate(
            KillReason.MANUAL,
            reason,
            trigger_value=0,
            threshold_value=0
        )

    def _activate(
        self,
        reason: KillReason,
        message: str,
        trigger_value: float,
        threshold_value: float
    ):
        """
        Internal activation method.

        Performs all emergency actions.
        """
        self.is_active = True
        self.activation_reason = reason
        self.activated_at = datetime.now()
        self.activation_message = message

        actions_taken = []
        positions_closed = 0
        orders_cancelled = 0
        alerts_sent = []

        # 1. Cancel all orders
        try:
            orders_cancelled = self._cancel_all_orders()
            actions_taken.append(f"Cancelled {orders_cancelled} orders")
        except Exception as e:
            actions_taken.append(f"Order cancellation failed: {e}")

        # 2. Close all positions
        try:
            positions_closed = self._close_all_positions()
            actions_taken.append(f"Closed {positions_closed} positions")
        except Exception as e:
            actions_taken.append(f"Position closure failed: {e}")

        # 3. Send alerts
        alert_message = self._format_alert_message(
            reason, message, trigger_value, threshold_value,
            positions_closed, orders_cancelled
        )

        if self.discord_callback:
            try:
                self.discord_callback(alert_message)
                alerts_sent.append("discord")
            except Exception as e:
                actions_taken.append(f"Discord alert failed: {e}")

        if self.telegram_callback:
            try:
                self.telegram_callback(alert_message)
                alerts_sent.append("telegram")
            except Exception as e:
                actions_taken.append(f"Telegram alert failed: {e}")

        if self.email_callback:
            try:
                self.email_callback("NEXUS KILL SWITCH ACTIVATED", alert_message)
                alerts_sent.append("email")
            except Exception as e:
                actions_taken.append(f"Email alert failed: {e}")

        if self.sms_callback:
            try:
                self.sms_callback(f"NEXUS KILL SWITCH: {reason.value} - {message[:100]}")
                alerts_sent.append("sms")
            except Exception as e:
                actions_taken.append(f"SMS alert failed: {e}")

        # Log event
        event = KillEvent(
            timestamp=self.activated_at,
            reason=reason,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            message=message,
            actions_taken=actions_taken,
            positions_closed=positions_closed,
            orders_cancelled=orders_cancelled,
            alerts_sent=alerts_sent,
        )
        self.kill_events.append(event)

        print(f"\n{'='*60}")
        print("[!!] KILL SWITCH ACTIVATED [!!]")
        print(f"{'='*60}")
        print(f"Reason: {reason.value}")
        print(f"Message: {message}")
        print(f"Positions closed: {positions_closed}")
        print(f"Orders cancelled: {orders_cancelled}")
        print(f"Alerts sent: {alerts_sent}")
        print(f"{'='*60}\n")

    def _cancel_all_orders(self) -> int:
        """Cancel all pending orders across all brokers."""
        total_cancelled = 0
        for name, broker in self.brokers.items():
            if hasattr(broker, 'cancel_all_orders'):
                try:
                    count = broker.cancel_all_orders()
                    total_cancelled += count if count else 0
                except Exception as e:
                    print(f"Error cancelling orders on {name}: {e}")
        return total_cancelled

    def _close_all_positions(self) -> int:
        """Close all positions across all brokers."""
        total_closed = 0
        for name, broker in self.brokers.items():
            if hasattr(broker, 'close_all_positions'):
                try:
                    count = broker.close_all_positions()
                    total_closed += count if count else 0
                except Exception as e:
                    print(f"Error closing positions on {name}: {e}")
        return total_closed

    def _format_alert_message(
        self,
        reason: KillReason,
        message: str,
        trigger_value: float,
        threshold_value: float,
        positions_closed: int,
        orders_cancelled: int
    ) -> str:
        """Format emergency alert message."""
        return f"""
🚨🚨🚨 NEXUS KILL SWITCH ACTIVATED 🚨🚨🚨

REASON: {reason.value.upper()}
MESSAGE: {message}

TRIGGER: {trigger_value}
THRESHOLD: {threshold_value}

ACTIONS TAKEN:
- Positions closed: {positions_closed}
- Orders cancelled: {orders_cancelled}

TIME: {datetime.now().isoformat()}

⚠️ MANUAL INTERVENTION REQUIRED ⚠️
Trading is DISABLED until manual reset.
"""

    def reset(self, confirmation: str = ""):
        """
        Reset the kill switch (requires confirmation).

        Args:
            confirmation: Must be "CONFIRM_RESET" to actually reset
        """
        if confirmation != "CONFIRM_RESET":
            print("Reset requires confirmation='CONFIRM_RESET'")
            return False

        self.is_active = False
        self.activation_reason = None
        self.activated_at = None
        self.activation_message = ""

        print("Kill switch reset. Trading enabled.")
        return True

    def arm(self):
        """Arm the kill switch (enable automatic triggering)."""
        self.is_armed = True
        print("Kill switch ARMED")

    def disarm(self):
        """Disarm the kill switch (disable automatic triggering)."""
        self.is_armed = False
        print("Kill switch DISARMED - manual activation still possible")

    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            "is_active": self.is_active,
            "is_armed": self.is_armed,
            "reason": self.activation_reason.value if self.activation_reason else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "message": self.activation_message,
            "can_trade": not self.is_active,
            "total_events": len(self.kill_events),
        }

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent kill switch events."""
        return [e.to_dict() for e in self.kill_events[-limit:]]


# Test the kill switch
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS KILL SWITCH TEST")
    print("=" * 60)

    # Mock alert callbacks for testing
    alerts_received = []

    def mock_discord(msg):
        alerts_received.append(("discord", msg[:50]))
        print(f"[DISCORD] Alert sent")

    def mock_telegram(msg):
        alerts_received.append(("telegram", msg[:50]))
        print(f"[TELEGRAM] Alert sent")

    kill_switch = KillSwitch(
        daily_loss_threshold=-3.0,
        weekly_loss_threshold=-6.0,
        max_drawdown_threshold=-10.0,
        connection_timeout_seconds=300,
        data_stale_seconds=30,
        max_broker_errors=5,
        discord_callback=mock_discord,
        telegram_callback=mock_telegram,
    )

    # Test 1: All clear
    print("\n--- Test 1: All Clear ---")
    state = SystemState(
        daily_pnl_pct=1.0,
        weekly_pnl_pct=3.0,
        drawdown_pct=-2.0,
        seconds_since_heartbeat=5,
        data_age_seconds=2,
        broker_error_count=0,
        position_count=3,
        expected_position_count=3,
    )
    status = kill_switch.check_conditions(state)
    print(f"Is active: {status.is_active}")
    print(f"Can trade: {status.can_trade}")
    print(f"Message: {status.message}")

    # Test 2: Daily loss trigger
    print("\n--- Test 2: Daily Loss Trigger ---")
    state.daily_pnl_pct = -3.5
    status = kill_switch.check_conditions(state)
    print(f"Is active: {status.is_active}")
    print(f"Reason: {status.reason}")
    print(f"Can trade: {status.can_trade}")

    # Reset for more tests
    kill_switch.reset("CONFIRM_RESET")

    # Test 3: Connection loss
    print("\n--- Test 3: Connection Loss ---")
    state = SystemState(
        daily_pnl_pct=1.0,
        weekly_pnl_pct=2.0,
        drawdown_pct=-3.0,
        seconds_since_heartbeat=400,  # > 300 threshold
        data_age_seconds=5,
        broker_error_count=0,
        position_count=2,
        expected_position_count=2,
    )
    status = kill_switch.check_conditions(state)
    print(f"Is active: {status.is_active}")
    print(f"Reason: {status.reason}")

    # Reset
    kill_switch.reset("CONFIRM_RESET")

    # Test 4: Manual activation
    print("\n--- Test 4: Manual Activation ---")
    kill_switch.activate_manual("Testing panic button")
    print(f"Is active: {kill_switch.is_active}")

    # Test 5: Reset without confirmation
    print("\n--- Test 5: Reset Without Confirmation ---")
    result = kill_switch.reset("wrong_confirmation")
    print(f"Reset successful: {result}")
    print(f"Still active: {kill_switch.is_active}")

    # Test 6: Proper reset
    print("\n--- Test 6: Proper Reset ---")
    result = kill_switch.reset("CONFIRM_RESET")
    print(f"Reset successful: {result}")
    print(f"Is active: {kill_switch.is_active}")

    # Test 7: Disarm and check
    print("\n--- Test 7: Disarmed Kill Switch ---")
    kill_switch.disarm()
    state = SystemState(
        daily_pnl_pct=-5.0,  # Would trigger if armed
        weekly_pnl_pct=-8.0,
        drawdown_pct=-12.0,
        seconds_since_heartbeat=500,
        data_age_seconds=60,
        broker_error_count=10,
        position_count=5,
        expected_position_count=3,
    )
    status = kill_switch.check_conditions(state)
    print(f"Is armed: {status.is_armed}")
    print(f"Is active: {status.is_active}")
    print(f"Conditions triggered: {[k for k, v in status.conditions_checked.items() if v]}")
    print("(Kill switch did NOT activate because it's disarmed)")

    # Re-arm
    kill_switch.arm()

    # Test 8: Position mismatch
    print("\n--- Test 8: Position Mismatch ---")
    kill_switch.reset("CONFIRM_RESET")
    state = SystemState(
        daily_pnl_pct=1.0,
        weekly_pnl_pct=2.0,
        drawdown_pct=-3.0,
        seconds_since_heartbeat=5,
        data_age_seconds=2,
        broker_error_count=0,
        position_count=5,
        expected_position_count=2,  # Mismatch!
    )
    status = kill_switch.check_conditions(state)
    print(f"Is active: {status.is_active}")
    print(f"Reason: {status.reason}")

    # Test 9: History
    print("\n--- Test 9: Kill Event History ---")
    history = kill_switch.get_history(5)
    print(f"Total events: {len(history)}")
    for event in history:
        print(f"  - {event['reason']}: {event['message'][:50]}...")

    # Test 10: Alerts received
    print("\n--- Test 10: Alerts Sent ---")
    print(f"Total alerts: {len(alerts_received)}")
    for channel, preview in alerts_received[:3]:
        safe_preview = preview.encode("ascii", "replace").decode("ascii")
        print(f"  [{channel}] {safe_preview}...")

    print("\n" + "=" * 60)
    print("KILL SWITCH TEST COMPLETE [OK]")
    print("=" * 60)
