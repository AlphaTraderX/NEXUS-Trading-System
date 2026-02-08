"""
NEXUS Kill Switch - Emergency shutdown system for risk management.

When triggered:
1. Cancels all pending orders
2. Closes all open positions
3. Disables new trade entry
4. Sends emergency alerts (via logging; integrate with Discord/Telegram in your app)
5. Requires manual reset after cooldown

CRITICAL: This is the last line of defense. It must ALWAYS work.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from nexus.config.settings import get_settings
from nexus.core.enums import KillSwitchAction, KillSwitchTrigger
from nexus.core.exceptions import KillSwitchError
from nexus.core.models import KillSwitchState, SystemHealth
from nexus.risk.state_persistence import get_risk_persistence

logger = logging.getLogger(__name__)

UTC = timezone.utc


def _build_system_status(
    last_heartbeat: Optional[datetime],
    last_data_update: Optional[datetime],
    active_errors: List[str],
    drawdown_pct: float = 0.0,
    is_connected: bool = True,
) -> Dict[str, Any]:
    """Build system_status dict from internal state."""
    return {
        "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
        "last_data_update": last_data_update.isoformat() if last_data_update else None,
        "drawdown_pct": drawdown_pct,
        "is_connected": is_connected,
        "active_errors": list(active_errors),
    }


class KillSwitch:
    """
    Emergency shutdown system for NEXUS.

    When triggered:
    1. Cancels all pending orders
    2. Closes all open positions
    3. Disables new trade entry
    4. Sends emergency alerts
    5. Requires manual reset

    CRITICAL: This is the last line of defense. It must ALWAYS work.
    """

    def __init__(self, settings: Any = None):
        self.settings = settings or get_settings()

        # Thresholds
        self.max_drawdown = getattr(self.settings, "max_drawdown", -10.0)
        self.connection_timeout = getattr(self.settings, "connection_timeout_seconds", 300)
        self.data_stale_threshold = getattr(self.settings, "data_stale_threshold_seconds", 30)
        self.cooldown_minutes = getattr(self.settings, "kill_switch_cooldown_minutes", 60)
        self.enabled = getattr(self.settings, "kill_switch_enabled", True)

        # State
        self._is_triggered = False
        self._trigger: KillSwitchTrigger = KillSwitchTrigger.NONE
        self._action_taken: KillSwitchAction = KillSwitchAction.NONE
        self._triggered_at: Optional[datetime] = None
        self._message: str = ""

        # Health tracking
        self._last_heartbeat: Optional[datetime] = None
        self._last_data_update: Optional[datetime] = None
        self._active_errors: List[str] = []

        self._lock = threading.RLock()

    def check_conditions(self, health: SystemHealth) -> KillSwitchState:
        """
        Check all kill switch conditions.

        Call this regularly (every few seconds) to monitor system health.

        Args:
            health: Current system health metrics

        Returns:
            KillSwitchState with current status
        """
        with self._lock:
            try:
                if not self.enabled:
                    return self._get_state_locked(health)

                # Check persisted kill switch state first
                try:
                    persistence = get_risk_persistence()
                    if persistence._state.get("kill_switch_active"):
                        self._is_triggered = True
                        self._trigger = KillSwitchTrigger.SYSTEM_ERROR
                        self._action_taken = KillSwitchAction.DISABLE_NEW_TRADES
                        self._message = persistence._state.get(
                            "kill_switch_reason",
                            "Kill switch was previously activated",
                        )
                        ts = persistence._state.get("kill_switch_triggered_at")
                        if ts:
                            try:
                                self._triggered_at = datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                self._triggered_at = datetime.now(UTC)
                        else:
                            self._triggered_at = datetime.now(UTC)
                        return self._get_state_locked(health)
                except Exception as e:
                    logger.error("Failed to check persisted kill switch state: %s", e)

                if self._is_triggered:
                    return self._get_state_locked(health)

                system_status = health.to_dict()

                # 1. Max drawdown check
                if health.drawdown_pct <= self.max_drawdown:
                    msg = (
                        f"Max drawdown {health.drawdown_pct}% exceeded limit of {self.max_drawdown}%"
                    )
                    self._trigger_locked(
                        KillSwitchTrigger.MAX_DRAWDOWN,
                        msg,
                        KillSwitchAction.FULL_SHUTDOWN,
                    )
                    return self._get_state_locked(health)

                # 2. Connection loss check
                if health.seconds_since_heartbeat > self.connection_timeout:
                    msg = (
                        f"Connection lost for {health.seconds_since_heartbeat:.0f}s "
                        f"(limit: {self.connection_timeout}s)"
                    )
                    self._trigger_locked(
                        KillSwitchTrigger.CONNECTION_LOSS,
                        msg,
                        KillSwitchAction.CANCEL_ALL_ORDERS,
                    )
                    return self._get_state_locked(health)

                # 3. Stale data check
                if health.seconds_since_data > self.data_stale_threshold:
                    msg = (
                        f"Market data stale for {health.seconds_since_data:.0f}s "
                        f"(limit: {self.data_stale_threshold}s)"
                    )
                    self._trigger_locked(
                        KillSwitchTrigger.STALE_DATA,
                        msg,
                        KillSwitchAction.DISABLE_NEW_TRADES,
                    )
                    return self._get_state_locked(health)

                # 4. Multiple errors check (>= 5)
                if len(health.active_errors) >= 5:
                    msg = f"Multiple system errors detected: {len(health.active_errors)} active errors"
                    self._trigger_locked(
                        KillSwitchTrigger.SYSTEM_ERROR,
                        msg,
                        KillSwitchAction.DISABLE_NEW_TRADES,
                    )
                    return self._get_state_locked(health)

                # Approaching limits - WARNING
                if health.drawdown_pct <= self.max_drawdown * 0.8:
                    logger.warning(
                        "Kill switch: drawdown approaching limit (%.2f%% / %.2f%%)",
                        health.drawdown_pct,
                        self.max_drawdown,
                    )
                if health.seconds_since_heartbeat > self.connection_timeout * 0.8:
                    logger.warning(
                        "Kill switch: connection heartbeat approaching timeout (%.0fs / %ds)",
                        health.seconds_since_heartbeat,
                        self.connection_timeout,
                    )
                if health.seconds_since_data > self.data_stale_threshold * 0.8:
                    logger.warning(
                        "Kill switch: data age approaching stale threshold (%.0fs / %ds)",
                        health.seconds_since_data,
                        self.data_stale_threshold,
                    )

                return self._get_state_locked(health)

            except Exception as e:
                logger.critical("Kill switch check_conditions failed: %s", e, exc_info=True)
                self._trigger_locked(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    f"Kill switch check failed: {e}",
                    KillSwitchAction.DISABLE_NEW_TRADES,
                )
                return self._get_state_locked(health)

    def _trigger_locked(
        self,
        reason: KillSwitchTrigger,
        message: str,
        action: KillSwitchAction,
    ) -> None:
        """Set triggered state (caller must hold _lock). Idempotent: if already triggered, only update message."""
        if self._is_triggered:
            self._message = message
            logger.critical("Kill switch re-triggered (idempotent): %s", message)
            return
        self._is_triggered = True
        self._trigger = reason
        self._action_taken = action
        self._triggered_at = datetime.now(UTC)
        self._message = message
        try:
            persistence = get_risk_persistence()
            persistence.activate_kill_switch(reason.value)
        except Exception as e:
            logger.error("Failed to persist kill switch state: %s", e)
        logger.critical(
            "KILL SWITCH TRIGGERED: %s | %s | action=%s",
            reason.value,
            message,
            action.value,
        )

    def trigger(
        self,
        reason: KillSwitchTrigger,
        message: str,
        action: KillSwitchAction = KillSwitchAction.FULL_SHUTDOWN,
    ) -> KillSwitchState:
        """
        Manually trigger the kill switch.

        Args:
            reason: Why the kill switch was triggered
            message: Human-readable description
            action: What action to take

        Returns:
            KillSwitchState after triggering
        """
        with self._lock:
            self._trigger_locked(reason, message, action)
            health_for_status = SystemHealth(
                last_heartbeat=self._last_heartbeat,
                last_data_update=self._last_data_update,
                seconds_since_heartbeat=0.0,
                seconds_since_data=0.0,
                drawdown_pct=0.0,
                is_connected=True,
                active_errors=list(self._active_errors),
            )
            return self._get_state_locked(health_for_status)

    def reset(self, force: bool = False) -> KillSwitchState:
        """
        Reset the kill switch after it's been triggered.

        Args:
            force: If True, ignore cooldown period (use with caution)

        Returns:
            KillSwitchState after reset attempt

        Raises:
            KillSwitchError if cooldown not elapsed and force=False
        """
        with self._lock:
            if not self._is_triggered:
                return self._get_state_locked(
                    SystemHealth(
                        last_heartbeat=self._last_heartbeat,
                        last_data_update=self._last_data_update,
                        seconds_since_heartbeat=0.0,
                        seconds_since_data=0.0,
                        drawdown_pct=0.0,
                        is_connected=True,
                        active_errors=list(self._active_errors),
                    )
                )
            now = datetime.now(UTC)
            cooldown_end = None
            if self._triggered_at is None:
                can_reset = True
            else:
                cooldown_end = self._triggered_at + timedelta(minutes=self.cooldown_minutes)
                can_reset = now >= cooldown_end
            if not can_reset and not force and cooldown_end is not None:
                remaining = (cooldown_end - now).total_seconds()
                raise KillSwitchError(
                    f"Cannot reset kill switch: cooldown not elapsed. "
                    f"Remaining: {max(0, int(remaining))}s"
                )
            self._is_triggered = False
            self._trigger = KillSwitchTrigger.NONE
            self._action_taken = KillSwitchAction.NONE
            self._triggered_at = None
            self._message = ""
            logger.warning("Kill switch RESET (force=%s). Trading enabled.", force)
            return self._get_state_locked(
                SystemHealth(
                    last_heartbeat=self._last_heartbeat,
                    last_data_update=self._last_data_update,
                    seconds_since_heartbeat=0.0,
                    seconds_since_data=0.0,
                    drawdown_pct=0.0,
                    is_connected=True,
                    active_errors=list(self._active_errors),
                )
            )

    def _get_state_locked(self, health: SystemHealth) -> KillSwitchState:
        """Build KillSwitchState (caller must hold _lock)."""
        now = datetime.now(UTC)
        if not self._is_triggered:
            can_reset = True
            cooldown_remaining_seconds = 0
        else:
            if self._triggered_at is None:
                can_reset = True
                cooldown_remaining_seconds = 0
            else:
                cooldown_end = self._triggered_at + timedelta(minutes=self.cooldown_minutes)
                can_reset = now >= cooldown_end
                cooldown_remaining_seconds = max(0, int((cooldown_end - now).total_seconds()))

        system_status = _build_system_status(
            self._last_heartbeat,
            self._last_data_update,
            self._active_errors,
            health.drawdown_pct,
            health.is_connected,
        )

        return KillSwitchState(
            is_triggered=self._is_triggered,
            trigger=self._trigger,
            action_taken=self._action_taken,
            triggered_at=self._triggered_at,
            message=self._message,
            can_reset=can_reset,
            cooldown_remaining_seconds=cooldown_remaining_seconds,
            system_status=system_status,
        )

    def get_state(self) -> KillSwitchState:
        """Get current kill switch state without checking conditions."""
        with self._lock:
            health = SystemHealth(
                last_heartbeat=self._last_heartbeat,
                last_data_update=self._last_data_update,
                seconds_since_heartbeat=0.0,
                seconds_since_data=0.0,
                drawdown_pct=0.0,
                is_connected=True,
                active_errors=list(self._active_errors),
            )
            return self._get_state_locked(health)

    def update_heartbeat(self) -> None:
        """Call this when broker connection is confirmed alive."""
        with self._lock:
            self._last_heartbeat = datetime.now(UTC)

    def update_data_timestamp(self) -> None:
        """Call this when fresh market data is received."""
        with self._lock:
            self._last_data_update = datetime.now(UTC)

    def record_error(self, error: str) -> None:
        """Record a system error (may trigger kill switch if too many)."""
        with self._lock:
            self._active_errors.append(error)

    def clear_errors(self) -> None:
        """Clear recorded errors."""
        with self._lock:
            self._active_errors.clear()

    def get_actions_to_execute(self) -> List[KillSwitchAction]:
        """
        Get list of actions that should be executed.

        Returns individual actions for FULL_SHUTDOWN:
        - CANCEL_ALL_ORDERS
        - CLOSE_ALL_POSITIONS
        - DISABLE_NEW_TRADES
        """
        with self._lock:
            if self._action_taken == KillSwitchAction.FULL_SHUTDOWN:
                return [
                    KillSwitchAction.CANCEL_ALL_ORDERS,
                    KillSwitchAction.CLOSE_ALL_POSITIONS,
                    KillSwitchAction.DISABLE_NEW_TRADES,
                ]
            if self._action_taken == KillSwitchAction.NONE:
                return []
            return [self._action_taken]

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is currently triggered."""
        with self._lock:
            return self._is_triggered

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (kill switch not triggered)."""
        with self._lock:
            return not self._is_triggered

    def get_system_health(self) -> SystemHealth:
        """Get current system health metrics (heartbeat/data staleness)."""
        now = datetime.now(UTC)
        with self._lock:
            heartbeat_age = 0.0
            if self._last_heartbeat:
                heartbeat_age = (now - self._last_heartbeat).total_seconds()
            data_age = 0.0
            if self._last_data_update:
                data_age = (now - self._last_data_update).total_seconds()
            return SystemHealth(
                last_heartbeat=self._last_heartbeat,
                last_data_update=self._last_data_update,
                seconds_since_heartbeat=heartbeat_age,
                seconds_since_data=data_age,
                drawdown_pct=0.0,
                is_connected=True,
                active_errors=list(self._active_errors),
            )


# Singleton instance
_kill_switch_instance: Optional[KillSwitch] = None


def get_kill_switch() -> Optional[KillSwitch]:
    """Get the global kill switch instance."""
    return _kill_switch_instance


def set_kill_switch(instance: KillSwitch) -> None:
    """Set the global kill switch instance."""
    global _kill_switch_instance
    _kill_switch_instance = instance
