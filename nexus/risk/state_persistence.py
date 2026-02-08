"""
Risk state persistence: circuit breaker status survives process restarts.

Persists to nexus_risk_state.json so FULL_STOP and other stops remain
in effect until explicitly reset.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Tuple

from nexus.core.enums import CircuitBreakerStatus

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "nexus_risk_state.json"
_impl: "RiskStatePersistence | None" = None
_lock = threading.Lock()


def get_risk_persistence(state_path: str | None = None) -> "RiskStatePersistence":
    """Return the singleton RiskStatePersistence instance."""
    global _impl
    with _lock:
        if _impl is None:
            _impl = RiskStatePersistence(state_path=state_path or DEFAULT_STATE_PATH)
        return _impl


class RiskStatePersistence:
    """
    Persists circuit breaker status to disk so it survives restarts.
    Used to enforce FULL_STOP and other stops across process lifecycles.
    """

    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self._path = state_path
        self._file_lock = threading.Lock()

    def _read(self) -> dict:
        with self._file_lock:
            if not os.path.exists(self._path):
                return {}
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read risk state file %s: %s", self._path, e)
                return {}

    def _write(self, data: dict) -> None:
        with self._file_lock:
            try:
                with open(self._path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except OSError as e:
                logger.error("Could not write risk state file %s: %s", self._path, e)
                raise

    def get_circuit_breaker_status(self) -> str:
        """
        Return current persisted circuit breaker status (enum name, e.g. 'FULL_STOP').
        Defaults to 'CLEAR' if no state file or invalid.
        """
        data = self._read()
        raw = data.get("circuit_breaker_status", CircuitBreakerStatus.CLEAR.value)
        try:
            status = CircuitBreakerStatus(raw)
            return status.name
        except ValueError:
            if raw.upper() in {s.name for s in CircuitBreakerStatus}:
                return raw.upper()
            return CircuitBreakerStatus.CLEAR.name

    def set_circuit_breaker_status(self, status: str, reason: str = "") -> None:
        """
        Persist circuit breaker status and optional reason.
        status: enum name (e.g. 'FULL_STOP') or value (e.g. 'full_stop').
        """
        s = CircuitBreakerStatus.CLEAR
        if isinstance(status, str) and status:
            norm = status.upper().replace("-", "_")
            if hasattr(CircuitBreakerStatus, norm):
                s = getattr(CircuitBreakerStatus, norm)
            else:
                try:
                    s = CircuitBreakerStatus(status)
                except ValueError:
                    pass
        data = self._read()
        data["circuit_breaker_status"] = s.value
        data["circuit_breaker_reason"] = reason or ""
        self._write(data)
        logger.info("Risk state persisted: status=%s reason=%s", s.name, reason or "(none)")

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Return (allowed, reason) based on persisted circuit breaker and kill switch.
        Trading is disallowed for FULL_STOP, DAILY_STOP, WEEKLY_STOP, or active kill switch.
        """
        data = self._read()
        if data.get("kill_switch_active"):
            return False, data.get("kill_switch_reason", "Kill switch was previously activated.")
        name = self.get_circuit_breaker_status()
        reason = data.get("circuit_breaker_reason", "")
        try:
            status = CircuitBreakerStatus[name]
        except KeyError:
            status = CircuitBreakerStatus.CLEAR
        if status == CircuitBreakerStatus.FULL_STOP:
            return False, reason or "Full stop - manual review required."
        if status == CircuitBreakerStatus.DAILY_STOP:
            return False, reason or "Daily stop - no new trades today."
        if status == CircuitBreakerStatus.WEEKLY_STOP:
            return False, reason or "Weekly stop - no new trades this week."
        return True, "Trading allowed"

    @property
    def _state(self) -> dict:
        """Current state from disk (for kill switch check)."""
        return self._read()

    def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch and persist."""
        data = self._read()
        data["kill_switch_active"] = True
        data["kill_switch_triggered_at"] = datetime.now(timezone.utc).isoformat()
        data["kill_switch_reason"] = reason
        data["requires_manual_reset"] = True
        data["manual_reset_reason"] = f"Kill switch: {reason}"
        self._write(data)
        logger.info("Kill switch state persisted: %s", reason)
