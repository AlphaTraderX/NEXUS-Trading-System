"""Test fixtures for risk module tests."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexus.risk.state_persistence import DEFAULT_STATE_PATH, reset_risk_persistence


@pytest.fixture(autouse=True)
def clean_persistence_state():
    """Clear persistence state before and after each test."""
    state_file = Path(DEFAULT_STATE_PATH)
    if state_file.exists():
        os.remove(state_file)
    reset_risk_persistence()

    yield

    if state_file.exists():
        os.remove(state_file)
    reset_risk_persistence()


@pytest.fixture(autouse=True)
def mock_persistence():
    """Mock persistence so tests see fresh state; no cross-test or within-test persistence."""
    mock_persist = MagicMock()
    mock_persist.is_trading_allowed.return_value = (True, "Testing")
    mock_persist.set_circuit_breaker_status = MagicMock()
    mock_persist.get_circuit_breaker_status.return_value = "CLEAR"
    mock_persist._state = {"kill_switch_active": False}
    mock_persist.activate_kill_switch = MagicMock()

    with patch("nexus.risk.circuit_breaker.get_risk_persistence", return_value=mock_persist), patch(
        "nexus.risk.kill_switch.get_risk_persistence", return_value=mock_persist
    ):
        yield mock_persist


@pytest.fixture
def fresh_circuit_breaker():
    """Create a fresh circuit breaker with no persisted state."""
    from nexus.config.settings import get_settings
    from nexus.risk.circuit_breaker import SmartCircuitBreaker
    return SmartCircuitBreaker(get_settings())


@pytest.fixture
def fresh_kill_switch():
    """Create a fresh kill switch with no persisted state."""
    from nexus.config.settings import get_settings
    from nexus.risk.kill_switch import KillSwitch
    return KillSwitch(get_settings())
