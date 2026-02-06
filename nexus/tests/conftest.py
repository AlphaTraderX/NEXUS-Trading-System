"""
NEXUS Test Configuration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_opportunity():
    """Sample opportunity for testing."""
    return {
        "symbol": "SPY",
        "direction": "LONG",
        "entry_price": 500.00,
        "stop_loss": 495.00,
        "take_profit": 510.00,
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "mode": "conservative",
        "base_risk_pct": 1.0,
        "max_drawdown": 10.0,
    }
