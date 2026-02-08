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


# --- Delivery layer fixtures ---
from datetime import datetime, timezone

from nexus.core.enums import Direction, EdgeType, Market, SignalStatus, SignalTier
from nexus.core.models import NexusSignal


def _base_signal(**overrides) -> NexusSignal:
    """Minimal valid NexusSignal; overrides applied on top."""
    base = {
        "signal_id": "sig-test-001",
        "created_at": datetime.now(timezone.utc),
        "opportunity_id": "opp-001",
        "symbol": "SPY",
        "market": Market.US_STOCKS,
        "direction": Direction.LONG,
        "entry_price": 502.50,
        "stop_loss": 498.00,
        "take_profit": 512.00,
        "position_size": 100.0,
        "position_value": 50250.0,
        "risk_amount": 450.0,
        "risk_percent": 1.0,
        "primary_edge": EdgeType.VWAP_DEVIATION,
        "secondary_edges": [EdgeType.RSI_EXTREME],
        "edge_score": 78,
        "tier": SignalTier.B,
        "gross_expected": 0.35,
        "costs": {},
        "net_expected": 0.28,
        "cost_ratio": 20.0,
        "ai_reasoning": "Strong VWAP deviation with RSI support.",
        "confluence_factors": ["Volume", "Trend"],
        "risk_factors": [],
        "market_context": "",
        "session": "us_regular",
        "valid_until": datetime.now(timezone.utc),
        "status": SignalStatus.PENDING,
    }
    base.update(overrides)
    return NexusSignal(**base)


@pytest.fixture
def sample_signal():
    """Standard Tier B LONG signal."""
    return _base_signal()


@pytest.fixture
def sample_signal_short():
    """SHORT signal."""
    return _base_signal(
        signal_id="sig-test-short",
        direction=Direction.SHORT,
        entry_price=500.0,
        stop_loss=505.0,
        take_profit=490.0,
        primary_edge=EdgeType.GAP_FILL,
    )


@pytest.fixture
def sample_signal_tier_a():
    """High conviction Tier A (score 92)."""
    return _base_signal(
        signal_id="sig-test-a",
        edge_score=92,
        tier=SignalTier.A,
        primary_edge=EdgeType.INSIDER_CLUSTER,
    )


@pytest.fixture
def sample_signal_tier_d():
    """Low conviction Tier D (score 42)."""
    return _base_signal(
        signal_id="sig-test-d",
        edge_score=42,
        tier=SignalTier.D,
        primary_edge=EdgeType.ORB,
    )
