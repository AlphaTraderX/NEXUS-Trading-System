"""
Tests for the Signal Generator.

The Signal Generator is the central coordinator that brings together
all risk checks and creates complete NexusSignal objects.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock
import uuid

from nexus.execution.signal_generator import (
    SignalGenerator,
    AccountState,
    MarketState,
    SystemState,
    RejectionReason,
)
from nexus.core.models import Opportunity, ScoredOpportunity, NexusSignal
from nexus.intelligence.cost_engine import CostBreakdown
from nexus.core.enums import (
    Market,
    Direction,
    EdgeType,
    MarketRegime,
    SignalTier,
    SignalStatus,
)


# ============== Fixtures ==============

@pytest.fixture
def mock_cost_engine():
    """Mock cost engine."""
    engine = Mock()
    engine.calculate_costs.return_value = CostBreakdown(
        spread=0.02,
        commission=0.01,
        slippage=0.02,
        overnight=0.01,
        fx_conversion=0.0,
        other=0.0,
    )
    engine.calculate_net_edge.return_value = {
        "gross_edge": 0.15,
        "total_costs": 0.06,
        "net_edge": 0.09,
        "cost_ratio": 40.0,
        "viable": True,
        "warnings": [],
    }
    return engine


@pytest.fixture
def mock_position_sizer():
    """Mock position sizer."""
    sizer = Mock()
    sizer.calculate_size.return_value = Mock(
        risk_pct=1.0,
        risk_amount=100.0,
        position_size=20.0,
        position_value=10000.0,
        can_trade=True,
        rejection_reason=None,
        to_dict=lambda: {
            "risk_pct": 1.0,
            "risk_amount": 100.0,
            "equity_used": 10000.0,
            "score_multiplier": 1.0,
            "heat_remaining": 20.0,
        },
    )
    return sizer


@pytest.fixture
def mock_heat_manager():
    """Mock heat manager."""
    manager = Mock()
    manager.can_add_position.return_value = Mock(
        allowed=True,
        current_heat=5.0,
        heat_after=6.0,
        heat_limit=25.0,
        to_dict=lambda: {
            "allowed": True,
            "current_heat": 5.0,
            "after_trade": 6.0,
            "limit": 25.0,
            "headroom": 19.0,
        },
    )
    return manager


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker."""
    breaker = Mock()
    breaker.check_status.return_value = Mock(
        can_trade=True,
        size_multiplier=1.0,
        message="No loss limits triggered",
        reason="No loss limits triggered",
        to_dict=lambda: {
            "status": "CLEAR",
            "action": "FULL_THROTTLE",
            "reason": "No loss limits triggered",
            "can_trade": True,
            "size_multiplier": 1.0,
        },
    )
    return breaker


@pytest.fixture
def mock_kill_switch():
    """Mock kill switch."""
    switch = Mock()
    switch.check_conditions.return_value = Mock(
        is_triggered=False,
        message="OK",
        to_dict=lambda: {"should_kill": False},
    )
    return switch


@pytest.fixture
def mock_correlation_monitor():
    """Mock correlation monitor (uses check_new_position in implementation)."""
    monitor = Mock()
    monitor.check_new_position.return_value = Mock(
        allowed=True,
        rejection_reasons=[],
        to_dict=lambda: {
            "can_trade": True,
            "warnings": [],
            "same_sector_count": 0,
            "same_direction_count": 0,
        },
    )
    return monitor


@pytest.fixture
def signal_generator(
    mock_cost_engine,
    mock_position_sizer,
    mock_heat_manager,
    mock_circuit_breaker,
    mock_kill_switch,
    mock_correlation_monitor,
):
    """Create a signal generator with mocked dependencies."""
    return SignalGenerator(
        cost_engine=mock_cost_engine,
        position_sizer=mock_position_sizer,
        heat_manager=mock_heat_manager,
        circuit_breaker=mock_circuit_breaker,
        kill_switch=mock_kill_switch,
        correlation_monitor=mock_correlation_monitor,
    )


@pytest.fixture
def sample_opportunity():
    """Create a sample opportunity."""
    return Opportunity(
        id=str(uuid.uuid4()),
        detected_at=datetime.utcnow(),
        scanner="VWAPDeviationScanner",
        symbol="SPY",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=500.0,
        stop_loss=497.0,
        take_profit=506.0,
        primary_edge=EdgeType.VWAP_DEVIATION,
        secondary_edges=[EdgeType.RSI_EXTREME],
        edge_data={"vwap": 499.0, "deviation_std": 2.5},
    )


@pytest.fixture
def sample_scored_opportunity(sample_opportunity):
    """Create a sample scored opportunity."""
    return ScoredOpportunity(
        opportunity=sample_opportunity,
        score=75,
        tier=SignalTier.B,
        factors=[
            "Primary edge (VWAP_DEVIATION): +30",
            "Trend alignment: +15",
            "Volume confirmation: +10",
            "Good R:R (2.0): +6",
        ],
        position_multiplier=1.25,
    )


@pytest.fixture
def sample_account_state():
    """Create a sample account state."""
    return AccountState(
        starting_balance=10000.0,
        current_equity=10500.0,
        daily_pnl=500.0,
        daily_pnl_pct=5.0,
        weekly_pnl_pct=8.0,
        drawdown_pct=0.0,
        portfolio_heat=5.0,
        win_streak=3,
        open_positions=[],
    )


@pytest.fixture
def sample_market_state():
    """Create a sample market state."""
    return MarketState(
        regime=MarketRegime.TRENDING_UP,
        vix=18.0,
        session="regular",
        summary="Bullish trend, low volatility",
    )


# ============== Signal Generation Tests ==============

class TestSignalGeneratorBasics:
    """Test basic signal generation functionality."""

    @pytest.mark.asyncio
    async def test_generates_signal_when_all_checks_pass(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that a signal is generated when all checks pass."""
        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is not None
        assert isinstance(signal, NexusSignal)
        assert signal.symbol == "SPY"
        assert signal.direction == Direction.LONG
        assert signal.status == SignalStatus.PENDING

    @pytest.mark.asyncio
    async def test_signal_has_all_required_fields(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that generated signal has all required fields."""
        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        # Core identifiers
        assert signal.signal_id is not None
        assert signal.opportunity_id == sample_scored_opportunity.opportunity.id

        # Trade parameters
        assert signal.entry_price == 500.0
        assert signal.stop_loss == 497.0
        assert signal.take_profit == 506.0

        # Position sizing
        assert signal.position_size > 0
        assert signal.position_value > 0
        assert signal.risk_amount > 0
        assert signal.risk_percent > 0

        # Edge info
        assert signal.primary_edge == EdgeType.VWAP_DEVIATION
        assert signal.edge_score == 75
        assert signal.tier == SignalTier.B

        # Costs
        assert signal.costs is not None
        assert signal.gross_expected > 0
        assert signal.net_expected > 0
        assert signal.cost_ratio > 0

        # Context
        assert signal.ai_reasoning is not None
        assert signal.confluence_factors is not None
        assert signal.risk_factors is not None
        assert signal.valid_until > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_signal_score_preserved(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that the original score and tier are preserved."""
        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal.edge_score == sample_scored_opportunity.score
        assert signal.tier == sample_scored_opportunity.tier


class TestKillSwitchRejection:
    """Test kill switch rejection scenarios."""

    @pytest.mark.asyncio
    async def test_rejects_when_kill_switch_triggered(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when kill switch triggers."""
        signal_generator.kill_switch.check_conditions.return_value = Mock(
            is_triggered=True,
            message="Daily loss limit exceeded",
            to_dict=lambda: {
                "should_kill": True,
                "reason": "DAILY_LOSS",
                "message": "Daily loss limit exceeded",
            },
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection is not None
        assert signal_generator.last_rejection.check_name == "kill_switch"


class TestCircuitBreakerRejection:
    """Test circuit breaker rejection scenarios."""

    @pytest.mark.asyncio
    async def test_rejects_when_circuit_breaker_stops(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when circuit breaker stops trading."""
        signal_generator.circuit_breaker.check_status.return_value = Mock(
            can_trade=False,
            size_multiplier=0,
            message="Daily loss 3.0% hit",
            reason="Daily loss 3.0% hit",
            to_dict=lambda: {},
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_reduces_size_when_circuit_breaker_warns(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that position size is reduced when circuit breaker warns."""
        signal_generator.circuit_breaker.check_status.return_value = Mock(
            can_trade=True,
            size_multiplier=0.5,
            message="Daily loss 2.0% - reducing size",
            reason="Daily loss 2.0% - reducing size",
            to_dict=lambda: {},
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        # Signal should still be generated but with reduced risk
        assert signal is not None
        # Risk should be reduced by the multiplier
        # Base risk is 1.0%, after 0.5x multiplier should be 0.5%
        assert signal.risk_percent < 1.0


class TestHeatManagerRejection:
    """Test heat manager rejection scenarios."""

    @pytest.mark.asyncio
    async def test_rejects_when_heat_capacity_exceeded(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when heat capacity exceeded."""
        signal_generator.heat_manager.can_add_position.return_value = Mock(
            allowed=False,
            current_heat=24.0,
            heat_after=26.0,
            heat_limit=25.0,
            to_dict=lambda: {
                "allowed": False,
                "current_heat": 24.0,
                "would_be": 26.0,
                "limit": 25.0,
                "reduce_to": 1.0,
            },
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "heat_manager"


class TestCorrelationRejection:
    """Test correlation rejection scenarios."""

    @pytest.mark.asyncio
    async def test_rejects_when_correlation_limit_exceeded(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when correlation limit exceeded."""
        signal_generator.correlation_monitor.check_new_position.return_value = Mock(
            allowed=False,
            rejection_reasons=["Already 3 LONG positions in US_STOCKS"],
            to_dict=lambda: {
                "can_trade": False,
                "warnings": ["Already 3 LONG positions in US_STOCKS"],
                "same_sector_count": 3,
                "same_direction_count": 3,
                "reason": "Too many correlated positions",
            },
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "correlation"


class TestCostViabilityRejection:
    """Test cost viability rejection scenarios."""

    @pytest.mark.asyncio
    async def test_rejects_when_not_viable_after_costs(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when trade not viable after costs."""
        signal_generator.cost_engine.calculate_net_edge.return_value = {
            "gross_edge": 0.10,
            "total_costs": 0.12,
            "net_edge": -0.02,
            "cost_ratio": 120.0,
            "viable": False,
            "warnings": ["Costs exceed edge"],
        }

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "cost_viability"

    @pytest.mark.asyncio
    async def test_rejects_when_net_edge_too_low(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when net edge below minimum."""
        signal_generator.cost_engine.calculate_net_edge.return_value = {
            "gross_edge": 0.08,
            "total_costs": 0.05,
            "net_edge": 0.03,  # Below default 0.05% minimum
            "cost_ratio": 62.5,
            "viable": True,
            "warnings": [],
        }

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "min_edge"

    @pytest.mark.asyncio
    async def test_rejects_when_cost_ratio_too_high(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that signal is rejected when cost ratio exceeds maximum."""
        signal_generator.cost_engine.calculate_net_edge.return_value = {
            "gross_edge": 0.10,
            "total_costs": 0.08,
            "net_edge": 0.02,
            "cost_ratio": 80.0,  # Above default 70% maximum
            "viable": True,
            "warnings": [],
        }

        # Need to also pass the min_edge check
        signal_generator.min_net_edge = 0.01

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        assert signal is None
        assert signal_generator.last_rejection.check_name == "cost_ratio"


class TestEdgeTypeHandling:
    """Test different edge types are handled correctly."""

    @pytest.mark.asyncio
    async def test_different_edges_have_different_expected_edges(
        self,
        signal_generator,
    ):
        """Test that different edge types have different expected edges."""
        # Insider cluster should have highest expected edge
        insider_edge = signal_generator._get_expected_edge(EdgeType.INSIDER_CLUSTER, 80)
        vwap_edge = signal_generator._get_expected_edge(EdgeType.VWAP_DEVIATION, 80)
        bollinger_edge = signal_generator._get_expected_edge(EdgeType.BOLLINGER_TOUCH, 80)

        assert insider_edge > vwap_edge > bollinger_edge

    @pytest.mark.asyncio
    async def test_different_edges_have_different_validity_periods(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that different edge types have different validity periods."""
        # Create opportunities with different edge types
        gap_opp = sample_scored_opportunity.opportunity
        gap_opp.primary_edge = EdgeType.GAP_FILL

        insider_opp = Opportunity(
            id=str(uuid.uuid4()),
            detected_at=datetime.utcnow(),
            scanner="InsiderClusterScanner",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=150.0,
            stop_loss=147.0,
            take_profit=156.0,
            primary_edge=EdgeType.INSIDER_CLUSTER,
            secondary_edges=[],
            edge_data={},
        )

        gap_validity = signal_generator._calculate_validity(gap_opp)
        insider_validity = signal_generator._calculate_validity(insider_opp)

        # Gap fill should have shorter validity than insider cluster
        gap_hours = (gap_validity - datetime.utcnow()).total_seconds() / 3600
        insider_hours = (insider_validity - datetime.utcnow()).total_seconds() / 3600

        assert gap_hours < insider_hours


class TestRiskFactorIdentification:
    """Test risk factor identification."""

    @pytest.mark.asyncio
    async def test_identifies_volatile_market_risk(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
    ):
        """Test that volatile market is identified as a risk."""
        volatile_state = MarketState(
            regime=MarketRegime.VOLATILE,
            vix=35.0,
            session="regular",
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            volatile_state,
        )

        assert signal is not None
        assert any("volatile" in r.lower() for r in signal.risk_factors)

    @pytest.mark.asyncio
    async def test_identifies_elevated_vix_risk(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
    ):
        """Test that elevated VIX is identified as a risk."""
        high_vix_state = MarketState(
            regime=MarketRegime.RANGING,
            vix=30.0,
            session="regular",
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            high_vix_state,
        )

        assert signal is not None
        assert any("vix" in r.lower() for r in signal.risk_factors)


class TestHelperMethods:
    """Test helper methods."""

    def test_broker_mapping(self, signal_generator):
        """Test correct broker is selected for each market."""
        assert signal_generator._get_broker_for_market(Market.US_STOCKS) == "ibkr"
        assert signal_generator._get_broker_for_market(Market.UK_STOCKS) == "ig"
        assert signal_generator._get_broker_for_market(Market.FOREX_MAJORS) == "oanda"
        assert signal_generator._get_broker_for_market(Market.US_FUTURES) == "ibkr"

    def test_expected_edge_scaling_by_score(self, signal_generator):
        """Test expected edge scales with score."""
        high_score = signal_generator._get_expected_edge(EdgeType.VWAP_DEVIATION, 85)
        mid_score = signal_generator._get_expected_edge(EdgeType.VWAP_DEVIATION, 65)
        low_score = signal_generator._get_expected_edge(EdgeType.VWAP_DEVIATION, 45)

        assert high_score > mid_score > low_score

    def test_risk_estimation_by_score(self, signal_generator, sample_account_state):
        """Test risk estimation scales with score."""
        high_risk = signal_generator._estimate_risk_percent(90, sample_account_state)
        mid_risk = signal_generator._estimate_risk_percent(70, sample_account_state)
        low_risk = signal_generator._estimate_risk_percent(45, sample_account_state)

        assert high_risk > mid_risk > low_risk


# ============== Integration-Style Tests ==============

class TestSignalGeneratorIntegration:
    """Integration-style tests for the signal generator."""

    @pytest.mark.asyncio
    async def test_full_signal_generation_flow(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test the complete signal generation flow."""
        # This test verifies all components work together
        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )

        # Verify all checks were called
        signal_generator.kill_switch.check_conditions.assert_called_once()
        signal_generator.circuit_breaker.check_status.assert_called_once()
        signal_generator.heat_manager.can_add_position.assert_called_once()
        signal_generator.correlation_monitor.check_new_position.assert_called_once()
        signal_generator.position_sizer.calculate_size.assert_called_once()
        signal_generator.cost_engine.calculate_costs.assert_called_once()
        signal_generator.cost_engine.calculate_net_edge.assert_called_once()

        # Verify signal is complete
        assert signal is not None
        assert signal.signal_id is not None
        assert signal.status == SignalStatus.PENDING

    @pytest.mark.asyncio
    async def test_rejection_tracking(
        self,
        signal_generator,
        sample_scored_opportunity,
        sample_account_state,
        sample_market_state,
    ):
        """Test that rejection reasons are tracked."""
        # First, generate a successful signal
        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )
        assert signal is not None
        assert signal_generator.last_rejection is None

        # Now trigger a rejection
        signal_generator.kill_switch.check_conditions.return_value = Mock(
            is_triggered=True,
            message="Test rejection",
            to_dict=lambda: {"should_kill": True, "message": "Test rejection"},
        )

        signal = await signal_generator.generate_signal(
            sample_scored_opportunity,
            sample_account_state,
            sample_market_state,
        )
        assert signal is None
        assert signal_generator.last_rejection is not None
        assert signal_generator.last_rejection.check_name == "kill_switch"
