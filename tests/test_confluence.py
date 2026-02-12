"""
Tests for confluence detection, merging, and scoring.

Validates:
- Opportunity model confluence fields and multiplier property
- Orchestrator _merge_confluence grouping and merging logic
- Scorer confluence bonus points and position multiplier scaling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from nexus.core.enums import Direction, EdgeType, Market, MarketRegime
from nexus.core.models import Opportunity
from nexus.intelligence.scorer import OpportunityScorer
from nexus.scanners.orchestrator import ScannerOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_opportunity(
    symbol: str = "AAPL",
    direction: Direction = Direction.LONG,
    primary_edge: EdgeType = EdgeType.VWAP_DEVIATION,
    entry_price: float = 180.0,
    stop_loss: float = 175.0,
    take_profit: float = 190.0,
    edge_data: dict = None,
    **overrides,
) -> Opportunity:
    """Helper to build an Opportunity with sensible defaults."""
    fields = dict(
        id="test-id",
        detected_at=datetime.now(timezone.utc),
        scanner="TestScanner",
        symbol=symbol,
        market=Market.US_STOCKS,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        primary_edge=primary_edge,
        edge_data=edge_data or {},
    )
    fields.update(overrides)
    return Opportunity(**fields)


@pytest.fixture
def scorer():
    return OpportunityScorer()


@pytest.fixture
def orchestrator():
    """Orchestrator with mock data provider (only _merge_confluence used)."""
    mock_provider = MagicMock()
    return ScannerOrchestrator(data_provider=mock_provider)


# Default scoring context — neutral across the board
NEUTRAL_TREND = {"alignment": "NEUTRAL"}
NEUTRAL_VOLUME = 1.0
NEUTRAL_REGIME = MarketRegime.RANGING
NEUTRAL_COST = {"cost_ratio": 50}


# ---------------------------------------------------------------------------
# Opportunity Model Tests
# ---------------------------------------------------------------------------

class TestConfluenceModel:
    """Test Opportunity confluence fields and multiplier property."""

    def test_defaults_no_confluence(self):
        opp = _make_opportunity()
        assert opp.is_confluence is False
        assert opp.confluence_count == 1
        assert opp.confluence_edges == []
        assert opp.confluence_multiplier == 1.0

    def test_two_edge_multiplier(self):
        opp = _make_opportunity(
            confluence_count=2,
            confluence_edges=[EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME],
            is_confluence=True,
        )
        assert opp.confluence_multiplier == 1.5

    def test_three_edge_multiplier(self):
        opp = _make_opportunity(
            confluence_count=3,
            confluence_edges=[
                EdgeType.VWAP_DEVIATION,
                EdgeType.RSI_EXTREME,
                EdgeType.GAP_FILL,
            ],
            is_confluence=True,
        )
        assert opp.confluence_multiplier == 2.0

    def test_four_edge_multiplier_capped(self):
        opp = _make_opportunity(confluence_count=4, is_confluence=True)
        assert opp.confluence_multiplier == 2.0  # >= 3 all return 2.0


# ---------------------------------------------------------------------------
# Orchestrator Merge Tests
# ---------------------------------------------------------------------------

class TestMergeConfluence:
    """Test orchestrator _merge_confluence grouping and merging."""

    def test_single_signal_passthrough(self, orchestrator):
        opp = _make_opportunity()
        merged = orchestrator._merge_confluence([opp])
        assert len(merged) == 1
        assert merged[0].is_confluence is False

    def test_two_edges_merge(self, orchestrator):
        opp1 = _make_opportunity(
            primary_edge=EdgeType.VWAP_DEVIATION,
            stop_loss=175.0,
            edge_data={"vwap_z": 2.5},
        )
        opp2 = _make_opportunity(
            primary_edge=EdgeType.RSI_EXTREME,
            stop_loss=176.0,
            edge_data={"rsi": 28},
        )
        merged = orchestrator._merge_confluence([opp1, opp2])

        assert len(merged) == 1
        m = merged[0]
        assert m.is_confluence is True
        assert m.confluence_count == 2
        assert m.scanner == "CONFLUENCE"

        # Both edges present
        edge_values = [
            e.value if hasattr(e, "value") else e for e in m.confluence_edges
        ]
        assert "vwap_deviation" in edge_values
        assert "rsi_extreme" in edge_values

        # Edge data merged from both
        assert m.edge_data["vwap_z"] == 2.5
        assert m.edge_data["rsi"] == 28

    def test_primary_edge_is_highest_scored(self, orchestrator):
        """INSIDER_CLUSTER (35) should beat RSI_EXTREME (20) as primary."""
        opp1 = _make_opportunity(primary_edge=EdgeType.RSI_EXTREME)
        opp2 = _make_opportunity(primary_edge=EdgeType.INSIDER_CLUSTER)
        merged = orchestrator._merge_confluence([opp1, opp2])

        primary = merged[0].primary_edge
        primary_val = primary.value if hasattr(primary, "value") else primary
        assert primary_val == "insider_cluster"

    def test_tightest_stop_long(self, orchestrator):
        """For LONG, tightest stop = highest stop_loss."""
        opp1 = _make_opportunity(stop_loss=174.0)
        opp2 = _make_opportunity(stop_loss=176.0)
        merged = orchestrator._merge_confluence([opp1, opp2])
        assert merged[0].stop_loss == 176.0

    def test_tightest_stop_short(self, orchestrator):
        """For SHORT, tightest stop = lowest stop_loss."""
        opp1 = _make_opportunity(
            direction=Direction.SHORT,
            entry_price=180.0,
            stop_loss=185.0,
            take_profit=170.0,
        )
        opp2 = _make_opportunity(
            direction=Direction.SHORT,
            entry_price=180.0,
            stop_loss=183.0,
            take_profit=170.0,
        )
        merged = orchestrator._merge_confluence([opp1, opp2])
        assert merged[0].stop_loss == 183.0

    def test_different_symbols_no_merge(self, orchestrator):
        opp1 = _make_opportunity(symbol="AAPL")
        opp2 = _make_opportunity(symbol="MSFT")
        merged = orchestrator._merge_confluence([opp1, opp2])
        assert len(merged) == 2

    def test_different_direction_no_merge(self, orchestrator):
        opp1 = _make_opportunity(direction=Direction.LONG)
        opp2 = _make_opportunity(
            direction=Direction.SHORT,
            entry_price=180.0,
            stop_loss=185.0,
            take_profit=170.0,
        )
        merged = orchestrator._merge_confluence([opp1, opp2])
        assert len(merged) == 2

    def test_three_edge_merge(self, orchestrator):
        opp1 = _make_opportunity(primary_edge=EdgeType.VWAP_DEVIATION)
        opp2 = _make_opportunity(primary_edge=EdgeType.RSI_EXTREME)
        opp3 = _make_opportunity(primary_edge=EdgeType.GAP_FILL)
        merged = orchestrator._merge_confluence([opp1, opp2, opp3])

        assert len(merged) == 1
        m = merged[0]
        assert m.confluence_count == 3
        assert m.is_confluence is True
        assert m.confluence_multiplier == 2.0


# ---------------------------------------------------------------------------
# Scorer Confluence Tests
# ---------------------------------------------------------------------------

class TestConfluenceScoring:
    """Test scorer confluence bonus and position multiplier."""

    def test_no_confluence_no_bonus(self, scorer):
        opp = _make_opportunity()
        result = scorer.score(opp, NEUTRAL_TREND, NEUTRAL_VOLUME, NEUTRAL_REGIME, NEUTRAL_COST)
        # No confluence factor in output
        assert not any("confluence" in f.lower() for f in result.factors)

    def test_two_edge_confluence_bonus(self, scorer):
        opp = _make_opportunity(
            primary_edge=EdgeType.VWAP_DEVIATION,
            is_confluence=True,
            confluence_count=2,
            confluence_edges=[EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME],
        )
        result = scorer.score(opp, NEUTRAL_TREND, NEUTRAL_VOLUME, NEUTRAL_REGIME, NEUTRAL_COST)
        assert any("2-edge confluence" in f for f in result.factors)

    def test_three_edge_confluence_bonus(self, scorer):
        opp = _make_opportunity(
            primary_edge=EdgeType.INSIDER_CLUSTER,
            is_confluence=True,
            confluence_count=3,
            confluence_edges=[
                EdgeType.INSIDER_CLUSTER,
                EdgeType.VWAP_DEVIATION,
                EdgeType.RSI_EXTREME,
            ],
        )
        result = scorer.score(opp, NEUTRAL_TREND, NEUTRAL_VOLUME, NEUTRAL_REGIME, NEUTRAL_COST)
        assert any("3-edge confluence" in f for f in result.factors)

    def test_confluence_score_higher_than_single(self, scorer):
        single = _make_opportunity(primary_edge=EdgeType.VWAP_DEVIATION)
        confluence = _make_opportunity(
            primary_edge=EdgeType.VWAP_DEVIATION,
            is_confluence=True,
            confluence_count=2,
            confluence_edges=[EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME],
        )
        single_result = scorer.score(
            single, NEUTRAL_TREND, NEUTRAL_VOLUME, NEUTRAL_REGIME, NEUTRAL_COST
        )
        confluence_result = scorer.score(
            confluence, NEUTRAL_TREND, NEUTRAL_VOLUME, NEUTRAL_REGIME, NEUTRAL_COST
        )
        # 2-edge confluence adds 10 flat + RSI_EXTREME(20)*0.6 = 22 points
        assert confluence_result.score >= single_result.score + 20

    def test_position_multiplier_no_confluence(self, scorer):
        assert scorer._get_position_multiplier(70) == 1.0
        assert scorer._get_position_multiplier(85) == 1.5
        assert scorer._get_position_multiplier(30) == 0.0

    def test_position_multiplier_with_confluence(self, scorer):
        opp = _make_opportunity(
            is_confluence=True,
            confluence_count=2,
            confluence_edges=[EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME],
        )
        # score=70 -> base 1.0, confluence 1.5x => 1.5
        assert scorer._get_position_multiplier(70, opp) == 1.5
        # score=85 -> base 1.5, confluence 1.5x => 2.25
        assert scorer._get_position_multiplier(85, opp) == 2.25

    def test_position_multiplier_capped_at_2_5(self, scorer):
        opp = _make_opportunity(
            is_confluence=True,
            confluence_count=3,
            confluence_edges=[
                EdgeType.VWAP_DEVIATION,
                EdgeType.RSI_EXTREME,
                EdgeType.GAP_FILL,
            ],
        )
        # score=85 -> base 1.5, confluence 2.0x => 3.0 capped to 2.5
        assert scorer._get_position_multiplier(85, opp) == 2.5

    def test_position_multiplier_f_tier_ignores_confluence(self, scorer):
        opp = _make_opportunity(is_confluence=True, confluence_count=2)
        assert scorer._get_position_multiplier(30, opp) == 0.0
