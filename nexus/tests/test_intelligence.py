"""
NEXUS Intelligence Layer Validation Tests
Tests for Cost Engine and Opportunity Scorer
"""

import pytest
from datetime import datetime, timezone
from nexus.intelligence import CostEngine, CostBreakdown, OpportunityScorer, ScoredOpportunity
from nexus.core.enums import EdgeType, Direction, Market, MarketRegime
from nexus.core.models import Opportunity


class TestCostBreakdown:
    """Test the CostBreakdown dataclass."""

    def test_total_calculation(self):
        """Total should sum all cost components."""
        costs = CostBreakdown(
            spread=0.02,
            commission=0.01,
            slippage=0.02,
            overnight=0.02,
            fx_conversion=0.01,
            other=0.005
        )
        expected = 0.02 + 0.01 + 0.02 + 0.02 + 0.01 + 0.005
        assert abs(costs.total - expected) < 0.0001

    def test_to_dict(self):
        """to_dict should return all components."""
        costs = CostBreakdown(
            spread=0.02,
            commission=0.01,
            slippage=0.02,
            overnight=0.02,
            fx_conversion=0.01,
            other=0.0
        )
        d = costs.to_dict()
        assert "spread" in d
        assert "total" in d
        assert d["spread"] == 0.02


class TestCostEngine:
    """Test the CostEngine class."""

    @pytest.fixture
    def engine(self):
        return CostEngine()

    # === Market Cost Tests ===

    def test_us_stocks_costs(self, engine):
        """US stocks should have lowest base costs."""
        costs = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 1)
        assert costs.spread > 0
        assert costs.total < 0.15  # Should be under 0.15%

    def test_uk_stocks_higher_costs(self, engine):
        """UK stocks should have higher spreads than US."""
        us_costs = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 1)
        uk_costs = engine.calculate_costs("HSBA", Market.UK_STOCKS, "ibkr", 1000, 1)
        assert uk_costs.spread > us_costs.spread

    def test_forex_majors_tight_costs(self, engine):
        """Forex majors should have tight spreads."""
        costs = engine.calculate_costs("EUR/USD", Market.FOREX_MAJORS, "oanda", 1000, 1)
        assert costs.spread < 0.05  # Under 0.05%

    def test_forex_crosses_wider_than_majors(self, engine):
        """Forex crosses should be wider than majors."""
        major = engine.calculate_costs("EUR/USD", Market.FOREX_MAJORS, "oanda", 1000, 1)
        cross = engine.calculate_costs("EUR/GBP", Market.FOREX_CROSSES, "oanda", 1000, 1)
        assert cross.spread > major.spread

    def test_futures_no_overnight(self, engine):
        """Futures should have zero overnight costs."""
        costs = engine.calculate_costs("ES", Market.US_FUTURES, "ibkr", 1000, 5)
        assert costs.overnight == 0

    # === Broker Cost Tests ===

    def test_ig_spread_markup(self, engine):
        """IG should apply 20% spread markup."""
        ibkr_costs = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 1)
        ig_costs = engine.calculate_costs("SPY", Market.US_STOCKS, "ig", 1000, 1)
        # IG has 1.2x spread markup
        assert ig_costs.spread > ibkr_costs.spread

    def test_oanda_no_markup(self, engine):
        """OANDA should have no spread markup."""
        costs = engine.calculate_costs("EUR/USD", Market.FOREX_MAJORS, "oanda", 1000, 1)
        # Base spread for forex majors is 0.015, round trip = 0.03
        assert costs.spread == pytest.approx(0.03, rel=0.1)

    # === Hold Period Tests ===

    def test_overnight_scales_with_days(self, engine):
        """Overnight costs should scale with hold period."""
        one_day = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 1)
        five_days = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 5)
        assert five_days.overnight == pytest.approx(one_day.overnight * 5, rel=0.01)

    def test_zero_hold_days(self, engine):
        """Zero hold days = no overnight cost (day trade)."""
        costs = engine.calculate_costs("SPY", Market.US_STOCKS, "ibkr", 1000, 0)
        assert costs.overnight == 0

    # === Net Edge Tests ===

    def test_viable_trade(self, engine):
        """High edge, low costs = viable."""
        costs = CostBreakdown(spread=0.02, commission=0.01, slippage=0.02,
                             overnight=0.01, fx_conversion=0, other=0)
        result = engine.calculate_net_edge(0.20, costs)
        assert result["viable"] is True
        assert result["net_edge"] > 0.10

    def test_unviable_high_cost_ratio(self, engine):
        """Costs > 70% of edge = not viable."""
        costs = CostBreakdown(spread=0.08, commission=0.02, slippage=0.02,
                             overnight=0.02, fx_conversion=0, other=0)
        result = engine.calculate_net_edge(0.15, costs)
        assert result["cost_ratio"] > 70
        assert result["viable"] is False

    def test_unviable_low_net_edge(self, engine):
        """Net edge < 0.05% = not viable."""
        costs = CostBreakdown(spread=0.05, commission=0.02, slippage=0.03,
                             overnight=0.02, fx_conversion=0, other=0)
        result = engine.calculate_net_edge(0.15, costs)
        assert result["net_edge"] < 0.05
        assert result["viable"] is False

    def test_negative_net_edge_warning(self, engine):
        """Negative net edge should produce warning."""
        costs = CostBreakdown(spread=0.10, commission=0.05, slippage=0.05,
                             overnight=0.02, fx_conversion=0, other=0)
        result = engine.calculate_net_edge(0.15, costs)
        assert result["net_edge"] < 0
        assert any("UNPROFITABLE" in w for w in result["warnings"])

    # === Edge Estimate Tests ===

    def test_edge_estimates_exist(self, engine):
        """All edge types should have estimates."""
        for edge_type in EdgeType:
            estimate = engine.get_edge_estimate(edge_type)
            assert estimate > 0

    def test_insider_cluster_highest_edge(self, engine):
        """Insider cluster should have highest edge estimate."""
        insider = engine.get_edge_estimate(EdgeType.INSIDER_CLUSTER)
        vwap = engine.get_edge_estimate(EdgeType.VWAP_DEVIATION)
        assert insider > vwap


class TestOpportunityScorer:
    """Test the OpportunityScorer class."""

    @pytest.fixture
    def scorer(self):
        return OpportunityScorer()

    @pytest.fixture
    def base_opportunity(self):
        """Create a base opportunity for testing."""
        return Opportunity(
            id="test-001",
            detected_at=datetime.now(timezone.utc),
            scanner="TestScanner",
            symbol="SPY",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=500.0,
            stop_loss=495.0,
            take_profit=510.0,
            primary_edge=EdgeType.VWAP_DEVIATION,
            secondary_edges=[],
            edge_data={}
        )

    @pytest.fixture
    def cost_analysis_good(self):
        """Good cost analysis (low costs)."""
        return {"cost_ratio": 15, "viable": True, "net_edge": 0.15}

    @pytest.fixture
    def cost_analysis_bad(self):
        """Bad cost analysis (high costs)."""
        return {"cost_ratio": 75, "viable": False, "net_edge": 0.02}

    # === Primary Edge Scoring ===

    def test_insider_cluster_highest_score(self, scorer, base_opportunity, cost_analysis_good):
        """Insider cluster should get 35 points."""
        base_opportunity.primary_edge = EdgeType.INSIDER_CLUSTER
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("35" in f and "INSIDER" in f.upper() for f in scored.factors)

    def test_lower_edges_get_lower_scores(self, scorer, base_opportunity, cost_analysis_good):
        """Lower-tier edges should score less."""
        base_opportunity.primary_edge = EdgeType.LONDON_OPEN  # 15 points
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("15" in f and "LONDON" in f.upper() for f in scored.factors)

    # === Secondary Edge Scoring ===

    def test_secondary_edges_add_points(self, scorer, base_opportunity, cost_analysis_good):
        """Secondary edges should add up to 25 points."""
        base_opportunity.secondary_edges = [EdgeType.RSI_EXTREME, EdgeType.POWER_HOUR]
        scored_with = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )

        base_opportunity.secondary_edges = []
        scored_without = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )

        assert scored_with.score > scored_without.score

    def test_secondary_edges_capped_at_25(self, scorer, base_opportunity, cost_analysis_good):
        """Secondary edges should cap at 25 points total."""
        # Add many high-value secondary edges
        base_opportunity.secondary_edges = [
            EdgeType.INSIDER_CLUSTER,  # Would be 35 * 0.4 = 14
            EdgeType.VWAP_DEVIATION,   # Would be 30 * 0.4 = 12
            EdgeType.TURN_OF_MONTH,    # Would be 30 * 0.4 = 12
        ]
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        # Secondary should be capped, not 14+12+12=38
        secondary_factor = [f for f in scored.factors if "Secondary" in f]
        assert len(secondary_factor) == 1
        # Extract points from factor string
        points = int(secondary_factor[0].split("+")[1].strip())
        assert points <= 25

    # === Trend Alignment Scoring ===

    def test_strong_trend_alignment_long(self, scorer, base_opportunity, cost_analysis_good):
        """Strong bullish + LONG = 15 points."""
        base_opportunity.direction = Direction.LONG
        scored = scorer.score(
            base_opportunity,
            {"alignment": "STRONG_BULLISH"},
            1.0,
            MarketRegime.TRENDING_UP,
            cost_analysis_good
        )
        assert any("15" in f and "Strong trend" in f for f in scored.factors)

    def test_strong_trend_alignment_short(self, scorer, base_opportunity, cost_analysis_good):
        """Strong bearish + SHORT = 15 points."""
        base_opportunity.direction = Direction.SHORT
        scored = scorer.score(
            base_opportunity,
            {"alignment": "STRONG_BEARISH"},
            1.0,
            MarketRegime.TRENDING_DOWN,
            cost_analysis_good
        )
        assert any("15" in f and "Strong trend" in f for f in scored.factors)

    def test_conflicting_trend_no_points(self, scorer, base_opportunity, cost_analysis_good):
        """Conflicting trend = 0 points."""
        scored = scorer.score(
            base_opportunity,
            {"alignment": "CONFLICTING"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("Conflicting" in f and "+0" in f for f in scored.factors)

    # === Volume Scoring ===

    def test_high_volume_10_points(self, scorer, base_opportunity, cost_analysis_good):
        """Volume >= 2.0x = 10 points."""
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            2.5,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("10" in f and "High volume" in f for f in scored.factors)

    def test_elevated_volume_7_points(self, scorer, base_opportunity, cost_analysis_good):
        """Volume 1.5-2.0x = 7 points."""
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.7,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("7" in f and "Elevated volume" in f for f in scored.factors)

    def test_normal_volume_0_points(self, scorer, base_opportunity, cost_analysis_good):
        """Volume < 1.2x = 0 points."""
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("Normal volume" in f and "+0" in f for f in scored.factors)

    # === Regime Alignment Scoring ===

    def test_regime_aligned_10_points(self, scorer, base_opportunity, cost_analysis_good):
        """Edge that works in regime = 10 points."""
        base_opportunity.primary_edge = EdgeType.VWAP_DEVIATION  # Works in RANGING
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("10" in f and "Regime aligned" in f for f in scored.factors)

    def test_regime_not_aligned_0_points(self, scorer, base_opportunity, cost_analysis_good):
        """Edge that doesn't work in regime = 0 points."""
        base_opportunity.primary_edge = EdgeType.TURN_OF_MONTH  # Doesn't work in RANGING
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        assert any("Regime neutral" in f and "+0" in f for f in scored.factors)

    # === Tier Classification ===

    def test_tier_a_threshold(self, scorer, base_opportunity, cost_analysis_good):
        """Score >= 80 = Tier A."""
        # Stack everything for high score
        base_opportunity.primary_edge = EdgeType.INSIDER_CLUSTER  # 35
        base_opportunity.secondary_edges = [EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME]  # ~20
        base_opportunity.take_profit = 520.0  # Better R:R

        scored = scorer.score(
            base_opportunity,
            {"alignment": "STRONG_BULLISH"},  # +15
            2.5,  # +10
            MarketRegime.VOLATILE,  # Insider works here +10
            {"cost_ratio": 10}  # +5
        )
        assert scored.tier == "A"
        assert scored.position_multiplier == 1.5

    def test_tier_b_threshold(self, scorer, base_opportunity, cost_analysis_good):
        """Score 65-79 = Tier B."""
        base_opportunity.primary_edge = EdgeType.VWAP_DEVIATION  # 30
        scored = scorer.score(
            base_opportunity,
            {"alignment": "STRONG_BULLISH"},  # +15
            1.8,  # +7
            MarketRegime.RANGING,  # +10
            {"cost_ratio": 30}  # +3
        )
        # 30 + 15 + 7 + 10 + 3 = 65+
        assert scored.tier == "B"
        assert scored.position_multiplier == 1.25

    def test_tier_f_low_score(self, scorer, base_opportunity, cost_analysis_bad):
        """Score < 40 = Tier F (don't trade)."""
        base_opportunity.primary_edge = EdgeType.LONDON_OPEN  # 15
        base_opportunity.take_profit = 502.0  # Poor R:R

        scored = scorer.score(
            base_opportunity,
            {"alignment": "CONFLICTING"},  # +0
            0.8,  # +0
            MarketRegime.VOLATILE,  # London doesn't work here, +0
            {"cost_ratio": 80}  # +0
        )
        assert scored.tier == "F"
        assert scored.position_multiplier == 0.0

    # === Score Capping ===

    def test_score_capped_at_100(self, scorer, base_opportunity, cost_analysis_good):
        """Score should never exceed 100."""
        # Max out everything
        base_opportunity.primary_edge = EdgeType.INSIDER_CLUSTER  # 35
        base_opportunity.secondary_edges = [
            EdgeType.VWAP_DEVIATION,
            EdgeType.TURN_OF_MONTH,
            EdgeType.RSI_EXTREME
        ]  # 25
        base_opportunity.take_profit = 530.0  # Excellent R:R

        scored = scorer.score(
            base_opportunity,
            {"alignment": "STRONG_BULLISH"},  # +15
            3.0,  # +10
            MarketRegime.VOLATILE,  # +10
            {"cost_ratio": 5}  # +5
        )
        assert scored.score <= 100

    # === to_dict Output ===

    def test_scored_opportunity_to_dict(self, scorer, base_opportunity, cost_analysis_good):
        """to_dict should contain all required fields."""
        scored = scorer.score(
            base_opportunity,
            {"alignment": "NEUTRAL"},
            1.0,
            MarketRegime.RANGING,
            cost_analysis_good
        )
        d = scored.to_dict()

        assert "opportunity_id" in d
        assert "symbol" in d
        assert "score" in d
        assert "tier" in d
        assert "factors" in d
        assert "position_multiplier" in d
        assert d["symbol"] == "SPY"


class TestIntegration:
    """Test Cost Engine + Scorer working together."""

    def test_full_pipeline(self):
        """Test complete cost → score pipeline."""
        # Create opportunity
        opp = Opportunity(
            id="integration-test",
            detected_at=datetime.now(timezone.utc),
            scanner="VWAPDeviationScanner",
            symbol="SPY",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=500.0,
            stop_loss=495.0,
            take_profit=512.50,
            primary_edge=EdgeType.VWAP_DEVIATION,
            secondary_edges=[EdgeType.RSI_EXTREME],
            edge_data={}
        )

        # Calculate costs
        cost_engine = CostEngine()
        costs = cost_engine.calculate_costs(
            opp.symbol,
            opp.market,
            "ibkr",
            5000,
            1
        )

        # Get edge estimate and calculate net edge
        gross_edge = cost_engine.get_edge_estimate(opp.primary_edge)
        cost_analysis = cost_engine.calculate_net_edge(gross_edge, costs)

        # Score
        scorer = OpportunityScorer()
        scored = scorer.score(
            opp,
            {"alignment": "STRONG_BULLISH"},
            1.5,
            MarketRegime.RANGING,
            cost_analysis
        )

        # Verify outputs
        assert costs.total > 0
        assert cost_analysis["viable"] in [True, False]
        assert 0 <= scored.score <= 100
        assert scored.tier in ["A", "B", "C", "D", "F"]
        assert scored.position_multiplier >= 0

        print("\n=== Integration Test Results ===")
        print(f"Symbol: {opp.symbol}")
        print(f"Total Costs: {costs.total:.4f}%")
        print(f"Gross Edge: {gross_edge:.4f}%")
        print(f"Net Edge: {cost_analysis['net_edge']:.4f}%")
        print(f"Viable: {cost_analysis['viable']}")
        print(f"Score: {scored.score}/100 (Tier {scored.tier})")
        print(f"Position Multiplier: {scored.position_multiplier}x")


# Run with: pytest nexus/tests/test_intelligence.py -v
