"""
NEXUS Opportunity Scorer
Scores opportunities 0-100 for signal generation.

SCORING PHILOSOPHY:
- Primary edge: 30-40 points max
- Secondary edges: 5-15 points each (up to 25 total)
- Confirmations: 5-10 points each (up to 20 total)
- Risk/Reward: 5-15 points
- Cost efficiency: 5 points

TIERS:
- A (80-100): Maximum conviction, 1.5x position
- B (65-79): High conviction, 1.25x position
- C (50-64): Standard, 1.0x position
- D (40-49): Lower conviction, 0.5x position
- F (0-39): Don't trade
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import EdgeType, Direction, Market


class SignalTier(Enum):
    """Signal quality tiers."""
    A = "A"  # 80-100: Maximum conviction
    B = "B"  # 65-79: High conviction
    C = "C"  # 50-64: Standard
    D = "D"  # 40-49: Lower conviction
    F = "F"  # 0-39: Don't trade


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class ScoredOpportunity:
    """An opportunity with its score and analysis."""
    opportunity: Any  # The original Opportunity object
    score: int
    tier: SignalTier
    factors: List[str]  # List of scoring factors applied
    position_multiplier: float
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class OpportunityScorer:
    """
    Score opportunities 0-100 for signal generation.

    Higher scores = higher conviction = larger positions.
    """

    # Base scores for each edge type (validated edges only)
    EDGE_BASE_SCORES = {
        EdgeType.INSIDER_CLUSTER: 35,    # Strongest documented (2.1% monthly abnormal returns)
        EdgeType.VWAP_DEVIATION: 30,     # Strong academic backing (Sharpe 2.1)
        EdgeType.TURN_OF_MONTH: 30,      # 100% of equity premium in 4-day window
        EdgeType.MONTH_END: 25,          # $7.5T pension fund flows
        EdgeType.GAP_FILL: 25,          # 60-92% fill rate documented
        EdgeType.RSI_EXTREME: 20,        # Works with correct params (2-period, 20/80)
        EdgeType.POWER_HOUR: 20,         # U-shaped volume pattern confirmed
        EdgeType.ASIAN_RANGE: 20,        # ICT framework validated
        EdgeType.ORB: 18,                # Needs volume + VWAP filters
        EdgeType.BOLLINGER_TOUCH: 15,    # Regime dependent (ranging only)
        EdgeType.LONDON_OPEN: 15,        # Needs confirmation
        EdgeType.NY_OPEN: 15,            # Needs confirmation
        EdgeType.EARNINGS_DRIFT: 15,     # Small/mid-cap only
    }

    # Regime compatibility for edges
    REGIME_EDGE_COMPATIBILITY = {
        MarketRegime.TRENDING_UP: [
            EdgeType.TURN_OF_MONTH,
            EdgeType.MONTH_END,
            EdgeType.INSIDER_CLUSTER,
            EdgeType.ORB,
            EdgeType.POWER_HOUR,
            EdgeType.NY_OPEN,
            EdgeType.EARNINGS_DRIFT,
        ],
        MarketRegime.TRENDING_DOWN: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.GAP_FILL,
            EdgeType.INSIDER_CLUSTER,
        ],
        MarketRegime.RANGING: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.GAP_FILL,
            EdgeType.ASIAN_RANGE,
            EdgeType.LONDON_OPEN,
        ],
        MarketRegime.VOLATILE: [
            # Very selective in volatile regime
            EdgeType.INSIDER_CLUSTER,
            EdgeType.TURN_OF_MONTH,
        ],
    }

    def __init__(self, min_score_to_trade: int = 50):
        """
        Initialize scorer.

        Args:
            min_score_to_trade: Minimum score to consider trading (default 50)
        """
        self.min_score_to_trade = min_score_to_trade

    def score(
        self,
        opportunity: Any,
        trend_alignment: Optional[Dict] = None,
        volume_ratio: float = 1.0,
        regime: MarketRegime = MarketRegime.RANGING,
        cost_analysis: Optional[Dict] = None
    ) -> ScoredOpportunity:
        """
        Score an opportunity and determine tier.

        Args:
            opportunity: Opportunity object with primary_edge, secondary_edges, etc.
            trend_alignment: Dict with 'alignment' key (STRONG_BULLISH, STRONG_BEARISH, PARTIAL, NONE)
            volume_ratio: Current volume / average volume
            regime: Current market regime
            cost_analysis: Dict from CostEngine.calculate_net_edge()

        Returns:
            ScoredOpportunity with score, tier, and factors
        """
        score = 0
        factors = []
        warnings = []

        # Get primary edge (handle both enum and string)
        primary_edge = opportunity.primary_edge
        if isinstance(primary_edge, str):
            try:
                primary_edge = EdgeType(primary_edge)
            except ValueError:
                primary_edge = None

        # Get direction (handle both enum and string)
        direction = opportunity.direction
        if isinstance(direction, str):
            try:
                direction = Direction(direction)
            except ValueError:
                direction = None

        # 1. Primary edge score (30-40 points max)
        edge_score = self.EDGE_BASE_SCORES.get(primary_edge, 10)
        score += edge_score
        edge_name = primary_edge.value if primary_edge else "unknown"
        factors.append(f"Primary edge ({edge_name}): +{edge_score}")

        # 2. Secondary edges (max 25 points, up to 3 edges counted)
        secondary_edges = getattr(opportunity, 'secondary_edges', []) or []
        secondary_score = 0
        for edge in secondary_edges[:3]:
            # Handle string or enum
            if isinstance(edge, str):
                try:
                    edge = EdgeType(edge)
                except ValueError:
                    continue
            edge_pts = self.EDGE_BASE_SCORES.get(edge, 5) * 0.4
            secondary_score += edge_pts
        secondary_score = min(secondary_score, 25)
        if secondary_score > 0:
            score += int(secondary_score)
            factors.append(f"Secondary edges ({len(secondary_edges[:3])}): +{int(secondary_score)}")

        # 3. Trend alignment (max 15 points)
        if trend_alignment:
            alignment = trend_alignment.get("alignment", "NONE")

            if alignment == "STRONG_BULLISH" and direction == Direction.LONG:
                score += 15
                factors.append("Strong bullish trend alignment: +15")
            elif alignment == "STRONG_BEARISH" and direction == Direction.SHORT:
                score += 15
                factors.append("Strong bearish trend alignment: +15")
            elif alignment == "PARTIAL":
                score += 8
                factors.append("Partial trend alignment: +8")
            elif alignment in ["STRONG_BULLISH", "STRONG_BEARISH"]:
                # Trading against trend
                score -= 5
                factors.append("Trading against trend: -5")
                warnings.append("Signal is against the prevailing trend")

        # 4. Volume confirmation (max 10 points)
        if volume_ratio >= 2.0:
            score += 10
            factors.append(f"High volume ({volume_ratio:.1f}x avg): +10")
        elif volume_ratio >= 1.5:
            score += 7
            factors.append(f"Elevated volume ({volume_ratio:.1f}x avg): +7")
        elif volume_ratio >= 1.2:
            score += 4
            factors.append(f"Above avg volume ({volume_ratio:.1f}x avg): +4")
        elif volume_ratio < 0.8:
            score -= 3
            factors.append(f"Low volume ({volume_ratio:.1f}x avg): -3")
            warnings.append("Below average volume - weak conviction")

        # 5. Regime alignment (max 10 points)
        if self._regime_aligns(primary_edge, regime):
            score += 10
            factors.append(f"Regime aligned ({regime.value}): +10")
        else:
            # Edge not ideal for current regime
            score -= 5
            factors.append(f"Regime mismatch ({regime.value}): -5")
            warnings.append(f"Edge {edge_name} not ideal in {regime.value} regime")

        # 6. Risk/Reward ratio (max 10 points)
        rr = getattr(opportunity, 'risk_reward_ratio', None)
        if rr is None:
            # Calculate if we have the data
            entry = getattr(opportunity, 'entry_price', 0)
            stop = getattr(opportunity, 'stop_loss', 0)
            target = getattr(opportunity, 'take_profit', 0)
            if entry and stop and target:
                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr = reward / risk if risk > 0 else 0

        if rr:
            if rr >= 3.0:
                score += 10
                factors.append(f"Excellent R:R ({rr:.1f}:1): +10")
            elif rr >= 2.0:
                score += 6
                factors.append(f"Good R:R ({rr:.1f}:1): +6")
            elif rr >= 1.5:
                score += 3
                factors.append(f"Acceptable R:R ({rr:.1f}:1): +3")
            elif rr < 1.0:
                score -= 5
                factors.append(f"Poor R:R ({rr:.1f}:1): -5")
                warnings.append("Risk/reward below 1:1")

        # 7. Cost efficiency (max 5 points)
        if cost_analysis:
            cost_ratio = cost_analysis.get("cost_ratio", 50)
            if cost_ratio < 20:
                score += 5
                factors.append(f"Low costs ({cost_ratio:.0f}% of edge): +5")
            elif cost_ratio < 35:
                score += 3
                factors.append(f"Moderate costs ({cost_ratio:.0f}% of edge): +3")
            elif cost_ratio > 50:
                score -= 3
                factors.append(f"High costs ({cost_ratio:.0f}% of edge): -3")
                warnings.append("Costs eating significant portion of edge")

            if not cost_analysis.get("viable", True):
                score -= 10
                factors.append("Trade not viable after costs: -10")
                warnings.append("Net edge below minimum threshold")

        # Ensure score is in valid range
        score = max(0, min(score, 100))

        # Determine tier
        tier = self._get_tier(score)

        # Get position multiplier
        position_multiplier = self._get_position_multiplier(score)

        return ScoredOpportunity(
            opportunity=opportunity,
            score=score,
            tier=tier,
            factors=factors,
            position_multiplier=position_multiplier,
            warnings=warnings
        )

    def _regime_aligns(self, edge: EdgeType, regime: MarketRegime) -> bool:
        """Check if edge is compatible with current regime."""
        if edge is None:
            return False
        compatible_edges = self.REGIME_EDGE_COMPATIBILITY.get(regime, [])
        return edge in compatible_edges

    def _get_tier(self, score: int) -> SignalTier:
        """Determine tier from score."""
        if score >= 80:
            return SignalTier.A
        elif score >= 65:
            return SignalTier.B
        elif score >= 50:
            return SignalTier.C
        elif score >= 40:
            return SignalTier.D
        else:
            return SignalTier.F

    def _get_position_multiplier(self, score: int) -> float:
        """
        Get position size multiplier based on score.

        Higher conviction = larger position.
        """
        if score >= 85:
            return 1.5   # 1.5x base risk
        elif score >= 75:
            return 1.25  # 1.25x base risk
        elif score >= 65:
            return 1.0   # Base risk
        elif score >= 50:
            return 0.75  # Reduced risk
        elif score >= 40:
            return 0.5   # Half risk
        else:
            return 0     # Don't trade

    def should_trade(self, scored: ScoredOpportunity) -> bool:
        """Check if a scored opportunity should be traded."""
        return scored.score >= self.min_score_to_trade and scored.tier != SignalTier.F

    def rank_opportunities(self, scored_list: List[ScoredOpportunity]) -> List[ScoredOpportunity]:
        """Rank opportunities by score (highest first)."""
        return sorted(scored_list, key=lambda x: x.score, reverse=True)


# Test the scorer
if __name__ == "__main__":
    from dataclasses import dataclass

    print("=" * 60)
    print("NEXUS OPPORTUNITY SCORER TEST")
    print("=" * 60)

    # Create a mock opportunity class for testing
    @dataclass
    class MockOpportunity:
        primary_edge: str
        direction: str
        secondary_edges: List[str] = None
        entry_price: float = 100.0
        stop_loss: float = 98.0
        take_profit: float = 105.0

        def __post_init__(self):
            if self.secondary_edges is None:
                self.secondary_edges = []

        @property
        def risk_reward_ratio(self):
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            return reward / risk if risk > 0 else 0

    scorer = OpportunityScorer(min_score_to_trade=50)

    # Test 1: High quality opportunity (A tier)
    print("\n--- Test 1: High Quality Opportunity ---")
    opp1 = MockOpportunity(
        primary_edge="insider_cluster",
        direction="long",
        secondary_edges=["vwap_deviation", "rsi_extreme"],
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=106.0  # 3:1 R:R
    )

    scored1 = scorer.score(
        opportunity=opp1,
        trend_alignment={"alignment": "STRONG_BULLISH"},
        volume_ratio=2.5,
        regime=MarketRegime.TRENDING_UP,
        cost_analysis={"cost_ratio": 15, "viable": True}
    )

    print(f"Score: {scored1.score}/100")
    print(f"Tier: {scored1.tier.value}")
    print(f"Position multiplier: {scored1.position_multiplier}x")
    print(f"Should trade: {scorer.should_trade(scored1)}")
    print("Factors:")
    for f in scored1.factors:
        print(f"  - {f}")

    # Test 2: Medium quality opportunity (C tier)
    print("\n--- Test 2: Medium Quality Opportunity ---")
    opp2 = MockOpportunity(
        primary_edge="gap_fill",
        direction="long",
        entry_price=100.0,
        stop_loss=98.5,
        take_profit=102.5  # 1.67:1 R:R
    )

    scored2 = scorer.score(
        opportunity=opp2,
        trend_alignment={"alignment": "PARTIAL"},
        volume_ratio=1.3,
        regime=MarketRegime.RANGING,
        cost_analysis={"cost_ratio": 30, "viable": True}
    )

    print(f"Score: {scored2.score}/100")
    print(f"Tier: {scored2.tier.value}")
    print(f"Position multiplier: {scored2.position_multiplier}x")
    print(f"Should trade: {scorer.should_trade(scored2)}")

    # Test 3: Poor opportunity (F tier)
    print("\n--- Test 3: Poor Opportunity ---")
    opp3 = MockOpportunity(
        primary_edge="london_open",
        direction="long",
        entry_price=100.0,
        stop_loss=97.0,
        take_profit=101.0  # 0.33:1 R:R - terrible!
    )

    scored3 = scorer.score(
        opportunity=opp3,
        trend_alignment={"alignment": "STRONG_BEARISH"},  # Trading against trend!
        volume_ratio=0.6,  # Low volume
        regime=MarketRegime.VOLATILE,
        cost_analysis={"cost_ratio": 60, "viable": False}
    )

    print(f"Score: {scored3.score}/100")
    print(f"Tier: {scored3.tier.value}")
    print(f"Position multiplier: {scored3.position_multiplier}x")
    print(f"Should trade: {scorer.should_trade(scored3)}")
    if scored3.warnings:
        print("Warnings:")
        for w in scored3.warnings:
            print(f"  - {w}")

    # Test 4: Ranking multiple opportunities
    print("\n--- Test 4: Ranking Opportunities ---")
    all_scored = [scored1, scored2, scored3]
    ranked = scorer.rank_opportunities(all_scored)

    print("Ranked by score:")
    for i, s in enumerate(ranked, 1):
        edge = s.opportunity.primary_edge
        print(f"  {i}. {edge}: {s.score}/100 (Tier {s.tier.value})")

    # Test 5: Edge scores reference
    print("\n--- Test 5: Edge Base Scores ---")
    print("Edge base scores (highest to lowest):")
    sorted_edges = sorted(scorer.EDGE_BASE_SCORES.items(), key=lambda x: x[1], reverse=True)
    for edge, score in sorted_edges:
        print(f"  {edge.value}: {score} points")

    print("\n" + "=" * 60)
    print("SCORER TEST COMPLETE [OK]")
    print("=" * 60)
