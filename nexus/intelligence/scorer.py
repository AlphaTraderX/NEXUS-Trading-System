"""
NEXUS Opportunity Scorer — Phase 5.2
Scores opportunities 0-100 for signal generation.

SCORING BREAKDOWN:
- Primary edge: 30-40 points max
- Secondary edges: 5-15 points each (up to 25 total)
- Trend alignment: up to 15 points
- Volume confirmation: up to 10 points
- Regime alignment: up to 10 points
- Risk/Reward: up to 10 points
- Cost efficiency: up to 5 points
"""

from dataclasses import dataclass
from typing import Dict, List

from nexus.core.enums import EdgeType, Direction, MarketRegime
from nexus.core.models import Opportunity


@dataclass
class ScoredOpportunity:
    """An opportunity with scoring applied."""

    opportunity: Opportunity
    score: int  # 0-100
    tier: str  # A, B, C, D, F
    factors: List[str]  # Human-readable scoring factors
    position_multiplier: float  # Size multiplier based on score

    def to_dict(self) -> dict:
        direction = self.opportunity.direction
        primary_edge = self.opportunity.primary_edge
        return {
            "opportunity_id": self.opportunity.id,
            "symbol": self.opportunity.symbol,
            "direction": getattr(direction, "value", direction),
            "primary_edge": getattr(primary_edge, "value", primary_edge),
            "score": self.score,
            "tier": self.tier,
            "factors": self.factors,
            "position_multiplier": self.position_multiplier,
        }


class OpportunityScorer:
    """
    Score opportunities 0-100 for signal generation.

    SCORING BREAKDOWN:
    - Primary edge: 30-40 points max
    - Secondary edges: 5-15 points each (up to 25 total)
    - Trend alignment: up to 15 points
    - Volume confirmation: up to 10 points
    - Regime alignment: up to 10 points
    - Risk/Reward: up to 10 points
    - Cost efficiency: up to 5 points
    """

    EDGE_BASE_SCORES = {
        EdgeType.INSIDER_CLUSTER: 35,
        EdgeType.VWAP_DEVIATION: 30,
        EdgeType.TURN_OF_MONTH: 30,
        EdgeType.MONTH_END: 25,
        EdgeType.GAP_FILL: 25,
        EdgeType.RSI_EXTREME: 20,
        EdgeType.POWER_HOUR: 20,
        EdgeType.ASIAN_RANGE: 20,
        EdgeType.ORB: 18,
        EdgeType.BOLLINGER_TOUCH: 15,
        EdgeType.LONDON_OPEN: 15,
        EdgeType.NY_OPEN: 15,
        EdgeType.EARNINGS_DRIFT: 15,
    }

    TIER_THRESHOLDS = {
        "A": 80,
        "B": 65,
        "C": 50,
        "D": 40,
    }

    TIER_MULTIPLIERS = {
        "A": 1.5,
        "B": 1.25,
        "C": 1.0,
        "D": 0.5,
        "F": 0.0,
    }

    def score(
        self,
        opportunity: Opportunity,
        trend_alignment: Dict,
        volume_ratio: float,
        regime: MarketRegime,
        cost_analysis: Dict,
    ) -> ScoredOpportunity:
        score = 0
        factors = []

        # Normalize primary_edge to EdgeType (handle Pydantic use_enum_values)
        primary_edge = opportunity.primary_edge
        if isinstance(primary_edge, str):
            try:
                primary_edge = EdgeType(primary_edge)
            except ValueError:
                primary_edge = None
        edge_name = getattr(primary_edge, "value", primary_edge) or str(primary_edge)

        # 1. Primary edge score (30-40 points)
        edge_score = self.EDGE_BASE_SCORES.get(primary_edge, 10) if primary_edge else 10
        score += edge_score
        factors.append(f"Primary edge ({edge_name}): +{edge_score}")

        # 2. Secondary edges (max 25 points, up to 3 edges at 40% value)
        secondary_edges = getattr(opportunity, "secondary_edges", []) or []
        secondary_score = 0
        for edge in secondary_edges[:3]:
            e = EdgeType(edge) if isinstance(edge, str) else edge
            edge_pts = self.EDGE_BASE_SCORES.get(e, 5) * 0.4
            secondary_score += edge_pts
        secondary_score = min(secondary_score, 25)
        score += int(secondary_score)
        if secondary_score > 0:
            factors.append(
                f"Secondary edges ({len(secondary_edges[:3])}): +{int(secondary_score)}"
            )

        # 3. Trend alignment (max 15 points)
        alignment = trend_alignment.get("alignment", "NEUTRAL")
        direction = opportunity.direction
        if isinstance(direction, str):
            try:
                direction = Direction(direction)
            except ValueError:
                direction = None
        direction_matches = self._check_trend_direction_match(alignment, direction)
        if alignment in ["STRONG_BULLISH", "STRONG_BEARISH"] and direction_matches:
            score += 15
            factors.append("Strong trend alignment: +15")
        elif alignment == "PARTIAL" and direction_matches:
            score += 8
            factors.append("Partial trend alignment: +8")
        elif alignment == "CONFLICTING":
            factors.append("Conflicting trend: +0")
        elif alignment == "NEUTRAL":
            factors.append("Neutral trend: +0")

        # 4. Volume confirmation (max 10 points)
        if volume_ratio >= 2.0:
            score += 10
            factors.append(f"High volume ({volume_ratio:.1f}x): +10")
        elif volume_ratio >= 1.5:
            score += 7
            factors.append(f"Elevated volume ({volume_ratio:.1f}x): +7")
        elif volume_ratio >= 1.2:
            score += 4
            factors.append(f"Above avg volume ({volume_ratio:.1f}x): +4")
        else:
            factors.append(f"Normal volume ({volume_ratio:.1f}x): +0")

        # 5. Regime alignment (max 10 points)
        if primary_edge and self._regime_aligns(primary_edge, regime):
            score += 10
            factors.append(f"Regime aligned ({regime.value}): +10")
        else:
            factors.append(f"Regime neutral ({regime.value}): +0")

        # 6. Risk/Reward ratio (max 10 points)
        rr = opportunity.risk_reward_ratio
        if rr >= 3.0:
            score += 10
            factors.append(f"Excellent R:R ({rr:.1f}:1): +10")
        elif rr >= 2.0:
            score += 6
            factors.append(f"Good R:R ({rr:.1f}:1): +6")
        elif rr >= 1.5:
            score += 3
            factors.append(f"Acceptable R:R ({rr:.1f}:1): +3")
        else:
            factors.append(f"Poor R:R ({rr:.1f}:1): +0")

        # 7. Cost efficiency (max 5 points)
        cost_ratio = cost_analysis.get("cost_ratio", 100)
        if cost_ratio < 20:
            score += 5
            factors.append(f"Low costs ({cost_ratio:.0f}%): +5")
        elif cost_ratio < 35:
            score += 3
            factors.append(f"Moderate costs ({cost_ratio:.0f}%): +3")
        else:
            factors.append(f"High costs ({cost_ratio:.0f}%): +0")

        # Cap at 100
        score = min(score, 100)

        # Determine tier
        tier = self._get_tier(score)
        multiplier = self.TIER_MULTIPLIERS[tier]

        return ScoredOpportunity(
            opportunity=opportunity,
            score=score,
            tier=tier,
            factors=factors,
            position_multiplier=multiplier,
        )

    def _get_tier(self, score: int) -> str:
        if score >= self.TIER_THRESHOLDS["A"]:
            return "A"
        elif score >= self.TIER_THRESHOLDS["B"]:
            return "B"
        elif score >= self.TIER_THRESHOLDS["C"]:
            return "C"
        elif score >= self.TIER_THRESHOLDS["D"]:
            return "D"
        else:
            return "F"

    def _check_trend_direction_match(self, alignment: str, direction: Direction) -> bool:
        """Check if trend alignment matches trade direction."""
        if alignment == "STRONG_BULLISH" and direction == Direction.LONG:
            return True
        if alignment == "STRONG_BEARISH" and direction == Direction.SHORT:
            return True
        if alignment == "PARTIAL":
            return True  # Partial always gives some credit
        return False

    def _regime_aligns(self, edge_type: EdgeType, regime: MarketRegime) -> bool:
        """Check if edge type works well in current regime."""

        REGIME_EDGES = {
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
                EdgeType.INSIDER_CLUSTER,
                EdgeType.TURN_OF_MONTH,
            ],
        }

        allowed_edges = REGIME_EDGES.get(regime, [])
        return edge_type in allowed_edges
