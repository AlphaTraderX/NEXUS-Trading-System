"""
NEXUS AI Reasoning Engine
Uses Groq's Llama 3.1 to generate trade explanations.

WHY THIS MATTERS:
- Helps you understand the trade logic
- Builds confidence in the system
- Aids post-trade analysis
- Makes signals more actionable

GROQ: Fast, cheap LLM inference
- Llama 3.1 70B for quality reasoning
- ~100ms response time
- Very low cost
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional Groq import - will work without it
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from nexus.core.enums import Direction, EdgeType
from .regime import MarketRegime


@dataclass
class ReasoningResult:
    """Result from AI reasoning generation."""
    explanation: str
    confidence_factors: list
    risk_warnings: list
    action_summary: str
    model_used: str
    generation_time_ms: float
    tokens_used: int

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "confidence_factors": self.confidence_factors,
            "risk_warnings": self.risk_warnings,
            "action_summary": self.action_summary,
            "model_used": self.model_used,
            "generation_time_ms": round(self.generation_time_ms, 2),
            "tokens_used": self.tokens_used,
        }


class ReasoningEngine:
    """
    Generate AI-powered trade explanations using Groq.

    Falls back to template-based reasoning if Groq unavailable.
    """

    # Edge descriptions for context
    EDGE_DESCRIPTIONS = {
        EdgeType.INSIDER_CLUSTER: "Multiple company insiders purchasing shares within a short timeframe, indicating strong internal confidence",
        EdgeType.VWAP_DEVIATION: "Price has deviated significantly from Volume Weighted Average Price, suggesting mean reversion opportunity",
        EdgeType.TURN_OF_MONTH: "Turn of Month effect - historically 100% of equity premium occurs in the 4-day window around month end/start",
        EdgeType.MONTH_END: "Month-end rebalancing flows from pension funds ($7.5T in assets) create predictable price movements",
        EdgeType.GAP_FILL: "Price gap from previous close tends to fill with 60-92% probability for small gaps",
        EdgeType.RSI_EXTREME: "RSI at extreme levels (below 20 or above 80) indicating oversold/overbought conditions ripe for reversal",
        EdgeType.POWER_HOUR: "Final hour of US trading shows increased volume and momentum continuation patterns",
        EdgeType.ASIAN_RANGE: "Break of Asian session range during London open, a validated ICT concept",
        EdgeType.ORB: "Opening Range Breakout - break of first 15-30 minute range with volume confirmation",
        EdgeType.BOLLINGER_TOUCH: "Price touching Bollinger Band in ranging market, 88% mean reversion probability",
        EdgeType.LONDON_OPEN: "London session open breakout capturing European institutional flow",
        EdgeType.NY_OPEN: "New York session open capturing US institutional participation",
        EdgeType.EARNINGS_DRIFT: "Post-earnings announcement drift - momentum continues in direction of surprise",
    }

    # Regime descriptions
    REGIME_DESCRIPTIONS = {
        MarketRegime.TRENDING_UP: "bullish trending market favoring momentum strategies",
        MarketRegime.TRENDING_DOWN: "bearish trending market requiring defensive positioning",
        MarketRegime.RANGING: "range-bound market ideal for mean reversion strategies",
        MarketRegime.VOLATILE: "highly volatile conditions requiring reduced position sizes",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-70b-versatile",
        max_tokens: int = 300,
        temperature: float = 0.3,
    ):
        """
        Initialize reasoning engine.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model to use (default: llama-3.1-70b-versatile)
            max_tokens: Max response tokens
            temperature: Response temperature (lower = more focused)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client = None
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Groq client: {e}")

        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.fallback_count = 0

    @property
    def is_available(self) -> bool:
        """Check if Groq is available."""
        return self.client is not None

    def generate(
        self,
        symbol: str,
        direction: Direction,
        primary_edge: EdgeType,
        secondary_edges: list,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        score: int,
        tier: str,
        regime: MarketRegime,
        net_edge: float,
        cost_ratio: float,
        volume_ratio: float = 1.0,
        trend_aligned: bool = True,
        edge_data: Dict = None,
    ) -> ReasoningResult:
        """
        Generate AI reasoning for a trade.

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            primary_edge: Main edge type
            secondary_edges: Additional edges
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            score: Opportunity score (0-100)
            tier: Signal tier (A/B/C/D)
            regime: Market regime
            net_edge: Net edge after costs (%)
            cost_ratio: Costs as % of gross edge
            volume_ratio: Current volume vs average
            trend_aligned: Whether trade aligns with trend
            edge_data: Additional edge-specific data

        Returns:
            ReasoningResult with explanation and analysis
        """
        start_time = datetime.now()

        # Get direction string
        dir_str = direction.value if hasattr(direction, 'value') else str(direction)
        edge_str = primary_edge.value if hasattr(primary_edge, 'value') else str(primary_edge)
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)

        # Calculate R:R
        if dir_str == "long":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        rr_ratio = reward / risk if risk > 0 else 0

        # Try Groq first
        if self.is_available:
            try:
                result = self._generate_with_groq(
                    symbol=symbol,
                    direction=dir_str,
                    primary_edge=edge_str,
                    secondary_edges=[e.value if hasattr(e, 'value') else str(e) for e in secondary_edges],
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=score,
                    tier=tier,
                    regime=regime_str,
                    net_edge=net_edge,
                    cost_ratio=cost_ratio,
                    rr_ratio=rr_ratio,
                    volume_ratio=volume_ratio,
                    trend_aligned=trend_aligned,
                    edge_data=edge_data or {},
                )

                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                result.generation_time_ms = elapsed_ms
                self.total_calls += 1
                self.total_tokens += result.tokens_used

                return result

            except Exception as e:
                print(f"Groq generation failed, using fallback: {e}")

        # Fallback to template-based reasoning
        self.fallback_count += 1
        return self._generate_fallback(
            symbol=symbol,
            direction=dir_str,
            primary_edge=primary_edge,
            secondary_edges=secondary_edges,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            score=score,
            tier=tier,
            regime=regime,
            net_edge=net_edge,
            cost_ratio=cost_ratio,
            rr_ratio=rr_ratio,
            volume_ratio=volume_ratio,
            trend_aligned=trend_aligned,
            edge_data=edge_data or {},
            start_time=start_time,
        )

    def _generate_with_groq(
        self,
        symbol: str,
        direction: str,
        primary_edge: str,
        secondary_edges: list,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        score: int,
        tier: str,
        regime: str,
        net_edge: float,
        cost_ratio: float,
        rr_ratio: float,
        volume_ratio: float,
        trend_aligned: bool,
        edge_data: Dict,
    ) -> ReasoningResult:
        """Generate reasoning using Groq API."""

        # Resolve primary_edge string to EdgeType for description lookup
        try:
            edge_type_key = EdgeType(primary_edge) if primary_edge in [e.value for e in EdgeType] else None
        except (ValueError, TypeError):
            edge_type_key = None
        edge_desc = self.EDGE_DESCRIPTIONS.get(edge_type_key, f"Statistical edge: {primary_edge}")

        prompt = f"""You are a professional trading analyst explaining a trade signal. Be concise and actionable.

TRADE SIGNAL:
- Symbol: {symbol}
- Direction: {direction.upper()}
- Primary Edge: {primary_edge.replace('_', ' ').title()}
- Edge Description: {edge_desc}
- Secondary Edges: {', '.join([e.replace('_', ' ').title() for e in secondary_edges]) if secondary_edges else 'None'}
- Entry: ${entry_price:.2f}
- Stop Loss: ${stop_loss:.2f}
- Take Profit: ${take_profit:.2f}
- Risk/Reward: {rr_ratio:.1f}:1
- Score: {score}/100 (Tier {tier})
- Market Regime: {regime.replace('_', ' ').title()}
- Net Edge: {net_edge:.2f}% (after {cost_ratio:.0f}% costs)
- Volume: {volume_ratio:.1f}x average
- Trend Aligned: {'Yes' if trend_aligned else 'No'}
{f'- Additional Data: {edge_data}' if edge_data else ''}

Provide a 2-3 sentence explanation of WHY this is a valid trade opportunity. Focus on the edge, the setup quality, and key risk factors. Be direct and professional."""

        # Call Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional trading analyst. Be concise, direct, and focus on actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        explanation = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        # Extract confidence factors
        confidence_factors = []
        if score >= 80:
            confidence_factors.append(f"High conviction score ({score}/100)")
        if rr_ratio >= 2.0:
            confidence_factors.append(f"Favorable R:R ({rr_ratio:.1f}:1)")
        if trend_aligned:
            confidence_factors.append("Trade aligned with higher timeframe trend")
        if volume_ratio > 1.5:
            confidence_factors.append(f"Strong volume confirmation ({volume_ratio:.1f}x)")
        if secondary_edges:
            confidence_factors.append(f"Multiple edges ({len(secondary_edges) + 1} total)")

        # Extract risk warnings
        risk_warnings = []
        if not trend_aligned:
            risk_warnings.append("Trading against higher timeframe trend")
        if regime == "volatile":
            risk_warnings.append("Elevated market volatility")
        if cost_ratio > 40:
            risk_warnings.append(f"Higher friction costs ({cost_ratio:.0f}% of edge)")
        if score < 65:
            risk_warnings.append(f"Moderate conviction score ({score}/100)")
        if volume_ratio < 1.0:
            risk_warnings.append("Below average volume")

        # Action summary
        action_summary = f"{direction.upper()} {symbol} @ ${entry_price:.2f} | Stop ${stop_loss:.2f} | Target ${take_profit:.2f}"

        return ReasoningResult(
            explanation=explanation,
            confidence_factors=confidence_factors,
            risk_warnings=risk_warnings,
            action_summary=action_summary,
            model_used=self.model,
            generation_time_ms=0,  # Will be set by caller
            tokens_used=tokens_used,
        )

    def _generate_fallback(
        self,
        symbol: str,
        direction: str,
        primary_edge: EdgeType,
        secondary_edges: list,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        score: int,
        tier: str,
        regime: MarketRegime,
        net_edge: float,
        cost_ratio: float,
        rr_ratio: float,
        volume_ratio: float,
        trend_aligned: bool,
        edge_data: Dict,
        start_time: datetime,
    ) -> ReasoningResult:
        """Generate template-based reasoning when Groq unavailable."""

        # Get edge description
        edge_desc = self.EDGE_DESCRIPTIONS.get(primary_edge, f"Statistical edge detected")
        regime_desc = self.REGIME_DESCRIPTIONS.get(regime, "current market conditions")

        # Get string values
        edge_str = primary_edge.value if hasattr(primary_edge, 'value') else str(primary_edge)
        dir_str = direction if isinstance(direction, str) else direction.value

        # Build explanation
        explanation = f"{edge_str.replace('_', ' ').title()} setup on {symbol}. {edge_desc}. "
        explanation += f"Score {score}/100 (Tier {tier}) in {regime_desc}. "
        explanation += f"Net edge of {net_edge:.2f}% after costs with {rr_ratio:.1f}:1 reward-to-risk."

        # Confidence factors
        confidence_factors = []
        if score >= 80:
            confidence_factors.append(f"High conviction ({score}/100)")
        elif score >= 65:
            confidence_factors.append(f"Good conviction ({score}/100)")

        if rr_ratio >= 2.5:
            confidence_factors.append(f"Excellent R:R ({rr_ratio:.1f}:1)")
        elif rr_ratio >= 2.0:
            confidence_factors.append(f"Good R:R ({rr_ratio:.1f}:1)")

        if trend_aligned:
            confidence_factors.append("Trend aligned")

        if volume_ratio > 1.5:
            confidence_factors.append(f"High volume ({volume_ratio:.1f}x)")

        if secondary_edges:
            confidence_factors.append(f"{len(secondary_edges) + 1} edges stacked")

        # Risk warnings
        risk_warnings = []
        if not trend_aligned:
            risk_warnings.append("Counter-trend trade")

        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        if regime_str == "volatile":
            risk_warnings.append("Volatile market")

        if cost_ratio > 40:
            risk_warnings.append(f"Higher costs ({cost_ratio:.0f}%)")

        if score < 65:
            risk_warnings.append("Moderate conviction")

        if rr_ratio < 1.5:
            risk_warnings.append(f"Lower R:R ({rr_ratio:.1f}:1)")

        # Action summary
        action_summary = f"{dir_str.upper()} {symbol} @ ${entry_price:.2f} | Stop ${stop_loss:.2f} | Target ${take_profit:.2f}"

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return ReasoningResult(
            explanation=explanation,
            confidence_factors=confidence_factors,
            risk_warnings=risk_warnings,
            action_summary=action_summary,
            model_used="fallback_template",
            generation_time_ms=elapsed_ms,
            tokens_used=0,
        )

    def get_statistics(self) -> Dict:
        """Get reasoning engine statistics."""
        return {
            "groq_available": self.is_available,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "fallback_count": self.fallback_count,
            "model": self.model if self.is_available else "fallback_template",
        }


# Test the reasoning engine
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS AI REASONING ENGINE TEST")
    print("=" * 60)

    engine = ReasoningEngine()

    print(f"\nGroq available: {engine.is_available}")
    print(f"Model: {engine.model if engine.is_available else 'fallback_template'}")

    # Test 1: Generate reasoning (will use fallback if no API key)
    print("\n--- Test 1: High Quality Trade ---")

    result = engine.generate(
        symbol="AAPL",
        direction=Direction.LONG,
        primary_edge=EdgeType.INSIDER_CLUSTER,
        secondary_edges=[EdgeType.RSI_EXTREME],
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=162.0,
        score=85,
        tier="A",
        regime=MarketRegime.TRENDING_UP,
        net_edge=0.28,
        cost_ratio=22.0,
        volume_ratio=1.8,
        trend_aligned=True,
        edge_data={"insider_count": 4, "total_value": 2500000},
    )

    print(f"Explanation: {result.explanation}")
    print(f"Confidence: {result.confidence_factors}")
    print(f"Warnings: {result.risk_warnings}")
    print(f"Action: {result.action_summary}")
    print(f"Model: {result.model_used}")
    print(f"Time: {result.generation_time_ms:.1f}ms")

    # Test 2: Counter-trend trade
    print("\n--- Test 2: Counter-Trend Trade ---")

    result = engine.generate(
        symbol="MSFT",
        direction=Direction.SHORT,
        primary_edge=EdgeType.RSI_EXTREME,
        secondary_edges=[],
        entry_price=380.0,
        stop_loss=390.0,
        take_profit=365.0,
        score=58,
        tier="C",
        regime=MarketRegime.TRENDING_UP,
        net_edge=0.12,
        cost_ratio=45.0,
        volume_ratio=0.9,
        trend_aligned=False,
    )

    print(f"Explanation: {result.explanation}")
    print(f"Confidence: {result.confidence_factors}")
    print(f"Warnings: {result.risk_warnings}")

    # Test 3: Volatile market
    print("\n--- Test 3: Volatile Market ---")

    result = engine.generate(
        symbol="NVDA",
        direction=Direction.LONG,
        primary_edge=EdgeType.VWAP_DEVIATION,
        secondary_edges=[EdgeType.BOLLINGER_TOUCH],
        entry_price=450.0,
        stop_loss=440.0,
        take_profit=470.0,
        score=72,
        tier="B",
        regime=MarketRegime.VOLATILE,
        net_edge=0.18,
        cost_ratio=28.0,
        volume_ratio=2.2,
        trend_aligned=True,
    )

    print(f"Explanation: {result.explanation}")
    print(f"Warnings: {result.risk_warnings}")

    # Test 4: Statistics
    print("\n--- Test 4: Engine Statistics ---")
    stats = engine.get_statistics()
    print(f"Stats: {stats}")

    print("\n" + "=" * 60)
    print("REASONING ENGINE TEST COMPLETE [OK]")
    print("=" * 60)
