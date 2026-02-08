"""
NEXUS AI Reasoning Engine
Generates human-readable explanations for trading signals using Groq LLM.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from nexus.core.enums import EdgeType, Direction, MarketRegime
from nexus.core.models import Opportunity

# Optional Groq import - gracefully handle if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


@dataclass
class ReasoningResult:
    """Result from AI reasoning generation."""
    explanation: str
    risk_factors: List[str]
    confluence_summary: str
    confidence_note: str
    generated_by: str  # "ai" or "template"

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "risk_factors": self.risk_factors,
            "confluence_summary": self.confluence_summary,
            "confidence_note": self.confidence_note,
            "generated_by": self.generated_by,
        }


class ReasoningEngine:
    """
    Generate AI-powered explanations for trading signals.

    Uses Groq API with Llama 3.1 70B for fast, free inference.
    Falls back to template-based reasoning if API unavailable.
    """

    # Edge type descriptions for context
    EDGE_DESCRIPTIONS = {
        EdgeType.INSIDER_CLUSTER: "Multiple corporate insiders buying shares within 14 days - strongest documented edge with 2.1% monthly abnormal returns",
        EdgeType.VWAP_DEVIATION: "Price has deviated significantly from Volume Weighted Average Price, suggesting mean reversion opportunity",
        EdgeType.TURN_OF_MONTH: "Turn of Month effect - historically 100% of equity premium earned in 4-day window around month end/start",
        EdgeType.MONTH_END: "Month-end rebalancing flows from $7.5T pension fund assets create predictable price pressure",
        EdgeType.GAP_FILL: "Gap detected with high probability of fill - 60-92% of small gaps fill within the session",
        EdgeType.RSI_EXTREME: "RSI at extreme levels (below 20 or above 80) indicating oversold/overbought condition",
        EdgeType.POWER_HOUR: "Power Hour (final hour of US session) - U-shaped volume pattern creates momentum opportunities",
        EdgeType.ASIAN_RANGE: "Asian session range established - London open breakout setup forming",
        EdgeType.ORB: "Opening Range Breakout - break of first 15-30 minute range with volume confirmation",
        EdgeType.BOLLINGER_TOUCH: "Price touched Bollinger Band in ranging market - 88% mean reversion probability",
        EdgeType.LONDON_OPEN: "London session open - major liquidity injection creates breakout opportunities",
        EdgeType.NY_OPEN: "New York session open - highest probability period for establishing daily high/low",
        EdgeType.EARNINGS_DRIFT: "Post-earnings announcement drift - momentum continuation after earnings surprise",
    }

    # Regime descriptions
    REGIME_DESCRIPTIONS = {
        MarketRegime.TRENDING_UP: "strong uptrend with momentum favoring longs",
        MarketRegime.TRENDING_DOWN: "downtrend with defensive positioning recommended",
        MarketRegime.RANGING: "range-bound conditions favoring mean reversion",
        MarketRegime.VOLATILE: "high volatility requiring reduced position sizes",
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-70b-versatile"):
        """
        Initialize reasoning engine.

        Args:
            api_key: Groq API key. If None, tries GROQ_API_KEY env var.
            model: Model to use (default: llama-3.1-70b-versatile)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Groq client: {e}")
                self.client = None

    @property
    def ai_available(self) -> bool:
        """Check if AI reasoning is available."""
        return self.client is not None

    def generate(
        self,
        opportunity: Opportunity,
        scored_result: Dict[str, Any],
        regime_data: Dict[str, Any],
        cost_analysis: Dict[str, Any],
    ) -> ReasoningResult:
        """
        Generate reasoning for a trading opportunity.

        Args:
            opportunity: The Opportunity object
            scored_result: Output from OpportunityScorer.score().to_dict()
            regime_data: Output from RegimeDetector.detect_regime()
            cost_analysis: Output from CostEngine.calculate_net_edge()

        Returns:
            ReasoningResult with explanation, risks, and confluence
        """

        # Try AI generation first
        if self.ai_available:
            try:
                return self._generate_ai_reasoning(
                    opportunity, scored_result, regime_data, cost_analysis
                )
            except Exception as e:
                print(f"AI reasoning failed, falling back to template: {e}")

        # Fallback to template-based reasoning
        return self._generate_template_reasoning(
            opportunity, scored_result, regime_data, cost_analysis
        )

    def _generate_ai_reasoning(
        self,
        opportunity: Opportunity,
        scored_result: Dict[str, Any],
        regime_data: Dict[str, Any],
        cost_analysis: Dict[str, Any],
    ) -> ReasoningResult:
        """Generate reasoning using Groq LLM."""

        # Build context for LLM
        primary_edge = opportunity.primary_edge
        if isinstance(primary_edge, str):
            primary_edge = EdgeType(primary_edge)

        edge_desc = self.EDGE_DESCRIPTIONS.get(primary_edge, "Statistical edge detected")

        regime = regime_data.get("regime", MarketRegime.RANGING)
        if isinstance(regime, str):
            regime = MarketRegime(regime)
        regime_desc = self.REGIME_DESCRIPTIONS.get(regime, "current market conditions")

        direction = opportunity.direction
        if isinstance(direction, str):
            direction = Direction(direction)

        # Calculate R:R
        if direction == Direction.LONG:
            risk = opportunity.entry_price - opportunity.stop_loss
            reward = opportunity.take_profit - opportunity.entry_price
        else:
            risk = opportunity.stop_loss - opportunity.entry_price
            reward = opportunity.entry_price - opportunity.take_profit
        rr_ratio = reward / risk if risk > 0 else 0

        factors = scored_result.get("factors", [])
        factors_text = "\n".join("- " + f for f in factors) if factors else "N/A"

        prompt = f"""You are a professional trading analyst. Generate a brief, actionable explanation for this trading signal.

SIGNAL DETAILS:
- Symbol: {opportunity.symbol}
- Direction: {direction.value.upper()}
- Entry: {opportunity.entry_price}
- Stop Loss: {opportunity.stop_loss}
- Take Profit: {opportunity.take_profit}
- Risk:Reward: {rr_ratio:.1f}:1

PRIMARY EDGE:
{edge_desc}

SCORE: {scored_result.get('score', 0)}/100 (Tier {scored_result.get('tier', 'C')})

SCORING FACTORS:
{factors_text}

MARKET REGIME:
{regime.value} - {regime_desc}
Regime Reasoning: {regime_data.get('reasoning', 'N/A')}

COST ANALYSIS:
- Net Edge: {cost_analysis.get('net_edge', 0):.3f}%
- Cost Ratio: {cost_analysis.get('cost_ratio', 0):.1f}%
- Viable: {cost_analysis.get('viable', False)}

Respond in JSON format:
{{
    "explanation": "2-3 sentence explanation of why this trade makes sense",
    "risk_factors": ["risk 1", "risk 2", "risk 3"],
    "confluence_summary": "One sentence summarizing the edge confluence",
    "confidence_note": "Brief note on conviction level based on score/tier"
}}

Be specific and actionable. No fluff."""

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional trading analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )

        # Parse response
        content = response.choices[0].message.content.strip()

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        return ReasoningResult(
            explanation=result.get("explanation", "AI reasoning generated."),
            risk_factors=result.get("risk_factors", []),
            confluence_summary=result.get("confluence_summary", ""),
            confidence_note=result.get("confidence_note", ""),
            generated_by="ai"
        )

    def _generate_template_reasoning(
        self,
        opportunity: Opportunity,
        scored_result: Dict[str, Any],
        regime_data: Dict[str, Any],
        cost_analysis: Dict[str, Any],
    ) -> ReasoningResult:
        """Generate reasoning using templates (fallback when AI unavailable)."""

        primary_edge = opportunity.primary_edge
        if isinstance(primary_edge, str):
            primary_edge = EdgeType(primary_edge)

        edge_desc = self.EDGE_DESCRIPTIONS.get(primary_edge, "Statistical edge detected")

        regime = regime_data.get("regime", MarketRegime.RANGING)
        if isinstance(regime, str):
            regime = MarketRegime(regime)
        regime_desc = self.REGIME_DESCRIPTIONS.get(regime, "current conditions")

        direction = opportunity.direction
        if isinstance(direction, str):
            direction = Direction(direction)

        score = scored_result.get('score', 0)
        tier = scored_result.get('tier', 'C')

        # Build explanation
        direction_word = "long" if direction == Direction.LONG else "short"
        explanation = f"{primary_edge.value} edge detected on {opportunity.symbol}. {edge_desc}. "
        explanation += f"Market regime is {regime_desc}, supporting this {direction_word} setup. "
        explanation += f"Score: {score}/100 (Tier {tier})."

        # Build risk factors
        risk_factors = []

        cost_ratio = cost_analysis.get('cost_ratio', 0)
        if cost_ratio > 50:
            risk_factors.append(f"High cost ratio ({cost_ratio:.0f}%) reduces net edge")

        if regime == MarketRegime.VOLATILE:
            risk_factors.append("Volatile regime - consider reduced position size")

        net_edge = cost_analysis.get('net_edge', 0)
        if net_edge < 0.10:
            risk_factors.append(f"Thin net edge ({net_edge:.2f}%) leaves little margin for error")

        if not risk_factors:
            risk_factors.append("Standard market risks apply")

        # Confluence summary
        factors = scored_result.get('factors', [])
        positive_factors = [f for f in factors if '+0' not in f and 'Conflicting' not in f and 'Normal' not in f]
        confluence_summary = f"{len(positive_factors)} positive factors aligned for this setup."

        # Confidence note
        if tier == 'A':
            confidence_note = "High conviction setup - full position size recommended."
        elif tier == 'B':
            confidence_note = "Good conviction - standard to slightly elevated size."
        elif tier == 'C':
            confidence_note = "Moderate conviction - standard position size."
        elif tier == 'D':
            confidence_note = "Lower conviction - reduced position size recommended."
        else:
            confidence_note = "Low conviction - consider skipping this trade."

        return ReasoningResult(
            explanation=explanation,
            risk_factors=risk_factors,
            confluence_summary=confluence_summary,
            confidence_note=confidence_note,
            generated_by="template"
        )

    def generate_quick_summary(self, opportunity: Opportunity, score: int, tier: str) -> str:
        """
        Generate a one-line summary for quick display.

        Returns: "LONG SPY | VWAP_DEVIATION | Score: 78 (B) | Mean reversion setup"
        """
        primary_edge = opportunity.primary_edge
        if isinstance(primary_edge, str):
            primary_edge = EdgeType(primary_edge)

        direction = opportunity.direction
        if isinstance(direction, str):
            direction = Direction(direction)

        edge_short = {
            EdgeType.INSIDER_CLUSTER: "Insider buying cluster",
            EdgeType.VWAP_DEVIATION: "VWAP mean reversion",
            EdgeType.TURN_OF_MONTH: "Turn of month effect",
            EdgeType.MONTH_END: "Month-end rebalancing",
            EdgeType.GAP_FILL: "Gap fill setup",
            EdgeType.RSI_EXTREME: "RSI extreme reversal",
            EdgeType.POWER_HOUR: "Power hour momentum",
            EdgeType.ASIAN_RANGE: "Asian range breakout",
            EdgeType.ORB: "Opening range breakout",
            EdgeType.BOLLINGER_TOUCH: "Bollinger band reversion",
            EdgeType.LONDON_OPEN: "London open breakout",
            EdgeType.NY_OPEN: "NY open momentum",
            EdgeType.EARNINGS_DRIFT: "Post-earnings drift",
        }.get(primary_edge, "Edge detected")

        return f"{direction.value.upper()} {opportunity.symbol} | {primary_edge.value} | Score: {score} ({tier}) | {edge_short}"

    def format_for_discord(self, result: ReasoningResult) -> str:
        """Format reasoning for Discord message."""
        risks = "\n".join(f"⚠️ {r}" for r in result.risk_factors)

        return f"""**AI Analysis**
{result.explanation}

**Risk Factors**
{risks}

**Confluence:** {result.confluence_summary}
**Conviction:** {result.confidence_note}
"""

    def format_for_telegram(self, result: ReasoningResult) -> str:
        """Format reasoning for Telegram message."""
        risks = "\n".join(f"• {r}" for r in result.risk_factors)

        return f"""🤖 *AI Analysis*
{result.explanation}

⚠️ *Risk Factors*
{risks}

📊 {result.confluence_summary}
💡 {result.confidence_note}
"""
