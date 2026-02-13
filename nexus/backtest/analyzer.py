"""
AI-Powered Backtest Analyzer

Uses Groq (Llama 3.1 70B) to analyze backtest results and provide insights.
Falls back to rule-based analysis when API key is unavailable.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from nexus.core.enums import EdgeType, Timeframe, Market

logger = logging.getLogger(__name__)


@dataclass
class EdgeAnalysis:
    """AI analysis for a single edge."""

    edge_type: EdgeType
    is_profitable: bool
    recommendation: str  # KEEP, DROP, MODIFY, NEEDS_MORE_DATA
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    suggested_improvements: List[str] = field(default_factory=list)
    optimal_timeframes: List[Timeframe] = field(default_factory=list)
    optimal_markets: List[Market] = field(default_factory=list)


class AIAnalyzer:
    """
    AI-powered analysis of backtest results.

    Uses Groq LLM to:
    - Analyze edge performance
    - Identify patterns
    - Recommend improvements
    - Explain why edges work or don't
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = "llama-3.1-70b-versatile"

    async def analyze_results(
        self,
        result: "MultiBacktestResult",
    ) -> Dict[str, Any]:
        """
        Analyze complete backtest results.

        Returns comprehensive analysis including:
        - Edge-by-edge analysis
        - Timeframe recommendations
        - Market recommendations
        - Overall strategy assessment
        """
        prompt = self._build_analysis_prompt(result)

        if self.api_key:
            analysis_text = await self._call_groq(prompt)
        else:
            analysis_text = self._rule_based_analysis(result)

        return self._parse_analysis(analysis_text, result)

    def _build_analysis_prompt(self, result: "MultiBacktestResult") -> str:
        edge_summary = []
        for edge_key, perf in result.edge_performance.items():
            edge_summary.append(
                f"- {edge_key}: {perf.total_trades} trades, "
                f"{perf.win_rate:.1f}% win rate, "
                f"${perf.net_pnl:,.2f} net P&L, "
                f"{perf.expectancy:.2f}R expectancy, "
                f"{perf.statistical_significance} significance"
            )

        tf_summary = []
        for tf_key, perf in result.timeframe_performance.items():
            if perf.total_trades > 0:
                tf_summary.append(
                    f"- {tf_key}: {perf.total_trades} trades, "
                    f"{perf.win_rate:.1f}% win rate, "
                    f"${perf.net_pnl:,.2f} net P&L"
                )

        prompt = f"""
You are an expert quantitative trading analyst. Analyze these backtest results and provide actionable insights.

## BACKTEST SUMMARY
- Period: {result.start_date.date()} to {result.end_date.date()}
- Starting Balance: ${result.starting_balance:,.2f}
- Ending Balance: ${result.ending_balance:,.2f}
- Net Return: {result.net_return_pct:.2f}%
- Total Trades: {result.total_trades}
- Win Rate: {result.win_rate:.1f}%
- Max Drawdown: {result.max_drawdown_pct:.2f}%

## PERFORMANCE BY EDGE TYPE
{chr(10).join(edge_summary) if edge_summary else "No edge data available"}

## PERFORMANCE BY TIMEFRAME
{chr(10).join(tf_summary) if tf_summary else "No timeframe data available"}

## ANALYSIS REQUIRED

For each edge type, provide:
1. VERDICT: KEEP (profitable with good stats), DROP (unprofitable or poor stats), MODIFY (needs adjustment), or NEEDS_MORE_DATA (insufficient trades)
2. REASONING: Why this edge works or doesn't work
3. IMPROVEMENTS: Specific suggestions to improve performance

Then provide:
4. OVERALL ASSESSMENT: Is this a viable trading system?
5. TOP 3 RECOMMENDATIONS: Most important changes to make
6. RISK WARNINGS: Any concerning patterns

Be specific and quantitative. Reference actual numbers from the results.
"""

        return prompt

    async def _call_groq(self, prompt: str) -> str:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert quantitative trading analyst with deep knowledge of edge-based trading strategies, statistical significance, and risk management."
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000,
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Groq API error: {response.status_code}")
                    return self._rule_based_analysis_text()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return self._rule_based_analysis_text()

    def _rule_based_analysis(self, result: "MultiBacktestResult") -> str:
        lines = ["## AUTOMATED ANALYSIS (Rule-Based)\n"]

        for edge_key, perf in result.edge_performance.items():
            verdict = "KEEP" if perf.is_profitable and perf.win_rate >= 50 else "DROP"
            if perf.total_trades < 30:
                verdict = "NEEDS_MORE_DATA"

            lines.append(f"\n### {edge_key}")
            lines.append(f"- Verdict: {verdict}")
            lines.append(f"- Win Rate: {perf.win_rate:.1f}%")
            lines.append(f"- Net P&L: ${perf.net_pnl:,.2f}")
            lines.append(f"- Trades: {perf.total_trades}")

            if perf.is_profitable:
                lines.append(f"- Reasoning: Positive expectancy with {perf.expectancy:.2f}R per trade")
            else:
                lines.append(f"- Reasoning: Negative expectancy, costs eating edge")

        return "\n".join(lines)

    def _rule_based_analysis_text(self) -> str:
        return """
## ANALYSIS (API Unavailable)

Unable to perform AI analysis. Please check your GROQ_API_KEY.

General recommendations:
1. Edges with win rate < 50% and negative expectancy should be dropped
2. Edges with < 30 trades need more data before conclusions
3. Focus on edges with positive R-multiple expectancy
4. Monitor cost ratios - if costs > 30% of gross P&L, edge may not be viable
"""

    def _parse_analysis(
        self,
        analysis_text: str,
        result: "MultiBacktestResult",
    ) -> Dict[str, Any]:
        edge_analyses = {}

        for edge_key, perf in result.edge_performance.items():
            # Determine recommendation based on stats
            if perf.total_trades < 30:
                recommendation = "NEEDS_MORE_DATA"
                confidence = "LOW"
            elif perf.is_profitable and perf.win_rate >= 50 and perf.expectancy > 0:
                recommendation = "KEEP"
                confidence = "HIGH" if perf.total_trades >= 100 else "MEDIUM"
            elif perf.is_profitable:
                recommendation = "MODIFY"
                confidence = "MEDIUM"
            else:
                recommendation = "DROP"
                confidence = "HIGH" if perf.total_trades >= 50 else "MEDIUM"

            edge_analyses[edge_key] = EdgeAnalysis(
                edge_type=perf.edge_type,
                is_profitable=perf.is_profitable,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=f"Win rate: {perf.win_rate:.1f}%, Expectancy: {perf.expectancy:.2f}R, Trades: {perf.total_trades}",
            )

        return {
            "raw_analysis": analysis_text,
            "edge_analyses": edge_analyses,
            "summary": {
                "total_edges_tested": len(result.edge_performance),
                "profitable_edges": sum(1 for p in result.edge_performance.values() if p.is_profitable),
                "edges_to_keep": sum(1 for e in edge_analyses.values() if e.recommendation == "KEEP"),
                "edges_to_drop": sum(1 for e in edge_analyses.values() if e.recommendation == "DROP"),
                "overall_viable": result.net_pnl > 0 and result.win_rate >= 45,
            },
        }

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        lines = [
            "=" * 70,
            "NEXUS BACKTEST ANALYSIS REPORT",
            "=" * 70,
            "",
            "## SUMMARY",
            f"- Edges Tested: {analysis['summary']['total_edges_tested']}",
            f"- Profitable Edges: {analysis['summary']['profitable_edges']}",
            f"- Edges to Keep: {analysis['summary']['edges_to_keep']}",
            f"- Edges to Drop: {analysis['summary']['edges_to_drop']}",
            f"- Overall Viable: {'YES' if analysis['summary']['overall_viable'] else 'NO'}",
            "",
            "## EDGE RECOMMENDATIONS",
        ]

        for edge_key, edge_analysis in analysis["edge_analyses"].items():
            status = "[KEEP]" if edge_analysis.recommendation == "KEEP" else "[DROP]" if edge_analysis.recommendation == "DROP" else "[?]"
            lines.append(f"\n{status} {edge_key}")
            lines.append(f"   Recommendation: {edge_analysis.recommendation}")
            lines.append(f"   Confidence: {edge_analysis.confidence}")
            lines.append(f"   Reasoning: {edge_analysis.reasoning}")

        lines.extend([
            "",
            "## AI ANALYSIS",
            analysis["raw_analysis"],
        ])

        return "\n".join(lines)
