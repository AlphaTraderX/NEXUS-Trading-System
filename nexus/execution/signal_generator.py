"""
NEXUS Signal Generator
Converts scored opportunities into complete, actionable NexusSignals.

This is where everything comes together:
- Opportunity from scanner
- Score from scorer
- Costs from cost engine
- Size from position sizer
- Context from regime/trend
- Reasoning from AI

OUTPUT: A complete signal with EVERYTHING needed to execute.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from uuid import uuid4
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.core.enums import Market, Direction, EdgeType, SignalStatus
from nexus.core.models import Opportunity, NexusSignal
from nexus.intelligence.cost_engine import CostEngine
from nexus.intelligence.scorer import OpportunityScorer, ScoredOpportunity, SignalTier, MarketRegime
from nexus.intelligence.trend_filter import TrendAlignment
from nexus.risk.position_sizer import DynamicPositionSizer, RiskMode
from nexus.risk.heat_manager import DynamicHeatManager, HeatAnalysis
from nexus.risk.circuit_breaker import SmartCircuitBreaker, BreakerState


@dataclass
class AccountState:
    """Current account state for signal generation."""
    starting_balance: float
    current_equity: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float
    win_streak: int = 0
    lose_streak: int = 0
    trades_today: int = 0
    portfolio_heat: float = 0.0  # Current heat %


@dataclass
class MarketState:
    """Current market state for context."""
    regime: MarketRegime
    trend_alignment: Optional[TrendAlignment] = None
    vix_level: float = 20.0
    session: str = "us_regular"
    upcoming_events: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        t = getattr(self.trend_alignment, "alignment", self.trend_alignment)
        value = getattr(t, "value", str(t)) if t else "unknown"
        return f"Regime: {self.regime.value}, Trend: {value}, VIX: {self.vix_level}"


@dataclass
class GenerationResult:
    """Result of signal generation attempt."""
    success: bool
    signal: Optional[NexusSignal] = None
    rejected_reason: Optional[str] = None
    checks_passed: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "signal_id": self.signal.signal_id if self.signal else None,
            "rejected_reason": self.rejected_reason,
            "checks_passed": self.checks_passed,
        }


class SignalGenerator:
    """
    Generate complete trading signals from scored opportunities.

    This is the integration point for all NEXUS components.
    """

    # Expected edge returns by type (gross, before costs)
    EDGE_EXPECTED_RETURNS = {
        EdgeType.INSIDER_CLUSTER: 0.40,
        EdgeType.VWAP_DEVIATION: 0.20,
        EdgeType.TURN_OF_MONTH: 0.30,
        EdgeType.MONTH_END: 0.25,
        EdgeType.GAP_FILL: 0.20,
        EdgeType.RSI_EXTREME: 0.15,
        EdgeType.POWER_HOUR: 0.12,
        EdgeType.ASIAN_RANGE: 0.12,
        EdgeType.ORB: 0.15,
        EdgeType.BOLLINGER_TOUCH: 0.12,
        EdgeType.LONDON_OPEN: 0.12,
        EdgeType.NY_OPEN: 0.12,
        EdgeType.EARNINGS_DRIFT: 0.20,
    }

    # Estimated hold times by edge (in days)
    EDGE_HOLD_TIMES = {
        EdgeType.INSIDER_CLUSTER: 5.0,
        EdgeType.VWAP_DEVIATION: 0.5,
        EdgeType.TURN_OF_MONTH: 3.0,
        EdgeType.MONTH_END: 2.0,
        EdgeType.GAP_FILL: 0.25,
        EdgeType.RSI_EXTREME: 1.0,
        EdgeType.POWER_HOUR: 0.1,
        EdgeType.ASIAN_RANGE: 0.25,
        EdgeType.ORB: 0.25,
        EdgeType.BOLLINGER_TOUCH: 0.5,
        EdgeType.LONDON_OPEN: 0.25,
        EdgeType.NY_OPEN: 0.25,
        EdgeType.EARNINGS_DRIFT: 3.0,
    }

    # Signal validity periods (hours)
    EDGE_VALIDITY_HOURS = {
        EdgeType.INSIDER_CLUSTER: 24,
        EdgeType.VWAP_DEVIATION: 2,
        EdgeType.TURN_OF_MONTH: 8,
        EdgeType.MONTH_END: 8,
        EdgeType.GAP_FILL: 4,
        EdgeType.RSI_EXTREME: 4,
        EdgeType.POWER_HOUR: 1,
        EdgeType.ASIAN_RANGE: 2,
        EdgeType.ORB: 2,
        EdgeType.BOLLINGER_TOUCH: 4,
        EdgeType.LONDON_OPEN: 2,
        EdgeType.NY_OPEN: 2,
        EdgeType.EARNINGS_DRIFT: 12,
    }

    # Market to broker mapping
    MARKET_BROKERS = {
        Market.US_STOCKS: "ibkr",
        Market.UK_STOCKS: "ibkr",
        Market.EU_STOCKS: "ibkr",
        Market.FOREX_MAJORS: "oanda",
        Market.FOREX_CROSSES: "oanda",
        Market.US_FUTURES: "ibkr",
        Market.COMMODITIES: "ibkr",
    }

    def __init__(
        self,
        cost_engine: CostEngine,
        scorer: OpportunityScorer,
        position_sizer: DynamicPositionSizer,
        heat_manager: DynamicHeatManager,
        circuit_breaker: SmartCircuitBreaker,
        ai_reasoning_callback: Optional[Callable] = None,
        min_score_to_trade: int = 50,
        min_net_edge: float = 0.05,
        max_cost_ratio: float = 70.0,
    ):
        """
        Initialize signal generator.

        Args:
            cost_engine: For calculating trading costs
            scorer: For scoring opportunities
            position_sizer: For calculating position sizes
            heat_manager: For checking portfolio heat
            circuit_breaker: For checking trading allowed
            ai_reasoning_callback: Optional function to generate AI reasoning
            min_score_to_trade: Minimum score required (default 50 = C tier)
            min_net_edge: Minimum net edge after costs (default 0.05%)
            max_cost_ratio: Maximum cost as % of edge (default 70%)
        """
        self.cost_engine = cost_engine
        self.scorer = scorer
        self.position_sizer = position_sizer
        self.heat_manager = heat_manager
        self.circuit_breaker = circuit_breaker
        self.ai_reasoning = ai_reasoning_callback

        self.min_score = min_score_to_trade
        self.min_net_edge = min_net_edge
        self.max_cost_ratio = max_cost_ratio

        # Statistics
        self.signals_generated = 0
        self.signals_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}

    def _trend_alignment_dict(self, trend_alignment: Optional[Any]) -> dict:
        """Build trend_alignment dict for scorer."""
        if trend_alignment is None:
            return {"alignment": "NONE"}
        if hasattr(trend_alignment, "alignment"):
            return {"alignment": trend_alignment.alignment.value}
        return {"alignment": getattr(trend_alignment, "value", "NONE")}

    def generate(
        self,
        opportunity: Opportunity,
        account_state: AccountState,
        market_state: MarketState,
        trend_alignment: Optional[Any] = None,
        volume_ratio: float = 1.0,
    ) -> GenerationResult:
        """
        Generate a complete signal from an opportunity.

        Args:
            opportunity: The opportunity to convert
            account_state: Current account state
            market_state: Current market state
            trend_alignment: Optional pre-calculated trend alignment (TrendAlignment or TrendAnalysis)
            volume_ratio: Current volume vs average

        Returns:
            GenerationResult with signal or rejection reason
        """
        checks = {}
        ta_dict = self._trend_alignment_dict(trend_alignment or market_state.trend_alignment)

        # 1. Check circuit breaker
        breaker_state = self.circuit_breaker.check(
            daily_pnl_pct=account_state.daily_pnl_pct,
            weekly_pnl_pct=account_state.weekly_pnl_pct,
            drawdown_pct=account_state.drawdown_pct,
        )
        checks["circuit_breaker"] = breaker_state.can_trade

        if not breaker_state.can_trade:
            return self._reject(f"Circuit breaker: {breaker_state.reason}", checks)

        # 2. Score the opportunity
        scored = self.scorer.score(
            opportunity=opportunity,
            trend_alignment=ta_dict,
            volume_ratio=volume_ratio,
            regime=market_state.regime,
            cost_analysis={"cost_ratio": 0},
        )
        checks["score_minimum"] = scored.score >= self.min_score

        if scored.score < self.min_score:
            return self._reject(f"Score {scored.score} below minimum {self.min_score}", checks)

        if getattr(scored.tier, "value", scored.tier) == "F":
            return self._reject("Tier F - do not trade", checks)

        # 3. Calculate costs
        broker = self.MARKET_BROKERS.get(opportunity.market, "ibkr")
        hold_days = self.EDGE_HOLD_TIMES.get(opportunity.primary_edge, 1.0)

        risk_pct = getattr(opportunity, "risk_percent", 1.0)
        estimated_position_value = (
            account_state.current_equity * (1.0 / risk_pct) if risk_pct > 0
            else account_state.current_equity * 0.1
        )

        costs = self.cost_engine.calculate_costs(
            symbol=opportunity.symbol,
            market=opportunity.market,
            broker=broker,
            position_value=estimated_position_value,
            hold_days=hold_days,
        )

        # 4. Check net edge viability
        gross_edge = self.EDGE_EXPECTED_RETURNS.get(opportunity.primary_edge, 0.15)
        score_factor = scored.score / 100
        adjusted_gross = gross_edge * (0.5 + 0.5 * score_factor)

        net_analysis = self.cost_engine.calculate_net_edge(adjusted_gross, costs)
        checks["net_edge_viable"] = net_analysis["viable"]
        checks["cost_ratio_ok"] = net_analysis["cost_ratio"] <= self.max_cost_ratio

        if not net_analysis["viable"]:
            return self._reject(
                f"Net edge {net_analysis['net_edge']:.3f}% not viable (costs: {net_analysis['cost_ratio']:.1f}%)",
                checks,
            )

        if net_analysis["cost_ratio"] > self.max_cost_ratio:
            return self._reject(
                f"Cost ratio {net_analysis['cost_ratio']:.1f}% exceeds max {self.max_cost_ratio}%",
                checks,
            )

        # 5. Re-score with actual costs
        scored = self.scorer.score(
            opportunity=opportunity,
            trend_alignment=ta_dict,
            volume_ratio=volume_ratio,
            regime=market_state.regime,
            cost_analysis=net_analysis,
        )

        # 6. Calculate position size
        position = self.position_sizer.calculate(
            entry_price=opportunity.entry_price,
            stop_loss=opportunity.stop_loss,
            starting_balance=account_state.starting_balance,
            current_equity=account_state.current_equity,
            score=scored.score,
            regime=market_state.regime.value,
            current_heat=account_state.portfolio_heat,
            win_streak=account_state.win_streak,
        )
        checks["position_valid"] = position.units > 0

        if position.units <= 0:
            return self._reject(f"Position size zero: {position.cap_reason}", checks)

        # 7. Check heat capacity
        new_heat = position.risk_percent
        heat_analysis = self.heat_manager.analyze(
            equity=account_state.current_equity,
            daily_pnl_pct=account_state.daily_pnl_pct,
        )
        checks["heat_capacity"] = (heat_analysis.heat_percent + new_heat) <= heat_analysis.heat_limit

        if (heat_analysis.heat_percent + new_heat) > heat_analysis.heat_limit:
            return self._reject(
                f"Heat limit: current {heat_analysis.heat_percent:.1f}% + new {new_heat:.1f}% > limit {heat_analysis.heat_limit:.1f}%",
                checks,
            )

        # 8. Apply circuit breaker size multiplier
        final_units = position.units * breaker_state.size_multiplier
        final_risk_amount = position.risk_amount * breaker_state.size_multiplier
        final_risk_pct = position.risk_percent * breaker_state.size_multiplier

        # 9. Generate AI reasoning (if callback provided)
        if self.ai_reasoning:
            try:
                ai_reasoning = self.ai_reasoning(opportunity, scored, market_state)
            except Exception as e:
                ai_reasoning = f"AI reasoning unavailable: {e}"
        else:
            ai_reasoning = self._generate_basic_reasoning(opportunity, scored, market_state, net_analysis)

        # 10. Identify risk factors
        risk_factors = self._identify_risk_factors(opportunity, market_state, scored)

        # 11. Calculate validity period
        validity_hours = self.EDGE_VALIDITY_HOURS.get(opportunity.primary_edge, 4)
        valid_until = datetime.now() + timedelta(hours=validity_hours)

        # 12. Determine session
        session = self._get_current_session()

        # 13. Build the signal
        signal = NexusSignal(
            signal_id=str(uuid4()),
            created_at=datetime.now(),
            opportunity_id=opportunity.id,
            symbol=opportunity.symbol,
            market=opportunity.market,
            direction=opportunity.direction,
            entry_price=opportunity.entry_price,
            stop_loss=opportunity.stop_loss,
            take_profit=opportunity.take_profit,
            position_size=final_units,
            position_value=final_units * opportunity.entry_price,
            risk_amount=final_risk_amount,
            risk_percent=final_risk_pct,
            primary_edge=opportunity.primary_edge,
            secondary_edges=opportunity.secondary_edges,
            edge_score=scored.score,
            tier=scored.tier,
            gross_expected=adjusted_gross,
            costs=costs,
            net_expected=net_analysis["net_edge"],
            cost_ratio=net_analysis["cost_ratio"],
            ai_reasoning=ai_reasoning,
            confluence_factors=scored.factors,
            risk_factors=risk_factors,
            market_context=market_state.summary,
            session=session,
            valid_until=valid_until,
            regime=market_state.regime,
            status=SignalStatus.PENDING,
        )

        self.signals_generated += 1
        return GenerationResult(success=True, signal=signal, checks_passed=checks)

    def _reject(self, reason: str, checks: Dict[str, bool]) -> GenerationResult:
        """Record a rejection."""
        self.signals_rejected += 1
        reason_key = reason.split(":")[0] if ":" in reason else reason[:30]
        self.rejection_reasons[reason_key] = self.rejection_reasons.get(reason_key, 0) + 1
        return GenerationResult(success=False, rejected_reason=reason, checks_passed=checks)

    def _generate_basic_reasoning(
        self,
        opportunity: Opportunity,
        scored: ScoredOpportunity,
        market_state: MarketState,
        net_analysis: Dict,
    ) -> str:
        """Generate basic reasoning without AI."""
        edge_val = getattr(opportunity.primary_edge, "value", opportunity.primary_edge) or str(opportunity.primary_edge)
        reasoning = f"{str(edge_val).replace('_', ' ').title()} setup detected on {opportunity.symbol}. "
        reasoning += f"Score: {scored.score}/100 (Tier {scored.tier.value}). "
        if scored.factors:
            reasoning += f"Key factors: {', '.join(scored.factors[:3])}. "
        reasoning += f"Market regime: {market_state.regime.value}. "
        reasoning += f"Net edge after costs: {net_analysis['net_edge']:.2f}%. "
        rr = opportunity.risk_reward_ratio
        reasoning += f"Risk/reward: {rr:.1f}:1."
        return reasoning

    def _identify_risk_factors(
        self,
        opportunity: Opportunity,
        market_state: MarketState,
        scored: ScoredOpportunity,
    ) -> List[str]:
        """Identify risk factors for the trade."""
        risks = []
        if market_state.regime == MarketRegime.VOLATILE:
            risks.append("Volatile market regime - increased uncertainty")
        trend_align = getattr(market_state.trend_alignment, "alignment", market_state.trend_alignment)
        if trend_align and getattr(trend_align, "value", None) == "conflicting":
            risks.append("Timeframes showing conflicting signals")
        if market_state.vix_level > 30:
            risks.append(f"High VIX ({market_state.vix_level}) - extreme volatility")
        elif market_state.vix_level > 25:
            risks.append(f"Elevated VIX ({market_state.vix_level}) - expect larger moves")
        if market_state.upcoming_events:
            risks.append(f"Upcoming events: {', '.join(market_state.upcoming_events[:2])}")
        if scored.score < 60:
            risks.append(f"Lower conviction score ({scored.score})")
        if opportunity.risk_reward_ratio < 1.5:
            risks.append(f"Below-optimal R:R ({opportunity.risk_reward_ratio:.1f}:1)")
        return risks

    def _get_current_session(self) -> str:
        """Determine current trading session (UK time based)."""
        hour = datetime.now().hour
        if 0 <= hour < 7:
            return "asia_overnight"
        if 7 <= hour < 8:
            return "london_open"
        if 8 <= hour < 14:
            return "european"
        if 14 <= hour < 15:
            return "us_premarket"
        if 15 <= hour < 16:
            return "us_open"
        if 16 <= hour < 20:
            return "us_regular"
        if 20 <= hour < 21:
            return "power_hour"
        return "us_close"

    def get_statistics(self) -> Dict:
        """Get generation statistics."""
        total = self.signals_generated + self.signals_rejected
        acceptance_rate = (self.signals_generated / total * 100) if total > 0 else 0
        return {
            "signals_generated": self.signals_generated,
            "signals_rejected": self.signals_rejected,
            "acceptance_rate": round(acceptance_rate, 1),
            "rejection_breakdown": self.rejection_reasons,
        }

    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self.signals_generated = 0
        self.signals_rejected = 0
        self.rejection_reasons = {}


if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS SIGNAL GENERATOR TEST")
    print("=" * 60)

    cost_engine = CostEngine()
    scorer = OpportunityScorer()
    position_sizer = DynamicPositionSizer(mode=RiskMode.STANDARD)
    heat_manager = DynamicHeatManager()
    circuit_breaker = SmartCircuitBreaker()

    generator = SignalGenerator(
        cost_engine=cost_engine,
        scorer=scorer,
        position_sizer=position_sizer,
        heat_manager=heat_manager,
        circuit_breaker=circuit_breaker,
        min_score_to_trade=50,
    )

    print("\n--- Test 1: High Quality Opportunity ---")
    opportunity = Opportunity(
        id="test-001",
        detected_at=datetime.now(),
        scanner="InsiderClusterScanner",
        symbol="AAPL",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=162.0,
        primary_edge=EdgeType.INSIDER_CLUSTER,
        secondary_edges=[EdgeType.RSI_EXTREME],
        edge_data={"insider_count": 4, "total_value": 2500000},
    )

    account_state = AccountState(
        starting_balance=10000.0,
        current_equity=10500.0,
        daily_pnl=500.0,
        daily_pnl_pct=5.0,
        weekly_pnl_pct=8.0,
        drawdown_pct=-2.0,
        win_streak=3,
        portfolio_heat=10.0,
    )

    market_state = MarketState(
        regime=MarketRegime.TRENDING_UP,
        trend_alignment=None,
        vix_level=18.0,
        session="us_regular",
    )

    result = generator.generate(
        opportunity=opportunity,
        account_state=account_state,
        market_state=market_state,
        volume_ratio=1.8,
    )

    print(f"Success: {result.success}")
    if result.success and result.signal:
        sig = result.signal
        print(f"Signal ID: {sig.signal_id[:8]}...")
        print(f"Symbol: {sig.symbol} {(getattr(sig.direction, 'value', sig.direction) or sig.direction).upper()}")
        print(f"Entry: ${sig.entry_price} | Stop: ${sig.stop_loss} | Target: ${sig.take_profit}")
        print(f"Position: {sig.position_size:.2f} units (${sig.position_value:.2f})")
        print(f"Risk: ${sig.risk_amount:.2f} ({sig.risk_percent:.2f}%)")
        print(f"Score: {sig.edge_score}/100 (Tier {getattr(sig.tier, 'value', sig.tier)})")
        print(f"Net Edge: {sig.net_expected:.3f}% (Cost ratio: {sig.cost_ratio:.1f}%)")
        print(f"R:R: {sig.risk_reward_ratio:.1f}:1")
        print(f"Reasoning: {sig.ai_reasoning[:100]}...")
    else:
        print(f"Rejected: {result.rejected_reason}")
    print(f"Checks: {result.checks_passed}")

    print("\n--- Test 2: Low Score Opportunity ---")
    low_opp = Opportunity(
        id="test-002",
        detected_at=datetime.now(),
        scanner="TestScanner",
        symbol="XYZ",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=50.0,
        stop_loss=48.0,
        take_profit=52.0,
        primary_edge=EdgeType.LONDON_OPEN,
        secondary_edges=[],
        edge_data={},
    )
    bad_market = MarketState(
        regime=MarketRegime.VOLATILE,
        trend_alignment=None,
        vix_level=35.0,
    )
    result = generator.generate(
        opportunity=low_opp,
        account_state=account_state,
        market_state=bad_market,
        volume_ratio=0.7,
    )
    print(f"Success: {result.success}")
    print(f"Rejected: {result.rejected_reason}")
    print(f"Checks: {result.checks_passed}")

    print("\n--- Test 3: Circuit Breaker Active ---")
    bad_account = AccountState(
        starting_balance=10000.0,
        current_equity=9600.0,
        daily_pnl=-400.0,
        daily_pnl_pct=-4.0,
        weekly_pnl_pct=-5.0,
        drawdown_pct=-6.0,
        portfolio_heat=15.0,
    )
    result = generator.generate(
        opportunity=opportunity,
        account_state=bad_account,
        market_state=market_state,
        volume_ratio=1.5,
    )
    print(f"Success: {result.success}")
    print(f"Rejected: {result.rejected_reason}")

    print("\n--- Test 4: Near Heat Limit ---")
    high_heat_account = AccountState(
        starting_balance=10000.0,
        current_equity=10000.0,
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        weekly_pnl_pct=0.0,
        drawdown_pct=-3.0,
        portfolio_heat=24.0,
    )
    result = generator.generate(
        opportunity=opportunity,
        account_state=high_heat_account,
        market_state=market_state,
        volume_ratio=1.5,
    )
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Rejected: {result.rejected_reason}")
    else:
        print("Signal generated with reduced size due to heat")

    print("\n--- Test 5: Generation Statistics ---")
    stats = generator.get_statistics()
    print(f"Generated: {stats['signals_generated']}")
    print(f"Rejected: {stats['signals_rejected']}")
    print(f"Acceptance rate: {stats['acceptance_rate']}%")
    print(f"Rejection breakdown: {stats['rejection_breakdown']}")

    print("\n" + "=" * 60)
    print("SIGNAL GENERATOR TEST COMPLETE [OK]")
    print("=" * 60)
