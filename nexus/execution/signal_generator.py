"""
Signal Generator - The Central Coordinator

Converts scored opportunities into complete NexusSignal objects by:
1. Running all risk checks (kill switch, circuit breaker, heat, correlation)
2. Calculating position size
3. Calculating costs
4. Checking trade viability
5. Creating the final signal

Think of this as the General Contractor - coordinates all specialists
into a final deliverable.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from nexus.core.models import (
    NexusSignal,
    Opportunity,
    ScoredOpportunity,
    SystemHealth,
)
from nexus.core.enums import (
    SignalStatus,
    SignalTier,
    Market,
    Direction,
    EdgeType,
    MarketRegime,
)
from nexus.intelligence.cost_engine import CostEngine, CostBreakdown
from nexus.execution.cooldown_manager import get_cooldown_manager
from nexus.risk.position_sizer import DynamicPositionSizer
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.circuit_breaker import SmartCircuitBreaker
from nexus.risk.kill_switch import KillSwitch
from nexus.risk.correlation import CorrelationMonitor
from nexus.storage.service import get_storage_service


logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any) -> Any:
    """Get attribute or dict key for both objects and dicts (e.g. mocks)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_edge(e: Any) -> EdgeType:
    """Normalize primary_edge to EdgeType (Pydantic may give str when use_enum_values=True)."""
    if isinstance(e, str):
        try:
            return EdgeType(e)
        except ValueError:
            return EdgeType.VWAP_DEVIATION
    return e


def _normalize_regime(r: Any) -> MarketRegime:
    """Normalize regime to MarketRegime."""
    if isinstance(r, str):
        try:
            return MarketRegime(r)
        except ValueError:
            return MarketRegime.RANGING
    return r


@dataclass
class AccountState:
    """Current account state for signal generation."""
    starting_balance: float
    current_equity: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float
    portfolio_heat: float
    win_streak: int = 0
    open_positions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.open_positions is None:
            self.open_positions = []


@dataclass
class MarketState:
    """Current market state for signal generation."""
    regime: MarketRegime
    vix: float = 20.0
    session: str = "regular"
    summary: str = ""

    @property
    def is_volatile(self) -> bool:
        regime_val = getattr(self.regime, "value", self.regime)
        return regime_val == MarketRegime.VOLATILE.value or regime_val == "volatile" or self.vix > 30


@dataclass
class SystemState:
    """System state for kill switch checks."""
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float
    seconds_since_heartbeat: int = 0
    data_age_seconds: int = 0
    broker_connected: bool = True


@dataclass
class RejectionReason:
    """Details about why a signal was rejected."""
    check_name: str
    reason: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SignalGenerator:
    """
    Converts scored opportunities into complete NexusSignal objects.

    This is the central coordinator that brings together:
    - Risk checks (kill switch, circuit breaker, heat, correlation)
    - Position sizing
    - Cost calculation
    - AI reasoning

    Only creates a signal if ALL checks pass.
    """

    # Expected edge by type (gross, before costs)
    EXPECTED_EDGES = {
        EdgeType.INSIDER_CLUSTER: 0.30,
        EdgeType.VWAP_DEVIATION: 0.15,
        EdgeType.TURN_OF_MONTH: 0.25,
        EdgeType.MONTH_END: 0.20,
        EdgeType.GAP_FILL: 0.15,
        EdgeType.RSI_EXTREME: 0.12,
        EdgeType.POWER_HOUR: 0.10,
        EdgeType.ASIAN_RANGE: 0.10,
        EdgeType.ORB: 0.15,
        EdgeType.BOLLINGER_TOUCH: 0.12,
        EdgeType.LONDON_OPEN: 0.12,
        EdgeType.NY_OPEN: 0.12,
        EdgeType.EARNINGS_DRIFT: 0.20,
        EdgeType.SENTIMENT_SPIKE: 0.15,
    }

    # Estimated hold time by edge type (in days)
    HOLD_TIMES = {
        EdgeType.INSIDER_CLUSTER: 5.0,
        EdgeType.VWAP_DEVIATION: 0.5,
        EdgeType.TURN_OF_MONTH: 4.0,
        EdgeType.MONTH_END: 2.0,
        EdgeType.GAP_FILL: 0.25,
        EdgeType.RSI_EXTREME: 1.0,
        EdgeType.POWER_HOUR: 0.1,
        EdgeType.ASIAN_RANGE: 0.5,
        EdgeType.ORB: 0.25,
        EdgeType.BOLLINGER_TOUCH: 1.0,
        EdgeType.LONDON_OPEN: 0.5,
        EdgeType.NY_OPEN: 0.5,
        EdgeType.EARNINGS_DRIFT: 5.0,
        EdgeType.SENTIMENT_SPIKE: 1.0,
    }

    # Signal validity duration by edge type (in hours)
    VALIDITY_HOURS = {
        EdgeType.INSIDER_CLUSTER: 24,
        EdgeType.VWAP_DEVIATION: 2,
        EdgeType.TURN_OF_MONTH: 8,
        EdgeType.MONTH_END: 8,
        EdgeType.GAP_FILL: 1,
        EdgeType.RSI_EXTREME: 4,
        EdgeType.POWER_HOUR: 1,
        EdgeType.ASIAN_RANGE: 2,
        EdgeType.ORB: 1,
        EdgeType.BOLLINGER_TOUCH: 4,
        EdgeType.LONDON_OPEN: 2,
        EdgeType.NY_OPEN: 2,
        EdgeType.EARNINGS_DRIFT: 24,
        EdgeType.SENTIMENT_SPIKE: 4,
    }

    def __init__(
        self,
        cost_engine: CostEngine,
        position_sizer: DynamicPositionSizer,
        heat_manager: DynamicHeatManager,
        circuit_breaker: SmartCircuitBreaker,
        kill_switch: KillSwitch,
        correlation_monitor: CorrelationMonitor,
        min_net_edge: float = 0.05,  # Minimum net edge after costs
        max_cost_ratio: float = 70.0,  # Max costs as % of gross edge
    ):
        """
        Initialize the Signal Generator.

        Args:
            cost_engine: Calculates true trade costs
            position_sizer: Determines position size
            heat_manager: Tracks portfolio heat
            circuit_breaker: Checks loss limits
            kill_switch: Emergency shutdown
            correlation_monitor: Checks concentration
            min_net_edge: Minimum acceptable net edge (default 0.05%)
            max_cost_ratio: Max costs as percentage of gross edge
        """
        self.cost_engine = cost_engine
        self.position_sizer = position_sizer
        self.heat_manager = heat_manager
        self.circuit_breaker = circuit_breaker
        self.kill_switch = kill_switch
        self.correlation_monitor = correlation_monitor
        self.min_net_edge = min_net_edge
        self.max_cost_ratio = max_cost_ratio

        # Track rejection reasons for debugging
        self._last_rejection: Optional[RejectionReason] = None

    @property
    def last_rejection(self) -> Optional[RejectionReason]:
        """Get the reason for the last rejected signal."""
        return self._last_rejection

    async def generate_signal(
        self,
        scored_opp: ScoredOpportunity,
        account_state: AccountState,
        market_state: MarketState,
    ) -> Optional[NexusSignal]:
        """
        Generate a complete NexusSignal from a scored opportunity.

        Runs through all risk checks and only creates a signal if
        all checks pass. Returns None if any check fails.

        Args:
            scored_opp: The scored opportunity from Intelligence Layer
            account_state: Current account metrics
            market_state: Current market conditions

        Returns:
            NexusSignal if all checks pass, None otherwise
        """
        opp = scored_opp.opportunity
        self._last_rejection = None

        logger.info(
            "Generating signal symbol=%s edge=%s score=%s",
            opp.symbol,
            getattr(opp.primary_edge, "value", opp.primary_edge),
            scored_opp.score,
        )

        # Step 1: Kill Switch Check
        system_state = SystemState(
            daily_pnl_pct=account_state.daily_pnl_pct,
            weekly_pnl_pct=account_state.weekly_pnl_pct,
            drawdown_pct=account_state.drawdown_pct,
        )
        health = SystemHealth(
            last_heartbeat=None,
            last_data_update=None,
            seconds_since_heartbeat=float(system_state.seconds_since_heartbeat),
            seconds_since_data=float(system_state.data_age_seconds),
            drawdown_pct=system_state.drawdown_pct,
            is_connected=system_state.broker_connected,
            active_errors=[],
        )
        kill_state = self.kill_switch.check_conditions(health)
        if _get(kill_state, "is_triggered", False):
            msg = _get(kill_state, "message", "Kill switch triggered")
            self._reject("kill_switch", msg, kill_state.to_dict() if hasattr(kill_state, "to_dict") else {})
            return None

        # Normalize enums (Pydantic use_enum_values can yield strings)
        opp_market = opp.market if hasattr(opp.market, "value") else Market(opp.market) if isinstance(opp.market, str) else opp.market
        opp_direction = opp.direction if hasattr(opp.direction, "value") else Direction(opp.direction) if isinstance(opp.direction, str) else opp.direction
        primary_edge = _normalize_edge(opp.primary_edge)

        # Step 2: Circuit Breaker Check
        circuit_result = self.circuit_breaker.check_status(
            account_state.daily_pnl_pct,
            account_state.weekly_pnl_pct,
            account_state.drawdown_pct,
        )
        if not _get(circuit_result, "can_trade", True):
            reason = _get(circuit_result, "reason", None) or _get(circuit_result, "message", "Circuit breaker triggered")
            details = circuit_result.to_dict() if hasattr(circuit_result, "to_dict") else {}
            self._reject("circuit_breaker", reason, details)
            return None

        size_multiplier = _get(circuit_result, "size_multiplier", 1.0)

        # Step 3: Heat Capacity Check
        estimated_risk = self._estimate_risk_percent(scored_opp.score, account_state)
        sector = opp.edge_data.get("sector") if opp.edge_data else None
        heat_result = self.heat_manager.can_add_position(
            new_risk_pct=estimated_risk,
            market=opp_market,
            daily_pnl_pct=account_state.daily_pnl_pct,
            sector=sector,
        )
        if not _get(heat_result, "allowed", True):
            details = heat_result.to_dict() if hasattr(heat_result, "to_dict") else {}
            self._reject("heat_manager", "Insufficient heat capacity", details)
            return None

        # Step 4: Correlation Check
        correlation_result = self.correlation_monitor.check_new_position(
            symbol=opp.symbol,
            market=opp_market,
            direction=opp_direction,
            risk_pct=estimated_risk,
            sector=sector,
        )
        if not _get(correlation_result, "allowed", True):
            reasons = _get(correlation_result, "rejection_reasons", [])
            reason = "; ".join(reasons) if reasons else _get(correlation_result, "reason", "Correlation limit")
            details = correlation_result.to_dict() if hasattr(correlation_result, "to_dict") else {}
            self._reject("correlation", reason, details)
            return None

        # Step 4b: Economic Calendar / News Filter
        news_size_multiplier = 1.0
        try:
            from nexus.data.forex_factory import get_forex_factory_client

            ff = get_forex_factory_client()
            news_check = await ff.is_safe_to_trade(opp.symbol)

            if not news_check.get("safe", True):
                self._reject(
                    "news_filter",
                    f"High-impact news: {news_check.get('reason', 'upcoming event')}",
                    {"minutes_until": news_check.get("minutes_until")},
                )
                return None

            if news_check.get("reduce_size", False):
                news_size_multiplier = 0.5
                logger.info(
                    "News filter: reducing size 50%% for %s - %s",
                    opp.symbol,
                    news_check.get("reason", ""),
                )
        except Exception as e:
            logger.debug("News check failed (continuing): %s", e)

        # Step 5: Calculate Position Size
        position_result = self.position_sizer.calculate_size(
            starting_balance=account_state.starting_balance,
            current_equity=account_state.current_equity,
            entry_price=opp.entry_price,
            stop_loss=opp.stop_loss,
            score=scored_opp.score,
            current_heat=account_state.portfolio_heat,
            win_streak=account_state.win_streak,
            regime=_normalize_regime(market_state.regime),
            symbol=opp.symbol,
            market=opp_market,
        )

        if not _get(position_result, "can_trade", True):
            reason = _get(position_result, "rejection_reason", "Position size rejected")
            details = position_result.to_dict() if hasattr(position_result, "to_dict") else {}
            self._reject("position_size", reason, details)
            return None

        combined_multiplier = size_multiplier * news_size_multiplier
        risk_pct = _get(position_result, "risk_pct", 1.0) * combined_multiplier
        risk_amount = _get(position_result, "risk_amount", 0.0) * combined_multiplier
        stop_distance_pct = abs(opp.entry_price - opp.stop_loss) / opp.entry_price if opp.entry_price else 0
        position_value = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0

        if position_value <= 0:
            details = position_result.to_dict() if hasattr(position_result, "to_dict") else {}
            self._reject("position_size", "Position value calculated as zero", details)
            return None

        # Step 6: Calculate Costs
        broker = self._get_broker_for_market(opp_market)
        hold_days = self.HOLD_TIMES.get(primary_edge, 1.0)

        costs = self.cost_engine.calculate_costs(
            symbol=opp.symbol,
            market=opp_market,
            broker=broker,
            position_value=position_value,
            hold_days=hold_days,
        )

        # Step 7: Check Trade Viability
        gross_edge = self._get_expected_edge(primary_edge, scored_opp.score)
        net_analysis = self.cost_engine.calculate_net_edge(gross_edge, costs)

        if not net_analysis.get("viable", True):
            self._reject("cost_viability", "Trade not viable after costs", net_analysis)
            return None

        if net_analysis["net_edge"] < self.min_net_edge:
            self._reject(
                "min_edge",
                f"Net edge {net_analysis['net_edge']:.3f}% below minimum {self.min_net_edge}%",
                net_analysis,
            )
            return None

        if net_analysis["cost_ratio"] > self.max_cost_ratio:
            self._reject(
                "cost_ratio",
                f"Cost ratio {net_analysis['cost_ratio']:.1f}% exceeds maximum {self.max_cost_ratio}%",
                net_analysis,
            )
            return None

        # Step 8: Generate AI Reasoning (placeholder for Groq integration)
        reasoning = self._generate_reasoning(opp, scored_opp, market_state, net_analysis, primary_edge, opp_direction)

        # Step 9: Build the Signal
        tier = scored_opp.tier
        if hasattr(tier, "value"):
            tier = tier
        elif isinstance(tier, str) and tier.upper() in ("A", "B", "C", "D", "F"):
            tier = SignalTier(tier.upper())

        signal = NexusSignal(
            signal_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            opportunity_id=opp.id,
            symbol=opp.symbol,
            market=opp_market,
            direction=opp_direction,
            entry_price=opp.entry_price,
            stop_loss=opp.stop_loss,
            take_profit=opp.take_profit,
            position_size=self._calculate_units(opp, position_value),
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            primary_edge=primary_edge,
            secondary_edges=opp.secondary_edges,
            edge_score=scored_opp.score,
            tier=tier,
            gross_expected=gross_edge,
            costs=costs,
            net_expected=net_analysis["net_edge"],
            cost_ratio=net_analysis["cost_ratio"],
            ai_reasoning=reasoning["explanation"],
            confluence_factors=scored_opp.factors,
            risk_factors=self._identify_risks(opp, market_state, primary_edge),
            market_context=market_state.summary or self._generate_market_context(market_state),
            session=market_state.session,
            valid_until=self._calculate_validity(opp),
            status=SignalStatus.PENDING,
        )

        logger.info(
            "Signal generated successfully signal_id=%s symbol=%s score=%s tier=%s net_edge=%s",
            signal.signal_id,
            signal.symbol,
            signal.edge_score,
            getattr(signal.tier, "value", signal.tier),
            signal.net_expected,
        )

        # Record cooldown
        cooldown = get_cooldown_manager()
        cooldown.record_signal(signal.symbol, opp_direction, primary_edge)

        # Save signal to database
        try:
            storage = get_storage_service()
            if storage._initialized:
                await storage.save_signal(signal)
        except Exception as e:
            logger.warning("Failed to save signal to database: %s", e)

        return signal

    def _reject(self, check_name: str, reason: str, details: Dict[str, Any] = None):
        """Record rejection reason."""
        self._last_rejection = RejectionReason(
            check_name=check_name,
            reason=reason,
            details=details or {},
        )
        logger.info("Signal rejected check=%s reason=%s", check_name, reason)

    def _estimate_risk_percent(self, score: int, account_state: AccountState) -> float:
        """Estimate risk percentage for heat capacity check."""
        base_risk = 1.0
        if score >= 85:
            return base_risk * 1.5
        elif score >= 75:
            return base_risk * 1.25
        elif score >= 65:
            return base_risk
        elif score >= 50:
            return base_risk * 0.75
        else:
            return base_risk * 0.5

    def _get_broker_for_market(self, market: Market) -> str:
        """Get the appropriate broker for a market."""
        broker_map = {
            Market.US_STOCKS: "ibkr",
            Market.UK_STOCKS: "ig",
            Market.EU_STOCKS: "ig",
            Market.FOREX_MAJORS: "oanda",
            Market.FOREX_CROSSES: "oanda",
            Market.US_FUTURES: "ibkr",
            Market.COMMODITIES: "ibkr",
        }
        return broker_map.get(market, "ibkr")

    def _get_expected_edge(self, edge_type: EdgeType, score: int) -> float:
        """
        Get expected gross edge based on edge type and score.

        Higher scores indicate stronger setups, so we adjust the
        expected edge accordingly.
        """
        base_edge = self.EXPECTED_EDGES.get(edge_type, 0.10)

        # Score adjustment: 80+ score = full edge, lower = reduced
        if score >= 80:
            multiplier = 1.1
        elif score >= 70:
            multiplier = 1.0
        elif score >= 60:
            multiplier = 0.9
        elif score >= 50:
            multiplier = 0.8
        else:
            multiplier = 0.7

        return base_edge * multiplier

    def _calculate_units(self, opp: Opportunity, position_value: float) -> float:
        """Calculate position size in units (shares, lots, contracts)."""
        if opp.entry_price <= 0:
            return 0
        return position_value / opp.entry_price

    def _generate_reasoning(
        self,
        opp: Opportunity,
        scored_opp: ScoredOpportunity,
        market_state: MarketState,
        net_analysis: Dict[str, Any],
        primary_edge: EdgeType,
        opp_direction: Direction,
    ) -> Dict[str, str]:
        """
        Generate AI reasoning for the signal.

        This is a placeholder that generates template-based reasoning.
        Will be replaced with Groq LLM integration later.
        """
        direction_word = "buying" if opp_direction == Direction.LONG else "selling"
        regime_val = getattr(market_state.regime, "value", market_state.regime)
        regime_word = str(regime_val).replace("_", " ").lower()

        # Build reasoning from factors
        factors_text = "; ".join(scored_opp.factors[:3]) if scored_opp.factors else "multiple confirmations"
        edge_val = getattr(primary_edge, "value", primary_edge)

        explanation = (
            f"{str(edge_val).replace('_', ' ').title()} edge detected for {opp.symbol}. "
            f"Signal scored {scored_opp.score}/100 (Tier {getattr(scored_opp.tier, 'value', scored_opp.tier)}) based on {factors_text}. "
            f"Market regime is {regime_word}. "
            f"Risk/reward is {opp.risk_reward_ratio:.1f}:1. "
            f"Net edge after costs: {net_analysis['net_edge']:.2f}%."
        )

        return {"explanation": explanation}

    def _identify_risks(self, opp: Opportunity, market_state: MarketState, primary_edge: EdgeType) -> List[str]:
        """Identify risk factors for the signal."""
        risks = []

        # Regime risk
        if market_state.is_volatile:
            risks.append("Market is volatile - consider reduced position size")

        # VIX risk
        if market_state.vix > 25:
            risks.append(f"Elevated VIX ({market_state.vix:.1f}) indicates market stress")

        # R:R risk
        if opp.risk_reward_ratio < 1.5:
            risks.append(f"Risk/reward ratio ({opp.risk_reward_ratio:.1f}:1) is below optimal")

        # Edge-specific risks
        edge_risks_map = {
            EdgeType.GAP_FILL: "Gap may not fill if large or news-driven",
            EdgeType.ORB: "Opening range breakouts have high false breakout rate",
            EdgeType.BOLLINGER_TOUCH: "Mean reversion fails in trending markets",
            EdgeType.LONDON_OPEN: "Stop hunts common at London open",
            EdgeType.EARNINGS_DRIFT: "Earnings edge only works in small/mid caps",
        }

        if primary_edge in edge_risks_map:
            risks.append(edge_risks_map[primary_edge])

        return risks if risks else ["No specific risk factors identified"]

    def _generate_market_context(self, market_state: MarketState) -> str:
        """Generate market context summary."""
        regime_val = getattr(market_state.regime, "value", market_state.regime)
        return (
            f"Regime: {str(regime_val).replace('_', ' ').title()}, "
            f"VIX: {market_state.vix:.1f}, "
            f"Session: {market_state.session}"
        )

    def _calculate_validity(self, opp: Opportunity, primary_edge: Optional[EdgeType] = None) -> datetime:
        """Calculate when the signal expires."""
        edge = primary_edge if primary_edge is not None else _normalize_edge(opp.primary_edge)
        hours = self.VALIDITY_HOURS.get(edge, 4)
        return datetime.utcnow() + timedelta(hours=hours)
