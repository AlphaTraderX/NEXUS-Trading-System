"""
NEXUS Trading System - Main Entry Point
The brain that orchestrates everything.

THIS IS IT. The culmination of 10 weeks of work.

FLOW:
1. Initialize all components
2. Connect to data providers
3. Run scan cycles
4. Generate signals
5. Manage positions
6. Deliver alerts
7. Track performance
8. Protect capital (circuit breakers, kill switch)
"""

import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.enums import Direction, EdgeType, Market
from core.models import Opportunity, NexusSignal

# Intelligence Layer
from intelligence.cost_engine import CostEngine
from intelligence.scorer import OpportunityScorer, SignalTier
from intelligence.regime import RegimeDetector, MarketRegime
from intelligence.trend_filter import TrendFilter
from intelligence.reasoning import ReasoningEngine

# Risk Layer
from risk.position_sizer import DynamicPositionSizer, RiskMode
from risk.heat_manager import DynamicHeatManager
from risk.circuit_breaker import SmartCircuitBreaker
from risk.kill_switch import KillSwitch, SystemState

# Execution Layer
from execution.signal_generator import SignalGenerator, AccountState, MarketState
from execution.position_manager import PositionManager, ExitReason

# Delivery Layer
from delivery.discord import DiscordDelivery
from delivery.telegram import TelegramDelivery


@dataclass
class NexusConfig:
    """NEXUS system configuration."""

    # Mode
    mode: str = "paper"  # paper, live
    risk_mode: str = "standard"  # conservative, standard, aggressive, maximum

    # Account
    starting_balance: float = 10000.0
    currency: str = "GBP"

    # Risk
    max_positions: int = 8
    max_per_market: int = 3
    max_heat_pct: float = 25.0

    # Circuit breakers
    daily_loss_warning: float = -1.5
    daily_loss_reduce: float = -2.0
    daily_loss_stop: float = -3.0
    weekly_loss_stop: float = -6.0
    max_drawdown: float = -10.0

    # Scoring
    min_score_to_trade: int = 50

    # Scan settings
    scan_interval_seconds: int = 60

    # Alerts
    discord_webhook: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # AI
    groq_api_key: Optional[str] = None


class NEXUS:
    """
    NEXUS Trading System

    The brain that orchestrates all components.
    """

    VERSION = "2.1.0"

    def __init__(self, config: NexusConfig = None):
        """Initialize NEXUS system."""

        self.config = config or NexusConfig()
        self.running = False
        self.started_at: Optional[datetime] = None

        # State
        self.current_equity = self.config.starting_balance
        self.daily_starting_equity = self.config.starting_balance
        self.weekly_starting_equity = self.config.starting_balance
        self.peak_equity = self.config.starting_balance

        # Initialize components
        self._init_intelligence()
        self._init_risk()
        self._init_execution()
        self._init_delivery()

        # Statistics
        self.cycles_run = 0
        self.signals_generated = 0
        self.signals_sent = 0

        print(f"[NEXUS v{self.VERSION}] Initialized")
        print(f"  Mode: {self.config.mode}")
        print(f"  Risk: {self.config.risk_mode}")
        print(f"  Balance: {self.config.currency} {self.config.starting_balance:,.2f}")

    def _init_intelligence(self):
        """Initialize intelligence layer."""
        print("[NEXUS] Initializing Intelligence Layer...")

        self.cost_engine = CostEngine()
        self.scorer = OpportunityScorer()
        self.regime_detector = RegimeDetector()
        self.trend_filter = TrendFilter()
        self.reasoning = ReasoningEngine(api_key=self.config.groq_api_key)

        print(f"  - Cost Engine: OK")
        print(f"  - Scorer: OK")
        print(f"  - Regime Detector: OK")
        print(f"  - Trend Filter: OK")
        print(f"  - AI Reasoning: {'Groq' if self.reasoning.is_available else 'Fallback'}")

    def _init_risk(self):
        """Initialize risk layer."""
        print("[NEXUS] Initializing Risk Layer...")

        # Map risk mode
        risk_mode_map = {
            "conservative": RiskMode.CONSERVATIVE,
            "standard": RiskMode.STANDARD,
            "aggressive": RiskMode.AGGRESSIVE,
            "maximum": RiskMode.MAXIMUM,
        }
        risk_mode = risk_mode_map.get(self.config.risk_mode, RiskMode.STANDARD)

        self.position_sizer = DynamicPositionSizer(mode=risk_mode)

        self.heat_manager = DynamicHeatManager(
            base_heat_limit=self.config.max_heat_pct,
        )

        self.circuit_breaker = SmartCircuitBreaker(
            daily_warning=self.config.daily_loss_warning,
            daily_reduce=self.config.daily_loss_reduce,
            daily_stop=self.config.daily_loss_stop,
            weekly_stop=self.config.weekly_loss_stop,
            drawdown_stop=self.config.max_drawdown,
        )

        self.kill_switch = KillSwitch(
            daily_loss_threshold=self.config.daily_loss_stop,
            weekly_loss_threshold=self.config.weekly_loss_stop,
            max_drawdown_threshold=self.config.max_drawdown,
        )

        print(f"  - Position Sizer: {risk_mode.value}")
        print(f"  - Heat Manager: {self.config.max_heat_pct}% max")
        print(f"  - Circuit Breaker: OK")
        print(f"  - Kill Switch: Armed")

    def _init_execution(self):
        """Initialize execution layer."""
        print("[NEXUS] Initializing Execution Layer...")

        # Create AI reasoning callback (must return str for NexusSignal.ai_reasoning)
        def ai_callback(opp, scored, market):
            result = self.reasoning.generate(
                symbol=opp.symbol,
                direction=opp.direction,
                primary_edge=opp.primary_edge,
                secondary_edges=opp.secondary_edges,
                entry_price=opp.entry_price,
                stop_loss=opp.stop_loss,
                take_profit=opp.take_profit,
                score=scored.score,
                tier=scored.tier.value if hasattr(scored.tier, "value") else str(scored.tier),
                regime=market.regime,
                net_edge=0.15,  # Will be calculated
                cost_ratio=25.0,
                volume_ratio=1.0,
                trend_aligned=True,
            )
            return result.explanation if hasattr(result, "explanation") else str(result)

        self.signal_generator = SignalGenerator(
            cost_engine=self.cost_engine,
            scorer=self.scorer,
            position_sizer=self.position_sizer,
            heat_manager=self.heat_manager,
            circuit_breaker=self.circuit_breaker,
            min_score_to_trade=self.config.min_score_to_trade,
            ai_reasoning_callback=ai_callback,
        )

        self.position_manager = PositionManager()

        print(f"  - Signal Generator: OK")
        print(f"  - Position Manager: OK")

    def _init_delivery(self):
        """Initialize delivery layer."""
        print("[NEXUS] Initializing Delivery Layer...")

        self.discord = DiscordDelivery(
            webhook_url=self.config.discord_webhook,
        )

        self.telegram = TelegramDelivery(
            bot_token=self.config.telegram_token,
            chat_id=self.config.telegram_chat_id,
            quiet_hours=(22, 7),
        )

        print(f"  - Discord: {'Configured' if self.discord.is_configured else 'Not configured'}")
        print(f"  - Telegram: {'Configured' if self.telegram.is_configured else 'Not configured'}")

    def _get_account_state(self) -> AccountState:
        """Get current account state."""
        daily_pnl = self.current_equity - self.daily_starting_equity
        daily_pnl_pct = (daily_pnl / self.daily_starting_equity * 100) if self.daily_starting_equity > 0 else 0

        weekly_pnl = self.current_equity - self.weekly_starting_equity
        weekly_pnl_pct = (weekly_pnl / self.weekly_starting_equity * 100) if self.weekly_starting_equity > 0 else 0

        drawdown = self.current_equity - self.peak_equity
        drawdown_pct = (drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0

        # Get heat from position manager
        open_positions = self.position_manager.get_open_positions()
        total_risk = sum(p.risk_to_stop for p in open_positions)
        heat_pct = (total_risk / self.current_equity * 100) if self.current_equity > 0 else 0

        return AccountState(
            starting_balance=self.config.starting_balance,
            current_equity=self.current_equity,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
            win_streak=0,  # TODO: Track from position manager
            lose_streak=0,
            trades_today=self.position_manager.total_trades,
            portfolio_heat=heat_pct,
        )

    def _get_market_state(self) -> MarketState:
        """Get current market state."""
        # Detect regime (simplified - would use real data)
        regime = MarketRegime.RANGING

        return MarketState(
            regime=regime,
            trend_alignment={"alignment": "neutral"},
            vix_level=18.0,  # TODO: Get real VIX
            session=self._get_current_session(),
            upcoming_events=[],
        )

    def _get_current_session(self) -> str:
        """Determine current trading session."""
        now = datetime.now()
        hour = now.hour

        # UK time based sessions
        if 0 <= hour < 7:
            return "asia_overnight"
        elif 7 <= hour < 8:
            return "london_open"
        elif 8 <= hour < 12:
            return "european"
        elif 12 <= hour < 14:
            return "us_premarket"
        elif 14 <= hour < 15:
            return "us_open"
        elif 15 <= hour < 20:
            return "us_regular"
        elif 20 <= hour < 21:
            return "power_hour"
        else:
            return "us_close"

    def _get_system_state(self) -> SystemState:
        """Get system state for kill switch."""
        account = self._get_account_state()
        open_count = len(self.position_manager.get_open_positions())

        return SystemState(
            daily_pnl_pct=account.daily_pnl_pct,
            weekly_pnl_pct=account.weekly_pnl_pct,
            drawdown_pct=account.drawdown_pct,
            seconds_since_heartbeat=0,  # TODO: Track actual heartbeat
            data_age_seconds=0,
            broker_error_count=0,
            position_count=open_count,
            expected_position_count=open_count,  # TODO: Track expected
            last_update=datetime.now(),
        )

    async def process_opportunity(self, opportunity: Opportunity) -> Optional[NexusSignal]:
        """Process a single opportunity through the full pipeline."""

        account_state = self._get_account_state()
        market_state = self._get_market_state()

        # Get trend alignment
        trend = self.trend_filter.get_trend_alignment(
            daily_trend="bullish",
            h4_trend="bullish",
            h1_trend="neutral",
        )

        # Generate signal
        result = self.signal_generator.generate(
            opportunity=opportunity,
            account_state=account_state,
            market_state=market_state,
            trend_alignment=trend,
            volume_ratio=1.2,
        )

        if not result.success:
            print(f"  [REJECTED] {opportunity.symbol}: {result.rejected_reason}")
            return None

        signal = result.signal
        self.signals_generated += 1

        dir_str = getattr(signal.direction, "value", signal.direction) or signal.direction
        tier_str = getattr(signal.tier, "value", signal.tier) or signal.tier
        print(f"  [SIGNAL] {signal.symbol} {str(dir_str).upper()} @ ${signal.entry_price:.2f}")
        print(f"           Score: {signal.edge_score}/100 | Tier: {tier_str}")
        print(f"           Net Edge: +{signal.net_expected:.2f}% | R:R: {signal.risk_reward_ratio:.1f}:1")

        return signal

    async def deliver_signal(self, signal: NexusSignal):
        """Deliver signal via all configured channels."""

        # Discord
        if self.discord.is_configured:
            result = self.discord.send_signal(signal)
            if result.success:
                print(f"  [DISCORD] Sent")
            else:
                print(f"  [DISCORD] Failed: {result.error}")

        # Telegram
        if self.telegram.is_configured:
            result = self.telegram.send_signal(signal)
            if result.success:
                print(f"  [TELEGRAM] Sent")
            else:
                print(f"  [TELEGRAM] Failed: {result.error}")

        self.signals_sent += 1

    async def run_cycle(self):
        """Run one scan/signal cycle."""

        self.cycles_run += 1
        print(f"\n[CYCLE {self.cycles_run}] {datetime.now().strftime('%H:%M:%S')}")

        # 1. Check kill switch
        system_state = self._get_system_state()
        kill_status = self.kill_switch.check_conditions(system_state)

        if kill_status.is_active:
            print(f"  [KILL SWITCH] Active: {kill_status.reason}")
            return

        # 2. Check circuit breaker
        account_state = self._get_account_state()
        breaker_state = self.circuit_breaker.check(
            daily_pnl_pct=account_state.daily_pnl_pct,
            weekly_pnl_pct=account_state.weekly_pnl_pct,
            drawdown_pct=account_state.drawdown_pct,
        )

        if not breaker_state.can_trade:
            print(f"  [CIRCUIT BREAKER] {breaker_state.status.value}: {breaker_state.reason}")
            return

        # 3. Simulate finding opportunities (replace with real scanner)
        # In production, this would call the scanner orchestrator
        print(f"  Session: {self._get_current_session()}")
        print(f"  Equity: ${self.current_equity:,.2f}")
        print(f"  Daily P&L: {account_state.daily_pnl_pct:+.2f}%")
        print(f"  Positions: {len(self.position_manager.get_open_positions())}/{self.config.max_positions}")

        # 4. For demo, create a sample opportunity
        if self.cycles_run == 1:
            sample_opp = Opportunity(
                id="demo-001",
                detected_at=datetime.now(),
                scanner="demo",
                symbol="AAPL",
                market=Market.US_STOCKS,
                direction=Direction.LONG,
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=162.0,
                primary_edge=EdgeType.INSIDER_CLUSTER,
                secondary_edges=[EdgeType.RSI_EXTREME],
                edge_data={"insider_count": 4},
                raw_score=0,
                adjusted_score=0,
            )

            signal = await self.process_opportunity(sample_opp)

            if signal:
                await self.deliver_signal(signal)

    async def start(self):
        """Start the NEXUS system."""

        print("\n" + "=" * 60)
        print(f"  NEXUS v{self.VERSION} - STARTING")
        print("=" * 60)

        self.running = True
        self.started_at = datetime.now()

        # Arm kill switch
        self.kill_switch.arm()

        print(f"\n[NEXUS] System armed and running")
        print(f"[NEXUS] Press Ctrl+C to stop\n")

        try:
            while self.running:
                await self.run_cycle()

                # Wait for next cycle
                await asyncio.sleep(self.config.scan_interval_seconds)

        except asyncio.CancelledError:
            print("\n[NEXUS] Shutdown requested...")

        await self.stop()

    async def stop(self):
        """Stop the NEXUS system."""

        print("\n[NEXUS] Stopping...")
        self.running = False

        # Close all positions if live mode
        if self.config.mode == "live":
            open_positions = self.position_manager.get_open_positions()
            if open_positions:
                print(f"[NEXUS] Closing {len(open_positions)} open positions...")
                # Would close via broker here

        # Send summary
        runtime = datetime.now() - self.started_at if self.started_at else None

        print("\n" + "=" * 60)
        print("  NEXUS SESSION SUMMARY")
        print("=" * 60)
        print(f"  Runtime: {runtime}")
        print(f"  Cycles: {self.cycles_run}")
        print(f"  Signals Generated: {self.signals_generated}")
        print(f"  Signals Sent: {self.signals_sent}")

        summary = self.position_manager.get_portfolio_summary()
        print(f"  Trades: {summary['total_trades']}")
        print(f"  Win Rate: {summary['win_rate']:.1f}%")
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print("=" * 60)

        print("\n[NEXUS] Goodbye!")

    def get_status(self) -> Dict:
        """Get current system status."""
        account = self._get_account_state()

        return {
            "version": self.VERSION,
            "running": self.running,
            "mode": self.config.mode,
            "risk_mode": self.config.risk_mode,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "cycles_run": self.cycles_run,
            "signals_generated": self.signals_generated,
            "signals_sent": self.signals_sent,
            "equity": self.current_equity,
            "daily_pnl_pct": account.daily_pnl_pct,
            "positions": len(self.position_manager.get_open_positions()),
            "kill_switch": self.kill_switch.get_status(),
        }


def main():
    """Run NEXUS."""

    # Create config from environment or defaults
    config = NexusConfig(
        mode=os.getenv("NEXUS_MODE", "paper"),
        risk_mode=os.getenv("NEXUS_RISK_MODE", "standard"),
        starting_balance=float(os.getenv("NEXUS_BALANCE", "10000")),
        discord_webhook=os.getenv("DISCORD_WEBHOOK_URL"),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        scan_interval_seconds=5,  # Fast for demo
    )

    # Create NEXUS instance
    nexus = NEXUS(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[NEXUS] Interrupt received...")
        nexus.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    try:
        asyncio.run(nexus.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
