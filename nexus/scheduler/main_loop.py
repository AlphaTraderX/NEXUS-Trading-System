"""
NEXUS Main Scheduler Loop

The foreman of the operation - tells scanners when to run.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from nexus.config.settings import settings
from nexus.core.models import Opportunity
from nexus.core.enums import SignalTier, Market, AlertPriority
from nexus.core.models import SystemHealth
from nexus.data.massive import MassiveProvider
from nexus.storage.service import get_storage_service
from nexus.data.oanda import OANDAProvider
from nexus.scanners.orchestrator import ScannerOrchestrator
from nexus.scheduler.market_hours import MarketHours
from nexus.intelligence.scorer import OpportunityScorer
from nexus.execution.signal_generator import SignalGenerator
from nexus.execution.order_manager import OrderManager
from nexus.execution.trade_executor import TradeExecutor, create_paper_executor
from nexus.delivery.alert_manager import AlertManager
from nexus.risk.circuit_breaker import SmartCircuitBreaker
from nexus.risk.kill_switch import KillSwitch, set_kill_switch
from nexus.risk.state_persistence import get_risk_persistence
from nexus.intelligence.cost_engine import CostEngine
from nexus.risk.position_sizer import DynamicPositionSizer
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.correlation import CorrelationMonitor
from nexus.execution.signal_generator import AccountState, MarketState
from nexus.core.enums import MarketRegime

# Conditional import for Windows compatibility
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")


class NexusScheduler:
    """
    Main NEXUS scheduler - runs continuously and triggers scans.
    
    Responsibilities:
    1. Connect to data providers when markets open
    2. Run scanners at appropriate intervals
    3. Handle market hours awareness
    4. Graceful shutdown on interrupt
    """
    
    def __init__(
        self,
        scan_interval: int = 60,  # seconds between scans
        verbose: bool = True,
    ):
        self.scan_interval = scan_interval
        self.verbose = verbose
        self.running = False
        self.settings = settings

        # Data providers (initialized on start)
        self.polygon: Optional[MassiveProvider] = None
        self.oanda: Optional[OANDAProvider] = None

        # Orchestrator (initialized with providers)
        self.orchestrator: Optional[ScannerOrchestrator] = None

        # Intelligence
        self.scorer = OpportunityScorer()

        # Risk Management
        self.circuit_breaker = SmartCircuitBreaker(self.settings)
        self.kill_switch = KillSwitch(self.settings)
        set_kill_switch(self.kill_switch)

        # Check persisted safety state on startup
        persistence = get_risk_persistence()
        allowed, reason = persistence.is_trading_allowed()
        if not allowed:
            logger.critical("🚨 TRADING BLOCKED ON STARTUP: %s", reason)
            logger.critical(
                "Manual reset required via: persistence.manual_reset('I_CONFIRM_RESET_AFTER_REVIEWING_LOSSES')"
            )
            self._startup_blocked = True
        else:
            self._startup_blocked = False
            logger.info("✅ Safety state check passed - trading allowed")

        # Cost / sizing / heat / correlation (for SignalGenerator)
        self.cost_engine = CostEngine()
        self.position_sizer = DynamicPositionSizer(self.settings)
        self.heat_manager = DynamicHeatManager(self.settings)
        self.correlation_monitor = CorrelationMonitor(self.settings)

        # Execution (paper trading mode)
        self.order_manager = OrderManager()
        self.trade_executor = TradeExecutor(order_manager=self.order_manager)
        # Register paper broker so execute_order can run
        paper_broker = create_paper_executor(
            account_balance=getattr(self.settings, "starting_balance", 10000.0),
        )
        self.trade_executor.register_broker("paper", paper_broker, markets=[Market.US_STOCKS, Market.FOREX], set_default=True)

        # Signal Generation
        self.signal_generator = SignalGenerator(
            cost_engine=self.cost_engine,
            position_sizer=self.position_sizer,
            heat_manager=self.heat_manager,
            circuit_breaker=self.circuit_breaker,
            kill_switch=self.kill_switch,
            correlation_monitor=self.correlation_monitor,
        )

        # Delivery
        self.alert_manager = AlertManager()

        # Stats
        self.scan_count = 0
        self.opportunities_found = 0
        self.start_time: Optional[datetime] = None
    
    async def initialize_providers(self) -> bool:
        """Initialize and connect to data providers."""
        try:
            # Initialize Polygon/Massive for US stocks
            self.polygon = MassiveProvider(api_key=settings.polygon_api_key)
            await self.polygon.connect()
            self._log("Connected to Polygon (US Stocks)")
            
            # Initialize OANDA for forex
            self.oanda = OANDAProvider(
                api_key=settings.oanda_api_key,
                account_id=settings.oanda_account_id,
                practice=True  # Use demo account
            )
            await self.oanda.connect()
            self._log("Connected to OANDA (Forex)")
            
            # Initialize orchestrator with providers
            self.orchestrator = ScannerOrchestrator(
                stock_provider=self.polygon,
                forex_provider=self.oanda,
            )
            self._log("Scanner orchestrator initialized")
            
            return True
            
        except Exception as e:
            self._log(f"Failed to initialize providers: {e}", level="error")
            return False
    
    async def shutdown_providers(self):
        """Gracefully disconnect from data providers."""
        try:
            if self.polygon:
                await self.polygon.disconnect()
                self._log("Disconnected from Polygon")
            
            if self.oanda:
                await self.oanda.disconnect()
                self._log("Disconnected from OANDA")
                
        except Exception as e:
            self._log(f"Error during shutdown: {e}", level="error")
    
    async def run_scan_cycle(self) -> None:
        """Run one complete scan → score → signal → execute → alert cycle."""
        # Check if blocked on startup
        if getattr(self, "_startup_blocked", False):
            logger.warning("Trading blocked - manual reset required")
            return

        cycle_start = datetime.now(timezone.utc)
        logger.info(f"{'='*60}")
        logger.info(f"SCAN CYCLE START: {cycle_start.strftime('%H:%M:%S')} UTC")

        try:
            # Update heartbeat to show scheduler is alive
            self.kill_switch.update_heartbeat()

            # Step 1: Check Kill Switch conditions (use real staleness from kill switch)
            health = self.kill_switch.get_system_health()
            kill_status = self.kill_switch.check_conditions(health)

            if kill_status.is_triggered:
                logger.critical(f"KILL SWITCH: {kill_status.message}")
                await self.alert_manager.send_alert(
                    f"KILL SWITCH ACTIVATED: {kill_status.message}",
                    priority=AlertPriority.CRITICAL,
                )
                return

            # Step 2: Check Circuit Breaker
            account_state_dict = await self._get_account_state()
            circuit_status = self.circuit_breaker.check_status(
                daily_pnl_pct=account_state_dict["daily_pnl_pct"],
                weekly_pnl_pct=account_state_dict["weekly_pnl_pct"],
                drawdown_pct=account_state_dict["drawdown_pct"],
            )

            if not circuit_status.can_trade:
                logger.warning(f"Circuit breaker: {circuit_status.message}")
                return

            # Market hours: skip scanners if no market open
            status = MarketHours.get_market_status()
            any_market_open = any(status["markets"].values())
            if not any_market_open:
                logger.info("Markets closed - skipping scanners")
                return

            # Step 3: Run scanners to find opportunities
            opportunities = await self.orchestrator.run_scan_cycle()
            logger.info(f"Found {len(opportunities)} raw opportunities")

            if not opportunities:
                logger.info("No opportunities this cycle")
                return

            self.opportunities_found += len(opportunities)

            # Step 4: Score each opportunity
            scored_opportunities = []
            for opp in opportunities:
                try:
                    trend_alignment = {"alignment": "NEUTRAL"}
                    volume_ratio = opp.edge_data.get("volume_ratio", 1.0) if getattr(opp, "edge_data", None) else 1.0
                    regime = getattr(self, "regime_detector", None)
                    regime_val = regime.current_regime if regime and hasattr(regime, "current_regime") else None
                    if regime_val is None:
                        regime_val = MarketRegime.RANGING

                    scored = self.scorer.score(
                        opportunity=opp,
                        trend_alignment=trend_alignment,
                        volume_ratio=volume_ratio,
                        regime=regime_val,
                        cost_analysis={"cost_ratio": 25},
                    )

                    # CRITICAL: Reject F-tier signals
                    tier_val = getattr(scored.tier, "value", scored.tier)
                    if tier_val == SignalTier.F.value or tier_val == "F":
                        logger.debug(f"Rejecting F-tier: {opp.symbol} (score: {scored.score})")
                        continue

                    scored_opportunities.append(scored)
                    logger.info(f"Scored {opp.symbol}: {scored.score}/100 (Tier {tier_val})")

                except Exception as e:
                    logger.error(f"Error scoring {opp.symbol}: {e}")
                    continue

            logger.info(f"{len(scored_opportunities)} opportunities passed scoring")

            if not scored_opportunities:
                return

            # Build AccountState and MarketState for signal generator
            account_state = AccountState(
                starting_balance=account_state_dict["starting_balance"],
                current_equity=account_state_dict["current_equity"],
                daily_pnl=0.0,
                daily_pnl_pct=account_state_dict["daily_pnl_pct"],
                weekly_pnl_pct=account_state_dict["weekly_pnl_pct"],
                drawdown_pct=account_state_dict["drawdown_pct"],
                portfolio_heat=0.0,
                win_streak=0,
                open_positions=[],
            )
            market_state = MarketState(regime=MarketRegime.RANGING)

            # Step 5: Generate signals for qualifying opportunities
            signals_generated = []
            for scored_opp in scored_opportunities:
                try:
                    signal = await self.signal_generator.generate_signal(
                        scored_opp=scored_opp,
                        account_state=account_state,
                        market_state=market_state,
                    )

                    if signal:
                        signals_generated.append(signal)
                        logger.info(f"SIGNAL: {signal.direction.value} {signal.symbol} @ {signal.entry_price}")

                except Exception as e:
                    logger.error(f"Error generating signal for {scored_opp.opportunity.symbol}: {e}")
                    continue

            logger.info(f"Generated {len(signals_generated)} signals")

            # Step 6: Execute signals (paper trading)
            for signal in signals_generated:
                try:
                    if self.settings.paper_trading:
                        result = await self.trade_executor.execute_signal(signal)

                        if result and result.get("success"):
                            logger.info(f"Order submitted: {signal.symbol}")
                            await self.alert_manager.send_signal_alert(signal)
                        else:
                            logger.warning(f"Order failed: {signal.symbol} - {result.get('error', 'Unknown')}")
                    else:
                        logger.info(f"LIVE MODE: Would execute {signal.symbol} (not implemented)")

                except Exception as e:
                    logger.error(f"Error executing signal {signal.symbol}: {e}")
                    continue

            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            logger.info(
                f"CYCLE COMPLETE: {elapsed:.2f}s | {len(opportunities)} scanned -> {len(scored_opportunities)} scored -> {len(signals_generated)} signals"
            )
            logger.info(f"{'='*60}")

        except Exception as e:
            logger.error(f"Scan cycle error: {e}", exc_info=True)

    async def _get_system_state(self) -> dict:
        """Get current system state for kill switch checks."""
        health = self.kill_switch.get_system_health()
        return {
            "daily_pnl_pct": 0.0,  # TODO: Get from position manager
            "weekly_pnl_pct": 0.0,
            "drawdown_pct": 0.0,
            "seconds_since_heartbeat": health.seconds_since_heartbeat,
            "data_age_seconds": health.seconds_since_data,
        }

    async def _get_account_state(self) -> dict:
        """Get current account state for circuit breaker."""
        balance = getattr(self.settings, "starting_balance", 10000.0)
        return {
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "drawdown_pct": 0.0,
            "current_equity": balance,
            "starting_balance": balance,
        }
    
    async def start(self):
        """
        Start the scheduler main loop.
        
        Runs continuously until interrupted.
        """
        self.running = True
        self.start_time = datetime.now(UTC)
        
        self._print_banner()

        # Initialize storage (database)
        storage = get_storage_service()
        if await storage.initialize():
            self._log("Database connected")
            state = await storage.get_system_state()
            if state is None:
                starting = getattr(settings, "starting_balance", 10000.0)
                await storage.initialize_system_state(starting)
                self._log(f"System state initialized with {starting:,.0f}")
        else:
            self._log("Database not connected - running in memory-only mode", level="warn")

        # Initialize providers
        if not await self.initialize_providers():
            self._log("Failed to initialize. Exiting.", level="error")
            return
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self._log(f"Scheduler started. Scanning every {self.scan_interval} seconds.")
        
        # Main loop
        try:
            while self.running:
                self.scan_count += 1
                
                # Run scan cycle
                await self.run_scan_cycle()
                
                # Wait for next cycle
                if self.running:
                    await asyncio.sleep(self.scan_interval)
                    
        except asyncio.CancelledError:
            self._log("Scheduler cancelled")
        except KeyboardInterrupt:
            self._log("Keyboard interrupt received")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        self._log("Shutting down scheduler...")
        self.running = False
        
        await self.shutdown_providers()
        
        self._print_summary()
    
    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            self._log(f"Received signal {signum}")
            self.running = False
        
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, handle_signal)
        
        # Handle termination (not available on Windows)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, handle_signal)
    
    def _print_banner(self):
        """Print startup banner."""
        banner = """
============================================================
    _   _________  ____  _______
   / | / / ____/ |/ / / / / ___/
  /  |/ / __/  |   / / / /\\__ \\
 / /|  / /___ /   / /_/ /___/ /
/_/ |_/_____//_/|_\\____//____/

        AUTOMATED TRADING SYSTEM
============================================================
"""
        print(banner)
        print(f"  Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Mode: {'PAPER' if settings.paper_trading else 'LIVE'}")
        print(f"  Scan Interval: {self.scan_interval}s")
        print("============================================================\n")
    
    def _print_summary(self):
        """Print session summary on shutdown."""
        if not self.start_time:
            return
        
        runtime = datetime.now(UTC) - self.start_time
        hours = runtime.total_seconds() / 3600
        
        print("\n============================================================")
        print("                    SESSION SUMMARY")
        print("============================================================")
        print(f"  Runtime: {runtime}")
        print(f"  Scans completed: {self.scan_count}")
        print(f"  Opportunities found: {self.opportunities_found}")
        print(f"  Avg per hour: {self.opportunities_found / max(hours, 0.01):.1f}")
        print("============================================================\n")
    
    def _log_market_status(self, status: dict):
        """Log current market status."""
        markets = status["markets"]
        sessions = status["active_sessions"]
        
        market_str = " | ".join([
            f"Forex: {'OPEN' if markets['forex'] else 'CLOSED'}",
            f"US: {'OPEN' if markets['us_stocks'] else 'CLOSED'}",
            f"UK: {'OPEN' if markets['uk_stocks'] else 'CLOSED'}",
        ])
        
        session_str = ", ".join(sessions) if sessions else "None"
        
        self._log(f"Markets: {market_str}")
        self._log(f"Active sessions: {session_str}")
    
    def _log_opportunities(self, opportunities: list[Opportunity]):
        """Log found opportunities."""
        print("\n" + "=" * 60)
        print(f"  OPPORTUNITIES FOUND: {len(opportunities)}")
        print("=" * 60)
        
        for opp in opportunities:
            direction = "LONG" if opp.direction.value == "long" else "SHORT"
            print(f"\n  {opp.primary_edge.value} | {opp.symbol}")
            print(f"  Direction: {direction}")
            print(f"  Entry: ${opp.entry_price:.2f}")
            print(f"  Stop: ${opp.stop_loss:.2f}")
            print(f"  Target: ${opp.take_profit:.2f}")
            print(f"  R:R: {opp.risk_reward_ratio:.2f}")
        
        print("\n" + "=" * 60 + "\n")
    
    def _log(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        prefix = {
            "info": "INFO ",
            "error": "ERROR",
            "warn": "WARN ",
        }.get(level, "INFO ")
        
        if self.verbose:
            print(f"{timestamp} | {prefix} | {message}")


async def run_scheduler(scan_interval: int = 60):
    """Convenience function to run the scheduler."""
    scheduler = NexusScheduler(scan_interval=scan_interval)
    await scheduler.start()


# Allow running directly
if __name__ == "__main__":
    asyncio.run(run_scheduler())
