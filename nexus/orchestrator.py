"""
NEXUS Main Orchestrator

The master control loop that coordinates all NEXUS components
for fully automated trading.

This is the "brain" that:
1. Runs continuous health checks
2. Monitors market regime
3. Scans for opportunities
4. Generates and validates signals
5. Executes trades with risk management
6. Monitors positions
7. Handles emergencies
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from nexus.config.settings import settings
from nexus.core.enums import AlertPriority, MarketRegime
from nexus.core.models import SystemHealth
from nexus.storage.service import get_storage_service, StorageService

# Scanners
from nexus.scanners.orchestrator import ScannerOrchestrator

# Intelligence
from nexus.intelligence.regime import RegimeDetector
from nexus.intelligence.regime_monitor import ContinuousRegimeMonitor
from nexus.intelligence.cost_engine import CostEngine
from nexus.intelligence.scorer import OpportunityScorer

# Risk
from nexus.risk.position_sizer import DynamicPositionSizer
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.circuit_breaker import SmartCircuitBreaker
from nexus.risk.kill_switch import KillSwitch
from nexus.risk.correlation import CorrelationMonitor
from nexus.risk.hardware_stops import HardwareStopManager

# Execution
from nexus.execution.signal_generator import (
    SignalGenerator,
    AccountState,
    MarketState,
)
from nexus.execution.trade_executor import TradeExecutor
from nexus.execution.position_manager import PositionManager
from nexus.execution.order_manager import OrderManager
from nexus.execution.reconciliation import ReconciliationEngine

# Delivery
from nexus.delivery.alert_manager import AlertManager

# Monitoring
from nexus.monitoring.health import HealthChecker
from nexus.monitoring.metrics import MetricsCollector
from nexus.monitoring.alerts import raise_alert

logger = logging.getLogger(__name__)


class NEXUSOrchestrator:
    """
    Main orchestrator for automated NEXUS trading.

    Usage:
        orchestrator = NEXUSOrchestrator()
        await orchestrator.initialize()
        await orchestrator.run()  # Runs until stopped
    """

    def __init__(self):
        self._running = False
        self._initialized = False

        # Core cycle timing
        self._scan_interval = 60  # Seconds between scans
        self._health_interval = 300  # Seconds between health checks
        self._reconcile_interval = 300  # Seconds between broker reconciliation

        # Components (initialized in initialize())
        self.storage: Optional[StorageService] = None
        self.scanner_orchestrator: Optional[ScannerOrchestrator] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.regime_monitor: Optional[ContinuousRegimeMonitor] = None
        self.cost_engine: Optional[CostEngine] = None
        self.scorer: Optional[OpportunityScorer] = None
        self.position_sizer: Optional[DynamicPositionSizer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.heat_manager: Optional[DynamicHeatManager] = None
        self.circuit_breaker: Optional[SmartCircuitBreaker] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.correlation_monitor: Optional[CorrelationMonitor] = None
        self.hardware_stops: Optional[HardwareStopManager] = None
        self.alert_manager: Optional[AlertManager] = None
        self.health_checker: Optional[HealthChecker] = None
        self.metrics: Optional[MetricsCollector] = None
        self.reconciliation: Optional[ReconciliationEngine] = None

        # Brokers
        self._brokers: Dict[str, Any] = {}
        self._data_providers: Dict[str, Any] = {}

        # State
        self._last_scan: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None
        self._last_reconcile: Optional[datetime] = None
        self._cycle_count = 0

    async def initialize(self) -> bool:
        """
        Initialize all components.

        Call this before run().
        """
        logger.info("Initializing NEXUS Orchestrator...")

        try:
            # Storage
            self.storage = get_storage_service()
            await self.storage.initialize()

            # Initialize system state if first run
            state = await self.storage.get_system_state()
            if not state:
                await self.storage.initialize_system_state(settings.starting_balance)

            # Risk components
            self.heat_manager = DynamicHeatManager(
                base_limit=settings.base_heat_limit,
                current_equity=settings.starting_balance,
            )

            self.circuit_breaker = SmartCircuitBreaker()
            self.kill_switch = KillSwitch()
            self.correlation_monitor = CorrelationMonitor()

            # Execution components
            self.position_manager = PositionManager(
                heat_manager=self.heat_manager,
                correlation_monitor=self.correlation_monitor,
            )

            self.order_manager = OrderManager()

            # Intelligence
            self.regime_detector = RegimeDetector(data_provider=None)  # Set provider later
            self.regime_monitor = ContinuousRegimeMonitor(
                regime_detector=self.regime_detector,
                check_interval_seconds=300,
            )

            self.cost_engine = CostEngine()
            self.scorer = OpportunityScorer()
            self.position_sizer = DynamicPositionSizer()

            # Signal generator (requires all risk components)
            self.signal_generator = SignalGenerator(
                cost_engine=self.cost_engine,
                position_sizer=self.position_sizer,
                heat_manager=self.heat_manager,
                circuit_breaker=self.circuit_breaker,
                kill_switch=self.kill_switch,
                correlation_monitor=self.correlation_monitor,
            )

            # Trade executor
            self.trade_executor = TradeExecutor(order_manager=self.order_manager)

            # Connect regime change to position adjustments
            self.regime_monitor.on_regime_change(self._handle_regime_change)

            # Alerts
            self.alert_manager = AlertManager()

            # Monitoring
            self.health_checker = HealthChecker()
            self.metrics = MetricsCollector()

            self._initialized = True
            logger.info("NEXUS Orchestrator initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False

    async def connect_brokers(self, brokers: Dict[str, Any]) -> bool:
        """
        Connect to brokers.

        Args:
            brokers: {"ibkr": IBKRProvider, "ig": IGProvider, ...}
        """
        self._brokers = brokers

        for name, broker in brokers.items():
            try:
                connected = await broker.connect()
                if connected:
                    logger.info(f"Connected to broker: {name}")
                    self.trade_executor.register_broker(name, broker)
                else:
                    logger.warning(f"Failed to connect to broker: {name}")
            except Exception as e:
                logger.error(f"Broker connection error ({name}): {e}")

        # Set up hardware stops on primary broker
        if "ibkr" in brokers:
            self.hardware_stops = HardwareStopManager(brokers["ibkr"])

        # Set up reconciliation engine
        self.reconciliation = ReconciliationEngine(
            position_manager=self.position_manager,
            order_manager=self.order_manager,
            trade_executor=self.trade_executor,
        )

        return True

    async def run(self) -> None:
        """
        Main run loop.

        Runs until stop() is called or kill switch triggers.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before run()")

        self._running = True
        logger.info("NEXUS Orchestrator starting main loop...")

        # Start background tasks
        await self.regime_monitor.start()
        await self.correlation_monitor.start_periodic_recalc(3600)

        # Alert that we're starting
        await self.alert_manager.send_alert(
            "NEXUS Orchestrator started",
            priority=AlertPriority.NORMAL,
        )

        try:
            while self._running:
                await self._run_cycle()
                await asyncio.sleep(self._scan_interval)

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            await self._emergency_shutdown(f"Fatal error: {e}")
        finally:
            await self._cleanup()

    async def _run_cycle(self) -> None:
        """Run one complete trading cycle."""
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        try:
            # 1. Check kill switch FIRST
            kill_status = self._check_kill_switch()
            if kill_status.get("triggered"):
                await self._emergency_shutdown(
                    kill_status.get("reason", "Kill switch triggered")
                )
                return

            # 2. Check circuit breaker
            circuit_state = self._check_circuit_breaker()
            if not circuit_state.get("can_trade", True):
                logger.info(f"Circuit breaker: {circuit_state.get('reason')}")
                # Still update positions, just don't take new trades
                await self._update_positions()
                return

            # 3. Run health checks (periodic)
            if self._should_run_health_check():
                await self._run_health_checks()

            # 4. Scan for opportunities
            opportunities = await self._scan_opportunities()

            # 5. Score and filter
            signals = await self._process_opportunities(opportunities)

            # 6. Execute valid signals
            for signal in signals:
                await self._execute_signal(signal)

            # 7. Update existing positions
            await self._update_positions()

            # 8. Reconcile with broker (periodic)
            if self._should_run_reconciliation():
                await self._reconcile_with_broker()

            # 9. Update metrics
            self.metrics.update_from_position_manager(self.position_manager)

            self._last_scan = cycle_start

        except Exception as e:
            logger.error(f"Error in cycle {self._cycle_count}: {e}")
            await raise_alert(
                "cycle_error", {"cycle": self._cycle_count, "error": str(e)}
            )

    def _check_kill_switch(self) -> dict:
        """Check all kill switch conditions."""
        if not self.kill_switch:
            return {"triggered": False}

        if self.kill_switch.is_triggered:
            return {"triggered": True, "reason": "Kill switch already active"}

        # Build SystemHealth from current state
        now = datetime.now(timezone.utc)
        health = SystemHealth(
            last_heartbeat=now,
            last_data_update=now,
            seconds_since_heartbeat=0,
            seconds_since_data=0,
            drawdown_pct=0,  # Updated from storage in full implementation
            is_connected=bool(self._brokers),
            active_errors=[],
        )

        state = self.kill_switch.check_conditions(health)
        if state.is_triggered:
            return {"triggered": True, "reason": state.message}

        return {"triggered": False}

    def _check_circuit_breaker(self) -> dict:
        """Check circuit breaker status."""
        if not self.circuit_breaker:
            return {"can_trade": True}

        # Get current P&L from storage (cached or default)
        state = self.circuit_breaker.check_status(
            daily_pnl_pct=0,  # Updated from storage in full implementation
            weekly_pnl_pct=0,
            drawdown_pct=0,
        )
        return {
            "can_trade": state.can_trade,
            "reason": state.message,
            "size_multiplier": state.size_multiplier,
        }

    async def _scan_opportunities(self) -> List[Any]:
        """Run all scanners and get opportunities."""
        if not self.scanner_orchestrator:
            return []

        try:
            opportunities = await self.scanner_orchestrator.run_scan_cycle()
            logger.debug(f"Found {len(opportunities)} opportunities")
            return opportunities
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []

    async def _process_opportunities(self, opportunities: List[Any]) -> List[Any]:
        """Score opportunities and generate signals."""
        signals = []

        for opp in opportunities:
            try:
                # Score
                scored = self.scorer.score(
                    opportunity=opp,
                    trend_alignment={"alignment": "PARTIAL"},
                    volume_ratio=1.0,
                    regime=self.regime_monitor.current_regime or MarketRegime.RANGING,
                    cost_analysis={"cost_ratio": 20},
                )

                # Filter by tier
                if scored.tier.value == "F":
                    continue

                # Build account state for signal generation
                sys_state = await self.storage.get_system_state() or {}
                account_state = AccountState(
                    starting_balance=settings.starting_balance,
                    current_equity=sys_state.get(
                        "current_equity", settings.starting_balance
                    ),
                    daily_pnl=sys_state.get("daily_pnl", 0),
                    daily_pnl_pct=sys_state.get("daily_pnl_pct", 0),
                    weekly_pnl_pct=sys_state.get("weekly_pnl_pct", 0),
                    drawdown_pct=sys_state.get("drawdown_pct", 0),
                    portfolio_heat=self.heat_manager.current_heat,
                    open_positions=[
                        p.to_dict()
                        for p in self.position_manager.open_positions
                    ],
                )

                market_state = MarketState(
                    regime=self.regime_monitor.current_regime or MarketRegime.RANGING,
                )

                # Generate signal (runs all risk checks internally)
                signal = await self.signal_generator.generate_signal(
                    scored, account_state, market_state
                )

                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error processing opportunity: {e}")

        return signals

    async def _execute_signal(self, signal) -> bool:
        """Execute a trading signal."""
        try:
            result = await self.trade_executor.execute_signal(signal)

            if result.get("success"):
                # Place hardware stop
                if self.hardware_stops:
                    await self.hardware_stops.place_hardware_stop(
                        position_id=result.get("order_id", ""),
                        symbol=signal.symbol,
                        direction=signal.direction.value,
                        entry_price=signal.entry_price,
                        software_stop=signal.stop_loss,
                        quantity=signal.position_size,
                    )

                # Send alert
                await self.alert_manager.send_signal(signal)

                # Persist signal
                await self.storage.save_signal(signal)

                return True
            else:
                logger.warning(f"Trade execution failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False

    async def _update_positions(self) -> None:
        """Update all open positions with current prices."""
        if not self.position_manager:
            return

        for position in self.position_manager.open_positions:
            try:
                # Get current price from data provider
                # TODO: Wire up data provider for live price updates
                pass
            except Exception as e:
                logger.debug(f"Error updating position {position.symbol}: {e}")

    async def _handle_regime_change(self, change) -> None:
        """Handle regime change - adjust positions if needed."""
        logger.warning(
            f"Regime changed: {change.previous.value} -> {change.current.value}"
        )

        # Get recommendations based on open position edges
        positions = [
            {
                "position_id": p.position_id,
                "symbol": p.symbol,
                "primary_edge": getattr(p, "edge_type", ""),
            }
            for p in self.position_manager.open_positions
        ]
        recommendations = self.regime_monitor.get_position_recommendations(positions)

        for rec in recommendations:
            logger.info(
                f"Regime recommendation: {rec['action']} {rec['symbol']} - {rec['reason']}"
            )
            await self.alert_manager.send_alert(
                f"Regime change: Consider {rec['action']} {rec['symbol']}",
                priority=AlertPriority.HIGH,
            )

    async def _emergency_shutdown(self, reason: str) -> None:
        """Emergency shutdown - close all positions."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        await self.alert_manager.send_alert(
            f"EMERGENCY SHUTDOWN: {reason}",
            priority=AlertPriority.CRITICAL,
        )

        # Cancel all hardware stops
        if self.hardware_stops:
            await self.hardware_stops.cancel_all_stops()

        self._running = False

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        logger.info("Cleaning up orchestrator...")

        if self.regime_monitor:
            await self.regime_monitor.stop()

        if self.correlation_monitor:
            await self.correlation_monitor.stop_periodic_recalc()

        for broker in self._brokers.values():
            try:
                await broker.disconnect()
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        logger.info("Stopping NEXUS Orchestrator...")
        self._running = False

    # ---- Helper methods ----

    def _should_run_health_check(self) -> bool:
        if not self._last_health_check:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_health_check
        ).total_seconds()
        return elapsed >= self._health_interval

    def _should_run_reconciliation(self) -> bool:
        if not self._last_reconcile:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_reconcile
        ).total_seconds()
        return elapsed >= self._reconcile_interval

    async def _run_health_checks(self) -> None:
        """Run system health checks."""
        self._last_health_check = datetime.now(timezone.utc)

        result = await self.health_checker.check_all(
            brokers=self._brokers,
            circuit_breaker=self.circuit_breaker,
            kill_switch=self.kill_switch,
            alert_manager=self.alert_manager,
        )

        if result.get("status") == "unhealthy":
            await raise_alert("system_unhealthy", result)

    async def _reconcile_with_broker(self) -> None:
        """Reconcile positions with broker."""
        self._last_reconcile = datetime.now(timezone.utc)

        if self.reconciliation:
            report = await self.reconciliation.reconcile()
            if report.has_critical:
                await raise_alert("reconciliation_mismatch", report.to_dict())
