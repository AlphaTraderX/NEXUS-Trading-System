"""
NEXUS GOD MODE Runner - FULL 426 INSTRUMENT VERSION

24/5 automated scanning across ALL markets using the DataOrchestrator
and the complete InstrumentRegistry.

Usage:
    python -m nexus.scripts.run_god_mode              # Single cycle (paper)
    python -m nexus.scripts.run_god_mode --loop        # Continuous (paper)
    python -m nexus.scripts.run_god_mode --loop --live # Live mode (DANGEROUS)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.config.settings import settings
from nexus.core.enums import AlertPriority, Direction, EdgeType, Market, SignalTier
from nexus.core.models import Opportunity, CircuitBreakerState, SystemHealth
from nexus.data.orchestrator import DataOrchestrator, get_orchestrator, TradingSession
from nexus.config.edge_config import get_enabled_edges
from nexus.execution.exit_manager import ExitManager, ExitResult
from nexus.execution.signal_cooldown import SignalCooldownManager
from nexus.risk.circuit_breaker import SmartCircuitBreaker
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.kill_switch import KillSwitch
from nexus.security import check_license, audit, AuditEventType

# All 11 profitable scanners
from nexus.scanners.gap import GapScanner
from nexus.scanners.overnight import OvernightPremiumScanner
from nexus.scanners.vwap import VWAPScanner
from nexus.scanners.rsi import RSIScanner
from nexus.scanners.session import PowerHourScanner, LondonOpenScanner, AsianRangeScanner
from nexus.scanners.turn_of_month import TurnOfMonthScanner
from nexus.scanners.orb import ORBScanner
from nexus.scanners.insider import InsiderScanner
from nexus.scanners.bollinger import BollingerScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("GOD_MODE")

STATE_FILE = Path("data/god_mode_state.json")

# EdgeType -> scanner class
SCANNER_MAP = {
    EdgeType.GAP_FILL: GapScanner,
    EdgeType.OVERNIGHT_PREMIUM: OvernightPremiumScanner,
    EdgeType.VWAP_DEVIATION: VWAPScanner,
    EdgeType.RSI_EXTREME: RSIScanner,
    EdgeType.POWER_HOUR: PowerHourScanner,
    EdgeType.LONDON_OPEN: LondonOpenScanner,
    EdgeType.ASIAN_RANGE: AsianRangeScanner,
    EdgeType.TURN_OF_MONTH: TurnOfMonthScanner,
    EdgeType.ORB: ORBScanner,
    EdgeType.INSIDER_CLUSTER: InsiderScanner,
    EdgeType.BOLLINGER_TOUCH: BollingerScanner,
}


def score_to_tier(score: int) -> SignalTier:
    if score >= 80:
        return SignalTier.A
    if score >= 65:
        return SignalTier.B
    if score >= 50:
        return SignalTier.C
    if score >= 40:
        return SignalTier.D
    return SignalTier.F


class GodModeState:
    """Persistent state across cycles."""

    def __init__(self, starting_balance: float = 10_000.0):
        self.starting_balance = starting_balance
        self.current_equity = starting_balance
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.open_positions: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.signals_generated = 0
        self.signals_taken = 0
        self.cycles_run = 0
        self.last_run: Optional[str] = None
        self.session_start: str = datetime.now(timezone.utc).isoformat()
        self.cooldowns = SignalCooldownManager()
        self.last_daily_reset: str = ""

    def daily_pnl_pct(self) -> float:
        return (self.daily_pnl / self.starting_balance) * 100 if self.starting_balance else 0.0

    def weekly_pnl_pct(self) -> float:
        return (self.weekly_pnl / self.starting_balance) * 100 if self.starting_balance else 0.0

    def drawdown_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return max(0, (self.starting_balance - self.current_equity) / self.starting_balance * 100)

    def maybe_reset_daily(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.last_daily_reset != today:
            if self.daily_pnl != 0.0:
                logger.info("New day (%s). Resetting daily P&L (was $%.2f)", today, self.daily_pnl)
            self.daily_pnl = 0.0
            self.last_daily_reset = today
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "starting_balance": self.starting_balance,
            "current_equity": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_pnl_pct": round(self.daily_pnl_pct(), 2),
            "weekly_pnl_pct": round(self.weekly_pnl_pct(), 2),
            "drawdown_pct": round(self.drawdown_pct(), 2),
            "open_positions_count": len(self.open_positions),
            "open_positions": self.open_positions,
            "closed_trades_count": len(self.closed_trades),
            "closed_trades": self.closed_trades[-50:],
            "signals_generated": self.signals_generated,
            "signals_taken": self.signals_taken,
            "cycles_run": self.cycles_run,
            "last_run": self.last_run,
            "session_start": self.session_start,
            "cooldowns": self.cooldowns.to_dict(),
            "last_daily_reset": self.last_daily_reset,
        }

    def save(self, path: Path = STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path = STATE_FILE, starting_balance: float = 10_000.0) -> "GodModeState":
        state = cls(starting_balance)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                state.current_equity = data.get("current_equity", starting_balance)
                state.daily_pnl = data.get("daily_pnl", 0.0)
                state.weekly_pnl = data.get("weekly_pnl", 0.0)
                raw_pos = data.get("open_positions", [])
                state.open_positions = raw_pos if isinstance(raw_pos, list) else []
                raw_trades = data.get("closed_trades", [])
                state.closed_trades = raw_trades if isinstance(raw_trades, list) else []
                state.signals_generated = data.get("signals_generated", 0)
                state.signals_taken = data.get("signals_taken", 0)
                state.cycles_run = data.get("cycles_run", 0)
                state.last_run = data.get("last_run")
                state.session_start = data.get("session_start", datetime.now(timezone.utc).isoformat())
                cd_data = data.get("cooldowns", {})
                if isinstance(cd_data, dict):
                    state.cooldowns = SignalCooldownManager.from_dict(cd_data)
                state.last_daily_reset = data.get("last_daily_reset", "")
                logger.info(
                    "Loaded state: %d open, %d closed, %d cooldowns",
                    len(state.open_positions), len(state.closed_trades),
                    len(state.cooldowns._last_signal),
                )
            except Exception as e:
                logger.warning("Failed to load state (%s), starting fresh", e)
        return state


class GodModeRunner:
    """
    GOD MODE: Multi-provider 24/5 scanning across 426 instruments.

    The orchestrator routes each scanner to the correct data provider
    based on the InstrumentRegistry's DataProvider field.
    """

    def __init__(self, live_mode: bool = False, starting_balance: float = 10_000.0):
        self.live_mode = live_mode
        self.state = GodModeState.load(starting_balance=starting_balance)
        self.orchestrator: Optional[DataOrchestrator] = None
        self._scanners: Dict[EdgeType, Any] = {}
        self.exit_manager: Optional[ExitManager] = None
        self.heat_manager: Optional[DynamicHeatManager] = None
        self.circuit_breaker: Optional[SmartCircuitBreaker] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.alert_manager: Any = None

    async def initialize(self) -> None:
        logger.info("=" * 60)
        logger.info("  GOD MODE INITIALIZING  [%s]", "LIVE" if self.live_mode else "PAPER")
        logger.info("=" * 60)

        if not check_license():
            logger.error("License check failed")
            return

        audit(
            event_type=AuditEventType.SYSTEM_START,
            component="god_mode",
            action="start",
            details={"live_mode": self.live_mode},
        )

        # Connect providers via orchestrator
        self.orchestrator = get_orchestrator()

        # Print registry summary before connecting
        self.orchestrator.registry.print_summary()

        connection_results = await self.orchestrator.connect_all()
        if not any(connection_results.values()):
            logger.error("No providers connected!")
            return

        # Log session status
        status = self.orchestrator.get_status()
        logger.info("Session: %s (%s)", status["current_session"], status["session_description"])
        logger.info("Session instruments: %d / %d total", status["session_instruments"], status["total_registry"])
        logger.info("Active edges: %d", len(status["active_edges"]))

        # Init scanners - each gets the orchestrator as its data_provider
        # so get_quote / get_bars auto-route to the correct provider
        self._init_scanners()

        # Exit manager
        self.exit_manager = ExitManager(data_provider=self.orchestrator)

        # Risk management
        try:
            self.circuit_breaker = SmartCircuitBreaker()
            self.kill_switch = KillSwitch()
            self.heat_manager = DynamicHeatManager()
            logger.info("Risk management initialized")
        except Exception as e:
            logger.warning("Risk management init failed: %s", e)

        # Alert manager
        try:
            from nexus.delivery.alert_manager import create_alert_manager
            self.alert_manager = create_alert_manager()
            channels = list(self.alert_manager._channels.keys())
            if channels:
                logger.info("Alert channels: %s", channels)
            else:
                self.alert_manager = None
        except Exception as e:
            logger.warning("Alert manager unavailable: %s", e)
            self.alert_manager = None

        logger.info("GOD MODE ready")

    def _init_scanners(self) -> None:
        """
        Initialize scanners. Each scanner receives the orchestrator as its
        data_provider so get_quote/get_bars auto-route per symbol.
        """
        enabled_edges = get_enabled_edges()

        for edge_type, scanner_class in SCANNER_MAP.items():
            if edge_type not in enabled_edges:
                continue
            try:
                self._scanners[edge_type] = scanner_class(data_provider=self.orchestrator)
                logger.info("  Scanner: %s", edge_type.value)
            except Exception as e:
                logger.error("  Failed: %s - %s", edge_type.value, e)

        logger.info("Initialized %d scanners", len(self._scanners))

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self) -> Optional[str]:
        if self.circuit_breaker is None:
            return None
        try:
            cb: CircuitBreakerState = self.circuit_breaker.check_status(
                daily_pnl_pct=self.state.daily_pnl_pct(),
                weekly_pnl_pct=self.state.weekly_pnl_pct(),
                drawdown_pct=self.state.drawdown_pct(),
            )
            if not cb.can_trade:
                return cb.message
        except Exception as e:
            logger.debug("CB check skipped: %s", e)
        return None

    def _check_kill_switch(self) -> Optional[str]:
        if self.kill_switch is None:
            return None
        try:
            health = SystemHealth(
                last_heartbeat=datetime.now(timezone.utc),
                last_data_update=datetime.now(timezone.utc),
                seconds_since_heartbeat=0.0,
                seconds_since_data=0.0,
                drawdown_pct=self.state.drawdown_pct(),
                is_connected=self.orchestrator is not None and self.orchestrator._connected,
                active_errors=[],
            )
            ks = self.kill_switch.check_conditions(health)
            if ks.is_triggered:
                return ks.message
        except Exception as e:
            logger.debug("KS check skipped: %s", e)
        return None

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> Dict:
        cycle_start = datetime.now(timezone.utc)
        result: Dict[str, Any] = {
            "timestamp": cycle_start.isoformat(),
            "session": "",
            "session_instruments": 0,
            "opportunities": 0,
            "signals": 0,
            "trades": 0,
            "exits": 0,
            "errors": [],
        }

        self.state.maybe_reset_daily()
        self.state.cooldowns.clear_expired()

        session_cfg = self.orchestrator.get_current_session()
        session_instruments = self.orchestrator.get_instruments_for_session()
        active_edges = self.orchestrator.get_edges_for_session()

        result["session"] = session_cfg.session.value
        result["session_instruments"] = len(session_instruments)

        logger.info(
            "Session: %s [%s] | Instruments: %d | Edges: %d",
            session_cfg.session.value, session_cfg.priority,
            len(session_instruments), len(active_edges),
        )

        # Phase 0: exits
        if self.state.open_positions and self.exit_manager:
            try:
                exits = await self.exit_manager.check_exits(self.state.open_positions)
                for er in exits:
                    await self._close_position(er)
                result["exits"] = len(exits)
            except Exception as e:
                logger.warning("Exit check error: %s", e)

        # Safety
        halt = self._check_kill_switch()
        if halt:
            logger.warning("Kill switch: %s", halt)
            result["errors"].append(f"Kill switch: {halt}")
            return result

        halt = self._check_circuit_breaker()
        if halt:
            logger.warning("Circuit breaker: %s", halt)
            result["errors"].append(f"Circuit breaker: {halt}")
            return result

        # Phase 1: run scanners
        all_opps: List[Opportunity] = []
        for edge_type in active_edges:
            if edge_type not in self._scanners:
                continue
            scanner = self._scanners[edge_type]
            name = scanner.__class__.__name__
            try:
                opps = await scanner.scan()
                if opps:
                    logger.info("  %s: %d opportunities", name, len(opps))
                    all_opps.extend(opps)
            except Exception as e:
                logger.error("  %s error: %s", name, e)
                result["errors"].append(f"{name}: {e}")

        result["opportunities"] = len(all_opps)
        self.state.signals_generated += len(all_opps)

        if not all_opps:
            logger.info("No opportunities this cycle")
            self._finish_cycle(cycle_start)
            return result

        # Phase 2: score and filter
        valid = []
        for opp in all_opps:
            score = getattr(opp, "raw_score", 0) or 0
            tier = score_to_tier(score)
            if tier != SignalTier.F:
                valid.append({"opp": opp, "score": score, "tier": tier})
        result["signals"] = len(valid)

        # Phase 3: execute
        for sig in valid:
            opp: Opportunity = sig["opp"]
            tier: SignalTier = sig["tier"]
            score: int = sig["score"]

            edge_type = opp.primary_edge
            if isinstance(edge_type, str):
                try:
                    edge_type = EdgeType(edge_type)
                except ValueError:
                    pass

            if isinstance(edge_type, EdgeType) and not self.state.cooldowns.can_signal(opp.symbol, edge_type):
                continue

            notional_pct = opp.edge_data.get("notional_pct", 16)
            tier_mult = {SignalTier.A: 1.5, SignalTier.B: 1.25, SignalTier.C: 1.0, SignalTier.D: 0.5}.get(tier, 1.0)
            position_value = self.state.current_equity * (notional_pct / 100) * tier_mult
            shares = int(position_value / opp.entry_price) if opp.entry_price > 0 else 0

            if self.heat_manager and opp.entry_price > 0 and shares > 0:
                stop = opp.stop_loss
                rps = abs(opp.entry_price - stop) if stop else opp.entry_price * 0.01
                new_risk = (rps * shares / self.state.current_equity) * 100
                market = opp.market if isinstance(opp.market, Market) else Market.US_STOCKS
                hc = self.heat_manager.can_add_position(
                    new_risk_pct=new_risk, market=market,
                    daily_pnl_pct=self.state.daily_pnl_pct(),
                )
                if not hc.allowed:
                    logger.info("Heat limit: skip %s (%s)", opp.symbol, "; ".join(hc.rejection_reasons))
                    continue

            logger.info(
                "TRADE: %s %s | $%.2f -> SL $%.2f TP $%.2f | %d(%s) | %d sh ($%.0f)",
                opp.symbol,
                opp.direction.value if hasattr(opp.direction, "value") else opp.direction,
                opp.entry_price, opp.stop_loss, opp.take_profit,
                score, tier.value, shares, position_value,
            )

            self.state.open_positions.append({
                "symbol": opp.symbol,
                "direction": opp.direction.value if hasattr(opp.direction, "value") else str(opp.direction),
                "entry_price": opp.entry_price,
                "stop_loss": opp.stop_loss,
                "take_profit": opp.take_profit,
                "shares": shares,
                "position_value": round(position_value, 2),
                "score": score,
                "tier": tier.value,
                "edge": opp.primary_edge.value if hasattr(opp.primary_edge, "value") else str(opp.primary_edge),
                "market": opp.market.value if hasattr(opp.market, "value") else str(opp.market),
                "opened_at": cycle_start.isoformat(),
            })

            if isinstance(edge_type, EdgeType):
                self.state.cooldowns.record_signal(opp.symbol, edge_type)

            if self.alert_manager:
                await self._send_alert(opp, score, tier, shares, position_value)

            self.state.signals_taken += 1
            result["trades"] += 1

        self._finish_cycle(cycle_start)
        return result

    def _finish_cycle(self, cycle_start: datetime) -> None:
        self.state.cycles_run += 1
        self.state.last_run = cycle_start.isoformat()
        self.state.save()

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def _send_alert(self, opp, score, tier, shares, position_value):
        direction_str = opp.direction.value if hasattr(opp.direction, "value") else str(opp.direction)
        edge_str = opp.primary_edge.value if hasattr(opp.primary_edge, "value") else str(opp.primary_edge)
        market_str = opp.market.value if hasattr(opp.market, "value") else str(opp.market)
        msg = (
            f"**NEXUS GOD MODE {'LIVE' if self.live_mode else 'PAPER'}**\n\n"
            f"**{direction_str.upper()}** {opp.symbol} ({market_str})\n"
            f"Entry: ${opp.entry_price:.2f} | SL: ${opp.stop_loss:.2f} | TP: ${opp.take_profit:.2f}\n"
            f"Edge: {edge_str} | Score: {score} ({tier.value})\n"
            f"Size: ~{shares} sh (${position_value:,.0f})"
        )
        try:
            await self.alert_manager.send_alert(msg, AlertPriority.NORMAL)
        except Exception as e:
            logger.warning("Alert failed: %s", e)

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------

    async def _close_position(self, exit_result: ExitResult) -> None:
        pos = exit_result.position
        self.state.current_equity += exit_result.net_pnl
        self.state.daily_pnl += exit_result.net_pnl
        self.state.weekly_pnl += exit_result.net_pnl

        self.state.closed_trades.append({
            **pos,
            "exit_price": exit_result.exit_price,
            "exit_reason": exit_result.exit_reason,
            "gross_pnl": round(exit_result.gross_pnl, 2),
            "costs": round(exit_result.costs, 4),
            "net_pnl": round(exit_result.net_pnl, 2),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })

        try:
            self.state.open_positions.remove(pos)
        except ValueError:
            self.state.open_positions = [
                p for p in self.state.open_positions
                if not (p.get("symbol") == pos.get("symbol")
                        and p.get("edge") == pos.get("edge")
                        and p.get("opened_at") == pos.get("opened_at"))
            ]

        logger.info(
            "CLOSED: %s %s | %s @ $%.2f | Net $%.2f | Equity $%.2f",
            pos.get("symbol", "?"), pos.get("edge", "?"),
            exit_result.exit_reason, exit_result.exit_price,
            exit_result.net_pnl, self.state.current_equity,
        )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run_loop(self, interval: int = 60) -> None:
        logger.info("GOD MODE loop starting (interval %ds)", interval)
        try:
            while True:
                session = self.orchestrator.get_current_session()
                logger.info(
                    "%s\nCYCLE %d | %s | Session: %s [%s]\n%s",
                    "=" * 60, self.state.cycles_run + 1,
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    session.session.value, session.priority, "=" * 60,
                )

                result = await self.run_cycle()

                logger.info(
                    "Result: %d exits | %d opps -> %d signals -> %d trades "
                    "(scanned %d instruments)",
                    result["exits"], result["opportunities"],
                    result["signals"], result["trades"],
                    result["session_instruments"],
                )
                if result["errors"]:
                    logger.warning("Errors: %s", result["errors"])

                logger.info(
                    "Totals: %d generated, %d taken, equity $%.2f\n",
                    self.state.signals_generated, self.state.signals_taken,
                    self.state.current_equity,
                )

                # Adaptive interval
                eff = interval
                if session.priority == "HIGH":
                    eff = max(30, interval // 2)
                elif session.priority == "LOW":
                    eff = interval * 3

                await asyncio.sleep(eff)

        except KeyboardInterrupt:
            logger.info("\nGOD MODE stopped by user")
        finally:
            self.state.save()
            if self.orchestrator:
                await self.orchestrator.disconnect_all()
            audit(event_type=AuditEventType.SYSTEM_STOP, component="god_mode", action="stop")

    async def shutdown(self) -> None:
        self.state.save()
        if self.orchestrator:
            await self.orchestrator.disconnect_all()
        if self.alert_manager:
            try:
                await self.alert_manager.close()
            except Exception:
                pass


async def main() -> int:
    parser = argparse.ArgumentParser(description="NEXUS GOD MODE")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval (sec)")
    parser.add_argument("--live", action="store_true", help="Live mode (DANGEROUS)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance")
    args = parser.parse_args()

    if args.live:
        confirm = input("LIVE MODE - Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return 1

    runner = GodModeRunner(live_mode=args.live, starting_balance=args.balance)
    await runner.initialize()

    if runner.orchestrator is None or not runner.orchestrator._connected:
        logger.error("Init failed")
        return 1

    if args.loop:
        await runner.run_loop(args.interval)
        return 0

    result = await runner.run_cycle()

    print(f"\n{'=' * 60}")
    print("GOD MODE RESULT")
    print(f"{'=' * 60}")
    print(f"Session:        {result['session']}")
    print(f"Instruments:    {result['session_instruments']}")
    print(f"Exits:          {result['exits']}")
    print(f"Opportunities:  {result['opportunities']}")
    print(f"Signals:        {result['signals']}")
    print(f"Trades:         {result['trades']}")
    print(f"Open positions: {len(runner.state.open_positions)}")
    print(f"Equity:         ${runner.state.current_equity:,.2f}")
    print(f"Daily P&L:      ${runner.state.daily_pnl:,.2f}")
    if result["errors"]:
        print(f"Errors:         {result['errors']}")

    await runner.shutdown()
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
