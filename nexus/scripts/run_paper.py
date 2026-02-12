"""
NEXUS Paper Trading Runner - full end-to-end simulation.

Runs the 4 validated scanners, generates signals, executes on paper,
tracks P&L, and sends alerts. Use this to validate live logic before
going live with real money.

Usage:
    python -m nexus.scripts.run_paper                       # Run once
    python -m nexus.scripts.run_paper --loop                # Run continuously (60s)
    python -m nexus.scripts.run_paper --loop --interval 300 # 5 min interval
    python -m nexus.scripts.run_paper --offline             # No Polygon API needed
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

# Exit manager (handles position exits — overnight MOO, RSI indicator, stop/target)
from nexus.execution.exit_manager import ExitManager, ExitResult

# Signal cooldown (prevents duplicate signals on same symbol/edge)
from nexus.execution.signal_cooldown import SignalCooldownManager

# Risk management (reuse instances for state persistence across cycles)
from nexus.risk.circuit_breaker import SmartCircuitBreaker
from nexus.risk.heat_manager import DynamicHeatManager
from nexus.risk.kill_switch import KillSwitch

# Scanners (the 4 fixed, validated ones)
from nexus.scanners.gap import GapScanner
from nexus.scanners.overnight import OvernightPremiumScanner
from nexus.scanners.vwap import VWAPScanner
from nexus.scanners.rsi import RSIScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

# State file for persistence across runs
STATE_FILE = Path("data/paper_trading_state.json")


class PaperTradingState:
    """Track paper trading state across runs."""

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
        if self.starting_balance == 0:
            return 0.0
        return (self.daily_pnl / self.starting_balance) * 100

    def weekly_pnl_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.weekly_pnl / self.starting_balance) * 100

    def maybe_reset_daily(self) -> bool:
        """Reset daily P&L if it's a new day. Returns True if reset."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self.last_daily_reset != today:
            if self.daily_pnl != 0.0:
                logger.info(
                    "New day (%s). Resetting daily P&L (was $%.2f)",
                    today, self.daily_pnl,
                )
            self.daily_pnl = 0.0
            self.last_daily_reset = today
            return True
        return False

    def drawdown_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return max(
            0,
            (self.starting_balance - self.current_equity)
            / self.starting_balance
            * 100,
        )

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
            "closed_trades": self.closed_trades[-50:],  # Keep last 50
            "signals_generated": self.signals_generated,
            "signals_taken": self.signals_taken,
            "cycles_run": self.cycles_run,
            "last_run": self.last_run,
            "session_start": self.session_start,
            "cooldowns": self.cooldowns.to_dict(),
            "last_daily_reset": self.last_daily_reset,
        }

    def save(self, path: Path = STATE_FILE) -> None:
        """Persist state to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path = STATE_FILE, starting_balance: float = 10_000.0) -> "PaperTradingState":
        """Load state from JSON, or create fresh."""
        state = cls(starting_balance)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                state.current_equity = data.get("current_equity", starting_balance)
                state.daily_pnl = data.get("daily_pnl", 0.0)
                state.weekly_pnl = data.get("weekly_pnl", 0.0)
                # Handle old format where positions/trades were counts (int)
                raw_positions = data.get("open_positions", [])
                state.open_positions = raw_positions if isinstance(raw_positions, list) else []
                raw_trades = data.get("closed_trades", [])
                state.closed_trades = raw_trades if isinstance(raw_trades, list) else []
                state.signals_generated = data.get("signals_generated", 0)
                state.signals_taken = data.get("signals_taken", 0)
                state.cycles_run = data.get("cycles_run", 0)
                state.last_run = data.get("last_run")
                state.session_start = data.get(
                    "session_start", datetime.now(timezone.utc).isoformat()
                )
                cooldown_data = data.get("cooldowns", {})
                if isinstance(cooldown_data, dict):
                    state.cooldowns = SignalCooldownManager.from_dict(cooldown_data)
                state.last_daily_reset = data.get("last_daily_reset", "")
                logger.info(
                    "Loaded state: %d open positions, %d closed trades, %d cooldowns",
                    len(state.open_positions), len(state.closed_trades),
                    len(state.cooldowns._last_signal),
                )
            except Exception as e:
                logger.warning("Failed to load state (%s), starting fresh", e)
        return state


def score_to_tier(score: int) -> SignalTier:
    """Map raw score to signal tier (matches backtest score-sizing tiers)."""
    if score >= 80:
        return SignalTier.A
    if score >= 65:
        return SignalTier.B
    if score >= 50:
        return SignalTier.C
    if score >= 40:
        return SignalTier.D
    return SignalTier.F


def is_market_open() -> bool:
    """Check if US market is currently open (rough UTC check)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


class PaperTradingRunner:
    """Run the paper trading loop with all validated scanners."""

    def __init__(self, offline: bool = False, starting_balance: float = 10_000.0):
        self.offline = offline
        self.state = PaperTradingState.load(starting_balance=starting_balance)
        self.data_provider: Any = None
        self.scanners: List[Any] = []
        self.alert_manager: Any = None
        self.exit_manager: Optional[ExitManager] = None
        self.heat_manager: Optional[DynamicHeatManager] = None
        self.circuit_breaker: Optional[SmartCircuitBreaker] = None
        self.kill_switch: Optional[KillSwitch] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing paper trading runner...")

        # Data provider
        if not self.offline and settings.polygon_api_key:
            try:
                from nexus.data.polygon import PolygonProvider

                provider = PolygonProvider()
                if await provider.connect():
                    self.data_provider = provider
                    logger.info("Polygon data provider connected")
                else:
                    logger.warning("Polygon connection failed — using placeholder data")
            except Exception as e:
                logger.warning("Polygon init failed (%s) — using placeholder data", e)

        if self.data_provider is None:
            logger.info("Using placeholder data (offline mode)")

        # Scanners
        dp = self.data_provider
        self.scanners = [
            GapScanner(data_provider=dp),
            OvernightPremiumScanner(data_provider=dp),
            VWAPScanner(data_provider=dp),
            RSIScanner(data_provider=dp),
        ]
        logger.info("Initialized %d validated scanners", len(self.scanners))

        # Exit manager (uses same data provider as scanners)
        self.exit_manager = ExitManager(data_provider=dp)

        # Risk management (reuse instances for state persistence across cycles)
        try:
            self.circuit_breaker = SmartCircuitBreaker()
            self.kill_switch = KillSwitch()
            self.heat_manager = DynamicHeatManager()
            logger.info("Risk management initialized (circuit breaker, kill switch, heat manager)")
        except Exception as e:
            logger.warning("Risk management init failed: %s (continuing without)", e)

        # Alert manager (best-effort — paper trading works without it)
        try:
            from nexus.delivery.alert_manager import create_alert_manager

            self.alert_manager = create_alert_manager()
            channels = list(self.alert_manager._channels.keys())
            if channels:
                logger.info("Alert channels: %s", channels)
            else:
                logger.info("No alert channels configured (alerts disabled)")
                self.alert_manager = None
        except Exception as e:
            logger.warning("Alert manager unavailable: %s", e)
            self.alert_manager = None

        logger.info("Paper trading runner ready")

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self) -> Optional[str]:
        """Check circuit breaker; return reason string if trading halted, else None."""
        if self.circuit_breaker is None:
            return None
        try:
            cb_state: CircuitBreakerState = self.circuit_breaker.check_status(
                daily_pnl_pct=self.state.daily_pnl_pct(),
                weekly_pnl_pct=self.state.weekly_pnl_pct(),
                drawdown_pct=self.state.drawdown_pct(),
            )
            if not cb_state.can_trade:
                return cb_state.message
        except Exception as e:
            logger.debug("Circuit breaker check skipped: %s", e)
        return None

    def _check_kill_switch(self) -> Optional[str]:
        """Check kill switch; return reason string if triggered, else None."""
        if self.kill_switch is None:
            return None
        try:
            health = SystemHealth(
                last_heartbeat=datetime.now(timezone.utc),
                last_data_update=datetime.now(timezone.utc),
                seconds_since_heartbeat=0.0,
                seconds_since_data=0.0,
                drawdown_pct=self.state.drawdown_pct(),
                is_connected=self.data_provider is not None,
                active_errors=[],
            )
            ks_state = self.kill_switch.check_conditions(health)
            if ks_state.is_triggered:
                return ks_state.message
        except Exception as e:
            logger.debug("Kill switch check skipped: %s", e)
        return None

    def _calculate_current_heat(self) -> float:
        """Calculate current portfolio heat as % of equity at risk."""
        if not self.state.open_positions or self.state.current_equity <= 0:
            return 0.0
        total_risk = 0.0
        for pos in self.state.open_positions:
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            shares = pos.get("shares", 0)
            if entry and stop and shares:
                risk_per_share = abs(entry - stop)
                total_risk += risk_per_share * shares
        return (total_risk / self.state.current_equity) * 100

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> Dict:
        """Run one complete scan / score / paper-execute cycle."""
        cycle_start = datetime.now(timezone.utc)
        result: Dict[str, Any] = {
            "timestamp": cycle_start.isoformat(),
            "market_open": is_market_open(),
            "opportunities": 0,
            "signals": 0,
            "trades": 0,
            "exits": 0,
            "errors": [],
        }

        # Reset daily P&L if new day
        self.state.maybe_reset_daily()

        # Clean up expired cooldowns periodically
        self.state.cooldowns.clear_expired()

        # Phase 0: Check exits on open positions BEFORE scanning
        if self.state.open_positions and self.exit_manager:
            exits = await self.exit_manager.check_exits(self.state.open_positions)
            for exit_result in exits:
                await self._close_position(exit_result)
            result["exits"] = len(exits)

        # Safety gates
        halt_reason = self._check_kill_switch()
        if halt_reason:
            logger.warning("Kill switch triggered: %s", halt_reason)
            result["errors"].append(f"Kill switch: {halt_reason}")
            return result

        halt_reason = self._check_circuit_breaker()
        if halt_reason:
            logger.warning("Circuit breaker halt: %s", halt_reason)
            result["errors"].append(f"Circuit breaker: {halt_reason}")
            return result

        # Run scanners
        all_opps: List[Opportunity] = []
        for scanner in self.scanners:
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
            self.state.cycles_run += 1
            self.state.last_run = cycle_start.isoformat()
            self.state.save()
            return result

        # Score and filter
        valid_signals = []
        for opp in all_opps:
            score = getattr(opp, "raw_score", 0) or 0
            tier = score_to_tier(score)
            if tier == SignalTier.F:
                continue
            valid_signals.append({"opportunity": opp, "score": score, "tier": tier})

        result["signals"] = len(valid_signals)

        # Paper-execute: log and alert (with cooldown + heat checks)
        for sig in valid_signals:
            opp: Opportunity = sig["opportunity"]
            tier: SignalTier = sig["tier"]
            score: int = sig["score"]

            edge_type = opp.primary_edge
            if isinstance(edge_type, str):
                try:
                    edge_type = EdgeType(edge_type)
                except ValueError:
                    pass

            # Check cooldown (prevents duplicate signals on same symbol/edge)
            if isinstance(edge_type, EdgeType) and not self.state.cooldowns.can_signal(opp.symbol, edge_type):
                logger.debug("Skipping %s %s — cooldown active", opp.symbol, edge_type.value)
                continue

            # Determine notional and size multiplier from tier
            notional_pct = opp.edge_data.get("notional_pct", 16)
            tier_mult = {
                SignalTier.A: 1.5,
                SignalTier.B: 1.25,
                SignalTier.C: 1.0,
                SignalTier.D: 0.5,
            }.get(tier, 1.0)

            position_value = self.state.current_equity * (notional_pct / 100) * tier_mult
            shares = int(position_value / opp.entry_price) if opp.entry_price > 0 else 0

            # Check portfolio heat before adding position
            if self.heat_manager and opp.entry_price > 0 and shares > 0:
                stop = opp.stop_loss
                risk_per_share = abs(opp.entry_price - stop) if stop else opp.entry_price * 0.01
                new_risk_pct = (risk_per_share * shares / self.state.current_equity) * 100
                market = opp.market if isinstance(opp.market, Market) else Market.US_STOCKS
                heat_check = self.heat_manager.can_add_position(
                    new_risk_pct=new_risk_pct,
                    market=market,
                    daily_pnl_pct=self.state.daily_pnl_pct(),
                )
                if not heat_check.allowed:
                    logger.info(
                        "Heat limit: skipping %s (%s)",
                        opp.symbol, "; ".join(heat_check.rejection_reasons),
                    )
                    continue

            logger.info(
                "PAPER TRADE: %s %s | Entry $%.2f | Stop $%.2f | "
                "Target $%.2f | Score %d (%s) | ~%d shares ($%.0f)",
                opp.symbol,
                opp.direction.value if hasattr(opp.direction, "value") else opp.direction,
                opp.entry_price,
                opp.stop_loss,
                opp.take_profit,
                score,
                tier.value,
                shares,
                position_value,
            )

            # Record open position
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
                "opened_at": cycle_start.isoformat(),
            })

            # Record cooldown after successful trade
            if isinstance(edge_type, EdgeType):
                self.state.cooldowns.record_signal(opp.symbol, edge_type)

            # Send alert
            if self.alert_manager:
                await self._send_paper_alert(opp, score, tier, shares, position_value)

            self.state.signals_taken += 1
            result["trades"] += 1

        self.state.cycles_run += 1
        self.state.last_run = cycle_start.isoformat()
        self.state.save()
        return result

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def _send_paper_alert(
        self,
        opp: Opportunity,
        score: int,
        tier: SignalTier,
        shares: int,
        position_value: float,
    ) -> None:
        """Send alert for a paper trade signal."""
        direction_str = opp.direction.value if hasattr(opp.direction, "value") else str(opp.direction)
        edge_str = opp.primary_edge.value if hasattr(opp.primary_edge, "value") else str(opp.primary_edge)

        message = (
            f"**NEXUS PAPER SIGNAL**\n\n"
            f"**{direction_str.upper()}** {opp.symbol}\n"
            f"Entry: ${opp.entry_price:.2f}\n"
            f"Stop: ${opp.stop_loss:.2f}\n"
            f"Target: ${opp.take_profit:.2f}\n\n"
            f"Edge: {edge_str}\n"
            f"Score: {score}/100 (Tier {tier.value})\n"
            f"Size: ~{shares} shares (${position_value:,.0f})\n\n"
            f"*Paper trading mode*"
        )

        try:
            await self.alert_manager.send_alert(message, AlertPriority.NORMAL)
        except Exception as e:
            logger.warning("Alert delivery failed: %s", e)

    # ------------------------------------------------------------------
    # Position exits
    # ------------------------------------------------------------------

    async def _close_position(self, exit_result: ExitResult) -> None:
        """Close a position and update P&L / equity."""
        pos = exit_result.position
        symbol = pos.get("symbol", "?")
        edge = pos.get("edge", "?")

        # Update equity and daily P&L
        self.state.current_equity += exit_result.net_pnl
        self.state.daily_pnl += exit_result.net_pnl
        self.state.weekly_pnl += exit_result.net_pnl

        # Move from open to closed
        closed_trade = {
            **pos,
            "exit_price": exit_result.exit_price,
            "exit_reason": exit_result.exit_reason,
            "gross_pnl": round(exit_result.gross_pnl, 2),
            "costs": round(exit_result.costs, 4),
            "net_pnl": round(exit_result.net_pnl, 2),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state.closed_trades.append(closed_trade)

        # Remove from open positions
        try:
            self.state.open_positions.remove(pos)
        except ValueError:
            # Position might have been modified; remove by symbol+edge+opened_at
            self.state.open_positions = [
                p for p in self.state.open_positions
                if not (
                    p.get("symbol") == pos.get("symbol")
                    and p.get("edge") == pos.get("edge")
                    and p.get("opened_at") == pos.get("opened_at")
                )
            ]

        logger.info(
            "CLOSED: %s %s | %s @ $%.2f | Net PnL $%.2f | "
            "Equity $%.2f | Daily P&L $%.2f",
            symbol, edge, exit_result.exit_reason,
            exit_result.exit_price, exit_result.net_pnl,
            self.state.current_equity, self.state.daily_pnl,
        )

        # Send exit alert
        if self.alert_manager:
            await self._send_exit_alert(exit_result)

    async def _send_exit_alert(self, exit_result: ExitResult) -> None:
        """Send alert for a position exit."""
        pos = exit_result.position
        direction_str = pos.get("direction", "?").upper()
        pnl_sign = "+" if exit_result.net_pnl >= 0 else ""

        message = (
            f"**NEXUS PAPER EXIT**\n\n"
            f"**{direction_str}** {pos.get('symbol', '?')} CLOSED\n"
            f"Entry: ${pos.get('entry_price', 0):.2f}\n"
            f"Exit: ${exit_result.exit_price:.2f}\n"
            f"Reason: {exit_result.exit_reason}\n\n"
            f"P&L: {pnl_sign}${exit_result.net_pnl:.2f}\n"
            f"Equity: ${self.state.current_equity:,.2f}\n\n"
            f"*Paper trading mode*"
        )

        try:
            await self.alert_manager.send_alert(message, AlertPriority.NORMAL)
        except Exception as e:
            logger.warning("Exit alert delivery failed: %s", e)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run_loop(self, interval: int = 60) -> None:
        """Run continuously with specified interval."""
        logger.info("Starting paper trading loop (interval: %ds)", interval)
        logger.info("Press Ctrl+C to stop\n")

        try:
            while True:
                self.state.cycles_run += 0  # just to access state
                header = (
                    f"{'=' * 55}\n"
                    f"CYCLE {self.state.cycles_run + 1} | "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
                    f"Market {'OPEN' if is_market_open() else 'CLOSED'}\n"
                    f"{'=' * 55}"
                )
                logger.info(header)

                result = await self.run_cycle()

                logger.info(
                    "Cycle result: %d exits | %d opps -> %d signals -> %d trades",
                    result["exits"],
                    result["opportunities"],
                    result["signals"],
                    result["trades"],
                )
                if result["errors"]:
                    logger.warning("Errors: %s", result["errors"])

                logger.info(
                    "Session: %d generated, %d taken, equity $%.2f\n",
                    self.state.signals_generated,
                    self.state.signals_taken,
                    self.state.current_equity,
                )

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\nPaper trading stopped by user")
        finally:
            self.state.save()
            if self.data_provider:
                await self.data_provider.disconnect()
            logger.info("Final state: %s", json.dumps(self.state.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.state.save()
        if self.data_provider:
            await self.data_provider.disconnect()
        if self.alert_manager:
            try:
                await self.alert_manager.close()
            except Exception:
                pass


async def main() -> int:
    parser = argparse.ArgumentParser(description="NEXUS Paper Trading Runner")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=int, default=60, help="Loop interval in seconds"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use placeholder data (no Polygon API)"
    )
    parser.add_argument(
        "--balance", type=float, default=10_000.0, help="Starting paper balance"
    )
    args = parser.parse_args()

    runner = PaperTradingRunner(offline=args.offline, starting_balance=args.balance)
    await runner.initialize()

    if args.loop:
        await runner.run_loop(args.interval)
        return 0

    # Single cycle
    result = await runner.run_cycle()

    print(f"\n{'=' * 55}")
    print("PAPER TRADING RESULT")
    print(f"{'=' * 55}")
    print(f"Exits:          {result['exits']}")
    print(f"Opportunities:  {result['opportunities']}")
    print(f"Valid signals:  {result['signals']}")
    print(f"Paper trades:   {result['trades']}")
    print(f"Open positions: {len(runner.state.open_positions)}")
    print(f"Equity:         ${runner.state.current_equity:,.2f}")
    print(f"Daily P&L:      ${runner.state.daily_pnl:,.2f}")
    if result["errors"]:
        print(f"Errors:         {result['errors']}")

    # Show summary (without full position/trade lists)
    summary = runner.state.to_dict()
    summary.pop("open_positions", None)
    summary.pop("closed_trades", None)
    print(f"\nState: {json.dumps(summary, indent=2)}")

    await runner.shutdown()
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
