"""
Scanner orchestrator: runs all scanners and aggregates results.
Runs each scanner, merges confluent signals by (symbol, direction), returns combined opportunities.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timezone

from collections import defaultdict

from .base import BaseScanner
from .calendar import TurnOfMonthScanner, MonthEndScanner
from .vwap import VWAPScanner
from .rsi import RSIScanner
from .gap import GapScanner
from .session import PowerHourScanner, LondonOpenScanner, NYOpenScanner, AsianRangeScanner, SessionScanner
from .orb import ORBScanner
from .bollinger import BollingerScanner
from .insider import InsiderScanner
from .earnings import EarningsDriftScanner
from .overnight import OvernightPremiumScanner
from .sentiment import SentimentScanner
from nexus.core.enums import Direction, EdgeType, Timeframe
from nexus.core.models import Opportunity
from nexus.data.base import BaseDataProvider
from nexus.execution.cooldown_manager import get_cooldown_manager

# Base scores for ranking which edge becomes primary in confluence
EDGE_BASE_SCORES: Dict[EdgeType, int] = {
    EdgeType.INSIDER_CLUSTER: 35,
    EdgeType.VWAP_DEVIATION: 30,
    EdgeType.TURN_OF_MONTH: 30,
    EdgeType.MONTH_END: 25,
    EdgeType.GAP_FILL: 25,
    EdgeType.RSI_EXTREME: 20,
    EdgeType.POWER_HOUR: 20,
    EdgeType.ASIAN_RANGE: 20,
    EdgeType.ORB: 18,
    EdgeType.BOLLINGER_TOUCH: 15,
    EdgeType.LONDON_OPEN: 15,
    EdgeType.NY_OPEN: 15,
    EdgeType.EARNINGS_DRIFT: 15,
    EdgeType.SENTIMENT_SPIKE: 20,
}

logger = logging.getLogger(__name__)


class ScannerOrchestrator:
    """
    Orchestrates all scanners and manages scan scheduling.

    Responsibilities:
    1. Initialize all scanner instances
    2. Run active scanners based on time/regime
    3. Aggregate and deduplicate opportunities
    4. Track scanner health/performance
    """

    def __init__(
        self,
        data_provider: Optional[BaseDataProvider] = None,
        stock_provider: Optional[BaseDataProvider] = None,
        forex_provider: Optional[BaseDataProvider] = None,
        crypto_provider: Optional[BaseDataProvider] = None,
        insider_provider: Optional[Any] = None,  # For insider scanner
        config: Optional[Dict] = None,
        edge_decay_monitor: Any = None,
    ):
        # Support both legacy (data_provider) and scheduler (stock_provider/forex_provider)
        if stock_provider is not None and forex_provider is not None:
            self.stock_provider = stock_provider
            self.forex_provider = forex_provider
            self.data_provider = stock_provider  # Scanners use data_provider for now
        elif data_provider is not None:
            self.stock_provider = data_provider
            self.forex_provider = data_provider
            self.data_provider = data_provider
        else:
            raise ValueError("Provide either data_provider or (stock_provider and forex_provider)")
        self.crypto_provider = crypto_provider
        self.insider_provider = insider_provider
        self.config = config or {}
        self.edge_decay_monitor = edge_decay_monitor

        # Initialize all scanners
        self.scanners = self._initialize_scanners()

        # Track disabled edges
        self._disabled_edges: Set[str] = set()

        # Scanner stats
        self.scan_count = 0
        self.last_scan_time = None
        self.scanner_stats = {}

    def _initialize_scanners(self) -> List[BaseScanner]:
        """Initialize all scanner instances."""

        scanners = [
            # Calendar scanners
            TurnOfMonthScanner(self.data_provider),
            MonthEndScanner(self.data_provider),

            # Mean reversion scanners
            VWAPScanner(self.data_provider),
            RSIScanner(self.data_provider),
            BollingerScanner(self.data_provider),

            # Gap scanners
            GapScanner(self.data_provider),

            # Session scanners
            PowerHourScanner(self.data_provider),
            LondonOpenScanner(self.data_provider),
            NYOpenScanner(self.data_provider),
            AsianRangeScanner(self.data_provider),
            SessionScanner(self.data_provider),

            # Breakout scanners
            ORBScanner(self.data_provider),

            # Fundamental scanners
            InsiderScanner(self.data_provider, insider_data_provider=self.insider_provider),
            EarningsDriftScanner(self.data_provider),

            # Overnight premium
            OvernightPremiumScanner(self.data_provider),

            # Sentiment (requires StockTwits API)
            SentimentScanner(self.data_provider),
        ]

        logger.info(f"Initialized {len(scanners)} scanners")
        return scanners

    def _is_edge_healthy(self, edge_type) -> bool:
        """Check if edge is healthy enough to trade."""
        edge_str = edge_type.value if hasattr(edge_type, 'value') else str(edge_type)

        # Check if manually disabled
        if edge_str in self._disabled_edges:
            return False

        # Check decay monitor
        if self.edge_decay_monitor:
            health = self.edge_decay_monitor.check_edge_health(edge_type)
            if health.get("status") == "DISABLED":
                self._disabled_edges.add(edge_str)
                logger.warning(f"Edge {edge_str} auto-disabled due to decay")
                return False
            if health.get("status") == "CRITICAL":
                logger.warning(f"Edge {edge_str} showing decay symptoms")
                # Don't disable yet, but could reduce position size

        return True

    def is_weekend(self) -> bool:
        """Check if it's currently weekend (Sat/Sun)."""
        now = datetime.now(timezone.utc)
        return now.weekday() >= 5  # 5 = Saturday, 6 = Sunday

    def is_crypto_only_mode(self) -> bool:
        """
        Check if we should only scan crypto.

        Returns True if it's weekend (all traditional markets closed).
        """
        return self.is_weekend()

    def get_crypto_instruments(self) -> List[str]:
        """Get crypto instruments from registry."""
        from nexus.data.instruments import get_instrument_registry, InstrumentType

        registry = get_instrument_registry()
        crypto = registry.get_by_type(InstrumentType.CRYPTO)
        return [c.symbol for c in crypto]

    async def scan_crypto_only(self) -> List[Opportunity]:
        """
        Scan only crypto markets (for weekends).

        Uses VWAP and RSI HF scanners on crypto instruments.
        """
        logger.info("Weekend mode: Scanning crypto only")

        if not self.crypto_provider:
            logger.warning("No crypto provider available for weekend trading")
            return []

        crypto_instruments = self.get_crypto_instruments()
        opportunities = []

        from .vwap_hf import VWAPHighFrequencyScanner
        from .rsi_hf import RSIHighFrequencyScanner

        scan_timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4]

        # VWAP HF scan
        vwap_scanner = VWAPHighFrequencyScanner(self.crypto_provider, self.settings if hasattr(self, 'settings') else None)
        for tf in scan_timeframes:
            try:
                opps = await vwap_scanner.scan_timeframe(tf, crypto_instruments)
                for opp in opps:
                    opp.edge_data["timeframe"] = tf.value
                    opp.edge_data["weekend_mode"] = True
                opportunities.extend(opps)
            except Exception as e:
                logger.error(f"VWAP crypto scan error ({tf.value}): {e}")

        # RSI HF scan
        rsi_scanner = RSIHighFrequencyScanner(self.crypto_provider, self.settings if hasattr(self, 'settings') else None)
        for tf in scan_timeframes:
            try:
                opps = await rsi_scanner.scan_timeframe(tf, crypto_instruments)
                for opp in opps:
                    opp.edge_data["timeframe"] = tf.value
                    opp.edge_data["weekend_mode"] = True
                opportunities.extend(opps)
            except Exception as e:
                logger.error(f"RSI crypto scan error ({tf.value}): {e}")

        logger.info(f"Weekend crypto scan found {len(opportunities)} opportunities")
        return opportunities

    async def run_all_scanners(self) -> List[Opportunity]:
        """
        Run all active scanners and aggregate results.

        Returns deduplicated list of opportunities.
        """
        self.scan_count += 1
        self.last_scan_time = datetime.now()

        all_opportunities = []

        for scanner in self.scanners:
            scanner_name = scanner.__class__.__name__

            try:
                # Check if scanner is active
                if not scanner.is_active(datetime.now(timezone.utc)):
                    logger.debug(f"{scanner_name}: Not active, skipping")
                    continue

                # Skip if edge is decaying/disabled
                if scanner.edge_type and not self._is_edge_healthy(scanner.edge_type):
                    logger.debug(f"{scanner_name}: Edge disabled, skipping")
                    continue

                # Run scanner
                logger.info(f"Running {scanner_name}...")
                opportunities = await scanner.scan()

                # Track stats
                self._update_stats(scanner_name, len(opportunities))

                if opportunities:
                    logger.info(f"{scanner_name}: Found {len(opportunities)} opportunities")
                    all_opportunities.extend(opportunities)

            except Exception as e:
                logger.error(f"{scanner_name} failed: {e}")
                self._update_stats(scanner_name, 0, error=str(e))

        # Merge confluent signals
        deduplicated = self._merge_confluence(all_opportunities)

        # Filter cooldowns
        filtered = self._filter_cooldowns(deduplicated)

        logger.info(
            f"Scan complete: {len(all_opportunities)} total, "
            f"{len(deduplicated)} after confluence merge, "
            f"{len(filtered)} after cooldown filter"
        )

        return filtered

    async def run_scan_cycle(self) -> List[Opportunity]:
        """
        Run one complete scan cycle.

        On weekends, only scans crypto markets.
        Used by NexusScheduler.
        """
        if self.is_crypto_only_mode() and self.crypto_provider:
            return await self.scan_crypto_only()
        return await self.run_all_scanners()

    async def run_scanner(self, scanner_name: str) -> List[Opportunity]:
        """Run a specific scanner by name."""

        for scanner in self.scanners:
            if scanner.__class__.__name__ == scanner_name:
                return await scanner.scan()

        logger.warning(f"Scanner not found: {scanner_name}")
        return []

    def _merge_confluence(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Merge opportunities when same symbol + direction detected.

        Instead of picking the best, COMBINE them:
        - Primary edge = highest scoring edge
        - confluence_edges = all edges that fired
        - confluence_count = number of edges
        - edge_data = merged from all opportunities
        """
        # Group by symbol + direction
        grouped: Dict[str, List[Opportunity]] = defaultdict(list)
        for opp in opportunities:
            dir_val = opp.direction.value if hasattr(opp.direction, 'value') else opp.direction
            key = f"{opp.symbol}_{dir_val}"
            grouped[key].append(opp)

        merged = []
        for key, opps in grouped.items():
            if len(opps) == 1:
                # No confluence - single signal
                merged.append(opps[0])
            else:
                # CONFLUENCE DETECTED - merge signals
                # Sort by edge base score to pick best as primary
                sorted_opps = sorted(
                    opps,
                    key=lambda o: EDGE_BASE_SCORES.get(
                        EdgeType(o.primary_edge) if isinstance(o.primary_edge, str) else o.primary_edge, 0
                    ),
                    reverse=True,
                )

                primary_opp = sorted_opps[0]

                # Collect all edges
                all_edges = [o.primary_edge for o in opps]

                # Merge edge_data from all opportunities
                merged_edge_data = {}
                for opp in opps:
                    merged_edge_data.update(opp.edge_data)

                # Use tightest stop (most conservative)
                dir_val = primary_opp.direction
                if isinstance(dir_val, str):
                    dir_val = Direction(dir_val)
                if dir_val == Direction.LONG:
                    best_stop = max(o.stop_loss for o in opps)
                else:
                    best_stop = min(o.stop_loss for o in opps)

                # Create merged opportunity
                merged_opp = Opportunity(
                    id=primary_opp.id,
                    detected_at=primary_opp.detected_at,
                    scanner="CONFLUENCE",
                    symbol=primary_opp.symbol,
                    market=primary_opp.market,
                    direction=primary_opp.direction,
                    entry_price=primary_opp.entry_price,
                    stop_loss=best_stop,
                    take_profit=primary_opp.take_profit,
                    primary_edge=primary_opp.primary_edge,
                    secondary_edges=[e for e in all_edges if e != primary_opp.primary_edge],
                    edge_data=merged_edge_data,
                    confluence_count=len(opps),
                    confluence_edges=all_edges,
                    is_confluence=True,
                )

                merged.append(merged_opp)

        return merged

    def _filter_cooldowns(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Filter out opportunities that are in cooldown."""
        cooldown = get_cooldown_manager()
        filtered = []

        for opp in opportunities:
            direction = opp.direction if hasattr(opp.direction, "value") else Direction(opp.direction)
            edge_type = opp.primary_edge if hasattr(opp.primary_edge, "value") else EdgeType(opp.primary_edge)

            can_signal, reason = cooldown.can_signal(
                opp.symbol,
                direction,
                edge_type,
            )
            if can_signal:
                filtered.append(opp)
            else:
                logger.debug("Cooldown filtered: %s - %s", opp.symbol, reason)

        return filtered

    def _update_stats(self, scanner_name: str, count: int, error: str = None):
        """Update scanner statistics."""

        if scanner_name not in self.scanner_stats:
            self.scanner_stats[scanner_name] = {
                "runs": 0,
                "opportunities": 0,
                "errors": 0,
                "last_error": None
            }

        stats = self.scanner_stats[scanner_name]
        stats["runs"] += 1
        stats["opportunities"] += count

        if error:
            stats["errors"] += 1
            stats["last_error"] = error

    def get_scanner_status(self) -> Dict:
        """Get status of all scanners."""

        status = {
            "total_scanners": len(self.scanners),
            "total_scans": self.scan_count,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "scanners": {}
        }

        for scanner in self.scanners:
            name = scanner.__class__.__name__
            status["scanners"][name] = {
                "edge_type": scanner.edge_type.value if scanner.edge_type else None,
                "active": scanner.is_active(datetime.now(timezone.utc)),
                "stats": self.scanner_stats.get(name, {})
            }

        return status

    def get_active_scanners(self) -> List[str]:
        """Get list of currently active scanner names."""
        now = datetime.now(timezone.utc)
        return [
            s.__class__.__name__
            for s in self.scanners
            if s.is_active(now)
        ]

    def disable_edge(self, edge_type) -> None:
        """Manually disable an edge."""
        edge_str = edge_type.value if hasattr(edge_type, 'value') else str(edge_type)
        self._disabled_edges.add(edge_str)
        logger.info(f"Edge {edge_str} manually disabled")

    def enable_edge(self, edge_type) -> None:
        """Re-enable a disabled edge."""
        edge_str = edge_type.value if hasattr(edge_type, 'value') else str(edge_type)
        self._disabled_edges.discard(edge_str)
        logger.info(f"Edge {edge_str} re-enabled")

    def get_edge_status(self) -> Dict[str, str]:
        """Get status of all edges."""
        status = {}
        for scanner in self.scanners:
            edge_str = scanner.edge_type.value if scanner.edge_type else None
            if edge_str is None:
                continue
            if edge_str in self._disabled_edges:
                status[edge_str] = "DISABLED"
            elif self.edge_decay_monitor:
                health = self.edge_decay_monitor.check_edge_health(scanner.edge_type)
                status[edge_str] = health.get("status", "HEALTHY")
            else:
                status[edge_str] = "HEALTHY"
        return status

    def record_trade_result(self, edge_type, trade_result: dict) -> None:
        """Record trade result for edge decay tracking."""
        if self.edge_decay_monitor:
            self.edge_decay_monitor.update_edge_performance(edge_type, trade_result)
