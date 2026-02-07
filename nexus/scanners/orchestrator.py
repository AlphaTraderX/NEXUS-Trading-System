"""
Scanner orchestrator: runs all scanners and aggregates results.
Runs each scanner, deduplicates by (symbol, direction), returns combined opportunities.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

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
from nexus.core.models import Opportunity
from nexus.data.base import BaseDataProvider

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
        insider_provider: Optional[Any] = None,  # For insider scanner
        config: Optional[Dict] = None
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
        self.insider_provider = insider_provider
        self.config = config or {}

        # Initialize all scanners
        self.scanners = self._initialize_scanners()

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
        ]

        logger.info(f"Initialized {len(scanners)} scanners")
        return scanners

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

        # Deduplicate
        deduplicated = self._deduplicate(all_opportunities)

        logger.info(
            f"Scan complete: {len(all_opportunities)} total, "
            f"{len(deduplicated)} after deduplication"
        )

        return deduplicated

    async def run_scan_cycle(self) -> List[Opportunity]:
        """
        Run one complete scan cycle (alias for run_all_scanners).
        Used by NexusScheduler.
        """
        return await self.run_all_scanners()

    async def run_scanner(self, scanner_name: str) -> List[Opportunity]:
        """Run a specific scanner by name."""

        for scanner in self.scanners:
            if scanner.__class__.__name__ == scanner_name:
                return await scanner.scan()

        logger.warning(f"Scanner not found: {scanner_name}")
        return []

    def _deduplicate(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Remove duplicate opportunities for same symbol/direction.

        Keeps the opportunity with the higher raw_score.
        """
        seen = {}

        for opp in opportunities:
            key = f"{opp.symbol}_{opp.direction}"

            if key not in seen:
                seen[key] = opp
            else:
                # Keep higher score
                existing_score = getattr(seen[key], 'raw_score', 0) or 0
                new_score = getattr(opp, 'raw_score', 0) or 0

                if new_score > existing_score:
                    seen[key] = opp

        return list(seen.values())

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
