"""
NEXUS Slippage Tracker

Tracks actual vs expected slippage and feeds back to CostEngine
for more accurate future estimates.
"""

import logging
from typing import Dict, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict

from nexus.core.enums import Market

logger = logging.getLogger(__name__)


@dataclass
class SlippageRecord:
    """Single slippage observation."""

    timestamp: datetime
    symbol: str
    market: Market
    expected_price: float
    actual_price: float
    slippage_pct: float
    direction: str
    session: str  # "pre_market", "regular", "after_hours"


class SlippageTracker:
    """
    Tracks slippage and provides feedback for cost estimation.

    Key insight: Slippage varies by:
    - Market (stocks vs forex vs futures)
    - Time of day (pre-market much worse)
    - Volatility conditions
    - Position size relative to volume
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._records: List[SlippageRecord] = []

        # Aggregated stats by market
        self._market_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "total_slippage": 0.0,
                "max_slippage": 0.0,
            }
        )

        # Stats by session
        self._session_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "total_slippage": 0.0,
            }
        )

    def record_execution(
        self,
        symbol: str,
        market: Market,
        expected_price: float,
        actual_price: float,
        direction: str,
        session: str = "regular",
    ) -> SlippageRecord:
        """Record an execution and its slippage."""
        slippage = abs(actual_price - expected_price)
        slippage_pct = (slippage / expected_price) * 100 if expected_price else 0.0

        record = SlippageRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=market,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_pct=slippage_pct,
            direction=direction,
            session=session,
        )

        self._records.append(record)
        self._update_stats(record)

        # Check for anomaly
        expected = self.get_expected_slippage(market, session)
        if slippage_pct > expected * 3:
            logger.warning(
                f"High slippage alert: {symbol} {slippage_pct:.3f}% "
                f"(expected ~{expected:.3f}%)"
            )

        return record

    def _update_stats(self, record: SlippageRecord) -> None:
        """Update aggregated statistics."""
        market_key = (
            record.market.value
            if hasattr(record.market, "value")
            else str(record.market)
        )

        self._market_stats[market_key]["count"] += 1
        self._market_stats[market_key]["total_slippage"] += record.slippage_pct
        self._market_stats[market_key]["max_slippage"] = max(
            self._market_stats[market_key]["max_slippage"],
            record.slippage_pct,
        )

        self._session_stats[record.session]["count"] += 1
        self._session_stats[record.session]["total_slippage"] += record.slippage_pct

    def get_average_slippage(self, market: Market) -> float:
        """Get average slippage for a market."""
        market_key = market.value if hasattr(market, "value") else str(market)
        stats = self._market_stats.get(market_key, {})

        if stats.get("count", 0) == 0:
            return 0.02  # Default 0.02%

        return stats["total_slippage"] / stats["count"]

    def get_expected_slippage(
        self, market: Market, session: str = "regular"
    ) -> float:
        """
        Get expected slippage based on historical data.

        Combines market average with session multiplier.
        """
        base = self.get_average_slippage(market)

        # Session multipliers (pre/after hours are worse)
        session_multipliers = {
            "pre_market": 2.5,
            "regular": 1.0,
            "after_hours": 2.0,
            "overnight": 1.5,
        }

        multiplier = session_multipliers.get(session, 1.0)

        return base * multiplier

    def get_cost_engine_overrides(self) -> Dict[str, float]:
        """
        Get slippage overrides for CostEngine.

        Returns dict of market -> adjusted slippage percentage.
        Only returns values that differ significantly from defaults.
        """
        overrides = {}

        # Default slippage assumptions (from CostEngine.MARKET_COSTS)
        defaults = {
            "us_stocks": 0.02,
            "uk_stocks": 0.03,
            "forex_majors": 0.01,
            "forex_crosses": 0.015,
            "us_futures": 0.01,
            "commodities": 0.015,
        }

        for market_key, stats in self._market_stats.items():
            if stats["count"] < 10:
                continue  # Not enough data

            avg = stats["total_slippage"] / stats["count"]
            default = defaults.get(market_key, 0.02)

            # If actual is >20% different from default, override
            if default > 0 and abs(avg - default) / default > 0.2:
                overrides[market_key] = avg
                logger.info(
                    f"Slippage override for {market_key}: "
                    f"{default:.3f}% -> {avg:.3f}%"
                )

        return overrides

    def get_stats(self) -> dict:
        """Get all slippage statistics."""
        return {
            "by_market": dict(self._market_stats),
            "by_session": dict(self._session_stats),
            "total_records": len(self._records),
            "overrides": self.get_cost_engine_overrides(),
        }

    def prune_old_records(self) -> int:
        """Remove records older than lookback period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        original_count = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        pruned = original_count - len(self._records)

        if pruned > 0:
            self._recalculate_stats()

        return pruned

    def _recalculate_stats(self) -> None:
        """Recalculate all stats from records."""
        self._market_stats.clear()
        self._session_stats.clear()

        for record in self._records:
            self._update_stats(record)
