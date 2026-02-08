"""
NEXUS Dynamic Heat Manager

Tracks portfolio-level risk (heat) and prevents overexposure.
Heat = sum of all position risks as % of equity.

- Tracks total portfolio heat, heat by market (max 10% per market), heat by sector
- Dynamically adjusts heat limit based on daily P&L
- Provides can_add_position checks before new trades
- Thread-safe position tracking
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from nexus.config.settings import get_settings
from nexus.core.enums import Market, Direction
from nexus.core.models import HeatCheckResult, HeatSummary, TrackedPosition

logger = logging.getLogger(__name__)

# Defaults (overridable via settings)
BASE_HEAT_LIMIT = 25.0
MAX_HEAT_LIMIT = 35.0
MIN_HEAT_LIMIT = 15.0
MAX_PER_MARKET = 10.0
MAX_CORRELATED = 3


@dataclass
class HeatStatus:
    """Current heat status for consumers (e.g. Position Manager)."""
    current_heat: float


class DynamicHeatManager:
    """
    Manage portfolio heat with dynamic limits.

    Heat limit expands when profitable, contracts when losing.
    Tracks heat by market and sector for concentration and correlation awareness.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.base_heat_limit = getattr(self.settings, "base_heat_limit", BASE_HEAT_LIMIT)
        self.max_heat_limit = getattr(self.settings, "max_heat_limit", MAX_HEAT_LIMIT)
        self.min_heat_limit = getattr(self.settings, "min_heat_limit", MIN_HEAT_LIMIT)
        self.max_per_market = getattr(self.settings, "max_per_market", MAX_PER_MARKET)
        self.max_correlated = getattr(self.settings, "max_correlated", MAX_CORRELATED)

        self._positions: Dict[str, TrackedPosition] = {}
        self._lock = threading.RLock()

    def add_position(self, position: TrackedPosition) -> None:
        """Add a position to track."""
        with self._lock:
            self._positions[position.position_id] = position
        logger.info(
            "Heat manager: added position %s %s (risk_pct=%.2f%%)",
            position.position_id,
            position.symbol,
            position.risk_pct,
        )

    def remove_position(self, position_id: str) -> Optional[TrackedPosition]:
        """Remove a position when closed."""
        with self._lock:
            removed = self._positions.pop(position_id, None)
        if removed:
            logger.info(
                "Heat manager: removed position %s %s",
                position_id,
                removed.symbol,
            )
        return removed

    def update_position_price(self, position_id: str, current_price: float) -> None:
        """Update current price for P&L calculation."""
        with self._lock:
            pos = self._positions.get(position_id)
            if pos:
                pos.current_price = current_price

    def get_current_heat(self) -> float:
        """Total portfolio heat as percentage (sum of all position risk_pct)."""
        with self._lock:
            return sum(p.risk_pct for p in self._positions.values())

    def get_heat_status(self) -> HeatStatus:
        """Return current heat status for Position Manager and other consumers."""
        return HeatStatus(current_heat=self.get_current_heat())

    def get_heat_by_market(self) -> Dict[Market, float]:
        """Heat breakdown by market."""
        with self._lock:
            by_market: Dict[Market, float] = {}
            for pos in self._positions.values():
                m = pos.market
                by_market[m] = by_market.get(m, 0.0) + pos.risk_pct
            return by_market

    def get_heat_by_sector(self) -> Dict[str, float]:
        """Heat breakdown by sector."""
        with self._lock:
            by_sector: Dict[str, float] = {}
            for pos in self._positions.values():
                sector = pos.sector or "unknown"
                by_sector[sector] = by_sector.get(sector, 0.0) + pos.risk_pct
            return by_sector

    def get_heat_limit(self, daily_pnl_pct: float) -> float:
        """
        Calculate current heat limit based on daily P&L.

        Rules:
        - Up 2%+ today: MAX_HEAT_LIMIT (35%)
        - Up 1-2% today: BASE + 5% (30%)
        - Flat (0 to 1%): BASE (25%)
        - Down 0-1%: BASE - 5% (20%)
        - Down 1%+: MIN_HEAT_LIMIT (15%)
        """
        if daily_pnl_pct >= 2.0:
            limit = self.max_heat_limit
        elif daily_pnl_pct >= 1.0:
            limit = self.base_heat_limit + 5.0
        elif daily_pnl_pct >= 0:
            limit = self.base_heat_limit
        elif daily_pnl_pct >= -1.0:
            limit = self.base_heat_limit - 5.0
        else:
            limit = self.min_heat_limit
        return max(self.min_heat_limit, min(limit, self.max_heat_limit))

    def can_add_position(
        self,
        new_risk_pct: float,
        market: Market,
        daily_pnl_pct: float,
        sector: Optional[str] = None,
    ) -> HeatCheckResult:
        """
        Check if a new position can be added.

        Returns HeatCheckResult with allowed, current_heat, heat_after, heat_limit,
        market_heat, market_heat_after, market_limit, rejection_reasons.
        """
        with self._lock:
            current_heat = sum(p.risk_pct for p in self._positions.values())
            heat_limit = self.get_heat_limit(daily_pnl_pct)
            heat_after = current_heat + new_risk_pct

            by_market = self.get_heat_by_market()
            market_heat = by_market.get(market, 0.0)
            market_heat_after = market_heat + new_risk_pct
            market_limit = self.max_per_market

            by_sector: Dict[str, int] = {}
            for p in self._positions.values():
                s = p.sector or "unknown"
                by_sector[s] = by_sector.get(s, 0) + 1

        rejection_reasons: List[str] = []

        if heat_after > heat_limit:
            rejection_reasons.append(
                f"Total heat would exceed limit: {heat_after:.1f}% > {heat_limit:.1f}%"
            )

        if market_heat_after > market_limit:
            rejection_reasons.append(
                f"Market {market.value} heat would exceed limit: "
                f"{market_heat_after:.1f}% > {market_limit:.1f}%"
            )

        if sector:
            sector_count = by_sector.get(sector, 0)
            if sector_count >= self.max_correlated:
                rejection_reasons.append(
                    f"Sector {sector} at max correlated positions: "
                    f"{sector_count} >= {self.max_correlated}"
                )

        allowed = len(rejection_reasons) == 0

        if not allowed and current_heat >= heat_limit * 0.9:
            logger.warning(
                "Heat manager: near limit (%.1f%% / %.1f%%), rejection: %s",
                current_heat,
                heat_limit,
                "; ".join(rejection_reasons),
            )

        return HeatCheckResult(
            allowed=allowed,
            current_heat=current_heat,
            heat_after=heat_after,
            heat_limit=heat_limit,
            market_heat=market_heat,
            market_heat_after=market_heat_after,
            market_limit=market_limit,
            rejection_reasons=rejection_reasons,
        )

    def get_portfolio_summary(self, daily_pnl_pct: float = 0.0) -> HeatSummary:
        """
        Get complete portfolio heat summary.

        Returns HeatSummary with total_heat, heat_limit, headroom, position_count,
        heat_by_market (str keys), heat_by_sector, positions (as dicts), warnings.
        """
        with self._lock:
            total_heat = sum(p.risk_pct for p in self._positions.values())
            heat_limit = self.get_heat_limit(daily_pnl_pct)
            headroom = max(0.0, heat_limit - total_heat)
            position_count = len(self._positions)

            by_market_raw = self.get_heat_by_market()
            heat_by_market = {
                (m.value if hasattr(m, "value") else str(m)): v
                for m, v in by_market_raw.items()
            }
            heat_by_sector = self.get_heat_by_sector()
            positions = [p.to_dict() for p in self._positions.values()]
            warnings: List[str] = []

        if total_heat >= heat_limit * 0.9:
            warnings.append(
                f"Near heat limit: {total_heat:.1f}% of {heat_limit:.1f}%"
            )
        for market_key, pct in heat_by_market.items():
            if pct > self.max_per_market:
                warnings.append(
                    f"Market {market_key} over limit: {pct:.1f}% > {self.max_per_market}%"
                )

        return HeatSummary(
            total_heat=total_heat,
            heat_limit=heat_limit,
            headroom=headroom,
            position_count=position_count,
            heat_by_market=heat_by_market,
            heat_by_sector=heat_by_sector,
            positions=positions,
            warnings=warnings,
        )
