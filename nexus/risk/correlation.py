"""
NEXUS Correlation Monitor – hidden concentration detector.

Tracks positions by sector and market, detects same-direction concentration,
calculates correlation-adjusted (effective) portfolio risk, and blocks or warns
when limits are approached.
"""

import asyncio
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nexus.config.settings import get_settings
from nexus.core.enums import Direction, Market
from nexus.core.models import CorrelationCheckResult

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Sector mapping (symbol -> sector or currency bucket)
# -----------------------------------------------------------------------------
SECTOR_MAPPING = {
    # US Tech
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # US Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "WFC": "Financials",
    # US Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    # US Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    # ETFs
    "SPY": "Index",
    "QQQ": "Index",
    "IWM": "Index",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    # Forex (by base currency)
    "EUR/USD": "EUR",
    "EUR/GBP": "EUR",
    "EUR/JPY": "EUR",
    "GBP/USD": "GBP",
    "GBP/JPY": "GBP",
    "USD/JPY": "JPY",
    "AUD/USD": "AUD",
    # UK Stocks
    "BP": "Energy",
    "SHEL": "Energy",
    "HSBC": "Financials",
    "LLOY": "Financials",
    "AZN": "Healthcare",
    "GSK": "Healthcare",
}

# Pairs with historically high correlation (> 0.7)
HIGH_CORRELATION_PAIRS = {
    ("AAPL", "MSFT"): 0.85,
    ("AAPL", "QQQ"): 0.90,
    ("MSFT", "QQQ"): 0.88,
    ("NVDA", "AMD"): 0.82,
    ("NVDA", "QQQ"): 0.85,
    ("SPY", "QQQ"): 0.92,
    ("JPM", "BAC"): 0.88,
    ("JPM", "XLF"): 0.90,
    ("XOM", "CVX"): 0.85,
    ("XOM", "XLE"): 0.92,
    ("EUR/USD", "EUR/GBP"): 0.75,
    ("EUR/USD", "GBP/USD"): -0.80,
    ("BP", "SHEL"): 0.88,
    ("HSBC", "LLOY"): 0.82,
}


@dataclass
class PositionCorrelationInfo:
    """Lightweight position info for correlation tracking."""

    position_id: str
    symbol: str
    market: Market
    direction: Direction
    risk_pct: float
    sector: str


class CorrelationMonitor:
    """
    Monitors position correlation to prevent hidden concentration risk.

    KEY INSIGHT: 6 positions at 1% each doesn't mean 6% risk if they're
    all correlated. With 0.85 correlation, effective risk could be higher.

    FORMULA (per spec):
    Effective Risk = Nominal Risk × √(N × Average Correlation)
    per sector, where N = number of positions, avg_corr = average pairwise correlation.
    """

    def __init__(self, settings: Any = None):
        self.settings = settings or get_settings()
        self.max_sector_positions = getattr(
            self.settings, "max_sector_positions", 3
        )
        self.max_same_direction = getattr(
            self.settings, "max_same_direction_per_market", 3
        )
        self.correlation_warning = getattr(
            self.settings, "correlation_warning_threshold", 0.7
        )
        self.max_risk_multiplier = getattr(
            self.settings, "max_effective_risk_multiplier", 2.0
        )
        self._positions: Dict[str, PositionCorrelationInfo] = {}
        self._lock = threading.RLock()

        # Dynamic correlation state
        self._correlation_cache: Dict[str, float] = {}
        self._last_recalc: Optional[datetime] = None
        self._recalc_running = False
        self._recalc_task: Optional[asyncio.Task] = None

    def add_position(
        self,
        position_id: str,
        symbol: str,
        market: Market,
        direction: Direction,
        risk_pct: float,
        sector: Optional[str] = None,
    ) -> None:
        """
        Add a position to track for correlation.

        Args:
            position_id: Unique identifier
            symbol: Trading symbol
            market: Market enum
            direction: LONG or SHORT
            risk_pct: Risk percentage for this position
            sector: Optional sector override (auto-detected if not provided)
        """
        sec = sector if sector is not None else self.get_sector(symbol)
        with self._lock:
            self._positions[position_id] = PositionCorrelationInfo(
                position_id=position_id,
                symbol=symbol,
                market=market,
                direction=direction,
                risk_pct=risk_pct,
                sector=sec,
            )

    def remove_position(self, position_id: str) -> None:
        """Remove a position from tracking."""
        with self._lock:
            self._positions.pop(position_id, None)

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol, or 'Unknown' if not mapped."""
        return SECTOR_MAPPING.get(symbol.upper(), "Unknown")

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols.

        Returns 0.0 if not in known pairs (assumes uncorrelated).
        Handles both orderings: (A,B) and (B,A).
        """
        a, b = symbol1.upper(), symbol2.upper()
        if a == b:
            return 1.0
        return HIGH_CORRELATION_PAIRS.get(
            (a, b), HIGH_CORRELATION_PAIRS.get((b, a), 0.0)
        )

    def calculate_effective_risk(
        self,
        positions: List[PositionCorrelationInfo],
        new_position: Optional[PositionCorrelationInfo] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate nominal and effective (correlation-adjusted) risk.

        Returns:
            (nominal_risk, effective_risk, multiplier)
        """
        all_positions = list(positions)
        if new_position is not None:
            all_positions.append(new_position)

        if not all_positions:
            return 0.0, 0.0, 1.0

        nominal = sum(p.risk_pct for p in all_positions)

        # Group by sector keeping positions (for symbol -> correlation lookup)
        sector_positions: Dict[str, List[PositionCorrelationInfo]] = {}
        for p in all_positions:
            if p.sector not in sector_positions:
                sector_positions[p.sector] = []
            sector_positions[p.sector].append(p)

        effective = 0.0
        for sector, positions_in_sector in sector_positions.items():
            n = len(positions_in_sector)
            sector_nominal = sum(p.risk_pct for p in positions_in_sector)
            if n <= 0:
                continue
            if n == 1:
                effective += sector_nominal
            else:
                # Actual pairwise correlations in this sector
                symbols = [p.symbol for p in positions_in_sector]
                correlations: List[float] = []
                for i, s1 in enumerate(symbols):
                    for s2 in symbols[i + 1 :]:
                        c = self.get_correlation(s1, s2)
                        correlations.append(abs(c))
                if len(correlations) > 0:
                    avg_corr = sum(correlations) / len(correlations)
                else:
                    avg_corr = 0.5  # Default assumption when no data
                if avg_corr <= 0:
                    avg_corr = 0.1  # Minimum assumption to avoid math errors
                # Effective Risk = Nominal Risk × √(N × Average Correlation)
                corr_factor = math.sqrt(n * avg_corr)
                effective += sector_nominal * corr_factor

        multiplier = effective / nominal if nominal > 0 else 1.0
        return nominal, effective, multiplier

    def check_new_position(
        self,
        symbol: str,
        market: Market,
        direction: Direction,
        risk_pct: float,
        sector: Optional[str] = None,
    ) -> CorrelationCheckResult:
        """
        Check if a new position would create dangerous concentration.

        Call this BEFORE opening a new position.

        Returns:
            CorrelationCheckResult with allowed status and analysis.
        """
        sec = sector if sector is not None else self.get_sector(symbol)
        new_info = PositionCorrelationInfo(
            position_id="_pending",
            symbol=symbol,
            market=market,
            direction=direction,
            risk_pct=risk_pct,
            sector=sec,
        )

        with self._lock:
            positions = list(self._positions.values())

        # Sector counts including new
        sector_count: Dict[str, int] = {}
        for p in positions:
            sector_count[p.sector] = sector_count.get(p.sector, 0) + 1
        sector_count[sec] = sector_count.get(sec, 0) + 1

        # Market+direction counts including new
        direction_count: Dict[str, int] = {}
        for p in positions:
            key = f"{p.market.value}_{p.direction.value}"
            direction_count[key] = direction_count.get(key, 0) + 1
        key_new = f"{market.value}_{direction.value}"
        direction_count[key_new] = direction_count.get(key_new, 0) + 1

        nominal, effective, multiplier = self.calculate_effective_risk(
            positions, new_info
        )

        high_correlation_pairs: List[Tuple[str, str, float]] = []
        for p in positions:
            corr = self.get_correlation(p.symbol, symbol)
            if abs(corr) >= self.correlation_warning:
                high_correlation_pairs.append((p.symbol, symbol, corr))

        warnings: List[str] = []
        rejection_reasons: List[str] = []

        # 1) Sector check
        sector_count_new = sector_count.get(sec, 0)
        if sector_count_new > self.max_sector_positions:
            rejection_reasons.append(
                f"Sector '{sec}' has {sector_count_new} positions "
                f"(max {self.max_sector_positions})"
            )
        elif sector_count_new == self.max_sector_positions:
            warnings.append(
                f"Sector '{sec}' at limit ({self.max_sector_positions} positions)"
            )

        # 2) Direction check
        dir_count = direction_count.get(key_new, 0)
        if dir_count > self.max_same_direction:
            rejection_reasons.append(
                f"Same market+direction has {dir_count} positions "
                f"(max {self.max_same_direction})"
            )
        elif dir_count == self.max_same_direction:
            warnings.append(
                f"Market+direction at limit ({self.max_same_direction})"
            )

        # 3) Correlation warnings (high-correlation pairs)
        for s1, s2, c in high_correlation_pairs:
            warnings.append(
                f"High correlation between {s1} and {s2} ({c:.2f})"
            )

        # 4) Effective risk check
        if nominal > 0 and multiplier > self.max_risk_multiplier:
            rejection_reasons.append(
                f"Effective risk multiplier {multiplier:.2f} exceeds "
                f"max {self.max_risk_multiplier}"
            )
        elif multiplier > 1.5:
            warnings.append(
                f"Effective risk multiplier elevated: {multiplier:.2f}"
            )

        allowed = len(rejection_reasons) == 0

        return CorrelationCheckResult(
            allowed=allowed,
            nominal_risk_pct=nominal,
            effective_risk_pct=effective,
            risk_multiplier=multiplier,
            sector_count=sector_count,
            direction_count=direction_count,
            high_correlation_pairs=high_correlation_pairs,
            warnings=warnings,
            rejection_reasons=rejection_reasons,
        )

    def get_concentration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current concentration.

        Returns:
            position_count, nominal_risk, effective_risk, by_sector,
            by_market_direction, high_correlation_pairs, warnings
        """
        with self._lock:
            positions = list(self._positions.values())

        if not positions:
            return {
                "position_count": 0,
                "nominal_risk": 0.0,
                "effective_risk": 0.0,
                "by_sector": {},
                "by_market_direction": {},
                "high_correlation_pairs": [],
                "warnings": [],
            }

        nominal, effective, _ = self.calculate_effective_risk(positions)

        by_sector: Dict[str, Dict[str, Any]] = {}
        for p in positions:
            if p.sector not in by_sector:
                by_sector[p.sector] = {"count": 0, "risk": 0.0}
            by_sector[p.sector]["count"] += 1
            by_sector[p.sector]["risk"] += p.risk_pct

        by_market_direction: Dict[str, Dict[str, Any]] = {}
        for p in positions:
            key = f"{p.market.value}_{p.direction.value}"
            if key not in by_market_direction:
                by_market_direction[key] = {"count": 0, "risk": 0.0}
            by_market_direction[key]["count"] += 1
            by_market_direction[key]["risk"] += p.risk_pct

        high_correlation_pairs: List[Tuple[str, str, float]] = []
        symbols = [p.symbol for p in positions]
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1 :]:
                corr = self.get_correlation(s1, s2)
                if abs(corr) >= self.correlation_warning:
                    high_correlation_pairs.append((s1, s2, corr))

        warnings: List[str] = []
        for sector, data in by_sector.items():
            if data["count"] >= self.max_sector_positions:
                warnings.append(
                    f"Sector {sector}: {data['count']} positions (limit "
                    f"{self.max_sector_positions})"
                )
        for key, data in by_market_direction.items():
            if data["count"] >= self.max_same_direction:
                warnings.append(
                    f"Market+direction {key}: {data['count']} positions "
                    f"(limit {self.max_same_direction})"
                )

        return {
            "position_count": len(positions),
            "nominal_risk": nominal,
            "effective_risk": effective,
            "by_sector": by_sector,
            "by_market_direction": by_market_direction,
            "high_correlation_pairs": high_correlation_pairs,
            "warnings": warnings,
        }

    # -----------------------------------------------------------------
    # Dynamic / real-time correlation methods
    # -----------------------------------------------------------------

    async def _calculate_correlation(self, sym1: str, sym2: str) -> float:
        """
        Calculate correlation between two symbols.

        Override this method to pull live data from a market data provider.
        Default implementation falls back to the static lookup table.
        """
        return self.get_correlation(sym1, sym2)

    async def recalculate_all_correlations(self) -> Dict[str, float]:
        """
        Recalculate correlation between all open positions.
        Call this periodically (e.g., every hour) or after adding positions.

        Returns:
            {"pair_key": correlation_value}
        """
        with self._lock:
            positions = list(self._positions.values())

        if len(positions) < 2:
            return {}

        symbols = list(set(p.symbol for p in positions))
        correlations: Dict[str, float] = {}

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                try:
                    corr = await self._calculate_correlation(sym1, sym2)
                    pair_key = f"{sym1}_{sym2}"
                    correlations[pair_key] = corr

                    # Alert if correlation spiked
                    if abs(corr) > 0.85:
                        logger.warning(
                            f"High correlation detected: {sym1} <-> {sym2} = {corr:.2f}"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not calculate correlation {sym1}/{sym2}: {e}"
                    )

        self._correlation_cache = correlations
        self._last_recalc = datetime.now(timezone.utc)

        return correlations

    def get_effective_portfolio_risk(self) -> dict:
        """
        Calculate effective portfolio risk accounting for correlations.

        Formula: Effective Risk = Nominal Risk * sqrt(N * Average Correlation)

        Returns:
            {
                "nominal_risk": float,
                "average_correlation": float,
                "correlation_factor": float,
                "effective_risk": float,
                "warning": str or None,
            }
        """
        with self._lock:
            positions = list(self._positions.values())

        if not positions:
            return {"nominal_risk": 0, "effective_risk": 0}

        # Sum nominal risk
        nominal_risk = sum(p.risk_pct for p in positions)

        n = len(positions)

        if n < 2 or not self._correlation_cache:
            return {
                "nominal_risk": nominal_risk,
                "average_correlation": 0,
                "correlation_factor": 1.0,
                "effective_risk": nominal_risk,
            }

        # Calculate average absolute correlation
        correlations = list(self._correlation_cache.values())
        avg_corr = (
            sum(abs(c) for c in correlations) / len(correlations)
            if correlations
            else 0
        )

        # Correlation factor: sqrt(N * avg_correlation)
        # This approximates how correlated moves amplify risk
        corr_factor = (n * avg_corr) ** 0.5 if avg_corr > 0 else 1.0

        effective_risk = nominal_risk * corr_factor

        warning = None
        if effective_risk > nominal_risk * 1.5:
            warning = f"Correlation amplifying risk by {corr_factor:.1f}x"

        return {
            "nominal_risk": nominal_risk,
            "position_count": n,
            "average_correlation": round(avg_corr, 3),
            "correlation_factor": round(corr_factor, 2),
            "effective_risk": round(effective_risk, 2),
            "risk_amplification": round((corr_factor - 1) * 100, 1),
            "warning": warning,
        }

    def should_block_new_position(
        self, symbol: str, risk_amount: float, max_effective_risk: float
    ) -> dict:
        """
        Check if adding a new position would exceed effective risk limits.

        Args:
            symbol: Symbol to add
            risk_amount: Risk amount of new position
            max_effective_risk: Maximum allowed effective risk

        Returns:
            {
                "allowed": bool,
                "reason": str,
                "current_effective_risk": float,
                "projected_effective_risk": float,
            }
        """
        current = self.get_effective_portfolio_risk()
        current_effective = current.get("effective_risk", 0)

        # Estimate correlation with existing positions
        # Assume new position has average correlation with existing
        avg_corr = current.get("average_correlation", 0.5)

        # Project new effective risk (simplified)
        new_nominal = current.get("nominal_risk", 0) + risk_amount
        new_n = current.get("position_count", 0) + 1
        new_corr_factor = (new_n * avg_corr) ** 0.5 if avg_corr > 0 else 1.0
        projected_effective = new_nominal * new_corr_factor

        if projected_effective > max_effective_risk:
            return {
                "allowed": False,
                "reason": f"Effective risk would be {projected_effective:.0f} > max {max_effective_risk:.0f}",
                "current_effective_risk": current_effective,
                "projected_effective_risk": projected_effective,
            }

        return {
            "allowed": True,
            "reason": "Within limits",
            "current_effective_risk": current_effective,
            "projected_effective_risk": projected_effective,
        }

    async def start_periodic_recalc(self, interval_seconds: int = 3600) -> None:
        """Start periodic correlation recalculation."""
        self._recalc_running = True

        async def recalc_loop():
            while self._recalc_running:
                try:
                    await self.recalculate_all_correlations()
                    risk = self.get_effective_portfolio_risk()
                    if risk.get("warning"):
                        logger.warning(f"Correlation warning: {risk['warning']}")
                except Exception as e:
                    logger.error(f"Correlation recalc failed: {e}")
                await asyncio.sleep(interval_seconds)

        self._recalc_task = asyncio.create_task(recalc_loop())

    async def stop_periodic_recalc(self) -> None:
        """Stop periodic recalculation."""
        self._recalc_running = False
        if self._recalc_task:
            self._recalc_task.cancel()
            try:
                await self._recalc_task
            except asyncio.CancelledError:
                pass

    def clear_positions(self) -> None:
        """Clear all tracked positions."""
        with self._lock:
            self._positions.clear()
