"""
NEXUS Correlation Monitor – hidden concentration detector.

Tracks positions by sector and market, detects same-direction concentration,
calculates correlation-adjusted (effective) portfolio risk, and blocks or warns
when limits are approached.
"""

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nexus.config.settings import get_settings
from nexus.core.enums import Direction, Market
from nexus.core.models import CorrelationCheckResult

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
    all correlated. With 0.85 correlation, effective risk could be 13%+.

    FORMULA (simplified):
    Effective Risk = Nominal Risk × √(1 + (N-1) × Average Correlation)
    per sector, where N = number of positions in that sector.
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

        sector_risks: Dict[str, List[float]] = {}
        for p in all_positions:
            if p.sector not in sector_risks:
                sector_risks[p.sector] = []
            sector_risks[p.sector].append(p.risk_pct)

        effective = 0.0
        for sector, risks in sector_risks.items():
            n = len(risks)
            sector_nominal = sum(risks)
            if n == 1:
                effective += sector_nominal
            else:
                avg_corr = 0.7
                corr_factor = math.sqrt(1 + (n - 1) * avg_corr)
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

    def clear_positions(self) -> None:
        """Clear all tracked positions."""
        with self._lock:
            self._positions.clear()
