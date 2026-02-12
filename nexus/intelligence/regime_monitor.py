"""
NEXUS Continuous Regime Monitor

Monitors market regime continuously and alerts on changes.
Can trigger position adjustments when regime shifts.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, List, Any
from dataclasses import dataclass

from nexus.core.enums import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RegimeChange:
    """Records a regime transition."""
    timestamp: datetime
    previous: MarketRegime
    current: MarketRegime
    symbol: str
    confidence: float  # 0-1, how confident in new regime

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "previous": self.previous.value,
            "current": self.current.value,
            "symbol": self.symbol,
            "confidence": self.confidence,
        }


class ContinuousRegimeMonitor:
    """
    Monitors regime continuously rather than point-in-time.

    Key features:
    - Checks regime every N minutes (default: 5)
    - Requires confirmation before declaring change (avoids whipsaws)
    - Triggers callbacks on regime change
    - Suggests position adjustments
    """

    # Regime-incompatible edges (should close or reduce when regime changes)
    REGIME_EDGE_CONFLICTS = {
        MarketRegime.TRENDING_UP: ["BOLLINGER_TOUCH", "RSI_EXTREME"],  # Mean reversion risky in trends
        MarketRegime.TRENDING_DOWN: ["ORB", "POWER_HOUR"],  # Momentum long risky in downtrends
        MarketRegime.VOLATILE: ["LONDON_OPEN", "NY_OPEN", "ORB"],  # Breakouts unreliable
        MarketRegime.RANGING: ["EARNINGS_DRIFT"],  # Drift needs trend
    }

    def __init__(
        self,
        regime_detector: Any,
        check_interval_seconds: int = 300,  # 5 minutes
        confirmation_count: int = 2,  # Need 2 consecutive readings to confirm change
    ):
        self.detector = regime_detector
        self.check_interval = check_interval_seconds
        self.confirmation_count = confirmation_count

        self._current_regime: Optional[MarketRegime] = None
        self._pending_regime: Optional[MarketRegime] = None
        self._pending_count: int = 0
        self._last_check: Optional[datetime] = None
        self._history: List[RegimeChange] = []

        # Callbacks
        self._on_change_callbacks: List[Callable[[RegimeChange], None]] = []

        self._running = False
        self._task: Optional[asyncio.Task] = None

    def on_regime_change(self, callback: Callable[[RegimeChange], None]) -> None:
        """Register callback for regime changes."""
        self._on_change_callbacks.append(callback)

    async def check_regime(self, symbol: str = "SPY") -> dict:
        """
        Check current regime and detect changes.

        Returns:
            {
                "current_regime": str,
                "changed": bool,
                "previous": str or None,
                "pending_change": str or None,
                "confidence": float,
            }
        """
        try:
            new_regime = await self.detector.detect_regime(symbol)
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {"current_regime": self._current_regime.value if self._current_regime else "unknown", "changed": False}

        self._last_check = datetime.now(timezone.utc)

        # First check - initialize
        if self._current_regime is None:
            self._current_regime = new_regime
            logger.info(f"Initial regime: {new_regime.value}")
            return {"current_regime": new_regime.value, "changed": False, "confidence": 1.0}

        # No change
        if new_regime == self._current_regime:
            self._pending_regime = None
            self._pending_count = 0
            return {"current_regime": new_regime.value, "changed": False, "confidence": 1.0}

        # Potential change - need confirmation
        if new_regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = new_regime
            self._pending_count = 1

        # Confirmed change
        if self._pending_count >= self.confirmation_count:
            previous = self._current_regime
            self._current_regime = new_regime
            self._pending_regime = None
            self._pending_count = 0

            change = RegimeChange(
                timestamp=datetime.now(timezone.utc),
                previous=previous,
                current=new_regime,
                symbol=symbol,
                confidence=1.0,
            )
            self._history.append(change)

            logger.warning(f"REGIME CHANGE: {previous.value} -> {new_regime.value}")

            # Fire callbacks
            for callback in self._on_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(change)
                    else:
                        callback(change)
                except Exception as e:
                    logger.error(f"Regime change callback failed: {e}")

            return {
                "current_regime": new_regime.value,
                "changed": True,
                "previous": previous.value,
                "confidence": 1.0,
            }

        # Pending change (not yet confirmed)
        confidence = self._pending_count / self.confirmation_count
        return {
            "current_regime": self._current_regime.value,
            "changed": False,
            "pending_change": new_regime.value,
            "pending_count": self._pending_count,
            "confirmation_needed": self.confirmation_count,
            "confidence": confidence,
        }

    def get_conflicting_edges(self) -> List[str]:
        """Get edges that conflict with current regime."""
        if self._current_regime is None:
            return []
        return self.REGIME_EDGE_CONFLICTS.get(self._current_regime, [])

    def get_position_recommendations(self, open_positions: List[dict]) -> List[dict]:
        """
        Get recommendations for open positions based on regime.

        Returns list of recommendations:
        [{"position_id": ..., "action": "CLOSE" | "REDUCE", "reason": ...}]
        """
        if self._current_regime is None:
            return []

        conflicting = set(self.get_conflicting_edges())
        recommendations = []

        for pos in open_positions:
            edge = pos.get("primary_edge", pos.get("edge_type", ""))
            if edge in conflicting:
                # In volatile regime, close positions
                if self._current_regime == MarketRegime.VOLATILE:
                    recommendations.append({
                        "position_id": pos.get("position_id", pos.get("id")),
                        "symbol": pos.get("symbol"),
                        "action": "CLOSE",
                        "reason": f"Regime changed to VOLATILE, {edge} edge unreliable",
                    })
                else:
                    # In other regimes, reduce size
                    recommendations.append({
                        "position_id": pos.get("position_id", pos.get("id")),
                        "symbol": pos.get("symbol"),
                        "action": "REDUCE",
                        "reason": f"{edge} edge conflicts with {self._current_regime.value} regime",
                    })

        return recommendations

    async def start(self, symbol: str = "SPY") -> None:
        """Start continuous monitoring loop."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting continuous regime monitor (interval: {self.check_interval}s)")

        async def monitor_loop():
            while self._running:
                try:
                    await self.check_regime(symbol)
                except Exception as e:
                    logger.error(f"Regime monitor error: {e}")
                await asyncio.sleep(self.check_interval)

        self._task = asyncio.create_task(monitor_loop())

    async def stop(self) -> None:
        """Stop monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Regime monitor stopped")

    @property
    def current_regime(self) -> Optional[MarketRegime]:
        return self._current_regime

    def get_history(self, limit: int = 20) -> List[dict]:
        """Get regime change history."""
        return [c.to_dict() for c in self._history[-limit:]]
