"""
NEXUS Dynamic Position Sizer

Calculates position sizes based on:
- ATR-based stop distance
- Signal score/tier multipliers
- Market regime adjustments
- Win streak momentum scaling
- Heat capacity limits
- Intraday compounding (current equity, not starting balance)
"""

import logging
from typing import Any, Optional

from nexus.core.enums import Market, MarketRegime
from nexus.core.models import PositionSize

logger = logging.getLogger(__name__)

# Default config (overridable via settings or constructor)
BASE_RISK_PCT = 1.0
MAX_RISK_PCT = 2.0
MAX_HEAT_LIMIT = 25.0
HEAT_BUFFER_PCT = 5.0  # Leave 5% buffer: only use up to (limit - 5)% for new position


class DynamicPositionSizer:
    """
    Calculate position sizes dynamically.

    Uses current_equity for intraday compounding. Applies score, regime,
    and momentum multipliers; enforces heat capacity and max risk cap.
    """

    # Score tier multipliers
    SCORE_MULTIPLIERS = [
        (85, 100, 1.5),   # Tier A - maximum conviction
        (75, 84, 1.25),   # Tier B - high conviction
        (65, 74, 1.0),    # Tier C - standard
        (50, 64, 0.75),   # Tier D - lower conviction
        (0, 49, 0.5),     # Marginal
    ]

    # Regime multipliers (MarketRegime enum)
    REGIME_MULTIPLIERS = {
        MarketRegime.TRENDING_UP: 1.0,    # Not 1.1 - trade both directions equally
        MarketRegime.TRENDING_DOWN: 1.0,  # Not 0.9 - trade both directions
        MarketRegime.RANGING: 1.0,
        MarketRegime.VOLATILE: 0.5,       # Correct per spec - half size in volatile markets
    }

    def __init__(
        self,
        settings: Any = None,
        base_risk_pct: Optional[float] = None,
        max_risk_pct: Optional[float] = None,
        max_heat_limit: Optional[float] = None,
    ):
        """
        Initialize position sizer.

        Args:
            settings: Optional settings object (e.g. from nexus.config.settings).
                     If provided, uses settings.base_risk_pct, max_risk_pct, max_heat_limit.
            base_risk_pct: Override base risk % (default 1.0).
            max_risk_pct: Override max risk % (default 2.0).
            max_heat_limit: Override max portfolio heat % (default 25.0).
        """
        if settings is not None:
            self.base_risk_pct = getattr(settings, "base_risk_pct", BASE_RISK_PCT)
            self.max_risk_pct = getattr(settings, "max_risk_pct", MAX_RISK_PCT)
            self.heat_limit = getattr(settings, "max_heat_limit", MAX_HEAT_LIMIT)
        else:
            self.base_risk_pct = base_risk_pct if base_risk_pct is not None else BASE_RISK_PCT
            self.max_risk_pct = max_risk_pct if max_risk_pct is not None else MAX_RISK_PCT
            self.heat_limit = max_heat_limit if max_heat_limit is not None else MAX_HEAT_LIMIT

    def calculate_size(
        self,
        starting_balance: float,
        current_equity: float,
        entry_price: float,
        stop_loss: float,
        score: int,
        current_heat: float,
        win_streak: int,
        regime: MarketRegime,
        symbol: str,
        market: Optional[Market] = None,
    ) -> PositionSize:
        """
        Calculate position size for a trade.

        Uses current_equity for risk amount (intraday compounding).
        Enforces heat capacity with 5% buffer and max risk cap.

        Args:
            starting_balance: Day's starting equity.
            current_equity: Current equity including today's P&L.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            score: Opportunity score (0-100).
            current_heat: Current portfolio heat (%).
            win_streak: Consecutive wins for momentum scaling.
            regime: Market regime.
            symbol: Symbol (for logging).
            market: Optional market type for rounding (stocks -> whole shares, forex -> 2 decimals).

        Returns:
            PositionSize with risk_pct, position_size, can_trade, etc.

        Raises:
            ValueError: If stop_loss equals entry_price.
        """
        # Stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            raise ValueError(
                f"[{symbol}] stop_loss equals entry_price: infinite position (entry={entry_price}, stop={stop_loss})"
            )
        stop_distance_pct = (stop_distance / entry_price) * 100.0

        # Heat capacity check
        if current_heat >= self.heat_limit:
            logger.warning(
                "[%s] Rejected: current_heat (%.1f%%) >= heat_limit (%.1f%%)",
                symbol, current_heat, self.heat_limit,
            )
            return self._reject(
                symbol=symbol,
                stop_distance=stop_distance,
                stop_distance_pct=stop_distance_pct,
                current_heat=current_heat,
                reason=f"Current heat {current_heat:.1f}% >= limit {self.heat_limit:.1f}%",
            )

        # Effective heat budget: leave 5% buffer
        usable_heat = min(
            self.heat_limit - current_heat,
            max(0, self.heat_limit - HEAT_BUFFER_PCT),
        )

        # Score multiplier
        score_mult = self._score_multiplier(score)
        regime_val = regime.value if hasattr(regime, "value") else str(regime)
        regime_enum = MarketRegime(regime_val) if isinstance(regime_val, str) else regime
        regime_mult = self.REGIME_MULTIPLIERS.get(regime_enum, 1.0)

        # Momentum: win_streak >= 2 -> boost 1 + (win_streak * 0.1), max 1.3
        if win_streak >= 2:
            momentum_mult = min(1.0 + (win_streak * 0.1), 1.3)
        else:
            momentum_mult = 1.0

        # Raw risk % before caps
        risk_pct = self.base_risk_pct * score_mult * regime_mult * momentum_mult

        # Cap by usable heat
        if risk_pct > usable_heat:
            risk_pct = usable_heat
            logger.debug(
                "[%s] Risk capped by heat budget: usable_heat=%.2f%%",
                symbol, usable_heat,
            )

        # Cap by max risk
        if risk_pct > self.max_risk_pct:
            risk_pct = self.max_risk_pct
            logger.debug("[%s] Risk capped at max_risk_pct=%.2f%%", symbol, self.max_risk_pct)

        # Risk amount and position size (use current_equity for intraday compounding)
        risk_amount = current_equity * (risk_pct / 100.0)
        position_size = risk_amount / stop_distance
        position_value = position_size * entry_price

        if position_size <= 0:
            logger.warning("[%s] Rejected: calculated position_size <= 0", symbol)
            return self._reject(
                symbol=symbol,
                stop_distance=stop_distance,
                stop_distance_pct=stop_distance_pct,
                current_heat=current_heat,
                reason="Calculated position size <= 0",
            )

        # Round position size: whole shares for stocks, 2 decimals for forex
        position_size = self._round_position_size(position_size, market, symbol)
        if position_size <= 0:
            logger.warning("[%s] Rejected: rounded position_size <= 0", symbol)
            return self._reject(
                symbol=symbol,
                stop_distance=stop_distance,
                stop_distance_pct=stop_distance_pct,
                current_heat=current_heat,
                reason="Rounded position size <= 0",
            )

        # Recompute position_value and risk_amount after rounding
        position_value = position_size * entry_price
        risk_amount = position_size * stop_distance
        risk_pct = (risk_amount / current_equity) * 100.0
        heat_after_trade = current_heat + risk_pct

        logger.debug(
            "[%s] size=%.4f value=%.2f risk_pct=%.2f risk_amount=%.2f "
            "score_mult=%.2f regime_mult=%.2f momentum_mult=%.2f heat_after=%.2f",
            symbol, position_size, position_value, risk_pct, risk_amount,
            score_mult, regime_mult, momentum_mult, heat_after_trade,
        )

        return PositionSize(
            risk_pct=risk_pct,
            risk_amount=risk_amount,
            position_size=position_size,
            position_value=position_value,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            score_multiplier=score_mult,
            regime_multiplier=regime_mult,
            momentum_multiplier=momentum_mult,
            heat_after_trade=heat_after_trade,
            can_trade=True,
            rejection_reason=None,
        )

    def _score_multiplier(self, score: int) -> float:
        for low, high, mult in self.SCORE_MULTIPLIERS:
            if low <= score <= high:
                return mult
        return 0.5  # fallback marginal

    def _round_position_size(self, size: float, market: Optional[Market], symbol: str) -> float:
        """Whole shares for stocks, 2 decimals for forex."""
        if market is not None:
            if market in (Market.FOREX_MAJORS, Market.FOREX_CROSSES):
                return round(size, 2)
        # Default: whole units (stocks)
        return round(size, 0)

    def _reject(
        self,
        symbol: str,
        stop_distance: float,
        stop_distance_pct: float,
        current_heat: float,
        reason: str,
    ) -> PositionSize:
        return PositionSize(
            risk_pct=0.0,
            risk_amount=0.0,
            position_size=0.0,
            position_value=0.0,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            score_multiplier=0.0,
            regime_multiplier=0.0,
            momentum_multiplier=0.0,
            heat_after_trade=current_heat,
            can_trade=False,
            rejection_reason=reason,
        )
