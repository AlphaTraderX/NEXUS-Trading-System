"""
NEXUS Dynamic Position Sizer
Calculates position size based on ATR, conviction, and market conditions.

KEY PRINCIPLES:
1. Risk a percentage of equity, not a fixed dollar amount
2. Adjust for volatility (ATR-based stops)
3. Scale with conviction (higher score = larger position)
4. Account for regime (volatile = smaller positions)
5. Compound intraday (use current equity, not starting balance)

FORMULA:
Position Size = (Equity × Risk%) / (Entry - Stop)
Risk% = Base Risk × Score Multiplier × Regime Multiplier × Momentum Multiplier
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RiskMode(Enum):
    """Trading risk modes."""
    CONSERVATIVE = "conservative"  # 0.75% base, careful
    STANDARD = "standard"          # 1.0% base, normal
    AGGRESSIVE = "aggressive"      # 1.25% base, confident
    MAXIMUM = "maximum"            # 1.5% base, high conviction only


@dataclass
class PositionSize:
    """Calculated position size and risk details."""
    units: float              # Number of shares/contracts/lots
    position_value: float     # Total position value
    risk_amount: float        # Dollar amount at risk
    risk_percent: float       # Percentage of equity at risk
    stop_distance: float      # Distance to stop loss
    stop_distance_pct: float  # Stop distance as percentage

    # Multipliers applied
    base_risk: float
    score_multiplier: float
    regime_multiplier: float
    momentum_multiplier: float
    heat_adjustment: float

    # Limits
    capped: bool              # Was position capped by limits?
    cap_reason: str           # Reason for cap if applied

    def to_dict(self) -> dict:
        return {
            "units": round(self.units, 4),
            "position_value": round(self.position_value, 2),
            "risk_amount": round(self.risk_amount, 2),
            "risk_percent": round(self.risk_percent, 4),
            "stop_distance": round(self.stop_distance, 4),
            "stop_distance_pct": round(self.stop_distance_pct, 4),
            "multipliers": {
                "base_risk": self.base_risk,
                "score": self.score_multiplier,
                "regime": self.regime_multiplier,
                "momentum": self.momentum_multiplier,
                "heat_adjustment": self.heat_adjustment,
            },
            "capped": self.capped,
            "cap_reason": self.cap_reason,
        }


class DynamicPositionSizer:
    """
    Calculate position sizes dynamically based on multiple factors.

    SCALING FACTORS:
    1. Base risk % (from mode)
    2. Score multiplier (conviction)
    3. Regime multiplier (market conditions)
    4. Momentum multiplier (win streak)
    5. Heat adjustment (portfolio capacity)
    """

    # Base risk by mode
    MODE_BASE_RISK = {
        RiskMode.CONSERVATIVE: 0.75,
        RiskMode.STANDARD: 1.0,
        RiskMode.AGGRESSIVE: 1.25,
        RiskMode.MAXIMUM: 1.5,
    }

    # Maximum risk caps by mode
    MODE_MAX_RISK = {
        RiskMode.CONSERVATIVE: 1.25,
        RiskMode.STANDARD: 1.5,
        RiskMode.AGGRESSIVE: 2.0,
        RiskMode.MAXIMUM: 2.5,
    }

    # Score-based multipliers
    SCORE_MULTIPLIERS = {
        (85, 100): 1.5,   # A+ tier: 1.5x
        (80, 84): 1.25,   # A tier: 1.25x
        (70, 79): 1.1,    # B+ tier: 1.1x
        (65, 69): 1.0,    # B tier: 1.0x
        (55, 64): 0.85,   # C+ tier: 0.85x
        (50, 54): 0.75,   # C tier: 0.75x
        (40, 49): 0.5,    # D tier: 0.5x
        (0, 39): 0.0,     # F tier: Don't trade
    }

    # Regime-based multipliers
    REGIME_MULTIPLIERS = {
        "trending_up": 1.0,
        "trending_down": 0.85,
        "ranging": 1.0,
        "volatile": 0.5,
    }

    def __init__(
        self,
        mode: RiskMode = RiskMode.STANDARD,
        scale_with_equity: bool = True,
        momentum_scaling: bool = True,
        max_position_pct: float = 20.0,  # Max 20% of equity in one position
    ):
        """
        Initialize position sizer.

        Args:
            mode: Risk mode (conservative, standard, aggressive, maximum)
            scale_with_equity: If True, use current equity (compounds intraday)
            momentum_scaling: If True, scale up slightly on win streaks
            max_position_pct: Maximum position size as % of equity
        """
        self.mode = mode
        self.scale_with_equity = scale_with_equity
        self.momentum_scaling = momentum_scaling
        self.max_position_pct = max_position_pct

        self.base_risk = self.MODE_BASE_RISK[mode]
        self.max_risk = self.MODE_MAX_RISK[mode]

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        starting_balance: float,
        current_equity: float = None,
        score: int = 50,
        regime: str = "ranging",
        win_streak: int = 0,
        current_heat: float = 0.0,
        max_heat: float = 25.0,
    ) -> PositionSize:
        """
        Calculate position size for a trade.

        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price
            starting_balance: Account starting balance
            current_equity: Current account equity (for intraday compounding)
            score: Opportunity score (0-100)
            regime: Market regime string
            win_streak: Current consecutive wins (for momentum scaling)
            current_heat: Current portfolio heat (% of equity at risk)
            max_heat: Maximum allowed portfolio heat

        Returns:
            PositionSize with all details
        """
        # Use current equity or starting balance
        if self.scale_with_equity and current_equity:
            equity = current_equity
        else:
            equity = starting_balance

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = (stop_distance / entry_price) * 100

        if stop_distance == 0:
            # Invalid stop - return zero position
            return self._zero_position("Stop loss equals entry price")

        # Start with base risk
        risk_pct = self.base_risk

        # 1. Score multiplier
        score_mult = self._get_score_multiplier(score)
        if score_mult == 0:
            return self._zero_position(f"Score {score} below minimum threshold")
        risk_pct *= score_mult

        # 2. Regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        risk_pct *= regime_mult

        # 3. Momentum multiplier (win streak bonus)
        momentum_mult = 1.0
        if self.momentum_scaling and win_streak >= 2:
            # Max 30% boost from win streak
            momentum_mult = min(1.0 + (win_streak * 0.1), 1.3)
            risk_pct *= momentum_mult

        # 4. Heat adjustment (don't exceed portfolio limits)
        heat_remaining = max_heat - current_heat
        heat_mult = 1.0
        cap_reason = ""
        capped = False

        if heat_remaining <= 0:
            return self._zero_position("Portfolio heat limit reached")

        # If adding full position would exceed heat limit, reduce
        if risk_pct > heat_remaining:
            heat_mult = heat_remaining / risk_pct
            risk_pct = heat_remaining
            capped = True
            cap_reason = f"Reduced to fit heat limit ({max_heat}%)"

        # 5. Apply absolute maximum
        if risk_pct > self.max_risk:
            risk_pct = self.max_risk
            capped = True
            cap_reason = f"Capped at mode maximum ({self.max_risk}%)"

        # Calculate position size
        risk_amount = equity * (risk_pct / 100)
        units = risk_amount / stop_distance
        position_value = units * entry_price

        # Check max position size limit
        max_position_value = equity * (self.max_position_pct / 100)
        if position_value > max_position_value:
            # Scale down to max position size
            scale_factor = max_position_value / position_value
            units *= scale_factor
            position_value = max_position_value
            risk_amount = units * stop_distance
            risk_pct = (risk_amount / equity) * 100
            capped = True
            cap_reason = f"Position size capped at {self.max_position_pct}% of equity"

        return PositionSize(
            units=units,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            base_risk=self.base_risk,
            score_multiplier=score_mult,
            regime_multiplier=regime_mult,
            momentum_multiplier=momentum_mult,
            heat_adjustment=heat_mult,
            capped=capped,
            cap_reason=cap_reason,
        )

    def _get_score_multiplier(self, score: int) -> float:
        """Get multiplier based on opportunity score."""
        for (low, high), mult in self.SCORE_MULTIPLIERS.items():
            if low <= score <= high:
                return mult
        return 0.0  # Invalid score

    def _zero_position(self, reason: str) -> PositionSize:
        """Return a zero position with reason."""
        return PositionSize(
            units=0,
            position_value=0,
            risk_amount=0,
            risk_percent=0,
            stop_distance=0,
            stop_distance_pct=0,
            base_risk=self.base_risk,
            score_multiplier=0,
            regime_multiplier=0,
            momentum_multiplier=0,
            heat_adjustment=0,
            capped=True,
            cap_reason=reason,
        )

    def calculate_from_atr(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        atr_multiplier: float = 1.5,
        **kwargs
    ) -> PositionSize:
        """
        Calculate position size using ATR for stop distance.

        Args:
            entry_price: Intended entry price
            atr: Average True Range value
            direction: "long" or "short"
            atr_multiplier: Stop distance as multiple of ATR (default 1.5)
            **kwargs: Additional args passed to calculate()

        Returns:
            PositionSize
        """
        stop_distance = atr * atr_multiplier

        if direction.lower() == "long":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        return self.calculate(
            entry_price=entry_price,
            stop_loss=stop_loss,
            **kwargs
        )

    def get_mode_info(self) -> Dict:
        """Get information about current mode settings."""
        return {
            "mode": self.mode.value,
            "base_risk_pct": self.base_risk,
            "max_risk_pct": self.max_risk,
            "scale_with_equity": self.scale_with_equity,
            "momentum_scaling": self.momentum_scaling,
            "max_position_pct": self.max_position_pct,
        }


# Test the position sizer
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS DYNAMIC POSITION SIZER TEST")
    print("=" * 60)

    # Test each mode
    for mode in RiskMode:
        print(f"\n--- {mode.value.upper()} Mode ---")
        sizer = DynamicPositionSizer(mode=mode)
        info = sizer.get_mode_info()
        print(f"Base risk: {info['base_risk_pct']}%")
        print(f"Max risk: {info['max_risk_pct']}%")

    # Use standard mode for detailed tests
    sizer = DynamicPositionSizer(
        mode=RiskMode.STANDARD,
        scale_with_equity=True,
        momentum_scaling=True
    )

    # Test 1: Basic position sizing
    print("\n--- Test 1: Basic Position Sizing ---")
    result = sizer.calculate(
        entry_price=100.0,
        stop_loss=98.0,      # $2 stop = 2%
        starting_balance=10000.0,
        current_equity=10000.0,
        score=65,            # B tier
        regime="ranging",
        win_streak=0,
        current_heat=0.0,
        max_heat=25.0,
    )

    print(f"Entry: $100, Stop: $98 (2% stop)")
    print(f"Score: 65 (B tier)")
    print(f"Risk: {result.risk_percent:.2f}% (${result.risk_amount:.2f})")
    print(f"Position: {result.units:.2f} units (${result.position_value:.2f})")
    print(f"Score multiplier: {result.score_multiplier}x")

    # Test 2: High conviction trade
    print("\n--- Test 2: High Conviction (A+ Tier) ---")
    result = sizer.calculate(
        entry_price=100.0,
        stop_loss=98.0,
        starting_balance=10000.0,
        current_equity=10500.0,  # Up 5% intraday
        score=90,               # A+ tier
        regime="trending_up",
        win_streak=3,           # On a streak
        current_heat=5.0,
        max_heat=25.0,
    )

    print(f"Score: 90 (A+ tier), Win streak: 3")
    print(f"Using current equity: $10,500")
    print(f"Risk: {result.risk_percent:.2f}% (${result.risk_amount:.2f})")
    print(f"Position: {result.units:.2f} units (${result.position_value:.2f})")
    print(f"Multipliers: score={result.score_multiplier}x, momentum={result.momentum_multiplier}x")

    # Test 3: Volatile market (reduced sizing)
    print("\n--- Test 3: Volatile Market ---")
    result = sizer.calculate(
        entry_price=100.0,
        stop_loss=95.0,         # Wider stop in volatility
        starting_balance=10000.0,
        current_equity=10000.0,
        score=70,
        regime="volatile",      # 0.5x multiplier
        win_streak=0,
        current_heat=0.0,
        max_heat=25.0,
    )

    print(f"Regime: volatile (0.5x multiplier)")
    print(f"Risk: {result.risk_percent:.2f}% (${result.risk_amount:.2f})")
    print(f"Position: {result.units:.2f} units")
    print(f"Regime multiplier: {result.regime_multiplier}x")

    # Test 4: Near heat limit
    print("\n--- Test 4: Near Heat Limit ---")
    result = sizer.calculate(
        entry_price=100.0,
        stop_loss=98.0,
        starting_balance=10000.0,
        current_equity=10000.0,
        score=80,
        regime="ranging",
        win_streak=0,
        current_heat=23.0,      # Only 2% heat remaining
        max_heat=25.0,
    )

    print(f"Current heat: 23%, Max heat: 25%")
    print(f"Risk: {result.risk_percent:.2f}%")
    print(f"Capped: {result.capped}")
    print(f"Reason: {result.cap_reason}")

    # Test 5: Low score (should not trade)
    print("\n--- Test 5: Low Score (No Trade) ---")
    result = sizer.calculate(
        entry_price=100.0,
        stop_loss=98.0,
        starting_balance=10000.0,
        current_equity=10000.0,
        score=35,               # F tier
        regime="ranging",
        win_streak=0,
        current_heat=0.0,
        max_heat=25.0,
    )

    print(f"Score: 35 (F tier)")
    print(f"Units: {result.units}")
    print(f"Capped: {result.capped}")
    print(f"Reason: {result.cap_reason}")

    # Test 6: ATR-based sizing
    print("\n--- Test 6: ATR-Based Sizing ---")
    result = sizer.calculate_from_atr(
        entry_price=100.0,
        atr=2.5,               # $2.50 ATR
        direction="long",
        atr_multiplier=1.5,    # 1.5 ATR stop = $3.75
        starting_balance=10000.0,
        current_equity=10000.0,
        score=65,
        regime="ranging",
    )

    print(f"ATR: $2.50, Multiplier: 1.5x")
    print(f"Stop distance: ${result.stop_distance:.2f}")
    print(f"Risk: {result.risk_percent:.2f}%")
    print(f"Position: {result.units:.2f} units")

    print("\n" + "=" * 60)
    print("POSITION SIZER TEST COMPLETE [OK]")
    print("=" * 60)
