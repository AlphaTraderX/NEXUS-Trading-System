"""
GOD MODE Configuration

Aggressive trading settings for maximum opportunity capture.
USE WITH CAUTION - only when system is validated and profitable.

Key differences from standard mode:
- Higher base risk (1.5% vs 1.0%)
- Dynamic heat expansion (up to 45% when profitable)
- Intraday compounding enabled
- All tiers tradeable (score >= 35)
- Momentum scaling on win streaks
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode presets."""
    CONSERVATIVE = "conservative"  # Learning/validating
    STANDARD = "standard"          # Normal operation
    AGGRESSIVE = "aggressive"      # High conviction
    GOD_MODE = "god_mode"          # Maximum opportunity


@dataclass
class ModeConfig:
    """Configuration for a trading mode."""

    # Risk per trade
    base_risk_pct: float
    max_risk_pct: float
    min_risk_pct: float

    # Portfolio heat
    base_heat_limit: float
    max_heat_limit: float
    heat_expansion_per_pct_profit: float

    # Position limits
    max_positions: int
    max_per_market: int
    max_correlated: int

    # Scoring thresholds
    min_score_to_trade: int

    # Score-based risk multipliers
    score_multipliers: Dict[str, float] = field(default_factory=dict)

    # Compounding
    intraday_compounding: bool = True
    scale_with_equity: bool = True
    momentum_scaling: bool = True

    # Win streak bonuses
    win_streak_bonus_per_win: float = 0.0
    max_win_streak_bonus: float = 0.0

    # Circuit breaker overrides
    daily_loss_stop: float = -3.0
    weekly_loss_stop: float = -6.0
    max_drawdown: float = -10.0


# =============================================================================
# MODE CONFIGURATIONS
# =============================================================================

CONSERVATIVE_CONFIG = ModeConfig(
    base_risk_pct=0.5,
    max_risk_pct=1.0,
    min_risk_pct=0.25,
    base_heat_limit=15.0,
    max_heat_limit=20.0,
    heat_expansion_per_pct_profit=1.0,
    max_positions=5,
    max_per_market=2,
    max_correlated=2,
    min_score_to_trade=60,
    score_multipliers={
        "A": 1.0,
        "B": 0.8,
        "C": 0.6,
        "D": 0.0,
        "F": 0.0,
    },
    intraday_compounding=False,
    scale_with_equity=False,
    momentum_scaling=False,
    win_streak_bonus_per_win=0.0,
    max_win_streak_bonus=0.0,
    daily_loss_stop=-2.0,
    weekly_loss_stop=-4.0,
    max_drawdown=-8.0,
)


STANDARD_CONFIG = ModeConfig(
    base_risk_pct=1.0,
    max_risk_pct=1.5,
    min_risk_pct=0.5,
    base_heat_limit=25.0,
    max_heat_limit=30.0,
    heat_expansion_per_pct_profit=2.0,
    max_positions=8,
    max_per_market=3,
    max_correlated=3,
    min_score_to_trade=50,
    score_multipliers={
        "A": 1.25,
        "B": 1.0,
        "C": 0.75,
        "D": 0.5,
        "F": 0.0,
    },
    intraday_compounding=True,
    scale_with_equity=True,
    momentum_scaling=False,
    win_streak_bonus_per_win=0.05,
    max_win_streak_bonus=0.15,
    daily_loss_stop=-3.0,
    weekly_loss_stop=-6.0,
    max_drawdown=-10.0,
)


AGGRESSIVE_CONFIG = ModeConfig(
    base_risk_pct=1.25,
    max_risk_pct=2.0,
    min_risk_pct=0.5,
    base_heat_limit=30.0,
    max_heat_limit=35.0,
    heat_expansion_per_pct_profit=2.5,
    max_positions=10,
    max_per_market=4,
    max_correlated=4,
    min_score_to_trade=45,
    score_multipliers={
        "A": 1.5,
        "B": 1.25,
        "C": 1.0,
        "D": 0.6,
        "F": 0.0,
    },
    intraday_compounding=True,
    scale_with_equity=True,
    momentum_scaling=True,
    win_streak_bonus_per_win=0.1,
    max_win_streak_bonus=0.25,
    daily_loss_stop=-3.0,
    weekly_loss_stop=-6.0,
    max_drawdown=-10.0,
)


GOD_MODE_CONFIG = ModeConfig(
    # Risk - HIGH
    base_risk_pct=1.5,
    max_risk_pct=2.5,
    min_risk_pct=0.75,
    # Heat - EXPANDABLE
    base_heat_limit=35.0,
    max_heat_limit=45.0,
    heat_expansion_per_pct_profit=3.0,  # +3% heat limit per 1% profit
    # Limits - GENEROUS
    max_positions=12,
    max_per_market=5,
    max_correlated=4,
    # Scoring - TRADE MORE
    min_score_to_trade=35,  # Take D-tier signals
    score_multipliers={
        "A": 1.75,   # Max conviction
        "B": 1.5,
        "C": 1.25,
        "D": 0.75,   # Still trade D-tier
        "F": 0.0,    # Skip F only
    },
    # Full compounding + momentum
    intraday_compounding=True,
    scale_with_equity=True,
    momentum_scaling=True,
    # Strong streak bonuses
    win_streak_bonus_per_win=0.15,  # +15% per consecutive win
    max_win_streak_bonus=0.45,      # Up to +45% bonus
    # Circuit breakers STILL ENFORCED
    daily_loss_stop=-3.0,
    weekly_loss_stop=-6.0,
    max_drawdown=-10.0,
)


# Mode lookup
MODE_CONFIGS = {
    TradingMode.CONSERVATIVE: CONSERVATIVE_CONFIG,
    TradingMode.STANDARD: STANDARD_CONFIG,
    TradingMode.AGGRESSIVE: AGGRESSIVE_CONFIG,
    TradingMode.GOD_MODE: GOD_MODE_CONFIG,
}


def get_mode_config(mode: TradingMode) -> ModeConfig:
    """Get configuration for a trading mode."""
    return MODE_CONFIGS[mode]


# =============================================================================
# GOD MODE POSITION SIZER
# =============================================================================

@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    risk_pct: float
    risk_amount: float
    position_size: float
    equity_used: float
    score_multiplier: float
    streak_bonus: float
    heat_remaining: float
    can_trade: bool
    reason: str = ""


class GodModePositionSizer:
    """
    Position sizer with GOD MODE capabilities.

    Features:
    - Score-based risk scaling
    - Win streak momentum bonuses
    - Intraday compounding
    - Dynamic heat expansion
    """

    def __init__(self, config: ModeConfig):
        self.config = config

    def calculate_size(
        self,
        starting_balance: float,
        current_equity: float,
        score: int,
        tier: str,
        current_heat: float,
        daily_pnl_pct: float,
        win_streak: int,
        entry_price: float,
        stop_loss: float,
    ) -> PositionSizeResult:
        """
        Calculate position size with all GOD MODE features.

        Args:
            starting_balance: Starting balance for the day
            current_equity: Current equity (with today's P&L)
            score: Signal score (0-100)
            tier: Signal tier (A/B/C/D/F)
            current_heat: Current portfolio heat %
            daily_pnl_pct: Today's P&L %
            win_streak: Consecutive winning trades
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            PositionSizeResult with all calculations
        """
        # Start with base risk
        risk_pct = self.config.base_risk_pct

        # 1. Apply score multiplier
        score_mult = self.config.score_multipliers.get(tier, 0.0)
        if score_mult == 0.0:
            return PositionSizeResult(
                risk_pct=0, risk_amount=0, position_size=0,
                equity_used=0, score_multiplier=0, streak_bonus=0,
                heat_remaining=0, can_trade=False,
                reason=f"Tier {tier} not tradeable in current mode",
            )

        risk_pct *= score_mult

        # 2. Apply win streak bonus
        streak_bonus = 0.0
        if self.config.momentum_scaling and win_streak >= 2:
            streak_bonus = min(
                win_streak * self.config.win_streak_bonus_per_win,
                self.config.max_win_streak_bonus,
            )
            risk_pct *= (1 + streak_bonus)

        # 3. Cap at max risk
        risk_pct = min(risk_pct, self.config.max_risk_pct)
        risk_pct = max(risk_pct, self.config.min_risk_pct)

        # 4. Calculate dynamic heat limit
        heat_limit = self.config.base_heat_limit
        if daily_pnl_pct > 0:
            heat_expansion = daily_pnl_pct * self.config.heat_expansion_per_pct_profit
            heat_limit = min(
                heat_limit + heat_expansion,
                self.config.max_heat_limit,
            )

        # 5. Check heat capacity
        heat_remaining = heat_limit - current_heat
        if heat_remaining <= 0:
            return PositionSizeResult(
                risk_pct=risk_pct, risk_amount=0, position_size=0,
                equity_used=current_equity, score_multiplier=score_mult,
                streak_bonus=streak_bonus, heat_remaining=0,
                can_trade=False,
                reason=f"Heat limit reached ({current_heat:.1f}% >= {heat_limit:.1f}%)",
            )

        # 6. Ensure we don't exceed heat capacity
        max_risk_from_heat = heat_remaining * 0.8  # Leave 20% buffer
        risk_pct = min(risk_pct, max_risk_from_heat)

        # 7. Calculate risk amount
        if self.config.scale_with_equity and self.config.intraday_compounding:
            equity_used = current_equity
        else:
            equity_used = starting_balance

        risk_amount = equity_used * (risk_pct / 100)

        # 8. Calculate position size
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return PositionSizeResult(
                risk_pct=risk_pct, risk_amount=risk_amount, position_size=0,
                equity_used=equity_used, score_multiplier=score_mult,
                streak_bonus=streak_bonus, heat_remaining=heat_remaining,
                can_trade=False,
                reason="Stop distance is zero",
            )

        position_size = risk_amount / stop_distance

        return PositionSizeResult(
            risk_pct=round(risk_pct, 3),
            risk_amount=round(risk_amount, 2),
            position_size=round(position_size, 4),
            equity_used=round(equity_used, 2),
            score_multiplier=round(score_mult, 2),
            streak_bonus=round(streak_bonus, 2),
            heat_remaining=round(heat_remaining, 2),
            can_trade=True,
        )


# =============================================================================
# INTRADAY COMPOUNDER
# =============================================================================

class IntradayCompounder:
    """
    Track intraday equity for compounding.

    After each winning trade, equity increases, so next
    position can be larger (compounding within the day).
    """

    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_equity = starting_balance
        self.daily_pnl = 0.0
        self.trades_today: List[float] = []
        self.win_streak = 0

    def update_equity(self, trade_pnl: float):
        """Update after each trade."""
        self.current_equity += trade_pnl
        self.daily_pnl += trade_pnl
        self.trades_today.append(trade_pnl)

        if trade_pnl > 0:
            self.win_streak += 1
        else:
            self.win_streak = 0

    def get_daily_stats(self) -> Dict:
        """Get daily statistics."""
        return {
            "starting": self.starting_balance,
            "current": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / self.starting_balance) * 100,
            "trades": len(self.trades_today),
            "winners": sum(1 for t in self.trades_today if t > 0),
            "losers": sum(1 for t in self.trades_today if t < 0),
            "win_streak": self.win_streak,
        }

    def reset_daily(self):
        """Reset at start of new day."""
        self.starting_balance = self.current_equity
        self.daily_pnl = 0.0
        self.trades_today = []
        self.win_streak = 0


# =============================================================================
# MODE COMPARISON
# =============================================================================

def compare_modes() -> str:
    """Generate comparison table of all modes."""
    rows = [
        ["Base Risk %", "0.5%", "1.0%", "1.25%", "1.5%"],
        ["Max Risk %", "1.0%", "1.5%", "2.0%", "2.5%"],
        ["Base Heat %", "15%", "25%", "30%", "35%"],
        ["Max Heat %", "20%", "30%", "35%", "45%"],
        ["Max Positions", "5", "8", "10", "12"],
        ["Min Score", "60", "50", "45", "35"],
        ["A-Tier Mult", "1.0x", "1.25x", "1.5x", "1.75x"],
        ["D-Tier Mult", "0.0x", "0.5x", "0.6x", "0.75x"],
        ["Compounding", "No", "Yes", "Yes", "Yes"],
        ["Momentum", "No", "No", "Yes", "Yes"],
        ["Streak Bonus", "0%", "5%/win", "10%/win", "15%/win"],
        ["Max Streak", "0%", "15%", "25%", "45%"],
    ]

    output = []
    output.append("=" * 70)
    output.append("MODE COMPARISON")
    output.append("=" * 70)
    output.append(
        f"{'Setting':<18} {'Conservative':<14} {'Standard':<12} {'Aggressive':<12} {'GOD MODE':<10}"
    )
    output.append("-" * 70)

    for row in rows:
        output.append(
            f"{row[0]:<18} {row[1]:<14} {row[2]:<12} {row[3]:<12} {row[4]:<10}"
        )

    output.append("=" * 70)
    return "\n".join(output)


if __name__ == "__main__":
    print(compare_modes())
