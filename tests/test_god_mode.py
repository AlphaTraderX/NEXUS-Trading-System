"""
Tests for GOD MODE configuration.

Tests all trading mode configurations, position sizer, and intraday compounder.
"""

import pytest

from nexus.config.god_mode import (
    TradingMode,
    ModeConfig,
    GodModePositionSizer,
    IntradayCompounder,
    PositionSizeResult,
    CONSERVATIVE_CONFIG,
    STANDARD_CONFIG,
    AGGRESSIVE_CONFIG,
    GOD_MODE_CONFIG,
    MODE_CONFIGS,
    get_mode_config,
    compare_modes,
)


# =============================================================================
# Trading Mode Enum
# =============================================================================


class TestTradingMode:
    """Test TradingMode enum."""

    def test_all_modes_exist(self):
        assert TradingMode.CONSERVATIVE == "conservative"
        assert TradingMode.STANDARD == "standard"
        assert TradingMode.AGGRESSIVE == "aggressive"
        assert TradingMode.GOD_MODE == "god_mode"

    def test_mode_count(self):
        assert len(TradingMode) == 4

    def test_mode_is_string_enum(self):
        assert isinstance(TradingMode.GOD_MODE, str)
        assert TradingMode.GOD_MODE == "god_mode"


# =============================================================================
# Mode Configs
# =============================================================================


class TestModeConfigs:
    """Test mode configuration lookup."""

    def test_all_modes_have_configs(self):
        for mode in TradingMode:
            assert mode in MODE_CONFIGS

    def test_get_mode_config(self):
        config = get_mode_config(TradingMode.GOD_MODE)
        assert config is GOD_MODE_CONFIG

    def test_get_standard_config(self):
        config = get_mode_config(TradingMode.STANDARD)
        assert config is STANDARD_CONFIG

    def test_get_conservative_config(self):
        config = get_mode_config(TradingMode.CONSERVATIVE)
        assert config is CONSERVATIVE_CONFIG

    def test_get_aggressive_config(self):
        config = get_mode_config(TradingMode.AGGRESSIVE)
        assert config is AGGRESSIVE_CONFIG


class TestModeConfigValues:
    """Test specific config values for each mode."""

    def test_conservative_is_cautious(self):
        c = CONSERVATIVE_CONFIG
        assert c.base_risk_pct == 0.5
        assert c.max_risk_pct == 1.0
        assert c.max_positions == 5
        assert c.min_score_to_trade == 60
        assert c.intraday_compounding is False
        assert c.momentum_scaling is False

    def test_standard_is_balanced(self):
        c = STANDARD_CONFIG
        assert c.base_risk_pct == 1.0
        assert c.max_risk_pct == 1.5
        assert c.max_positions == 8
        assert c.min_score_to_trade == 50

    def test_aggressive_is_higher(self):
        c = AGGRESSIVE_CONFIG
        assert c.base_risk_pct == 1.25
        assert c.max_risk_pct == 2.0
        assert c.max_positions == 10
        assert c.min_score_to_trade == 45
        assert c.momentum_scaling is True

    def test_god_mode_is_maximum(self):
        c = GOD_MODE_CONFIG
        assert c.base_risk_pct == 1.5
        assert c.max_risk_pct == 2.5
        assert c.max_positions == 12
        assert c.min_score_to_trade == 35
        assert c.momentum_scaling is True
        assert c.intraday_compounding is True
        assert c.scale_with_equity is True

    def test_risk_escalation(self):
        """Risk increases across modes."""
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        for i in range(len(configs) - 1):
            assert configs[i].base_risk_pct < configs[i + 1].base_risk_pct
            assert configs[i].max_risk_pct < configs[i + 1].max_risk_pct

    def test_heat_escalation(self):
        """Heat limits increase across modes."""
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        for i in range(len(configs) - 1):
            assert configs[i].base_heat_limit < configs[i + 1].base_heat_limit
            assert configs[i].max_heat_limit < configs[i + 1].max_heat_limit

    def test_position_limit_escalation(self):
        """Position limits increase across modes."""
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        for i in range(len(configs) - 1):
            assert configs[i].max_positions < configs[i + 1].max_positions

    def test_min_score_decreases(self):
        """Minimum score to trade decreases (more permissive)."""
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        for i in range(len(configs) - 1):
            assert configs[i].min_score_to_trade > configs[i + 1].min_score_to_trade

    def test_circuit_breakers_always_enforced(self):
        """Circuit breakers exist in all modes."""
        for mode, config in MODE_CONFIGS.items():
            assert config.daily_loss_stop < 0, f"{mode} missing daily loss stop"
            assert config.weekly_loss_stop < 0, f"{mode} missing weekly loss stop"
            assert config.max_drawdown < 0, f"{mode} missing max drawdown"

    def test_god_mode_score_multipliers(self):
        """GOD MODE has higher multipliers."""
        gm = GOD_MODE_CONFIG.score_multipliers
        std = STANDARD_CONFIG.score_multipliers
        assert gm["A"] > std["A"]
        assert gm["B"] > std["B"]
        assert gm["C"] > std["C"]
        assert gm["D"] > std["D"]

    def test_conservative_no_d_tier(self):
        """Conservative mode skips D-tier signals."""
        assert CONSERVATIVE_CONFIG.score_multipliers["D"] == 0.0

    def test_god_mode_trades_d_tier(self):
        """GOD MODE trades D-tier signals."""
        assert GOD_MODE_CONFIG.score_multipliers["D"] > 0.0

    def test_all_modes_skip_f_tier(self):
        """F-tier is never traded."""
        for mode, config in MODE_CONFIGS.items():
            assert config.score_multipliers["F"] == 0.0, f"{mode} should skip F-tier"

    def test_god_mode_streak_bonuses(self):
        """GOD MODE has highest streak bonuses."""
        gm = GOD_MODE_CONFIG
        assert gm.win_streak_bonus_per_win == 0.15
        assert gm.max_win_streak_bonus == 0.45

    def test_god_mode_heat_expansion(self):
        """GOD MODE has aggressive heat expansion."""
        gm = GOD_MODE_CONFIG
        assert gm.heat_expansion_per_pct_profit == 3.0
        assert gm.max_heat_limit == 45.0


# =============================================================================
# Position Sizer
# =============================================================================


class TestGodModePositionSizer:
    """Test GodModePositionSizer."""

    def setup_method(self):
        self.sizer = GodModePositionSizer(GOD_MODE_CONFIG)

    def test_basic_size_calculation(self):
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.can_trade is True
        assert result.position_size > 0
        assert result.risk_amount > 0
        assert result.score_multiplier == 1.75

    def test_f_tier_rejected(self):
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=20,
            tier="F",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.can_trade is False
        assert "not tradeable" in result.reason

    def test_heat_limit_blocks_trade(self):
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=45.0,  # At max heat
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.can_trade is False
        assert "Heat limit" in result.reason

    def test_zero_stop_distance_rejected(self):
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=100.0,  # Same as entry
        )
        assert result.can_trade is False
        assert "Stop distance" in result.reason

    def test_win_streak_increases_risk(self):
        """Win streak should increase position size."""
        result_no_streak = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        result_streak = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=3,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result_streak.risk_pct >= result_no_streak.risk_pct
        assert result_streak.streak_bonus > 0

    def test_streak_bonus_capped(self):
        """Win streak bonus should be capped."""
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=100,  # Huge streak
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.streak_bonus <= GOD_MODE_CONFIG.max_win_streak_bonus

    def test_heat_expansion_when_profitable(self):
        """Heat limit should expand when profitable."""
        # Not profitable - base heat limit applies
        result_flat = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=34.0,  # Just below base 35%
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )

        # Profitable - heat limit expands
        result_profit = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=102_000,
            score=80,
            tier="A",
            current_heat=34.0,
            daily_pnl_pct=2.0,  # 2% daily profit
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )

        assert result_profit.heat_remaining > result_flat.heat_remaining

    def test_equity_compounding(self):
        """With compounding, bigger equity = bigger position."""
        result_100k = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        result_120k = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=120_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result_120k.risk_amount > result_100k.risk_amount
        assert result_120k.equity_used == 120_000

    def test_score_multiplier_tiers(self):
        """Each tier gets correct multiplier."""
        tiers = {"A": 1.75, "B": 1.5, "C": 1.25, "D": 0.75}
        for tier, expected_mult in tiers.items():
            result = self.sizer.calculate_size(
                starting_balance=100_000,
                current_equity=100_000,
                score=80,
                tier=tier,
                current_heat=0,
                daily_pnl_pct=0,
                win_streak=0,
                entry_price=100.0,
                stop_loss=98.0,
            )
            assert result.score_multiplier == expected_mult

    def test_risk_capped_at_max(self):
        """Risk should never exceed max_risk_pct."""
        result = self.sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=99,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=10,  # Big streak + A tier = lots of risk
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.risk_pct <= GOD_MODE_CONFIG.max_risk_pct

    def test_conservative_sizer_rejects_d_tier(self):
        """Conservative mode should reject D-tier."""
        sizer = GodModePositionSizer(CONSERVATIVE_CONFIG)
        result = sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=45,
            tier="D",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=0,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.can_trade is False

    def test_conservative_no_momentum_scaling(self):
        """Conservative mode ignores win streaks."""
        sizer = GodModePositionSizer(CONSERVATIVE_CONFIG)
        result = sizer.calculate_size(
            starting_balance=100_000,
            current_equity=100_000,
            score=80,
            tier="A",
            current_heat=0,
            daily_pnl_pct=0,
            win_streak=5,
            entry_price=100.0,
            stop_loss=98.0,
        )
        assert result.streak_bonus == 0.0

    def test_position_size_result_fields(self):
        """PositionSizeResult has all expected fields."""
        result = PositionSizeResult(
            risk_pct=1.5,
            risk_amount=1500,
            position_size=750,
            equity_used=100_000,
            score_multiplier=1.75,
            streak_bonus=0.15,
            heat_remaining=20.0,
            can_trade=True,
        )
        assert result.risk_pct == 1.5
        assert result.risk_amount == 1500
        assert result.position_size == 750
        assert result.equity_used == 100_000
        assert result.score_multiplier == 1.75
        assert result.streak_bonus == 0.15
        assert result.heat_remaining == 20.0
        assert result.can_trade is True
        assert result.reason == ""


# =============================================================================
# Intraday Compounder
# =============================================================================


class TestIntradayCompounder:
    """Test IntradayCompounder."""

    def test_initialization(self):
        comp = IntradayCompounder(100_000)
        assert comp.starting_balance == 100_000
        assert comp.current_equity == 100_000
        assert comp.daily_pnl == 0.0
        assert comp.win_streak == 0

    def test_winning_trade_updates(self):
        comp = IntradayCompounder(100_000)
        comp.update_equity(500)
        assert comp.current_equity == 100_500
        assert comp.daily_pnl == 500
        assert comp.win_streak == 1

    def test_losing_trade_resets_streak(self):
        comp = IntradayCompounder(100_000)
        comp.update_equity(500)
        comp.update_equity(300)
        assert comp.win_streak == 2
        comp.update_equity(-200)
        assert comp.win_streak == 0

    def test_daily_stats(self):
        comp = IntradayCompounder(100_000)
        comp.update_equity(500)
        comp.update_equity(-200)
        comp.update_equity(300)

        stats = comp.get_daily_stats()
        assert stats["starting"] == 100_000
        assert stats["current"] == 100_600
        assert stats["daily_pnl"] == 600
        assert abs(stats["daily_pnl_pct"] - 0.6) < 0.01
        assert stats["trades"] == 3
        assert stats["winners"] == 2
        assert stats["losers"] == 1
        assert stats["win_streak"] == 1

    def test_daily_reset(self):
        comp = IntradayCompounder(100_000)
        comp.update_equity(1000)
        comp.reset_daily()

        assert comp.starting_balance == 101_000
        assert comp.current_equity == 101_000
        assert comp.daily_pnl == 0.0
        assert comp.win_streak == 0
        assert len(comp.trades_today) == 0

    def test_compounding_effect(self):
        """After winning, equity is higher so next position can be bigger."""
        comp = IntradayCompounder(100_000)
        initial_equity = comp.current_equity
        comp.update_equity(2000)  # Win 2%
        assert comp.current_equity > initial_equity
        # Next position would use 102,000 as base

    def test_multiple_day_compounding(self):
        """Equity carries over across days."""
        comp = IntradayCompounder(100_000)
        # Day 1: make 1%
        comp.update_equity(1000)
        comp.reset_daily()
        assert comp.starting_balance == 101_000

        # Day 2: make 1%
        comp.update_equity(1010)
        comp.reset_daily()
        assert comp.starting_balance == 102_010


# =============================================================================
# Compare Modes Output
# =============================================================================


class TestCompareModesOutput:
    """Test the compare_modes function."""

    def test_output_contains_all_modes(self):
        output = compare_modes()
        assert "Conservative" in output
        assert "Standard" in output
        assert "Aggressive" in output
        assert "GOD MODE" in output

    def test_output_contains_key_metrics(self):
        output = compare_modes()
        assert "Base Risk" in output
        assert "Max Heat" in output
        assert "Max Positions" in output
        assert "Min Score" in output

    def test_output_is_formatted(self):
        output = compare_modes()
        assert "=" * 70 in output
        assert "MODE COMPARISON" in output


# =============================================================================
# Scaling Progression
# =============================================================================


class TestScalingProgression:
    """Test that modes form a clear progression."""

    def test_a_tier_multiplier_increases(self):
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        a_mults = [c.score_multipliers["A"] for c in configs]
        for i in range(len(a_mults) - 1):
            assert a_mults[i] < a_mults[i + 1]

    def test_streak_bonus_increases(self):
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        bonuses = [c.win_streak_bonus_per_win for c in configs]
        for i in range(len(bonuses) - 1):
            assert bonuses[i] <= bonuses[i + 1]

    def test_heat_expansion_increases(self):
        configs = [CONSERVATIVE_CONFIG, STANDARD_CONFIG, AGGRESSIVE_CONFIG, GOD_MODE_CONFIG]
        expansions = [c.heat_expansion_per_pct_profit for c in configs]
        for i in range(len(expansions) - 1):
            assert expansions[i] < expansions[i + 1]
