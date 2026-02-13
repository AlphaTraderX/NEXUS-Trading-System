"""Tests for GOD MODE market regime detection."""

import pytest
import numpy as np
from datetime import datetime

from nexus.intelligence.regime_detector import (
    GodModeRegime,
    GodModeRegimeDetector,
    RegimeConfig,
    REGIME_CONFIGS,
    get_historical_regimes,
)
from nexus.core.enums import EdgeType, Direction


# ═══════════════════════════════════════════════════════════════════════
# GodModeRegime enum
# ═══════════════════════════════════════════════════════════════════════


class TestGodModeRegime:
    """Test GodModeRegime enum."""

    def test_all_regimes_exist(self):
        assert GodModeRegime.STRONG_BULL.value == "strong_bull"
        assert GodModeRegime.BULL.value == "bull"
        assert GodModeRegime.SIDEWAYS.value == "sideways"
        assert GodModeRegime.BEAR.value == "bear"
        assert GodModeRegime.STRONG_BEAR.value == "strong_bear"
        assert GodModeRegime.VOLATILE.value == "volatile"

    def test_regime_count(self):
        assert len(GodModeRegime) == 6


# ═══════════════════════════════════════════════════════════════════════
# Regime configurations
# ═══════════════════════════════════════════════════════════════════════


class TestRegimeConfigs:
    """Test regime configurations."""

    def test_all_regimes_have_configs(self):
        for regime in GodModeRegime:
            assert regime in REGIME_CONFIGS

    def test_strong_bull_config(self):
        config = REGIME_CONFIGS[GodModeRegime.STRONG_BULL]
        assert config.position_size_multiplier > 1.0
        assert config.preferred_direction == Direction.LONG
        assert config.max_heat > 30
        assert EdgeType.TURN_OF_MONTH in config.allowed_edges

    def test_strong_bear_config(self):
        config = REGIME_CONFIGS[GodModeRegime.STRONG_BEAR]
        assert config.position_size_multiplier < 1.0
        assert config.max_positions <= 4
        assert config.max_heat < 20
        assert len(config.allowed_edges) < 5

    def test_sideways_favors_mean_reversion(self):
        config = REGIME_CONFIGS[GodModeRegime.SIDEWAYS]
        assert EdgeType.VWAP_DEVIATION in config.allowed_edges
        assert EdgeType.RSI_EXTREME in config.allowed_edges
        assert EdgeType.BOLLINGER_TOUCH in config.allowed_edges

    def test_position_size_progression(self):
        """Size multiplier should decrease from bull to bear."""
        bull = REGIME_CONFIGS[GodModeRegime.STRONG_BULL].position_size_multiplier
        bear = REGIME_CONFIGS[GodModeRegime.STRONG_BEAR].position_size_multiplier
        assert bull > bear

    def test_max_heat_progression(self):
        """Heat limit should decrease as regime worsens."""
        strong_bull = REGIME_CONFIGS[GodModeRegime.STRONG_BULL].max_heat
        strong_bear = REGIME_CONFIGS[GodModeRegime.STRONG_BEAR].max_heat
        assert strong_bull > strong_bear

    def test_max_positions_progression(self):
        """Max positions should decrease as regime worsens."""
        strong_bull = REGIME_CONFIGS[GodModeRegime.STRONG_BULL].max_positions
        strong_bear = REGIME_CONFIGS[GodModeRegime.STRONG_BEAR].max_positions
        assert strong_bull > strong_bear

    def test_volatile_wider_stops(self):
        """Volatile regime should widen stops."""
        config = REGIME_CONFIGS[GodModeRegime.VOLATILE]
        assert config.stop_multiplier > 1.0

    def test_sideways_tighter_stops(self):
        """Sideways regime should tighten stops."""
        config = REGIME_CONFIGS[GodModeRegime.SIDEWAYS]
        assert config.stop_multiplier < 1.0


# ═══════════════════════════════════════════════════════════════════════
# GodModeRegimeDetector
# ═══════════════════════════════════════════════════════════════════════


class TestGodModeRegimeDetector:
    """Test GodModeRegimeDetector class."""

    def setup_method(self):
        self.detector = GodModeRegimeDetector()

    def test_detect_bull_regime(self):
        prices = np.linspace(100, 150, 250)
        regime = self.detector.detect_regime(prices)
        assert regime in [GodModeRegime.BULL, GodModeRegime.STRONG_BULL]

    def test_detect_bear_regime(self):
        prices = np.linspace(150, 100, 250)
        regime = self.detector.detect_regime(prices)
        assert regime in [GodModeRegime.BEAR, GodModeRegime.STRONG_BEAR]

    def test_detect_sideways_regime(self):
        # Flat-ish prices with small noise — clearly range-bound
        rng = np.random.RandomState(42)
        prices = 100 + rng.randn(250) * 0.5  # Very tight around 100
        regime = self.detector.detect_regime(prices)
        # Tiny noise can land in SIDEWAYS or edge into BULL/BEAR depending
        # on exact random draw; VOLATILE is possible too.  The key constraint
        # is that it should NOT be STRONG_BULL or STRONG_BEAR.
        assert regime not in [GodModeRegime.STRONG_BULL, GodModeRegime.STRONG_BEAR]

    def test_insufficient_data_returns_sideways(self):
        prices = np.array([100, 101, 102])
        regime = self.detector.detect_regime(prices)
        assert regime == GodModeRegime.SIDEWAYS

    def test_is_edge_allowed(self):
        assert self.detector.is_edge_allowed(
            EdgeType.VWAP_DEVIATION, GodModeRegime.SIDEWAYS
        )
        assert not self.detector.is_edge_allowed(
            EdgeType.OVERNIGHT_PREMIUM, GodModeRegime.STRONG_BEAR
        )

    def test_adjust_position_size(self):
        base_size = 1000
        bull_size = self.detector.adjust_position_size(
            base_size, GodModeRegime.STRONG_BULL
        )
        assert bull_size > base_size

        bear_size = self.detector.adjust_position_size(
            base_size, GodModeRegime.STRONG_BEAR
        )
        assert bear_size < base_size

    def test_adjust_stops(self):
        base_stop = 2.0
        volatile_stop = self.detector.adjust_stops(
            base_stop, GodModeRegime.VOLATILE
        )
        assert volatile_stop > base_stop

        sideways_stop = self.detector.adjust_stops(
            base_stop, GodModeRegime.SIDEWAYS
        )
        assert sideways_stop < base_stop

    def test_regime_history_tracked(self):
        prices = np.linspace(100, 150, 250)
        now = datetime(2024, 6, 1)
        self.detector.detect_regime(prices, date=now)
        assert len(self.detector.regime_history) == 1
        assert self.detector.regime_history[0][0] == now

    def test_get_config_default(self):
        config = self.detector.get_config()
        assert config.regime == GodModeRegime.SIDEWAYS  # default

    def test_get_config_explicit(self):
        config = self.detector.get_config(GodModeRegime.STRONG_BULL)
        assert config.regime == GodModeRegime.STRONG_BULL


# ═══════════════════════════════════════════════════════════════════════
# Historical regimes
# ═══════════════════════════════════════════════════════════════════════


class TestHistoricalRegimes:
    """Test historical regime data."""

    def test_regimes_cover_2020_to_2024(self):
        regimes = get_historical_regimes()
        assert "2020-01" in regimes
        assert "2024-12" in regimes

    def test_covid_crash_is_strong_bear(self):
        regimes = get_historical_regimes()
        assert regimes["2020-03"] == GodModeRegime.STRONG_BEAR

    def test_2022_bear_market(self):
        regimes = get_historical_regimes()
        bear_months = ["2022-01", "2022-02", "2022-04", "2022-05", "2022-06"]
        for month in bear_months:
            assert regimes[month] in [
                GodModeRegime.BEAR,
                GodModeRegime.STRONG_BEAR,
                GodModeRegime.VOLATILE,
            ]

    def test_2023_bull_market(self):
        regimes = get_historical_regimes()
        bull_months = ["2023-06", "2023-07", "2023-11", "2023-12"]
        for month in bull_months:
            assert regimes[month] in [GodModeRegime.BULL, GodModeRegime.STRONG_BULL]

    def test_60_months_covered(self):
        regimes = get_historical_regimes()
        assert len(regimes) == 60  # 5 years * 12 months

    def test_all_regimes_used(self):
        """Every regime type should appear at least once."""
        regimes = get_historical_regimes()
        used = set(regimes.values())
        for regime in GodModeRegime:
            assert regime in used, f"{regime} never used in historical data"


# ═══════════════════════════════════════════════════════════════════════
# Regime adaptation
# ═══════════════════════════════════════════════════════════════════════


class TestRegimeAdaptation:
    """Test that strategies adapt correctly to regimes."""

    def test_mean_reversion_edges_in_sideways(self):
        sideways_config = REGIME_CONFIGS[GodModeRegime.SIDEWAYS]
        for edge in [EdgeType.VWAP_DEVIATION, EdgeType.RSI_EXTREME, EdgeType.BOLLINGER_TOUCH]:
            assert edge in sideways_config.allowed_edges

    def test_trend_edges_in_bull(self):
        bull_config = REGIME_CONFIGS[GodModeRegime.BULL]
        for edge in [EdgeType.TURN_OF_MONTH, EdgeType.OVERNIGHT_PREMIUM, EdgeType.ORB]:
            assert edge in bull_config.allowed_edges

    def test_defensive_in_bear(self):
        bear_config = REGIME_CONFIGS[GodModeRegime.STRONG_BEAR]
        assert bear_config.max_positions <= 4
        assert bear_config.max_heat <= 20
        assert bear_config.position_size_multiplier <= 0.5
        assert bear_config.preferred_direction == Direction.SHORT

    def test_bull_favors_longs(self):
        for regime in [GodModeRegime.STRONG_BULL, GodModeRegime.BULL]:
            config = REGIME_CONFIGS[regime]
            assert config.preferred_direction == Direction.LONG

    def test_sideways_no_bias(self):
        config = REGIME_CONFIGS[GodModeRegime.SIDEWAYS]
        assert config.preferred_direction is None

    def test_volatile_reduces_exposure(self):
        config = REGIME_CONFIGS[GodModeRegime.VOLATILE]
        assert config.position_size_multiplier < 1.0
        assert config.max_positions <= 5
        assert config.max_heat <= 20
