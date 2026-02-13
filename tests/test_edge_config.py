"""Tests for edge configuration."""

import pytest
from nexus.core.enums import EdgeType
from nexus.config.edge_config import (
    EDGE_CONFIGS,
    get_enabled_edges,
    get_edge_weight,
    is_edge_enabled,
    get_edge_cooldown,
    get_edge_max_daily,
)


class TestEdgeConfig:
    def test_losing_edges_disabled(self):
        """Verify losing edges from backtest are disabled."""
        assert not is_edge_enabled(EdgeType.MONTH_END)
        assert not is_edge_enabled(EdgeType.NY_OPEN)

    def test_top_performers_have_high_weight(self):
        """Top performing edges should have weight > 1.0."""
        assert get_edge_weight(EdgeType.VWAP_DEVIATION) >= 1.4
        assert get_edge_weight(EdgeType.GAP_FILL) >= 1.3

    def test_marginal_edges_have_low_weight(self):
        """Marginal edges should have reduced weight."""
        assert get_edge_weight(EdgeType.BOLLINGER_TOUCH) < 1.0
        assert get_edge_weight(EdgeType.LONDON_OPEN) < 1.0

    def test_enabled_edges_count(self):
        """Should have 11 enabled edges (15 total - 4 disabled)."""
        enabled = get_enabled_edges()
        assert len(enabled) == 11

    def test_all_edges_have_config(self):
        """All EdgeType values should have configuration."""
        for edge in EdgeType:
            assert edge in EDGE_CONFIGS, f"Missing config for {edge}"

    def test_disabled_edges_zero_weight(self):
        """Disabled edges should have zero weight."""
        for edge, cfg in EDGE_CONFIGS.items():
            if not cfg.enabled:
                assert cfg.weight == 0.0, f"{edge} disabled but weight={cfg.weight}"

    def test_cooldown_values(self):
        """Key edges should have expected cooldowns."""
        assert get_edge_cooldown(EdgeType.RSI_EXTREME) == 120
        assert get_edge_cooldown(EdgeType.INSIDER_CLUSTER) == 1440
        assert get_edge_cooldown(EdgeType.VWAP_DEVIATION) == 60

    def test_max_daily_values(self):
        """Disabled edges should have zero max daily."""
        assert get_edge_max_daily(EdgeType.MONTH_END) == 0
        assert get_edge_max_daily(EdgeType.NY_OPEN) == 0
        assert get_edge_max_daily(EdgeType.VWAP_DEVIATION) > 0
