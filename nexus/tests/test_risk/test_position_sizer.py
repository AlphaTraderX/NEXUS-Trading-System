"""
NEXUS Dynamic Position Sizer tests.
"""

import pytest
from unittest.mock import MagicMock

from nexus.core.enums import Market, MarketRegime
from nexus.core.models import PositionSize
from nexus.risk.position_sizer import DynamicPositionSizer, BASE_RISK_PCT, MAX_RISK_PCT, MAX_HEAT_LIMIT


@pytest.fixture
def mock_settings():
    """Settings object with risk config."""
    s = MagicMock()
    s.base_risk_pct = 1.0
    s.max_risk_pct = 2.0
    s.max_heat_limit = 25.0
    return s


@pytest.fixture
def sizer(mock_settings):
    """Default position sizer using mock settings."""
    return DynamicPositionSizer(settings=mock_settings)


def test_basic_calculation(sizer):
    """Standard calculation with default inputs."""
    result = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.5,
        score=78,
        current_heat=0.0,
        win_streak=0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    assert result.can_trade is True
    assert result.rejection_reason is None
    assert result.position_size > 0
    assert result.position_value > 0
    assert result.risk_amount > 0
    assert result.risk_pct > 0
    assert result.stop_distance == 1.5
    assert result.score_multiplier == 1.25  # 78 -> Tier B
    assert result.regime_multiplier == 1.0
    assert result.momentum_multiplier == 1.0


def test_score_multipliers(sizer):
    """Each tier produces correct multiplier."""
    # Tier A: >= 85 -> 1.5x
    r = sizer.calculate_size(10000, 10000, 100, 98, 90, 0, 0, MarketRegime.RANGING, "SPY")
    assert r.can_trade and r.score_multiplier == 1.5
    # Tier B: >= 75 -> 1.25x
    r = sizer.calculate_size(10000, 10000, 100, 98, 78, 0, 0, MarketRegime.RANGING, "SPY")
    assert r.can_trade and r.score_multiplier == 1.25
    # Tier C: >= 65 -> 1.0x
    r = sizer.calculate_size(10000, 10000, 100, 98, 68, 0, 0, MarketRegime.RANGING, "SPY")
    assert r.can_trade and r.score_multiplier == 1.0
    # Tier D: >= 50 -> 0.75x
    r = sizer.calculate_size(10000, 10000, 100, 98, 55, 0, 0, MarketRegime.RANGING, "SPY")
    assert r.can_trade and r.score_multiplier == 0.75
    # Marginal: < 50 -> 0.5x
    r = sizer.calculate_size(10000, 10000, 100, 98, 40, 0, 0, MarketRegime.RANGING, "SPY")
    assert r.can_trade and r.score_multiplier == 0.5


def test_regime_multipliers(sizer):
    """Each regime produces correct multiplier."""
    base = dict(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=70,
        current_heat=0.0,
        win_streak=0,
        symbol="SPY",
    )
    r = sizer.calculate_size(**base, regime=MarketRegime.TRENDING_UP)
    assert r.regime_multiplier == 1.0  # Spec: trade both directions equally
    r = sizer.calculate_size(**base, regime=MarketRegime.TRENDING_DOWN)
    assert r.regime_multiplier == 1.0  # Spec: trade both directions
    r = sizer.calculate_size(**base, regime=MarketRegime.RANGING)
    assert r.regime_multiplier == 1.0
    r = sizer.calculate_size(**base, regime=MarketRegime.VOLATILE)
    assert r.regime_multiplier == 0.5  # Spec: half size in volatile markets


def test_momentum_scaling(sizer):
    """Win streak boosts correctly, caps at 1.3x."""
    base = dict(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=70,
        current_heat=0.0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    r0 = sizer.calculate_size(**base, win_streak=0)
    r1 = sizer.calculate_size(**base, win_streak=1)
    r2 = sizer.calculate_size(**base, win_streak=2)
    r5 = sizer.calculate_size(**base, win_streak=5)
    assert r0.momentum_multiplier == 1.0
    assert r1.momentum_multiplier == 1.0
    assert r2.momentum_multiplier == 1.2  # 1 + 2*0.1
    assert r5.momentum_multiplier == 1.3  # capped
    assert r2.position_size > r0.position_size
    assert r5.position_size >= r2.position_size


def test_heat_capacity_rejection(sizer):
    """Rejects when would exceed heat limit."""
    result = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=80,
        current_heat=25.0,
        win_streak=0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    assert result.can_trade is False
    assert result.rejection_reason is not None
    assert "heat" in result.rejection_reason.lower()
    assert result.position_size == 0
    assert result.risk_amount == 0


def test_heat_capacity_partial(sizer):
    """Reduces size to fit within heat limit."""
    # Use most of heat budget so only small room left
    result = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=90,
        current_heat=18.0,
        win_streak=2,
        regime=MarketRegime.TRENDING_UP,
        symbol="SPY",
    )
    assert result.can_trade is True
    # Usable heat = min(25-18, 20) = 7. So risk_pct should be capped at 7
    assert result.risk_pct <= 7.5  # allow small float tolerance
    assert result.heat_after_trade <= 25.0


def test_max_risk_cap(sizer):
    """Never exceeds MAX_RISK_PCT."""
    result = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=99.5,
        score=95,
        current_heat=0.0,
        win_streak=5,
        regime=MarketRegime.TRENDING_UP,
        symbol="SPY",
    )
    assert result.can_trade is True
    assert result.risk_pct <= MAX_RISK_PCT + 0.01  # 2.0% cap


def test_zero_stop_distance(sizer):
    """Raises ValueError when stop equals entry."""
    with pytest.raises(ValueError) as exc_info:
        sizer.calculate_size(
            starting_balance=10000.0,
            current_equity=10000.0,
            entry_price=100.0,
            stop_loss=100.0,
            score=70,
            current_heat=0.0,
            win_streak=0,
            regime=MarketRegime.RANGING,
            symbol="SPY",
        )
    assert "stop_loss" in str(exc_info.value).lower() or "entry" in str(exc_info.value).lower()


def test_intraday_compounding(sizer):
    """Uses current_equity not starting_balance."""
    result_low = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=70,
        current_heat=0.0,
        win_streak=0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    result_high = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10150.0,
        entry_price=100.0,
        stop_loss=98.0,
        score=70,
        current_heat=0.0,
        win_streak=0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    assert result_high.position_size > result_low.position_size
    assert result_high.risk_amount > result_low.risk_amount
    # Risk % should be same; risk_amount scales with equity
    assert abs(result_high.risk_pct - result_low.risk_pct) < 0.01


def test_negative_stop_distance_short(sizer):
    """Handles short positions correctly (stop above entry)."""
    result = sizer.calculate_size(
        starting_balance=10000.0,
        current_equity=10000.0,
        entry_price=100.0,
        stop_loss=102.0,
        score=70,
        current_heat=0.0,
        win_streak=0,
        regime=MarketRegime.RANGING,
        symbol="SPY",
    )
    assert result.can_trade is True
    assert result.stop_distance == 2.0
    assert result.position_size > 0
    assert result.risk_amount == result.position_size * 2.0


def test_position_size_dataclass_to_dict():
    """PositionSize.to_dict returns asdict."""
    from nexus.core.models import PositionSize
    p = PositionSize(
        risk_pct=1.0,
        risk_amount=100.0,
        position_size=50.0,
        position_value=5000.0,
        stop_distance=2.0,
        stop_distance_pct=2.0,
        score_multiplier=1.0,
        regime_multiplier=1.0,
        momentum_multiplier=1.0,
        heat_after_trade=1.0,
        can_trade=True,
        rejection_reason=None,
    )
    d = p.to_dict()
    assert d["risk_pct"] == 1.0
    assert d["position_size"] == 50.0
    assert d["can_trade"] is True
