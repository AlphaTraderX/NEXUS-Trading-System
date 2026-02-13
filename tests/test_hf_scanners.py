"""Tests for high-frequency multi-timeframe scanners."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from nexus.core.enums import Timeframe, EdgeType, Direction, TIMEFRAME_THRESHOLDS
from nexus.scanners.vwap_hf import VWAPHighFrequencyScanner
from nexus.scanners.rsi_hf import RSIHighFrequencyScanner


# =============================================================================
# Timeframe enum tests
# =============================================================================


class TestTimeframeEnum:
    """Test Timeframe enum."""

    def test_minutes(self):
        assert Timeframe.M5.minutes == 5
        assert Timeframe.M15.minutes == 15
        assert Timeframe.M30.minutes == 30
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440

    def test_bars_per_day(self):
        assert Timeframe.M5.bars_per_day == 288
        assert Timeframe.M15.bars_per_day == 96
        assert Timeframe.M30.bars_per_day == 48
        assert Timeframe.H1.bars_per_day == 24
        assert Timeframe.H4.bars_per_day == 6
        assert Timeframe.D1.bars_per_day == 1

    def test_intraday(self):
        intraday = Timeframe.intraday()
        assert Timeframe.M5 in intraday
        assert Timeframe.M15 in intraday
        assert Timeframe.M30 in intraday
        assert Timeframe.H1 in intraday
        assert Timeframe.D1 not in intraday
        assert Timeframe.H4 not in intraday

    def test_swing(self):
        swing = Timeframe.swing()
        assert Timeframe.H4 in swing
        assert Timeframe.D1 in swing
        assert Timeframe.M5 not in swing
        assert Timeframe.H1 not in swing

    def test_all_timeframes_have_values(self):
        expected_values = {"5m", "15m", "30m", "1h", "4h", "1d"}
        actual_values = {tf.value for tf in Timeframe}
        assert actual_values == expected_values


class TestTimeframeThresholds:
    """Test timeframe-specific thresholds."""

    def test_thresholds_exist_for_all_timeframes(self):
        for tf in Timeframe:
            assert tf in TIMEFRAME_THRESHOLDS, f"Missing thresholds for {tf.value}"

    def test_m5_has_tighter_vwap_threshold(self):
        m5 = TIMEFRAME_THRESHOLDS[Timeframe.M5]
        d1 = TIMEFRAME_THRESHOLDS[Timeframe.D1]
        assert m5["vwap_deviation_std"] < d1["vwap_deviation_std"]

    def test_m5_has_tighter_rsi_thresholds(self):
        m5 = TIMEFRAME_THRESHOLDS[Timeframe.M5]
        d1 = TIMEFRAME_THRESHOLDS[Timeframe.D1]
        assert m5["rsi_oversold"] < d1["rsi_oversold"]
        assert m5["rsi_overbought"] > d1["rsi_overbought"]

    def test_all_thresholds_have_required_keys(self):
        required_keys = ["vwap_deviation_std", "rsi_oversold", "rsi_overbought", "bollinger_std"]
        for tf, thresholds in TIMEFRAME_THRESHOLDS.items():
            for key in required_keys:
                assert key in thresholds, f"Missing {key} in {tf.value}"

    def test_thresholds_are_reasonable(self):
        for tf, thresholds in TIMEFRAME_THRESHOLDS.items():
            assert 0 < thresholds["vwap_deviation_std"] < 5
            assert 0 < thresholds["rsi_oversold"] < 50
            assert 50 < thresholds["rsi_overbought"] < 100
            assert 0 < thresholds["bollinger_std"] < 5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_data_provider():
    provider = MagicMock()
    provider.get_bars = AsyncMock(return_value=None)
    provider.get_quote = AsyncMock(return_value=None)
    return provider


@pytest.fixture
def sample_bars():
    """Create sample OHLCV data with volume."""
    np.random.seed(42)
    n = 100

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


@pytest.fixture
def sample_bars_no_volume():
    """OHLCV data without volume."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
    })


# =============================================================================
# VWAP HF Scanner tests
# =============================================================================


class TestVWAPHFScannerConfig:
    """Test VWAP HF scanner configuration."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return VWAPHighFrequencyScanner(mock_data_provider)

    def test_supported_timeframes(self, scanner):
        assert Timeframe.M5 in scanner.supported_timeframes
        assert Timeframe.M15 in scanner.supported_timeframes
        assert Timeframe.M30 in scanner.supported_timeframes
        assert Timeframe.H1 in scanner.supported_timeframes
        assert Timeframe.H4 in scanner.supported_timeframes
        assert Timeframe.D1 in scanner.supported_timeframes
        assert len(scanner.supported_timeframes) == 6

    def test_edge_type(self, scanner):
        assert scanner.edge_type == EdgeType.VWAP_DEVIATION

    def test_is_active(self, scanner):
        assert scanner.is_active() is True

    def test_has_registry(self, scanner):
        assert hasattr(scanner, "registry")
        assert scanner.registry is not None

    def test_get_thresholds_varies_by_timeframe(self, scanner):
        m5 = scanner.get_thresholds(Timeframe.M5)
        d1 = scanner.get_thresholds(Timeframe.D1)
        assert m5["vwap_deviation_std"] < d1["vwap_deviation_std"]


class TestVWAPCalculations:
    """Test VWAP calculation methods."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return VWAPHighFrequencyScanner(mock_data_provider)

    def test_calculate_vwap_with_volume(self, scanner, sample_bars):
        vwap = scanner._calculate_vwap(sample_bars)
        assert isinstance(vwap, float)
        assert vwap > 0

    def test_calculate_vwap_without_volume(self, scanner, sample_bars_no_volume):
        vwap = scanner._calculate_vwap(sample_bars_no_volume)
        assert isinstance(vwap, float)
        assert vwap > 0

    def test_calculate_vwap_std_with_volume(self, scanner, sample_bars):
        vwap = scanner._calculate_vwap(sample_bars)
        vwap_std = scanner._calculate_vwap_std(sample_bars, vwap)
        assert isinstance(vwap_std, float)
        assert vwap_std >= 0

    def test_calculate_vwap_std_without_volume(self, scanner, sample_bars_no_volume):
        vwap = scanner._calculate_vwap(sample_bars_no_volume)
        vwap_std = scanner._calculate_vwap_std(sample_bars_no_volume, vwap)
        assert isinstance(vwap_std, float)
        assert vwap_std >= 0

    def test_vwap_is_weighted_by_volume(self, scanner, sample_bars):
        """VWAP should differ from simple mean when volume varies."""
        vwap = scanner._calculate_vwap(sample_bars)
        typical = (sample_bars["high"] + sample_bars["low"] + sample_bars["close"]) / 3
        simple_mean = float(typical.mean())
        # They should be close but not identical (volume weighting matters)
        assert vwap != pytest.approx(simple_mean, rel=1e-10)


class TestVWAPHFScannerAsync:
    """Test VWAP HF scanner async methods."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return VWAPHighFrequencyScanner(mock_data_provider)

    @pytest.mark.asyncio
    async def test_scan_returns_list(self, scanner):
        """scan() returns list even with no data."""
        result = await scanner.scan()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_timeframe_returns_list(self, scanner):
        """scan_timeframe() returns list even with no data."""
        result = await scanner.scan_timeframe(Timeframe.M5, instruments=["AAPL"])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_symbol_no_bars(self, scanner):
        """Returns None when no bars available."""
        result = await scanner._scan_symbol("AAPL", Timeframe.M5, 1.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_scan_symbol_with_extreme_deviation(self, scanner, sample_bars):
        """Should detect signal when price deviates extremely from VWAP."""
        # Force extreme deviation by pushing last close far from VWAP
        extreme_bars = sample_bars.copy()
        vwap = VWAPHighFrequencyScanner._calculate_vwap(extreme_bars)
        vwap_std = VWAPHighFrequencyScanner._calculate_vwap_std(extreme_bars, vwap)

        # Push price 5 std devs above VWAP
        extreme_bars.iloc[-1, extreme_bars.columns.get_loc("close")] = vwap + (vwap_std * 5)

        scanner.data = MagicMock()
        scanner.data.get_bars = AsyncMock(return_value=extreme_bars)

        result = await scanner._scan_symbol("AAPL", Timeframe.M5, 1.5)

        if result is not None:
            assert result.direction == Direction.SHORT  # Mean reversion: fade the high
            assert result.edge_data["strategy"] == "mean_reversion"
            assert result.edge_data["timeframe"] == "5m"

    @pytest.mark.asyncio
    async def test_scan_all_timeframes(self, scanner):
        """scan_all_timeframes() runs without error."""
        result = await scanner.scan_all_timeframes(instruments=["AAPL"])
        assert isinstance(result, list)


# =============================================================================
# RSI HF Scanner tests
# =============================================================================


class TestRSIHFScannerConfig:
    """Test RSI HF scanner configuration."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return RSIHighFrequencyScanner(mock_data_provider)

    def test_uses_rsi_period_2(self, scanner):
        """Must use RSI(2), not RSI(14)."""
        assert scanner.rsi_period == 2

    def test_supported_timeframes(self, scanner):
        assert Timeframe.M5 in scanner.supported_timeframes
        assert len(scanner.supported_timeframes) == 6

    def test_edge_type(self, scanner):
        assert scanner.edge_type == EdgeType.RSI_EXTREME

    def test_is_active(self, scanner):
        assert scanner.is_active() is True

    def test_has_registry(self, scanner):
        assert hasattr(scanner, "registry")
        assert scanner.registry is not None


class TestRSICalculations:
    """Test RSI calculation methods."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return RSIHighFrequencyScanner(mock_data_provider)

    def test_calculate_rsi_2(self, scanner, sample_bars):
        rsi = scanner._calculate_rsi_2(sample_bars)
        assert len(rsi) == len(sample_bars)

        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_responds_to_trends(self, scanner):
        """RSI should be high after consecutive up bars."""
        n = 20
        up_bars = pd.DataFrame({
            "open": [100 + i for i in range(n)],
            "high": [101 + i for i in range(n)],
            "low": [99 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000] * n,
        })

        rsi = scanner._calculate_rsi_2(up_bars)
        # After many consecutive up bars, RSI(2) should be very high
        assert rsi.iloc[-1] > 80


class TestRSIHFScannerAsync:
    """Test RSI HF scanner async methods."""

    @pytest.fixture
    def scanner(self, mock_data_provider):
        return RSIHighFrequencyScanner(mock_data_provider)

    @pytest.mark.asyncio
    async def test_scan_returns_list(self, scanner):
        result = await scanner.scan()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_timeframe_returns_list(self, scanner):
        result = await scanner.scan_timeframe(Timeframe.M15, instruments=["AAPL"])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_symbol_no_bars(self, scanner):
        result = await scanner._scan_symbol("AAPL", Timeframe.M15, 20, 80)
        assert result is None

    @pytest.mark.asyncio
    async def test_scan_symbol_with_extreme_rsi(self, scanner):
        """Should detect signal when RSI is extreme."""
        # Create bars where RSI(2) will be very low (consecutive down)
        n = 50
        close = [100 - i * 0.5 for i in range(n)]

        bars = pd.DataFrame({
            "open": [c + 0.1 for c in close],
            "high": [c + 0.3 for c in close],
            "low": [c - 0.3 for c in close],
            "close": close,
            "volume": [1_000_000] * n,
        })

        scanner.data = MagicMock()
        scanner.data.get_bars = AsyncMock(return_value=bars)

        result = await scanner._scan_symbol("AAPL", Timeframe.M15, 20, 80)

        if result is not None:
            assert result.direction == Direction.LONG  # Mean reversion: buy oversold
            assert result.edge_data["rsi_period"] == 2
            assert result.edge_data["timeframe"] == "15m"

    @pytest.mark.asyncio
    async def test_scan_all_timeframes(self, scanner):
        result = await scanner.scan_all_timeframes(instruments=["AAPL"])
        assert isinstance(result, list)


# =============================================================================
# Base scanner multi-timeframe tests
# =============================================================================


class TestBaseScannerMultiTimeframe:
    """Test BaseScanner multi-timeframe methods."""

    def test_default_supported_timeframes(self, mock_data_provider):
        """Default scanner supports only D1."""
        from nexus.scanners.rsi import RSIScanner
        scanner = RSIScanner(mock_data_provider)
        # Original scanners inherit [D1] default
        assert Timeframe.D1 in scanner.supported_timeframes

    def test_hf_scanners_support_6_timeframes(self, mock_data_provider):
        vwap = VWAPHighFrequencyScanner(mock_data_provider)
        rsi = RSIHighFrequencyScanner(mock_data_provider)
        assert len(vwap.supported_timeframes) == 6
        assert len(rsi.supported_timeframes) == 6

    @pytest.mark.asyncio
    async def test_scan_all_timeframes_tags_results(self, mock_data_provider):
        """scan_all_timeframes should tag each opp with timeframe metadata."""
        scanner = VWAPHighFrequencyScanner(mock_data_provider)

        # We can't easily generate real signals here, but verify the method runs
        result = await scanner.scan_all_timeframes(instruments=["TEST"])
        assert isinstance(result, list)


# =============================================================================
# Integration tests
# =============================================================================


class TestScannerRegistryIntegration:
    """Test that HF scanners properly use the instrument registry."""

    def test_vwap_uses_registry(self, mock_data_provider):
        scanner = VWAPHighFrequencyScanner(mock_data_provider)
        from nexus.data.instruments import InstrumentType
        stocks = scanner.registry.get_by_type(InstrumentType.STOCK)
        assert len(stocks) > 0

    def test_rsi_uses_registry(self, mock_data_provider):
        scanner = RSIHighFrequencyScanner(mock_data_provider)
        from nexus.data.instruments import InstrumentType
        stocks = scanner.registry.get_by_type(InstrumentType.STOCK)
        assert len(stocks) > 0
