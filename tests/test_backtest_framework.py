"""Tests for multi-asset backtesting framework."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from nexus.backtest.models import (
    BacktestTrade, MultiBacktestResult, EdgePerformance, TradeOutcome
)
from nexus.backtest.historical_loader import HistoricalDataLoader
from nexus.backtest.simulator import BacktestSimulator
from nexus.backtest.multi_engine import MultiBacktestEngine
from nexus.backtest.analyzer import AIAnalyzer, EdgeAnalysis
from nexus.backtest.reporter import BacktestReporter
from nexus.core.enums import EdgeType, Direction, Market, Timeframe
from nexus.core.models import Opportunity


# =============================================================================
# BacktestTrade model
# =============================================================================


class TestBacktestTrade:
    """Test BacktestTrade model."""

    def test_create_winning_trade(self):
        trade = BacktestTrade(
            trade_id="test123",
            signal_id="sig456",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            edge_type=EdgeType.RSI_EXTREME,
            timeframe=Timeframe.H1,
            score=70,
            tier="B",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            exit_price=155.0,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            hold_duration_minutes=60,
            gross_pnl=500.0,
            gross_pnl_pct=3.33,
            costs=50.0,
            net_pnl=450.0,
            net_pnl_pct=3.0,
            risk_amount=500.0,
            risk_pct=1.0,
            r_multiple=0.9,
            outcome=TradeOutcome.WIN,
            exit_reason="take_profit",
        )

        assert trade.is_winner is True
        assert trade.is_loser is False

    def test_create_losing_trade(self):
        trade = BacktestTrade(
            trade_id="test",
            signal_id="sig",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            edge_type=EdgeType.RSI_EXTREME,
            timeframe=Timeframe.H1,
            score=70,
            tier="B",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            exit_price=145.0,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            hold_duration_minutes=30,
            gross_pnl=-500.0,
            gross_pnl_pct=-3.33,
            costs=50.0,
            net_pnl=-550.0,
            net_pnl_pct=-3.67,
            risk_amount=500.0,
            risk_pct=1.0,
            r_multiple=-1.1,
            outcome=TradeOutcome.LOSS,
            exit_reason="stop_loss",
        )

        assert trade.is_winner is False
        assert trade.is_loser is True

    def test_trade_outcomes(self):
        assert TradeOutcome.WIN == "win"
        assert TradeOutcome.LOSS == "loss"
        assert TradeOutcome.STOPPED == "stopped"
        assert TradeOutcome.TARGET == "target"
        assert TradeOutcome.TIME_EXIT == "time_exit"


# =============================================================================
# EdgePerformance model
# =============================================================================


class TestEdgePerformance:
    """Test EdgePerformance model."""

    def test_win_rate_calculation(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)
        perf.total_trades = 100
        perf.winners = 55
        perf.losers = 45

        assert abs(perf.win_rate - 55.0) < 0.01

    def test_zero_trades(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)

        assert perf.win_rate == 0.0
        assert perf.expectancy == 0.0
        assert perf.avg_net_pct == 0.0

    def test_statistical_significance(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)

        perf.total_trades = 10
        assert perf.statistical_significance == "INSUFFICIENT"

        perf.total_trades = 50
        assert perf.statistical_significance == "MODERATE"

        perf.total_trades = 150
        assert perf.statistical_significance == "GOOD"

        perf.total_trades = 500
        assert perf.statistical_significance == "STRONG"

    def test_is_profitable(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)
        perf.net_pnl = 1000
        assert perf.is_profitable is True

        perf.net_pnl = -500
        assert perf.is_profitable is False

    def test_expectancy(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)
        perf.total_trades = 10
        perf.total_r_multiple = 5.0

        assert perf.expectancy == 0.5

    def test_cost_ratio(self):
        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)
        perf.gross_pnl = 10000
        perf.total_costs = 2000

        assert perf.cost_ratio == 20.0


# =============================================================================
# MultiBacktestResult
# =============================================================================


class TestMultiBacktestResult:
    """Test MultiBacktestResult model."""

    def test_net_return_pct(self):
        result = MultiBacktestResult(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            starting_balance=100000,
            ending_balance=120000,
        )

        assert result.net_return_pct == 20.0

    def test_win_rate(self):
        result = MultiBacktestResult(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            starting_balance=100000,
            ending_balance=120000,
        )
        result.total_trades = 100
        result.total_winners = 60

        assert result.win_rate == 60.0

    def test_avg_trade_pnl(self):
        result = MultiBacktestResult(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            starting_balance=100000,
            ending_balance=120000,
        )
        result.total_trades = 100
        result.net_pnl = 20000

        assert result.avg_trade_pnl == 200.0

    def test_zero_starting_balance(self):
        result = MultiBacktestResult(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            starting_balance=0,
            ending_balance=0,
        )
        assert result.net_return_pct == 0.0


# =============================================================================
# HistoricalDataLoader
# =============================================================================


class TestHistoricalDataLoader:
    """Test HistoricalDataLoader."""

    @pytest.fixture
    def loader(self):
        return HistoricalDataLoader()

    def test_synthetic_data_generation(self, loader):
        df = loader._generate_synthetic_data(
            symbol="AAPL",
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "timestamp" in df.columns

    def test_synthetic_data_reproducibility(self, loader):
        """Same symbol should produce same data."""
        df1 = loader._generate_synthetic_data(
            "AAPL", Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        df2 = loader._generate_synthetic_data(
            "AAPL", Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_symbols_different_data(self, loader):
        df_aapl = loader._generate_synthetic_data(
            "AAPL", Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        df_msft = loader._generate_synthetic_data(
            "MSFT", Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )

        # Different symbols should have different prices
        assert not df_aapl["close"].equals(df_msft["close"])

    def test_cache_path(self, loader):
        path = loader.get_cache_path("AAPL", Timeframe.H1)
        assert "AAPL_1h.parquet" in str(path)

    def test_cache_path_forex(self, loader):
        path = loader.get_cache_path("EUR/USD", Timeframe.H1)
        assert "EUR_USD_1h.parquet" in str(path)

    def test_date_filtering(self, loader):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz=timezone.utc),
            "close": range(100),
        })

        filtered = loader._filter_date_range(
            df,
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        assert len(filtered) < len(df)

    def test_empty_date_filter(self, loader):
        df = pd.DataFrame(columns=["timestamp", "close"])
        result = loader._filter_date_range(
            df,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert result.empty

    def test_timeframe_bar_count(self, loader):
        """Different timeframes should produce different bar counts."""
        df_h1 = loader._generate_synthetic_data(
            "AAPL", Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        )
        df_h4 = loader._generate_synthetic_data(
            "AAPL", Timeframe.H4,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        # H1 should have ~4x more bars than H4
        assert len(df_h1) > len(df_h4)


# =============================================================================
# BacktestSimulator
# =============================================================================


class TestBacktestSimulator:
    """Test BacktestSimulator."""

    @pytest.fixture
    def simulator(self):
        return BacktestSimulator(slippage_pct=0.02)

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz=timezone.utc)
        return pd.DataFrame({
            "timestamp": dates,
            "open": [100 + i * 0.1 for i in range(100)],
            "high": [101 + i * 0.1 for i in range(100)],
            "low": [99 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000000] * 100,
        })

    @pytest.fixture
    def sample_opportunity(self):
        return Opportunity(
            id="test123",
            detected_at=datetime.now(timezone.utc),
            scanner="test",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            primary_edge=EdgeType.RSI_EXTREME,
            edge_data={"timeframe": "1h", "timeframe_minutes": 60},
            raw_score=70,
        )

    def test_slippage_long_entry(self, simulator):
        entry = simulator._apply_slippage(100.0, Direction.LONG, is_entry=True)
        assert entry > 100.0

    def test_slippage_long_exit(self, simulator):
        exit_p = simulator._apply_slippage(100.0, Direction.LONG, is_entry=False)
        assert exit_p < 100.0

    def test_slippage_short_entry(self, simulator):
        entry = simulator._apply_slippage(100.0, Direction.SHORT, is_entry=True)
        assert entry < 100.0

    def test_slippage_short_exit(self, simulator):
        exit_p = simulator._apply_slippage(100.0, Direction.SHORT, is_entry=False)
        assert exit_p > 100.0

    def test_find_exit_stop_loss(self, simulator):
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "high": [105, 104, 103, 102, 101, 100, 99, 98, 97, 96],
            "low": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            "close": [102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
        })

        result = simulator._find_exit(
            data=data,
            entry_idx=0,
            direction=Direction.LONG,
            stop_loss=95.0,
            take_profit=110.0,
        )

        assert result is not None
        assert result[2] == "stop_loss"

    def test_find_exit_take_profit(self, simulator):
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "high": [102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
            "low": [99, 101, 103, 105, 107, 109, 111, 113, 115, 117],
            "close": [101, 103, 105, 107, 109, 111, 113, 115, 117, 119],
        })

        result = simulator._find_exit(
            data=data,
            entry_idx=0,
            direction=Direction.LONG,
            stop_loss=95.0,
            take_profit=110.0,
        )

        assert result is not None
        assert result[2] == "take_profit"

    def test_find_exit_time_exit(self, simulator):
        """Time exit when no stop or target hit."""
        sim = BacktestSimulator(slippage_pct=0.02, max_hold_bars=5)
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
        })

        result = sim._find_exit(
            data=data,
            entry_idx=0,
            direction=Direction.LONG,
            stop_loss=90.0,  # Very far
            take_profit=120.0,  # Very far
        )

        assert result is not None
        assert result[2] == "time_exit"

    def test_simulate_trade(self, simulator, sample_data, sample_opportunity):
        trade = simulator.simulate_trade(
            opportunity=sample_opportunity,
            data=sample_data,
            entry_bar_idx=10,
            starting_equity=100000,
            risk_pct=1.0,
        )

        assert trade is not None
        assert trade.symbol == "AAPL"

    def test_simulate_trade_at_end_of_data(self, simulator, sample_data, sample_opportunity):
        """Trade at end of data should return None."""
        trade = simulator.simulate_trade(
            opportunity=sample_opportunity,
            data=sample_data,
            entry_bar_idx=len(sample_data) - 1,
            starting_equity=100000,
        )

        assert trade is None

    def test_score_to_tier(self, simulator):
        assert simulator._score_to_tier(85) == "A"
        assert simulator._score_to_tier(70) == "B"
        assert simulator._score_to_tier(55) == "C"
        assert simulator._score_to_tier(40) == "D"
        assert simulator._score_to_tier(20) == "F"


# =============================================================================
# MultiBacktestEngine
# =============================================================================


class TestMultiBacktestEngine:
    """Test MultiBacktestEngine."""

    @pytest.fixture
    def engine(self):
        return MultiBacktestEngine(starting_balance=100000)

    def test_initialization(self, engine):
        assert engine.starting_balance == 100000
        assert engine.risk_per_trade == 1.0

    def test_rsi_calculation(self, engine):
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = engine._calculate_rsi(prices, 2)

        assert len(rsi) == len(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_calculation(self, engine):
        data = pd.DataFrame({
            "high": [102, 103, 104, 105, 106],
            "low": [98, 99, 100, 101, 102],
            "close": [100, 101, 102, 103, 104],
        })

        atr = engine._calculate_atr(data, 3)
        assert len(atr) == len(data)

    def test_vwap_calculation(self, engine):
        data = pd.DataFrame({
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "close": [100, 101, 102],
            "volume": [1000, 2000, 1500],
        })

        vwap = engine._calculate_vwap(data)
        assert len(vwap) == len(data)
        assert not vwap.isna().any()

    def test_bollinger_calculation(self, engine):
        data = pd.DataFrame({
            "close": [100 + i for i in range(30)],
        })

        upper, lower = engine._calculate_bollinger(data, period=20)

        assert len(upper) == len(data)
        assert len(lower) == len(data)

        valid_idx = ~upper.isna() & ~lower.isna()
        assert (upper[valid_idx] > lower[valid_idx]).all()

    def test_check_rsi_extreme_oversold(self, engine):
        """RSI < 20 should generate LONG signal."""
        bar = pd.Series({
            "rsi_2": 15.0,
            "close": 100.0,
            "atr": 2.0,
            "timestamp": datetime(2024, 6, 15, tzinfo=timezone.utc),
        })
        prev_bar = pd.Series({"rsi_2": 25.0, "close": 102.0})

        signal = engine._check_rsi_extreme(bar, prev_bar, "AAPL", Market.US_STOCKS, Timeframe.H1)

        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_check_rsi_extreme_overbought(self, engine):
        """RSI > 80 should generate SHORT signal."""
        bar = pd.Series({
            "rsi_2": 85.0,
            "close": 100.0,
            "atr": 2.0,
            "timestamp": datetime(2024, 6, 15, tzinfo=timezone.utc),
        })
        prev_bar = pd.Series({"rsi_2": 75.0, "close": 98.0})

        signal = engine._check_rsi_extreme(bar, prev_bar, "AAPL", Market.US_STOCKS, Timeframe.H1)

        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_check_rsi_normal_no_signal(self, engine):
        """RSI between 20-80 should NOT generate signal."""
        bar = pd.Series({
            "rsi_2": 50.0,
            "close": 100.0,
            "atr": 2.0,
            "timestamp": datetime(2024, 6, 15, tzinfo=timezone.utc),
        })
        prev_bar = pd.Series({"rsi_2": 48.0, "close": 99.0})

        signal = engine._check_rsi_extreme(bar, prev_bar, "AAPL", Market.US_STOCKS, Timeframe.H1)
        assert signal is None

    def test_check_gap_down(self, engine):
        """Gap down > 0.5% should generate LONG signal."""
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz=timezone.utc),
            "open": [100.0, 98.5, 99.0],  # Gap down of 1.5%
            "high": [101, 99.5, 100],
            "low": [99, 98, 98.5],
            "close": [100.0, 99.0, 99.5],
            "atr": [2.0, 2.0, 2.0],
        })

        signal = engine._check_gap(data, 1, "AAPL", Market.US_STOCKS, Timeframe.H1)

        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_check_gap_too_small(self, engine):
        """Gap < 0.5% should NOT generate signal."""
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz=timezone.utc),
            "open": [100.0, 99.8, 100.0],  # Gap of only 0.2%
            "high": [101, 100.5, 101],
            "low": [99, 99.5, 99.8],
            "close": [100.0, 100.2, 100.5],
            "atr": [2.0, 2.0, 2.0],
        })

        signal = engine._check_gap(data, 1, "AAPL", Market.US_STOCKS, Timeframe.H1)
        assert signal is None

    def test_check_turn_of_month(self, engine):
        """TOM window (day 28-31 or 1-3) should generate signal on D1."""
        bar = pd.Series({
            "timestamp": datetime(2024, 1, 30, tzinfo=timezone.utc),
            "close": 100.0,
            "atr": 2.0,
        })

        signal = engine._check_turn_of_month(bar, "SPY", Market.US_STOCKS, Timeframe.D1)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_check_turn_of_month_mid_month_no_signal(self, engine):
        """Mid-month should NOT trigger TOM."""
        bar = pd.Series({
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "close": 100.0,
            "atr": 2.0,
        })

        signal = engine._check_turn_of_month(bar, "SPY", Market.US_STOCKS, Timeframe.D1)
        assert signal is None

    def test_check_turn_of_month_intraday_no_signal(self, engine):
        """TOM should only fire on D1 timeframe."""
        bar = pd.Series({
            "timestamp": datetime(2024, 1, 30, tzinfo=timezone.utc),
            "close": 100.0,
            "atr": 2.0,
        })

        signal = engine._check_turn_of_month(bar, "SPY", Market.US_STOCKS, Timeframe.H1)
        assert signal is None


# =============================================================================
# AIAnalyzer
# =============================================================================


class TestAIAnalyzer:
    """Test AIAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return AIAnalyzer()

    def test_rule_based_analysis(self, analyzer):
        result = MagicMock()
        result.start_date = datetime(2024, 1, 1)
        result.end_date = datetime(2024, 12, 31)
        result.starting_balance = 100000
        result.ending_balance = 120000
        result.net_return_pct = 20.0
        result.total_trades = 100
        result.win_rate = 55.0
        result.max_drawdown_pct = 5.0

        perf = MagicMock()
        perf.total_trades = 50
        perf.win_rate = 60.0
        perf.net_pnl = 10000
        perf.expectancy = 0.5
        perf.is_profitable = True
        perf.statistical_significance = "GOOD"

        result.edge_performance = {"rsi_extreme": perf}
        result.timeframe_performance = {}

        analysis = analyzer._rule_based_analysis(result)

        assert "rsi_extreme" in analysis

    def test_parse_analysis_keep(self, analyzer):
        result = MagicMock()

        perf = MagicMock()
        perf.total_trades = 100
        perf.win_rate = 60.0
        perf.expectancy = 0.5
        perf.is_profitable = True
        perf.edge_type = EdgeType.RSI_EXTREME

        result.edge_performance = {"rsi_extreme": perf}
        result.net_pnl = 10000
        result.win_rate = 60.0

        parsed = analyzer._parse_analysis("Test analysis", result)

        assert parsed["edge_analyses"]["rsi_extreme"].recommendation == "KEEP"
        assert parsed["summary"]["overall_viable"] is True

    def test_parse_analysis_drop(self, analyzer):
        result = MagicMock()

        perf = MagicMock()
        perf.total_trades = 100
        perf.win_rate = 40.0
        perf.expectancy = -0.3
        perf.is_profitable = False
        perf.edge_type = EdgeType.BOLLINGER_TOUCH

        result.edge_performance = {"bollinger_touch": perf}
        result.net_pnl = -5000
        result.win_rate = 40.0

        parsed = analyzer._parse_analysis("Test analysis", result)

        assert parsed["edge_analyses"]["bollinger_touch"].recommendation == "DROP"

    def test_parse_analysis_needs_data(self, analyzer):
        result = MagicMock()

        perf = MagicMock()
        perf.total_trades = 10  # < 30
        perf.win_rate = 70.0
        perf.expectancy = 1.0
        perf.is_profitable = True
        perf.edge_type = EdgeType.GAP_FILL

        result.edge_performance = {"gap_fill": perf}
        result.net_pnl = 1000
        result.win_rate = 70.0

        parsed = analyzer._parse_analysis("Test analysis", result)

        assert parsed["edge_analyses"]["gap_fill"].recommendation == "NEEDS_MORE_DATA"

    def test_generate_report(self, analyzer):
        analysis = {
            "raw_analysis": "Test analysis text",
            "edge_analyses": {
                "rsi_extreme": EdgeAnalysis(
                    edge_type=EdgeType.RSI_EXTREME,
                    is_profitable=True,
                    recommendation="KEEP",
                    confidence="HIGH",
                    reasoning="Good performance",
                ),
            },
            "summary": {
                "total_edges_tested": 1,
                "profitable_edges": 1,
                "edges_to_keep": 1,
                "edges_to_drop": 0,
                "overall_viable": True,
            },
        }

        report = analyzer.generate_report(analysis)

        assert "NEXUS BACKTEST ANALYSIS" in report
        assert "rsi_extreme" in report
        assert "KEEP" in report


# =============================================================================
# BacktestReporter
# =============================================================================


class TestBacktestReporter:
    """Test BacktestReporter."""

    @pytest.fixture
    def reporter(self):
        return BacktestReporter()

    @pytest.fixture
    def mock_result(self):
        result = MultiBacktestResult(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            starting_balance=100000,
            ending_balance=120000,
        )
        result.total_trades = 100
        result.total_winners = 55
        result.total_losers = 45
        result.gross_pnl = 25000
        result.total_costs = 5000
        result.net_pnl = 20000
        result.max_drawdown = 5000
        result.max_drawdown_pct = 5.0

        perf = EdgePerformance(edge_type=EdgeType.RSI_EXTREME)
        perf.total_trades = 50
        perf.winners = 30
        perf.losers = 20
        perf.net_pnl = 10000
        result.edge_performance = {"rsi_extreme": perf}
        result.timeframe_performance = {}
        result.market_performance = {}

        return result

    def test_generate_summary(self, reporter, mock_result):
        summary = reporter.generate_summary(mock_result)

        assert "BACKTEST SUMMARY" in summary
        assert "100,000" in summary
        assert "120,000" in summary
        assert "100" in summary  # total trades

    def test_generate_edge_report(self, reporter, mock_result):
        report = reporter.generate_edge_report(mock_result)

        assert "EDGE TYPE" in report
        assert "RSI_EXTREME" in report

    def test_generate_timeframe_report(self, reporter, mock_result):
        report = reporter.generate_timeframe_report(mock_result)
        assert "TIMEFRAME" in report

    def test_generate_full_report(self, reporter, mock_result):
        report = reporter.generate_full_report(mock_result)

        assert "BACKTEST SUMMARY" in report
        assert "EDGE TYPE" in report

    def test_full_report_with_ai_analysis(self, reporter, mock_result):
        ai_analysis = {"raw_analysis": "AI says this is good"}
        report = reporter.generate_full_report(mock_result, ai_analysis=ai_analysis)

        assert "AI ANALYSIS" in report
        assert "AI says this is good" in report


# =============================================================================
# Integration
# =============================================================================


class TestIntegration:
    """Integration tests for multi-asset backtest framework."""

    @pytest.mark.asyncio
    async def test_full_backtest_flow(self):
        """Test complete backtest flow with synthetic data."""
        engine = MultiBacktestEngine(starting_balance=100000)

        result = await engine.run(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            symbols=["AAPL", "MSFT"],
            timeframes=[Timeframe.H1],
            edge_types=[EdgeType.RSI_EXTREME],
        )

        assert result is not None
        assert result.starting_balance == 100000

    @pytest.mark.asyncio
    async def test_backtest_with_reporting(self):
        """Test backtest with full reporting."""
        engine = MultiBacktestEngine(starting_balance=100000)

        result = await engine.run(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            symbols=["AAPL"],
            timeframes=[Timeframe.D1],
            edge_types=[EdgeType.TURN_OF_MONTH],
        )

        reporter = BacktestReporter()
        report = reporter.generate_full_report(result)

        assert "BACKTEST SUMMARY" in report
        assert "100,000" in report

    @pytest.mark.asyncio
    async def test_backtest_multiple_edges(self):
        """Test backtest with multiple edge types."""
        engine = MultiBacktestEngine(starting_balance=100000)

        result = await engine.run(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            symbols=["AAPL"],
            timeframes=[Timeframe.H1],
            edge_types=[EdgeType.RSI_EXTREME, EdgeType.BOLLINGER_TOUCH],
        )

        assert result is not None
        assert result.total_trades >= 0

    @pytest.mark.asyncio
    async def test_data_loader_in_engine(self):
        """Verify engine loads data correctly."""
        engine = MultiBacktestEngine(starting_balance=100000)

        data = await engine.data_loader.load_data(
            symbol="AAPL",
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert data is not None
        assert not data.empty
        assert len(data) > 50
