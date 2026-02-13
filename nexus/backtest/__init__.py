"""NEXUS Backtesting Framework."""

# Existing backtest module
from .data_loader import BacktestDataLoader
from .engine import BacktestEngine, BacktestResult
from .statistics import BacktestStatistics, StatisticsCalculator
from .trade_simulator import ExitReason, SimulatedTrade, TradeSimulator

# Multi-asset backtesting framework
from .multi_engine import MultiBacktestEngine
from .historical_loader import HistoricalDataLoader
from .simulator import BacktestSimulator
from .reporter import BacktestReporter
from .analyzer import AIAnalyzer, EdgeAnalysis
from .models import (
    BacktestTrade,
    MultiBacktestResult,
    EdgePerformance,
    TradeOutcome,
)

__all__ = [
    # Existing
    "BacktestDataLoader",
    "BacktestEngine",
    "BacktestResult",
    "BacktestStatistics",
    "ExitReason",
    "SimulatedTrade",
    "StatisticsCalculator",
    "TradeSimulator",
    # Multi-asset framework
    "MultiBacktestEngine",
    "HistoricalDataLoader",
    "BacktestSimulator",
    "BacktestReporter",
    "AIAnalyzer",
    "EdgeAnalysis",
    "BacktestTrade",
    "MultiBacktestResult",
    "EdgePerformance",
    "TradeOutcome",
]
