"""
NEXUS Base Scanner

Abstract base class for all edge scanners. Each scanner detects one specific
type of trading edge and returns Opportunity objects for scoring and execution.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.enums import Direction, EdgeType, Market
from core.models import Opportunity
from intelligence.regime import MarketRegime


class BaseScanner(ABC):
    """
    Abstract base class for all edge scanners.

    Defines the contract: subclasses must implement scan() and is_active(), and
    may override edge_type, markets, instruments, and name. Provides shared
    helpers for ATR/RSI, entry/stop/target calculation, opportunity creation,
    and symbol-to-market inference.
    """

    def __init__(
        self,
        data_provider: Any = None,
        settings: Any = None,
    ) -> None:
        """
        Initialize the scanner with configuration.

        Args:
            data_provider: Data source for prices/bars. Optional for Phase 2
                (placeholder data used when None). Required in Phase 3 when
                connecting real providers (IBKR, Polygon, etc.).
            settings: NexusSettings configuration object (or dict). Stored
                as-is for use by subclasses.
        """
        self.data = data_provider
        self.settings = settings
        # Override in subclass or set after init
        self.edge_type: Optional[EdgeType] = None
        self.markets: List[Market] = []
        self.instruments: List[str] = []
        self.name: str = self.__class__.__name__

    async def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol.

        If data_provider is connected, use it.
        Otherwise return placeholder for testing.
        """
        if self.data is not None:
            quote = await self.data.get_quote(symbol)
            return quote.last

        # Placeholder prices for testing
        placeholders = {
            "SPY": 500.0,
            "QQQ": 450.0,
            "IWM": 200.0,
            "ES": 5000.0,
            "NQ": 18000.0,
        }
        return placeholders.get(symbol, 100.0)

    async def get_bars(
        self, symbol: str, timeframe: str = "1D", limit: int = 30
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for symbol.

        If data_provider is connected, use it.
        Otherwise return placeholder DataFrame for testing.
        """
        if self.data is not None:
            return await self.data.get_bars(symbol, timeframe, limit)

        # Placeholder DataFrame for testing
        import numpy as np

        dates = pd.date_range(end=datetime.now(), periods=limit, freq="D")
        base_price = await self.get_current_price(symbol)

        # Generate realistic-looking OHLCV data
        np.random.seed(42)  # Reproducible
        returns = np.random.normal(0, 0.02, limit)  # 2% daily vol
        closes = base_price * np.cumprod(1 + returns)

        return pd.DataFrame(
            {
                "open": closes * (1 + np.random.uniform(-0.01, 0.01, limit)),
                "high": closes * (1 + np.random.uniform(0, 0.02, limit)),
                "low": closes * (1 - np.random.uniform(0, 0.02, limit)),
                "close": closes,
                "volume": np.random.randint(1000000, 10000000, limit),
            },
            index=dates,
        )

    async def get_bars_safe(
        self, symbol: str, timeframe: str = "1D", limit: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV bars for symbol; return None on failure.
        """
        try:
            return await self.get_bars(symbol, timeframe, limit)
        except Exception:
            return None

    @abstractmethod
    async def scan(self) -> List[Opportunity]:
        """
        Run the scanner and return opportunities.

        Subclasses fetch data, apply edge logic, and return a list of
        Opportunity instances. Scoring and filtering happen downstream.

        Returns:
            List of Opportunity objects detected by this scanner.
        """
        pass

    @abstractmethod
    def is_active(self, timestamp: datetime) -> bool:
        """
        Check if the scanner should run at the given time.

        Used by the orchestrator to decide whether to call scan() (e.g.
        session-based scanners only active during certain hours).

        Args:
            timestamp: The time to check (typically current UTC).

        Returns:
            True if this scanner should be run at that time.
        """
        pass

    def calculate_entry_stop_target(
        self,
        current_price: float,
        direction: Direction,
        atr: float,
        stop_multiplier: float = 1.5,
        target_multiplier: float = 2.5,
    ) -> tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit using ATR.

        For LONG:
          - entry = current_price
          - stop = current_price - (atr * stop_multiplier)
          - target = current_price + (atr * target_multiplier)

        For SHORT:
          - entry = current_price
          - stop = current_price + (atr * stop_multiplier)
          - target = current_price - (atr * target_multiplier)

        Args:
            current_price: Current market price.
            direction: LONG or SHORT.
            atr: Average True Range value.
            stop_multiplier: ATR multiplier for stop loss.
            target_multiplier: ATR multiplier for take profit.

        Returns:
            (entry, stop_loss, take_profit).
        """
        if direction == Direction.LONG:
            entry = current_price
            stop = current_price - (atr * stop_multiplier)
            target = current_price + (atr * target_multiplier)
        else:
            entry = current_price
            stop = current_price + (atr * stop_multiplier)
            target = current_price - (atr * target_multiplier)
        return (entry, stop, target)

    def create_opportunity(
        self,
        symbol: str,
        market: Market,
        direction: Direction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        edge_data: Optional[Dict[str, Any]] = None,
        secondary_edges: Optional[List[EdgeType]] = None,
        valid_until: Optional[datetime] = None,
    ) -> Opportunity:
        """
        Create an Opportunity with all required fields.

        Sets id (UUID), detected_at (UTC now), scanner (self.name),
        primary_edge (self.edge_type), and raw_score/adjusted_score to 0 for
        the scorer to fill.

        Args:
            symbol: Instrument symbol.
            market: Market enum.
            direction: LONG or SHORT.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            edge_data: Optional scanner-specific edge data.
            secondary_edges: Optional list of additional edge types.
            valid_until: Optional expiry time for the opportunity.

        Returns:
            Opportunity instance ready for scoring.
        """
        return Opportunity(
            id=str(uuid.uuid4()),
            detected_at=datetime.now(timezone.utc),
            scanner=self.name,
            symbol=symbol,
            market=market,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            primary_edge=self.edge_type,
            secondary_edges=secondary_edges or [],
            edge_data=edge_data or {},
            raw_score=0,
            adjusted_score=0,
            valid_until=valid_until,
        )

    @staticmethod
    def calculate_atr(bars: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range from OHLC bars.

        True Range = max of:
          - high - low
          - abs(high - previous_close)
          - abs(low - previous_close)

        ATR = rolling mean of TR over period.

        Args:
            bars: DataFrame with columns high, low, close.
            period: Rolling window for ATR.

        Returns:
            The most recent ATR value.
        """
        if bars.empty or len(bars) < 2:
            return 0.0
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period, min_periods=1).mean()
        last = atr_series.iloc[-1]
        return float(last) if pd.notna(last) else 0.0

    @staticmethod
    def calculate_rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI from close prices.

        RSI = 100 - (100 / (1 + RS)), RS = avg_gain / avg_loss over period.

        Args:
            bars: DataFrame with column close.
            period: RSI period.

        Returns:
            Full RSI series (same length as bars; early values may be NaN).
        """
        if bars.empty or "close" not in bars.columns:
            return pd.Series(dtype=float)
        close = bars["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_market_for_symbol(self, symbol: str) -> Market:
        """
        Infer market from symbol.

        Rules:
          - If "/" in symbol (e.g. EUR/USD): FOREX_MAJORS
          - If symbol in ["ES", "NQ", "RTY", "CL", "GC"]: US_FUTURES
          - If symbol ends with ".L": UK_STOCKS
          - Default: US_STOCKS

        Args:
            symbol: Instrument symbol.

        Returns:
            Inferred Market enum.
        """
        s = symbol.strip().upper()
        if "/" in symbol:
            return Market.FOREX_MAJORS
        if s in ("ES", "NQ", "RTY", "CL", "GC"):
            return Market.US_FUTURES
        if symbol.endswith(".L"):
            return Market.UK_STOCKS
        return Market.US_STOCKS
