"""
Position Exit Manager - handles exit logic for all edge types.

Matches backtest engine exit logic exactly:
- Overnight premium: MOO exit (1-bar hold, exit at next open)
- RSI extreme: Indicator exit (RSI(2) > 50 OR close > SMA(5) for longs)
- Gap fill / VWAP deviation: Stop/target/time expiry
- All edges: Max hold time enforcement

Ported from BacktestEngine._rsi_exit_checker() and
_simulate_overnight_trades() to ensure live matches backtest.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Max hold days per edge (from backtest EDGE_MAX_HOLD_BARS on daily bars)
MAX_HOLD_DAYS: Dict[str, int] = {
    "overnight_premium": 1,   # 1 bar: close to next open
    "gap_fill": 5,            # 5 days for momentum continuation
    "vwap_deviation": 10,     # 10 days on daily (trend-following)
    "rsi_extreme": 10,        # 10 days on daily
    "insider_cluster": 20,    # Swing trade
    "london_open": 1,         # Intraday (16 bars on 15m = 4h)
}

# MOC/MOO auction spread (0.5 bps round-trip — near-zero for index ETFs)
AUCTION_SPREAD_PCT = 0.005

# Standard slippage for non-auction exits
STANDARD_SLIPPAGE_PCT = 0.02


class ExitResult:
    """Result of an exit check for a single position."""

    def __init__(
        self,
        position: Dict,
        exit_price: float,
        exit_reason: str,
        gross_pnl: float,
        costs: float,
    ):
        self.position = position
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.gross_pnl = gross_pnl
        self.costs = costs
        self.net_pnl = gross_pnl - costs

    def __repr__(self) -> str:
        return (
            f"ExitResult({self.position['symbol']} {self.exit_reason} "
            f"@ ${self.exit_price:.2f}, PnL ${self.net_pnl:.2f})"
        )


class ExitManager:
    """Manages position exits for paper/live trading.

    Checks all open positions for exit conditions each cycle:
    1. Overnight premium: always exit next cycle (MOO, 1-bar hold)
    2. RSI extreme: indicator-based exit (RSI>50 or close>SMA5)
    3. Gap fill / VWAP: stop-loss, take-profit, or time expiry
    4. All edges: max hold days enforcement
    """

    def __init__(self, data_provider: Any = None):
        self.data_provider = data_provider

    async def check_exits(self, positions: List[Dict]) -> List[ExitResult]:
        """Check all open positions for exit conditions.

        Args:
            positions: List of position dicts from PaperTradingState.

        Returns:
            List of ExitResult for positions that should be closed.
        """
        exits: List[ExitResult] = []

        for pos in positions:
            try:
                result = await self._check_position(pos)
                if result:
                    exits.append(result)
            except Exception as e:
                logger.warning(
                    "Exit check failed for %s (%s): %s",
                    pos.get("symbol", "?"),
                    pos.get("edge", "?"),
                    e,
                )

        return exits

    async def _check_position(self, pos: Dict) -> Optional[ExitResult]:
        """Check a single position for exit conditions."""
        edge = pos.get("edge", "")

        if edge == "overnight_premium":
            return await self._check_overnight_exit(pos)
        elif edge == "rsi_extreme":
            return await self._check_rsi_exit(pos)
        else:
            return await self._check_standard_exit(pos)

    # ------------------------------------------------------------------
    # Overnight premium exit (MOO)
    # ------------------------------------------------------------------

    async def _check_overnight_exit(self, pos: Dict) -> Optional[ExitResult]:
        """Overnight premium: exit at today's open (MOO).

        Matches _simulate_overnight_trades(): entry at bar's close,
        exit at next bar's open. Always exits on next cycle.
        """
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]
        shares = pos.get("shares", 0)

        # Get today's bar to get the open price
        bars = await self._get_bars(symbol, "1D", 2)
        if bars is None or len(bars) < 1:
            # No data — force exit at entry price (flat)
            logger.warning(
                "Overnight %s: no data for MOO exit, closing flat", symbol
            )
            return ExitResult(
                position=pos,
                exit_price=entry_price,
                exit_reason="moo_exit_no_data",
                gross_pnl=0.0,
                costs=0.0,
            )

        # Exit at latest bar's open (MOO price)
        exit_price = float(bars.iloc[-1]["open"])

        # P&L (overnight is always LONG)
        gross_pnl = (exit_price - entry_price) * shares

        # MOC/MOO auction costs (near-zero slippage)
        trade_value = entry_price * shares
        costs = trade_value * (AUCTION_SPREAD_PCT / 100)

        logger.info(
            "Overnight EXIT: %s MOO @ $%.2f (entry $%.2f) | "
            "PnL $%.2f | Shares %d",
            symbol, exit_price, entry_price, gross_pnl - costs, shares,
        )

        return ExitResult(
            position=pos,
            exit_price=exit_price,
            exit_reason="moo_exit",
            gross_pnl=gross_pnl,
            costs=costs,
        )

    # ------------------------------------------------------------------
    # RSI indicator exit
    # ------------------------------------------------------------------

    async def _check_rsi_exit(self, pos: Dict) -> Optional[ExitResult]:
        """RSI extreme: indicator-based exit.

        Matches _rsi_exit_checker() exactly:
        - Long exit:  RSI(2) > 50  OR  close > 5 SMA
        - Short exit: RSI(2) < 50  OR  close < 5 SMA

        Also checks stop-loss, take-profit, and max hold time.
        """
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]
        shares = pos.get("shares", 0)
        is_long = pos.get("direction", "long") == "long"

        # Need enough bars for RSI(2) + SMA(5) warmup
        bars = await self._get_bars(symbol, "1D", 30)
        if bars is None or len(bars) < 10:
            return None

        # Calculate RSI(2) and SMA(5) using the same method as scanner
        bars = self._calculate_rsi_exit_indicators(bars)
        current = bars.iloc[-1]

        rsi = current.get("rsi_2")
        sma_5 = current.get("sma_5")
        close = float(current["close"])

        # Check indicator exit (matches _rsi_exit_checker exactly)
        if not pd.isna(rsi) and not pd.isna(sma_5):
            should_exit = False
            if is_long:
                should_exit = rsi > 50 or close > sma_5
            else:
                should_exit = rsi < 50 or close < sma_5

            if should_exit:
                exit_price = close
                gross_pnl = self._calc_pnl(
                    entry_price, exit_price, shares, is_long
                )
                costs = abs(exit_price * shares) * (STANDARD_SLIPPAGE_PCT / 100)

                logger.info(
                    "RSI EXIT: %s %s indicator | RSI(2)=%.1f SMA5=$%.2f "
                    "Close=$%.2f | PnL $%.2f",
                    symbol,
                    "LONG" if is_long else "SHORT",
                    rsi, sma_5, close,
                    gross_pnl - costs,
                )

                return ExitResult(
                    position=pos,
                    exit_price=exit_price,
                    exit_reason="indicator_exit",
                    gross_pnl=gross_pnl,
                    costs=costs,
                )

        # Fall through to standard checks (stop/target/time)
        return self._check_stop_target_time(pos, close)

    # ------------------------------------------------------------------
    # Standard exit (gap, vwap, etc.)
    # ------------------------------------------------------------------

    async def _check_standard_exit(self, pos: Dict) -> Optional[ExitResult]:
        """Standard exit: stop-loss, take-profit, or time expiry."""
        symbol = pos["symbol"]

        bars = await self._get_bars(symbol, "1D", 2)
        if bars is None or len(bars) < 1:
            return None

        close = float(bars.iloc[-1]["close"])
        return self._check_stop_target_time(pos, close)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_stop_target_time(
        self, pos: Dict, current_price: float
    ) -> Optional[ExitResult]:
        """Check stop-loss, take-profit, and max hold time."""
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]
        stop_loss = pos.get("stop_loss", 0)
        take_profit = pos.get("take_profit", 0)
        shares = pos.get("shares", 0)
        edge = pos.get("edge", "")
        is_long = pos.get("direction", "long") == "long"

        # Stop-loss check
        if is_long and stop_loss > 0 and current_price <= stop_loss:
            exit_price = stop_loss  # Assume stop fills at stop price
            gross_pnl = self._calc_pnl(entry_price, exit_price, shares, is_long)
            costs = abs(exit_price * shares) * (STANDARD_SLIPPAGE_PCT / 100)

            logger.info(
                "STOP LOSS: %s %s @ $%.2f (stop $%.2f) | PnL $%.2f",
                symbol, "LONG" if is_long else "SHORT",
                current_price, stop_loss, gross_pnl - costs,
            )

            return ExitResult(
                position=pos,
                exit_price=exit_price,
                exit_reason="stop_loss",
                gross_pnl=gross_pnl,
                costs=costs,
            )

        if not is_long and stop_loss > 0 and current_price >= stop_loss:
            exit_price = stop_loss
            gross_pnl = self._calc_pnl(entry_price, exit_price, shares, is_long)
            costs = abs(exit_price * shares) * (STANDARD_SLIPPAGE_PCT / 100)

            logger.info(
                "STOP LOSS: %s SHORT @ $%.2f (stop $%.2f) | PnL $%.2f",
                symbol, current_price, stop_loss, gross_pnl - costs,
            )

            return ExitResult(
                position=pos,
                exit_price=exit_price,
                exit_reason="stop_loss",
                gross_pnl=gross_pnl,
                costs=costs,
            )

        # Take-profit check
        if is_long and take_profit > 0 and current_price >= take_profit:
            exit_price = take_profit
            gross_pnl = self._calc_pnl(entry_price, exit_price, shares, is_long)
            costs = abs(exit_price * shares) * (STANDARD_SLIPPAGE_PCT / 100)

            logger.info(
                "TAKE PROFIT: %s LONG @ $%.2f (target $%.2f) | PnL $%.2f",
                symbol, current_price, take_profit, gross_pnl - costs,
            )

            return ExitResult(
                position=pos,
                exit_price=exit_price,
                exit_reason="take_profit",
                gross_pnl=gross_pnl,
                costs=costs,
            )

        if not is_long and take_profit > 0 and current_price <= take_profit:
            exit_price = take_profit
            gross_pnl = self._calc_pnl(entry_price, exit_price, shares, is_long)
            costs = abs(exit_price * shares) * (STANDARD_SLIPPAGE_PCT / 100)

            logger.info(
                "TAKE PROFIT: %s SHORT @ $%.2f (target $%.2f) | PnL $%.2f",
                symbol, current_price, take_profit, gross_pnl - costs,
            )

            return ExitResult(
                position=pos,
                exit_price=exit_price,
                exit_reason="take_profit",
                gross_pnl=gross_pnl,
                costs=costs,
            )

        # Max hold time check
        max_days = MAX_HOLD_DAYS.get(edge, 10)
        opened_at_str = pos.get("opened_at")
        if opened_at_str:
            try:
                opened_at = datetime.fromisoformat(opened_at_str)
                now = datetime.now(timezone.utc)
                days_held = (now - opened_at).days
                if days_held >= max_days:
                    exit_price = current_price
                    gross_pnl = self._calc_pnl(
                        entry_price, exit_price, shares, is_long
                    )
                    costs = abs(exit_price * shares) * (
                        STANDARD_SLIPPAGE_PCT / 100
                    )

                    logger.info(
                        "TIME EXPIRY: %s %s after %d days (max %d) | PnL $%.2f",
                        symbol, edge, days_held, max_days, gross_pnl - costs,
                    )

                    return ExitResult(
                        position=pos,
                        exit_price=exit_price,
                        exit_reason="time_expiry",
                        gross_pnl=gross_pnl,
                        costs=costs,
                    )
            except (ValueError, TypeError):
                pass

        return None

    @staticmethod
    def _calc_pnl(
        entry: float, exit: float, shares: int, is_long: bool
    ) -> float:
        """Calculate gross P&L."""
        if is_long:
            return (exit - entry) * shares
        else:
            return (entry - exit) * shares

    @staticmethod
    def _calculate_rsi_exit_indicators(bars: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI(2) and SMA(5) for exit checking.

        Matches _prepare_rsi_indicators() in backtest engine.
        """
        delta = bars["close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=2, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=2, adjust=False).mean()
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        bars["rsi_2"] = 100 - (100 / (1 + rs))
        bars["sma_5"] = bars["close"].rolling(window=5).mean()
        return bars

    async def _get_bars(
        self, symbol: str, timeframe: str, count: int
    ) -> Optional[pd.DataFrame]:
        """Get bars from data provider, with fallback."""
        if self.data_provider is None:
            return None

        try:
            bars = await self.data_provider.get_bars(symbol, timeframe, count)
            if bars is not None and len(bars) > 0:
                return bars
        except Exception as e:
            logger.debug("Failed to get bars for %s: %s", symbol, e)

        return None
