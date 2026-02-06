"""
NEXUS Interactive Brokers Provider

Connects to IBKR TWS or Gateway for stocks, futures, and options.
Uses ib_insync for async operations.

IMPORTANT: IBKR has a daily reset window from 23:45-00:45 ET
where the API is unavailable. We handle this gracefully.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Contract, Stock, Forex, Future, Order as IBOrder
from ib_insync import MarketOrder, LimitOrder, StopOrder

from config.settings import settings
from data.base import (
    BaseBroker,
    Quote,
    AccountInfo,
    Position,
    Order,
    OrderResult,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger(__name__)

# Eastern timezone for IBKR reset window
ET = ZoneInfo("America/New_York")


class IBKRProvider(BaseBroker):
    """
    Interactive Brokers data and execution provider.
    
    Features:
    - Stocks, futures, forex, options
    - Real-time and historical data
    - Order execution with all order types
    - Handles daily reset window (23:45-00:45 ET)
    """
    
    # IBKR daily reset window (ET)
    RESET_START = time(23, 45)
    RESET_END = time(0, 45)
    
    def __init__(self):
        super().__init__()
        self.ib = IB()
        self._host = settings.ibkr_host
        self._port = settings.ibkr_port
        self._client_id = settings.ibkr_client_id
        self._timeout = settings.ibkr_timeout
        self._reconnect_attempts = 3
        self._subscriptions: Dict[str, int] = {}  # symbol -> reqId
        
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    async def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        if self.is_reset_window():
            logger.warning("IBKR in daily reset window, connection will wait")
            await self._wait_for_reset_end()
        
        try:
            await asyncio.wait_for(
                self.ib.connectAsync(
                    host=self._host,
                    port=self._port,
                    clientId=self._client_id,
                ),
                timeout=self._timeout,
            )
            
            self._connected = True
            self._account_id = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else None
            
            # Set up event handlers
            self.ib.disconnectedEvent += self._on_disconnected
            self.ib.errorEvent += self._on_error
            
            logger.info(f"Connected to IBKR (account: {self._account_id})")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"IBKR connection timeout after {self._timeout}s")
            return False
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect after disconnect."""
        for attempt in range(self._reconnect_attempts):
            logger.info(f"IBKR reconnect attempt {attempt + 1}/{self._reconnect_attempts}")
            await asyncio.sleep(5 * (attempt + 1))  # Backoff
            
            if await self.connect():
                return True
        
        logger.error("IBKR reconnection failed after all attempts")
        return False
    
    def _on_disconnected(self) -> None:
        """Handle disconnect event."""
        self._connected = False
        logger.warning("IBKR disconnected")
        
        # Don't reconnect during reset window
        if not self.is_reset_window():
            asyncio.create_task(self._reconnect())
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any) -> None:
        """Handle error event."""
        # Filter out non-critical errors
        if errorCode in [2104, 2106, 2158]:  # Market data farm messages
            return
        logger.error(f"IBKR error {errorCode}: {errorString}")
    
    # =========================================================================
    # RESET WINDOW HANDLING
    # =========================================================================
    
    def is_reset_window(self) -> bool:
        """Check if currently in IBKR daily reset window."""
        now_et = datetime.now(ET).time()
        
        # Handle midnight crossing
        if self.RESET_START <= now_et or now_et <= self.RESET_END:
            return True
        return False
    
    def get_time_until_reset_end(self) -> int:
        """Get minutes until reset window ends."""
        if not self.is_reset_window():
            return 0
        
        now_et = datetime.now(ET)
        
        # Calculate end time
        if now_et.time() > self.RESET_START:
            # After 23:45, end is tomorrow at 00:45
            end = now_et.replace(hour=0, minute=45, second=0) + timedelta(days=1)
        else:
            # Before 00:45, end is today at 00:45
            end = now_et.replace(hour=0, minute=45, second=0)
        
        return int((end - now_et).total_seconds() / 60)
    
    async def _wait_for_reset_end(self) -> None:
        """Wait for reset window to end."""
        minutes = self.get_time_until_reset_end()
        if minutes > 0:
            logger.info(f"Waiting {minutes} minutes for IBKR reset window to end")
            await asyncio.sleep(minutes * 60 + 60)  # Add 1 minute buffer
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote."""
        contract = self._make_contract(symbol)
        
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, '', False, False)
        
        # Wait for data
        await asyncio.sleep(0.5)
        
        return Quote(
            symbol=symbol,
            bid=ticker.bid if ticker.bid > 0 else ticker.last,
            ask=ticker.ask if ticker.ask > 0 else ticker.last,
            last=ticker.last if ticker.last > 0 else ticker.close,
            volume=int(ticker.volume) if ticker.volume else 0,
            timestamp=datetime.now(),
        )
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get OHLCV bars."""
        contract = self._make_contract(symbol)
        self.ib.qualifyContracts(contract)
        
        # Convert timeframe to IBKR format
        bar_size = self._convert_timeframe(timeframe)
        duration = self._calculate_duration(timeframe, limit)
        
        end_dt = end_date or datetime.now()
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,  # Include extended hours
            formatDate=1,
        )
        
        if not bars:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame([{
            'timestamp': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': int(bar.volume),
        } for bar in bars])
        
        return df.tail(limit)
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> bool:
        """Subscribe to real-time quotes."""
        for symbol in symbols:
            contract = self._make_contract(symbol)
            self.ib.qualifyContracts(contract)
            
            ticker = self.ib.reqMktData(contract, '', False, False)
            self._subscriptions[symbol] = ticker.contract.conId
            
            # Set up callback
            ticker.updateEvent += lambda t, s=symbol: callback(Quote(
                symbol=s,
                bid=t.bid if t.bid > 0 else t.last,
                ask=t.ask if t.ask > 0 else t.last,
                last=t.last,
                volume=int(t.volume) if t.volume else 0,
                timestamp=datetime.now(),
            ))
        
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self._subscriptions:
                contract = self._make_contract(symbol)
                self.ib.cancelMktData(contract)
                del self._subscriptions[symbol]
    
    # =========================================================================
    # ACCOUNT
    # =========================================================================
    
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        self.ib.reqAccountSummary()
        await asyncio.sleep(0.5)
        
        summary = {s.tag: float(s.value) for s in self.ib.accountSummary() 
                   if s.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 
                               'GrossPositionValue', 'MaintMarginReq', 'AvailableFunds']}
        
        return AccountInfo(
            account_id=self._account_id or "",
            balance=summary.get('TotalCashValue', 0),
            equity=summary.get('NetLiquidation', 0),
            margin_used=summary.get('MaintMarginReq', 0),
            margin_available=summary.get('AvailableFunds', 0),
            currency="USD",  # IBKR base currency
            unrealized_pnl=summary.get('GrossPositionValue', 0) - summary.get('TotalCashValue', 0),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        ib_positions = self.ib.positions()
        
        positions = []
        for pos in ib_positions:
            if pos.position != 0:
                # Get current price
                contract = pos.contract
                self.ib.qualifyContracts(contract)
                ticker = self.ib.reqMktData(contract, '', False, False)
                await asyncio.sleep(0.3)
                
                current_price = ticker.last if ticker.last > 0 else pos.avgCost
                
                pnl = (current_price - pos.avgCost) * pos.position
                pnl_pct = ((current_price / pos.avgCost) - 1) * 100 if pos.avgCost > 0 else 0
                
                positions.append(Position(
                    symbol=self.convert_symbol_from_broker(contract.symbol),
                    direction="long" if pos.position > 0 else "short",
                    size=abs(pos.position),
                    entry_price=pos.avgCost,
                    current_price=current_price,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl_pct,
                    market_value=abs(pos.position * current_price),
                ))
        
        return positions
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    # =========================================================================
    # ORDERS
    # =========================================================================
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order."""
        contract = self._make_contract(order.symbol)
        self.ib.qualifyContracts(contract)
        
        # Determine action
        action = "BUY" if order.direction == "long" else "SELL"
        
        # Create IBKR order
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.size)
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.size, order.limit_price)
        elif order.order_type == OrderType.STOP:
            ib_order = StopOrder(action, order.size, order.stop_price)
        else:
            ib_order = MarketOrder(action, order.size)
        
        # Submit
        trade = self.ib.placeOrder(contract, ib_order)
        
        # Wait for fill (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_fill(trade),
                timeout=settings.order_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Order timeout for {order.symbol}")
        
        # Build result
        return OrderResult(
            order_id=str(trade.order.orderId),
            status=self._convert_order_status(trade.orderStatus.status),
            symbol=order.symbol,
            direction=order.direction,
            requested_size=order.size,
            filled_size=trade.orderStatus.filled,
            fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.filled > 0 else None,
            fill_time=datetime.now() if trade.orderStatus.filled > 0 else None,
            commission=sum(f.commission for f in trade.fills) if trade.fills else 0,
            message=trade.orderStatus.status,
        )
    
    async def _wait_for_fill(self, trade) -> None:
        """Wait for order to fill."""
        while not trade.isDone():
            await asyncio.sleep(0.1)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                return True
        return False
    
    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get order status."""
        for trade in self.ib.trades():
            if str(trade.order.orderId) == order_id:
                return OrderResult(
                    order_id=order_id,
                    status=self._convert_order_status(trade.orderStatus.status),
                    symbol=trade.contract.symbol,
                    direction="long" if trade.order.action == "BUY" else "short",
                    requested_size=trade.order.totalQuantity,
                    filled_size=trade.orderStatus.filled,
                    fill_price=trade.orderStatus.avgFillPrice,
                    message=trade.orderStatus.status,
                )
        return None
    
    async def get_open_orders(self) -> List[OrderResult]:
        """Get all open orders."""
        return [
            OrderResult(
                order_id=str(t.order.orderId),
                status=self._convert_order_status(t.orderStatus.status),
                symbol=t.contract.symbol,
                direction="long" if t.order.action == "BUY" else "short",
                requested_size=t.order.totalQuantity,
                filled_size=t.orderStatus.filled,
            )
            for t in self.ib.openTrades()
        ]
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
    ) -> OrderResult:
        """Close a position."""
        position = await self.get_position(symbol)
        if not position:
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                symbol=symbol,
                direction="close",
                requested_size=0,
                message="No position found",
            )
        
        close_size = size or position.size
        direction = "short" if position.direction == "long" else "long"
        
        return await self.place_order(Order(
            symbol=symbol,
            direction=direction,
            size=close_size,
            order_type=OrderType.MARKET,
        ))
    
    async def close_all_positions(self) -> List[OrderResult]:
        """Close all positions."""
        positions = await self.get_positions()
        results = []
        
        for pos in positions:
            result = await self.close_position(pos.symbol)
            results.append(result)
        
        return results
    
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        open_orders = self.ib.openTrades()
        for trade in open_orders:
            self.ib.cancelOrder(trade.order)
        return len(open_orders)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _make_contract(self, symbol: str) -> Contract:
        """Create IBKR contract from symbol."""
        # Detect contract type from symbol format
        if "/" in symbol:
            # Forex: EUR/USD
            pair = symbol.replace("/", "")
            return Forex(pair)
        elif symbol in ["ES", "NQ", "RTY", "CL", "GC"]:
            # Futures (front month)
            return Future(symbol, exchange='CME' if symbol in ["ES", "NQ", "RTY"] else 'NYMEX')
        else:
            # Stock
            return Stock(symbol, 'SMART', 'USD')
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert NEXUS timeframe to IBKR bar size."""
        mapping = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1 week",
        }
        return mapping.get(timeframe, "1 hour")
    
    def _calculate_duration(self, timeframe: str, limit: int) -> str:
        """Calculate IBKR duration string."""
        # Rough estimates
        if timeframe in ["1m", "5m"]:
            days = max(1, limit // 100)
            return f"{days} D"
        elif timeframe in ["15m", "30m"]:
            days = max(1, limit // 30)
            return f"{days} D"
        elif timeframe in ["1h", "4h"]:
            days = max(1, limit // 6)
            return f"{days} D"
        elif timeframe == "1d":
            return f"{limit} D"
        elif timeframe == "1w":
            return f"{limit * 7} D"
        return "30 D"
    
    def _convert_order_status(self, ib_status: str) -> OrderStatus:
        """Convert IBKR status to NEXUS status."""
        mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
        }
        return mapping.get(ib_status, OrderStatus.PENDING)
