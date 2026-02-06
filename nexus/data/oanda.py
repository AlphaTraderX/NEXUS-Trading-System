"""
NEXUS OANDA Provider

Forex specialist broker with excellent API and tight spreads.
Uses REST API v20 for all operations.

Markets: Forex (majors, minors, exotics)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd

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


class OANDAProvider(BaseBroker):
    """
    OANDA forex broker provider.
    
    Features:
    - Forex majors, minors, exotics
    - Tight spreads on majors
    - Excellent REST API (v20)
    - Streaming prices available
    """
    
    # API URLs
    LIVE_URL = "https://api-fxtrade.oanda.com/v3"
    PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
    STREAM_LIVE = "https://stream-fxtrade.oanda.com/v3"
    STREAM_PRACTICE = "https://stream-fxpractice.oanda.com/v3"
    
    def __init__(self):
        super().__init__()
        self._account_id = settings.oanda_account_id
        self._api_key = settings.oanda_api_key
        self._practice = settings.oanda_practice
        
        self._base_url = self.PRACTICE_URL if self._practice else self.LIVE_URL
        self._stream_url = self.STREAM_PRACTICE if self._practice else self.STREAM_LIVE
        
        self._client = httpx.AsyncClient(timeout=30.0)
        self._subscriptions: Dict[str, Callable] = {}
    
    def _auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    async def connect(self) -> bool:
        """Connect to OANDA."""
        try:
            # Test connection by getting account
            response = await self._client.get(
                f"{self._base_url}/accounts/{self._account_id}",
                headers=self._auth_headers(),
            )
            
            if response.status_code != 200:
                logger.error(f"OANDA auth failed: {response.status_code} - {response.text}")
                return False
            
            self._connected = True
            logger.info(f"Connected to OANDA (account: {self._account_id}, practice: {self._practice})")
            return True
            
        except Exception as e:
            logger.error(f"OANDA connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OANDA."""
        self._connected = False
        self._subscriptions.clear()
        logger.info("Disconnected from OANDA")
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol."""
        instrument = self.convert_symbol(symbol)
        
        response = await self._client.get(
            f"{self._base_url}/accounts/{self._account_id}/pricing",
            headers=self._auth_headers(),
            params={"instruments": instrument},
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get quote: {response.text}")
        
        data = response.json()
        prices = data.get("prices", [])
        
        if not prices:
            raise Exception(f"No price data for {symbol}")
        
        price = prices[0]
        
        # OANDA returns arrays of bids/asks
        bids = price.get("bids", [{}])
        asks = price.get("asks", [{}])
        
        bid = float(bids[0].get("price", 0)) if bids else 0
        ask = float(asks[0].get("price", 0)) if asks else 0
        
        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,  # OANDA doesn't provide last
            volume=0,  # Forex doesn't have traditional volume
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
        instrument = self.convert_symbol(symbol)
        granularity = self._convert_timeframe(timeframe)
        
        params = {
            "granularity": granularity,
            "count": limit,
            "price": "MBA",  # Mid, Bid, Ask
        }
        
        if end_date:
            params["to"] = end_date.isoformat() + "Z"
        
        response = await self._client.get(
            f"{self._base_url}/instruments/{instrument}/candles",
            headers=self._auth_headers(),
            params=params,
        )
        
        if response.status_code != 200:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        data = response.json()
        candles = data.get("candles", [])
        
        if not candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Use mid prices
        df = pd.DataFrame([{
            'timestamp': c.get("time"),
            'open': float(c.get("mid", {}).get("o", 0)),
            'high': float(c.get("mid", {}).get("h", 0)),
            'low': float(c.get("mid", {}).get("l", 0)),
            'close': float(c.get("mid", {}).get("c", 0)),
            'volume': int(c.get("volume", 0)),
        } for c in candles if c.get("complete", True)])
        
        return df
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> bool:
        """Subscribe to real-time quotes."""
        for symbol in symbols:
            self._subscriptions[symbol] = callback
        
        # Note: Full implementation would use streaming endpoint
        logger.info(f"Subscribed to {len(symbols)} symbols (polling mode)")
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            self._subscriptions.pop(symbol, None)
    
    # =========================================================================
    # ACCOUNT
    # =========================================================================
    
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        response = await self._client.get(
            f"{self._base_url}/accounts/{self._account_id}/summary",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get account: {response.text}")
        
        data = response.json()
        account = data.get("account", {})
        
        return AccountInfo(
            account_id=self._account_id,
            balance=float(account.get("balance", 0)),
            equity=float(account.get("NAV", 0)),
            margin_used=float(account.get("marginUsed", 0)),
            margin_available=float(account.get("marginAvailable", 0)),
            currency=account.get("currency", "USD"),
            unrealized_pnl=float(account.get("unrealizedPL", 0)),
            realized_pnl_today=float(account.get("pl", 0)),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        response = await self._client.get(
            f"{self._base_url}/accounts/{self._account_id}/openPositions",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        positions = []
        
        for pos in data.get("positions", []):
            instrument = pos.get("instrument", "")
            
            # OANDA separates long and short
            long_data = pos.get("long", {})
            short_data = pos.get("short", {})
            
            if float(long_data.get("units", 0)) != 0:
                units = float(long_data.get("units", 0))
                avg_price = float(long_data.get("averagePrice", 0))
                unrealized = float(long_data.get("unrealizedPL", 0))
                
                positions.append(Position(
                    symbol=self.convert_symbol_from_broker(instrument),
                    direction="long",
                    size=abs(units),
                    entry_price=avg_price,
                    current_price=avg_price,  # Would need separate quote
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=(unrealized / (abs(units) * avg_price)) * 100 if avg_price > 0 else 0,
                    market_value=abs(units) * avg_price,
                ))
            
            if float(short_data.get("units", 0)) != 0:
                units = float(short_data.get("units", 0))
                avg_price = float(short_data.get("averagePrice", 0))
                unrealized = float(short_data.get("unrealizedPL", 0))
                
                positions.append(Position(
                    symbol=self.convert_symbol_from_broker(instrument),
                    direction="short",
                    size=abs(units),
                    entry_price=avg_price,
                    current_price=avg_price,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=(unrealized / (abs(units) * avg_price)) * 100 if avg_price > 0 else 0,
                    market_value=abs(units) * avg_price,
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
        instrument = self.convert_symbol(order.symbol)
        
        # OANDA uses negative units for sell
        units = order.size if order.direction == "long" else -order.size
        
        # Build order payload
        order_spec: Dict[str, Any] = {
            "instrument": instrument,
            "units": str(int(units)),  # OANDA wants string
            "type": "MARKET" if order.order_type == OrderType.MARKET else "LIMIT",
            "positionFill": "DEFAULT",
        }
        
        if order.order_type == OrderType.LIMIT and order.limit_price:
            order_spec["price"] = str(order.limit_price)
        
        if order.stop_loss:
            order_spec["stopLossOnFill"] = {
                "price": str(order.stop_loss),
            }
        
        if order.take_profit:
            order_spec["takeProfitOnFill"] = {
                "price": str(order.take_profit),
            }
        
        response = await self._client.post(
            f"{self._base_url}/accounts/{self._account_id}/orders",
            headers=self._auth_headers(),
            json={"order": order_spec},
        )
        
        if response.status_code not in [200, 201]:
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                requested_size=order.size,
                message=response.text,
            )
        
        data = response.json()
        
        # Check if filled immediately (market order)
        if "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            return OrderResult(
                order_id=fill.get("id", ""),
                status=OrderStatus.FILLED,
                symbol=order.symbol,
                direction=order.direction,
                requested_size=order.size,
                filled_size=abs(float(fill.get("units", 0))),
                fill_price=float(fill.get("price", 0)),
                fill_time=datetime.now(),
                commission=float(fill.get("commission", 0)),
                message="Filled",
            )
        
        # Pending order
        if "orderCreateTransaction" in data:
            create = data["orderCreateTransaction"]
            return OrderResult(
                order_id=create.get("id", ""),
                status=OrderStatus.SUBMITTED,
                symbol=order.symbol,
                direction=order.direction,
                requested_size=order.size,
                message="Order submitted",
            )
        
        return OrderResult(
            order_id="",
            status=OrderStatus.REJECTED,
            symbol=order.symbol,
            direction=order.direction,
            requested_size=order.size,
            message="Unknown response",
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        response = await self._client.put(
            f"{self._base_url}/accounts/{self._account_id}/orders/{order_id}/cancel",
            headers=self._auth_headers(),
        )
        return response.status_code == 200
    
    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get order status."""
        response = await self._client.get(
            f"{self._base_url}/accounts/{self._account_id}/orders/{order_id}",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        order = data.get("order", {})
        
        return OrderResult(
            order_id=order_id,
            status=self._convert_order_status(order.get("state", "")),
            symbol=self.convert_symbol_from_broker(order.get("instrument", "")),
            direction="long" if float(order.get("units", 0)) > 0 else "short",
            requested_size=abs(float(order.get("units", 0))),
            filled_size=abs(float(order.get("filledUnits", 0))),
        )
    
    async def get_open_orders(self) -> List[OrderResult]:
        """Get all open orders."""
        response = await self._client.get(
            f"{self._base_url}/accounts/{self._account_id}/pendingOrders",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        orders = []
        
        for order in data.get("orders", []):
            units = float(order.get("units", 0))
            orders.append(OrderResult(
                order_id=order.get("id", ""),
                status=OrderStatus.PENDING,
                symbol=self.convert_symbol_from_broker(order.get("instrument", "")),
                direction="long" if units > 0 else "short",
                requested_size=abs(units),
            ))
        
        return orders
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
    ) -> OrderResult:
        """Close a position."""
        instrument = self.convert_symbol(symbol)
        
        # Determine which side to close
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
        
        close_units = size or position.size
        
        if position.direction == "long":
            payload = {"longUnits": "ALL" if size is None else str(int(close_units))}
        else:
            payload = {"shortUnits": "ALL" if size is None else str(int(close_units))}
        
        response = await self._client.put(
            f"{self._base_url}/accounts/{self._account_id}/positions/{instrument}/close",
            headers=self._auth_headers(),
            json=payload,
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Get the close transaction
            if "longOrderFillTransaction" in data:
                fill = data["longOrderFillTransaction"]
            elif "shortOrderFillTransaction" in data:
                fill = data["shortOrderFillTransaction"]
            else:
                fill = {}
            
            return OrderResult(
                order_id=fill.get("id", ""),
                status=OrderStatus.FILLED,
                symbol=symbol,
                direction="close",
                requested_size=close_units,
                filled_size=abs(float(fill.get("units", close_units))),
                fill_price=float(fill.get("price", 0)),
                fill_time=datetime.now(),
                message="Position closed",
            )
        
        return OrderResult(
            order_id="",
            status=OrderStatus.REJECTED,
            symbol=symbol,
            direction="close",
            requested_size=close_units,
            message=response.text,
        )
    
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
        orders = await self.get_open_orders()
        count = 0
        
        for order in orders:
            if await self.cancel_order(order.order_id):
                count += 1
        
        return count
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def convert_symbol(self, nexus_symbol: str) -> str:
        """Convert NEXUS symbol to OANDA instrument."""
        # EUR/USD -> EUR_USD
        return nexus_symbol.replace("/", "_")
    
    def convert_symbol_from_broker(self, oanda_instrument: str) -> str:
        """Convert OANDA instrument to NEXUS symbol."""
        # EUR_USD -> EUR/USD
        return oanda_instrument.replace("_", "/")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert NEXUS timeframe to OANDA granularity."""
        mapping = {
            "1m": "M1",
            "5m": "M5",
            "15m": "M15",
            "30m": "M30",
            "1h": "H1",
            "4h": "H4",
            "1d": "D",
            "1w": "W",
        }
        return mapping.get(timeframe, "H1")
    
    def _convert_order_status(self, oanda_status: str) -> OrderStatus:
        """Convert OANDA status to NEXUS status."""
        mapping = {
            "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "TRIGGERED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
        }
        return mapping.get(oanda_status, OrderStatus.PENDING)
