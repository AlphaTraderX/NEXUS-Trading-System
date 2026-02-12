"""
NEXUS OANDA Data Provider

Connects to OANDA v20 REST API for forex market data and execution.
Supports both practice (demo) and live accounts.

API Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from .base import (
    BaseBroker,
    ReconnectionMixin,
    Quote,
    AccountInfo,
    Position,
    Order,
    OrderResult,
    normalize_timeframe,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


class OANDAProvider(ReconnectionMixin, BaseBroker):
    """
    OANDA v20 API provider for forex trading.
    
    Provides:
    - Real-time forex quotes
    - Historical OHLCV bars
    - Account management
    - Order execution
    
    Supports both practice (demo) and live environments.
    """
    
    # API endpoints
    PRACTICE_REST = "https://api-fxpractice.oanda.com"
    PRACTICE_STREAM = "https://stream-fxpractice.oanda.com"
    LIVE_REST = "https://api-fxtrade.oanda.com"
    LIVE_STREAM = "https://stream-fxtrade.oanda.com"
    
    # Timeframe mapping to OANDA granularity
    GRANULARITY_MAP = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1D": "D",
        "1W": "W",
        "1M": "M",
    }
    
    # Common forex pairs
    FOREX_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY",
    ]
    
    def __init__(
        self,
        api_key: str,
        account_id: str,
        practice: bool = True
    ):
        """
        Initialize OANDA provider.

        Args:
            api_key: OANDA API token
            account_id: OANDA account ID (e.g., "101-004-12345678-001")
            practice: Use practice/demo environment (default True)
        """
        super().__init__()
        self._init_reconnection()
        self.api_key = api_key
        self.account_id = account_id
        self.practice = practice
        
        # Set URLs based on environment
        self.rest_url = self.PRACTICE_REST if practice else self.LIVE_REST
        self.stream_url = self.PRACTICE_STREAM if practice else self.LIVE_STREAM
        
        self.client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
    
    async def connect(self) -> bool:
        """Initialize HTTP client and verify connection."""
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers=self._headers
            )
            
            # Test connection by fetching account
            account = await self.get_account()
            
            if account:
                self._connected = True
                logger.info(f"Connected to OANDA {'Practice' if self.practice else 'Live'} - Account: {self.account_id}")
                logger.info(f"Balance: {account.currency} {account.balance:,.2f}")
                return True
            else:
                logger.error("Failed to fetch OANDA account")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._connected = False
        logger.info("Disconnected from OANDA")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None
    ) -> Optional[Dict]:
        """Make authenticated request to OANDA API."""
        if not self.client:
            logger.error("OANDA not connected. Call connect() first.")
            return None

        url = f"{self.rest_url}{endpoint}"

        try:
            response = await self.client.request(
                method,
                url,
                params=params,
                json=json_data
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            logger.warning(f"OANDA timeout for {endpoint}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"OANDA HTTP error for {endpoint}: {e.response.status_code}")
            if e.response.status_code == 429:
                logger.warning("OANDA rate limit hit - backing off")
            return None
        except httpx.RequestError as e:
            logger.error(f"OANDA request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"OANDA JSON decode error for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"OANDA unexpected error for {endpoint}: {e}")
            return None
    
    # ==================== Data Methods ====================
    
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote for a forex pair.

        Args:
            symbol: Forex pair (e.g., "EUR/USD" or "EUR_USD")
        """
        instrument = self._normalize_symbol(symbol)

        endpoint = f"/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}

        data = await self._request("GET", endpoint, params=params)
        if data is None:
            return None

        prices = data.get("prices", [])
        if not prices:
            logger.warning(f"No pricing data for {symbol}")
            return None

        price = prices[0]

        # Get bid/ask from price buckets
        bids = price.get("bids", [{}])
        asks = price.get("asks", [{}])

        bid = float(bids[0].get("price", 0)) if bids else 0
        ask = float(asks[0].get("price", 0)) if asks else 0

        try:
            from nexus.risk.kill_switch import get_kill_switch
            kill_switch = get_kill_switch()
            if kill_switch:
                kill_switch.update_data_timestamp()
        except Exception:
            pass  # Don't crash on kill switch update failure

        try:
            ts = price.get("time", "").replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(ts) if ts else datetime.now(UTC)
        except Exception:
            timestamp = datetime.now(UTC)

        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,  # Mid price as "last"
            volume=0,  # Forex doesn't have volume in the same way
            timestamp=timestamp,
        )
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars for a forex pair.

        Args:
            symbol: Forex pair (e.g., "EUR/USD" or "EUR_USD")
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 4h, 1D)
            limit: Number of bars to fetch (max 5000)
            end_date: End date for bars (default: now)
        """
        instrument = self._normalize_symbol(symbol)
        tf = normalize_timeframe(timeframe)

        if tf not in self.GRANULARITY_MAP:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        granularity = self.GRANULARITY_MAP[tf]

        endpoint = f"/v3/instruments/{instrument}/candles"

        params = {
            "granularity": granularity,
            "count": min(limit, 5000),
            "price": "MBA",  # Mid, Bid, Ask
        }

        if end_date:
            params["to"] = end_date.isoformat()

        data = await self._request("GET", endpoint, params=params)
        if data is None:
            return None

        candles = data.get("candles", [])
        
        if not candles:
            logger.warning(f"No candle data for {symbol} {timeframe}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Parse candles into DataFrame
        rows = []
        for candle in candles:
            if not candle.get("complete", True):
                continue  # Skip incomplete candles
            
            mid = candle.get("mid", {})
            
            rows.append({
                "timestamp": datetime.fromisoformat(candle["time"].replace("Z", "+00:00")),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0)),
            })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = df.sort_values("timestamp").tail(limit).reset_index(drop=True)
        
        try:
            from nexus.risk.kill_switch import get_kill_switch
            kill_switch = get_kill_switch()
            if kill_switch:
                kill_switch.update_data_timestamp()
        except Exception:
            pass  # Don't crash on kill switch update failure

        return df
    
    # ==================== Account Methods ====================
    
    async def get_account(self) -> Optional[AccountInfo]:
        """Get account information."""
        endpoint = f"/v3/accounts/{self.account_id}"

        data = await self._request("GET", endpoint)
        if data is None:
            return None

        account = data.get("account", {})

        return AccountInfo(
            balance=float(account.get("balance", 0)),
            equity=float(account.get("NAV", 0)),
            margin_used=float(account.get("marginUsed", 0)),
            margin_available=float(account.get("marginAvailable", 0)),
            currency=account.get("currency", "USD"),
            unrealized_pnl=float(account.get("unrealizedPL", 0)),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        endpoint = f"/v3/accounts/{self.account_id}/openPositions"

        data = await self._request("GET", endpoint)
        if data is None:
            return []

        positions = data.get("positions", [])
        
        result = []
        for pos in positions:
            instrument = pos.get("instrument", "")
            
            # OANDA separates long and short
            long_units = int(pos.get("long", {}).get("units", 0))
            short_units = int(pos.get("short", {}).get("units", 0))
            
            if long_units > 0:
                long_data = pos.get("long", {})
                result.append(Position(
                    symbol=self._denormalize_symbol(instrument),
                    direction="long",
                    size=float(long_units),
                    entry_price=float(long_data.get("averagePrice", 0)),
                    current_price=0,  # Would need separate quote call
                    unrealized_pnl=float(long_data.get("unrealizedPL", 0)),
                    unrealized_pnl_pct=0,  # Calculate if needed
                    margin_used=float(long_data.get("marginUsed", 0)),
                ))
            
            if short_units < 0:
                short_data = pos.get("short", {})
                result.append(Position(
                    symbol=self._denormalize_symbol(instrument),
                    direction="short",
                    size=float(abs(short_units)),
                    entry_price=float(short_data.get("averagePrice", 0)),
                    current_price=0,
                    unrealized_pnl=float(short_data.get("unrealizedPL", 0)),
                    unrealized_pnl_pct=0,
                    margin_used=float(short_data.get("marginUsed", 0)),
                ))
        
        return result
    
    # ==================== Order Methods ====================
    
    async def place_order(self, order: Order) -> OrderResult:
        """
        Place an order.
        
        Supports market, limit, and stop orders.
        """
        instrument = self._normalize_symbol(order.symbol)
        
        # Determine units (positive for long, negative for short)
        units = order.size if order.direction == "long" else -order.size
        
        endpoint = f"/v3/accounts/{self.account_id}/orders"
        
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(int(units)),
                "type": order.order_type.upper(),
                "timeInForce": order.time_in_force,
            }
        }
        
        # Add price for limit/stop orders
        if order.order_type.lower() == "limit" and order.limit_price:
            order_data["order"]["price"] = str(order.limit_price)
        elif order.order_type.lower() == "stop" and order.stop_price:
            order_data["order"]["price"] = str(order.stop_price)
        
        # Add stop loss
        if order.stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(order.stop_loss)
            }
        
        # Add take profit
        if order.take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(order.take_profit)
            }
        
        try:
            data = await self._request("POST", endpoint, json_data=order_data)
            if data is None:
                return OrderResult(
                    order_id="",
                    status="rejected",
                    message="OANDA request failed (no response)",
                )

            # Check for fill
            if "orderFillTransaction" in data:
                fill = data["orderFillTransaction"]
                return OrderResult(
                    order_id=fill.get("id", ""),
                    status="filled",
                    fill_price=float(fill.get("price", 0)),
                    fill_time=datetime.fromisoformat(fill.get("time", "").replace("Z", "+00:00")),
                    filled_size=float(abs(int(fill.get("units", 0)))),
                    message="Order filled"
                )
            
            # Check for pending order
            if "orderCreateTransaction" in data:
                created = data["orderCreateTransaction"]
                return OrderResult(
                    order_id=created.get("id", ""),
                    status="pending",
                    message="Order created"
                )
            
            return OrderResult(
                order_id="",
                status="rejected",
                message=str(data)
            )
            
        except Exception as e:
            return OrderResult(
                order_id="",
                status="rejected",
                message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
        
        try:
            await self._request("PUT", endpoint)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def close_position(self, symbol: str, size: Optional[float] = None) -> OrderResult:
        """Close a position (full or partial)."""
        instrument = self._normalize_symbol(symbol)
        
        endpoint = f"/v3/accounts/{self.account_id}/positions/{instrument}/close"
        
        # Determine what to close
        if size:
            # Partial close - need to determine direction
            data = {"longUnits": str(int(size))}  # Assumes long, adjust as needed
        else:
            # Full close
            data = {"longUnits": "ALL", "shortUnits": "ALL"}
        
        try:
            response = await self._request("PUT", endpoint, json_data=data)
            if response is None:
                return OrderResult(
                    order_id="",
                    status="rejected",
                    message="OANDA request failed (no response)",
                )

            return OrderResult(
                order_id=response.get("relatedTransactionIDs", [""])[0],
                status="filled",
                message="Position closed"
            )
        except Exception as e:
            return OrderResult(
                order_id="",
                status="rejected",
                message=str(e)
            )
    
    async def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions (emergency)."""
        positions = await self.get_positions()
        results = []
        
        for pos in positions:
            result = await self.close_position(pos.symbol)
            results.append(result)
        
        return results
    
    # ==================== Helper Methods ====================
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Convert symbol to OANDA format.
        
        "EUR/USD" -> "EUR_USD"
        "EURUSD" -> "EUR_USD"
        """
        symbol = symbol.upper().replace("/", "_")
        
        # Handle no separator (EURUSD -> EUR_USD)
        if "_" not in symbol and len(symbol) == 6:
            symbol = f"{symbol[:3]}_{symbol[3:]}"
        
        return symbol
    
    def _denormalize_symbol(self, instrument: str) -> str:
        """
        Convert OANDA format to standard.
        
        "EUR_USD" -> "EUR/USD"
        """
        return instrument.replace("_", "/")
    
    # ==================== Indicator Helpers ====================
    
    @staticmethod
    def calculate_pips(symbol: str, price_diff: float) -> float:
        """
        Calculate pip value from price difference.
        
        JPY pairs: 1 pip = 0.01
        Others: 1 pip = 0.0001
        """
        if "JPY" in symbol.upper():
            return price_diff / 0.01
        return price_diff / 0.0001
    
    @staticmethod
    def pip_value(symbol: str) -> float:
        """Get pip value for a symbol."""
        if "JPY" in symbol.upper():
            return 0.01
        return 0.0001
