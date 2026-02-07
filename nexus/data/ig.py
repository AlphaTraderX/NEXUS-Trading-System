"""
NEXUS IG Markets Provider

Spread betting broker for TAX-FREE trading (UK).
Uses REST API for orders and Lightstreamer for streaming.

Markets: Forex, Indices, Commodities
Tax Status: SPREAD BETTING = TAX FREE (UK)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd

from nexus.config.settings import settings
from .base import (
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


class IGProvider(BaseBroker):
    """
    IG Markets spread betting provider.
    
    Features:
    - Spread betting (tax-free in UK)
    - Forex, indices, commodities
    - REST API for trading
    - Guaranteed stops available (extra cost)
    
    Note: Uses "epic" codes for instruments (e.g., CS.D.EURUSD.CFD.IP)
    """
    
    # API URLs
    LIVE_URL = "https://api.ig.com/gateway/deal"
    DEMO_URL = "https://demo-api.ig.com/gateway/deal"
    
    # Epic mappings (NEXUS symbol -> IG epic)
    EPIC_MAP = {
        # Forex
        "EUR/USD": "CS.D.EURUSD.CFD.IP",
        "GBP/USD": "CS.D.GBPUSD.CFD.IP",
        "USD/JPY": "CS.D.USDJPY.CFD.IP",
        "AUD/USD": "CS.D.AUDUSD.CFD.IP",
        "USD/CAD": "CS.D.USDCAD.CFD.IP",
        "EUR/GBP": "CS.D.EURGBP.CFD.IP",
        # Indices
        "US500": "IX.D.SPTRD.IFD.IP",  # S&P 500
        "US100": "IX.D.NASDAQ.IFD.IP",  # Nasdaq 100
        "UK100": "IX.D.FTSE.IFD.IP",    # FTSE 100
        "DE40": "IX.D.DAX.IFD.IP",       # DAX
        # Commodities
        "GOLD": "CS.D.USCGC.TODAY.IP",
        "OIL": "CC.D.CL.UNC.IP",
    }
    
    def __init__(self):
        super().__init__()
        self._api_key = settings.ig_api_key
        self._username = settings.ig_username
        self._password = settings.ig_password
        self._demo = settings.ig_demo
        
        self._base_url = self.DEMO_URL if self._demo else self.LIVE_URL
        self._cst: Optional[str] = None  # Client session token
        self._security_token: Optional[str] = None
        self._lightstreamer_endpoint: Optional[str] = None
        
        self._client = httpx.AsyncClient(timeout=30.0)
        self._subscriptions: Dict[str, Any] = {}
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    async def connect(self) -> bool:
        """Connect and authenticate with IG."""
        try:
            # Authentication request
            response = await self._client.post(
                f"{self._base_url}/session",
                headers={
                    "X-IG-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json; charset=UTF-8",
                    "Version": "2",
                },
                json={
                    "identifier": self._username,
                    "password": self._password,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"IG auth failed: {response.status_code} - {response.text}")
                return False
            
            # Extract tokens from headers
            self._cst = response.headers.get("CST")
            self._security_token = response.headers.get("X-SECURITY-TOKEN")
            
            # Get account info from response
            data = response.json()
            self._account_id = data.get("currentAccountId")
            self._lightstreamer_endpoint = data.get("lightstreamerEndpoint")
            
            self._connected = True
            logger.info(f"Connected to IG Markets (account: {self._account_id}, demo: {self._demo})")
            return True
            
        except Exception as e:
            logger.error(f"IG connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from IG."""
        if self._connected:
            try:
                await self._client.delete(
                    f"{self._base_url}/session",
                    headers=self._auth_headers(),
                )
            except Exception:
                pass
            
            self._connected = False
            self._cst = None
            self._security_token = None
            logger.info("Disconnected from IG Markets")
    
    def _auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        return {
            "X-IG-API-KEY": self._api_key,
            "CST": self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
        }
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol."""
        epic = self.convert_symbol(symbol)
        
        response = await self._client.get(
            f"{self._base_url}/markets/{epic}",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get quote: {response.text}")
        
        data = response.json()
        snapshot = data.get("snapshot", {})
        
        return Quote(
            symbol=symbol,
            bid=float(snapshot.get("bid", 0)),
            ask=float(snapshot.get("offer", 0)),
            last=float(snapshot.get("bid", 0)),  # IG doesn't provide last
            volume=0,  # Spread betting doesn't show volume
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
        epic = self.convert_symbol(symbol)
        resolution = self._convert_timeframe(timeframe)
        
        params = {
            "resolution": resolution,
            "max": limit,
            "pageSize": limit,
        }
        
        response = await self._client.get(
            f"{self._base_url}/prices/{epic}",
            headers={**self._auth_headers(), "Version": "3"},
            params=params,
        )
        
        if response.status_code != 200:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        data = response.json()
        prices = data.get("prices", [])
        
        if not prices:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame([{
            'timestamp': p.get("snapshotTime"),
            'open': float(p.get("openPrice", {}).get("bid", 0)),
            'high': float(p.get("highPrice", {}).get("bid", 0)),
            'low': float(p.get("lowPrice", {}).get("bid", 0)),
            'close': float(p.get("closePrice", {}).get("bid", 0)),
            'volume': int(p.get("lastTradedVolume", 0)),
        } for p in prices])
        
        return df
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> bool:
        """Subscribe to real-time quotes (simplified - would use Lightstreamer)."""
        # Note: Full implementation would use Lightstreamer
        # For now, we poll periodically
        for symbol in symbols:
            self._subscriptions[symbol] = callback
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
            f"{self._base_url}/accounts",
            headers=self._auth_headers(),
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get account: {response.text}")
        
        data = response.json()
        accounts = data.get("accounts", [])
        
        # Find spread betting account
        account = None
        for acc in accounts:
            if acc.get("accountId") == self._account_id:
                account = acc
                break
        
        if not account:
            account = accounts[0] if accounts else {}
        
        balance = account.get("balance", {})
        
        return AccountInfo(
            account_id=self._account_id or "",
            balance=float(balance.get("balance", 0)),
            equity=float(balance.get("balance", 0)) + float(balance.get("profitLoss", 0)),
            margin_used=float(balance.get("deposit", 0)),
            margin_available=float(balance.get("available", 0)),
            currency=account.get("currency", "GBP"),
            unrealized_pnl=float(balance.get("profitLoss", 0)),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        response = await self._client.get(
            f"{self._base_url}/positions",
            headers={**self._auth_headers(), "Version": "2"},
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        positions = []
        
        for pos in data.get("positions", []):
            position_data = pos.get("position", {})
            market_data = pos.get("market", {})
            
            direction = position_data.get("direction", "").lower()
            size = float(position_data.get("size", 0))
            entry = float(position_data.get("openLevel", 0))
            
            # Get current price
            current_bid = float(market_data.get("bid", entry))
            current_ask = float(market_data.get("offer", entry))
            current = current_bid if direction == "sell" else current_ask
            
            # Calculate P&L
            if direction == "buy":
                pnl = (current_bid - entry) * size
            else:
                pnl = (entry - current_ask) * size
            
            pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
            if direction == "sell":
                pnl_pct = -pnl_pct
            
            positions.append(Position(
                symbol=self.convert_symbol_from_broker(market_data.get("epic", "")),
                direction="long" if direction == "buy" else "short",
                size=size,
                entry_price=entry,
                current_price=current,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                market_value=size * current,
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
        """Place a spread bet order."""
        epic = self.convert_symbol(order.symbol)
        
        # Build order payload
        payload = {
            "epic": epic,
            "direction": "BUY" if order.direction == "long" else "SELL",
            "size": order.size,
            "orderType": "MARKET" if order.order_type == OrderType.MARKET else "LIMIT",
            "currencyCode": "GBP",
            "forceOpen": True,
            "guaranteedStop": False,
            "expiry": "DFB",  # Daily funded bet
        }
        
        if order.limit_price and order.order_type == OrderType.LIMIT:
            payload["level"] = order.limit_price
        
        if order.stop_loss:
            payload["stopLevel"] = order.stop_loss
            payload["stopDistance"] = None
        
        if order.take_profit:
            payload["limitLevel"] = order.take_profit
            payload["limitDistance"] = None
        
        response = await self._client.post(
            f"{self._base_url}/positions/otc",
            headers={**self._auth_headers(), "Version": "2"},
            json=payload,
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
        deal_reference = data.get("dealReference", "")
        
        # Confirm the deal
        await asyncio.sleep(0.5)  # Wait for deal to process
        
        confirm_response = await self._client.get(
            f"{self._base_url}/confirms/{deal_reference}",
            headers=self._auth_headers(),
        )
        
        if confirm_response.status_code == 200:
            confirm_data = confirm_response.json()
            deal_status = confirm_data.get("dealStatus", "")
            
            return OrderResult(
                order_id=confirm_data.get("dealId", deal_reference),
                status=OrderStatus.FILLED if deal_status == "ACCEPTED" else OrderStatus.REJECTED,
                symbol=order.symbol,
                direction=order.direction,
                requested_size=order.size,
                filled_size=order.size if deal_status == "ACCEPTED" else 0,
                fill_price=float(confirm_data.get("level", 0)),
                fill_time=datetime.now() if deal_status == "ACCEPTED" else None,
                message=confirm_data.get("reason", deal_status),
            )
        
        return OrderResult(
            order_id=deal_reference,
            status=OrderStatus.PENDING,
            symbol=order.symbol,
            direction=order.direction,
            requested_size=order.size,
            message="Awaiting confirmation",
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        response = await self._client.delete(
            f"{self._base_url}/workingorders/otc/{order_id}",
            headers={**self._auth_headers(), "Version": "2", "_method": "DELETE"},
        )
        return response.status_code == 200
    
    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get order status."""
        # IG doesn't have a direct order lookup - would need to search working orders
        return None
    
    async def get_open_orders(self) -> List[OrderResult]:
        """Get all open orders."""
        response = await self._client.get(
            f"{self._base_url}/workingorders",
            headers={**self._auth_headers(), "Version": "2"},
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        orders = []
        
        for wo in data.get("workingOrders", []):
            order_data = wo.get("workingOrderData", {})
            market_data = wo.get("marketData", {})
            
            orders.append(OrderResult(
                order_id=order_data.get("dealId", ""),
                status=OrderStatus.PENDING,
                symbol=self.convert_symbol_from_broker(market_data.get("epic", "")),
                direction="long" if order_data.get("direction") == "BUY" else "short",
                requested_size=float(order_data.get("size", 0)),
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
        # Get current position to find deal ID
        response = await self._client.get(
            f"{self._base_url}/positions",
            headers={**self._auth_headers(), "Version": "2"},
        )
        
        if response.status_code != 200:
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                symbol=symbol,
                direction="close",
                requested_size=0,
                message="Failed to get positions",
            )
        
        data = response.json()
        epic = self.convert_symbol(symbol)
        
        for pos in data.get("positions", []):
            if pos.get("market", {}).get("epic") == epic:
                deal_id = pos.get("position", {}).get("dealId")
                position_data = pos.get("position", {})
                
                close_size = size or float(position_data.get("size", 0))
                direction = "SELL" if position_data.get("direction") == "BUY" else "BUY"
                
                close_response = await self._client.post(
                    f"{self._base_url}/positions/otc",
                    headers={**self._auth_headers(), "Version": "2", "_method": "DELETE"},
                    json={
                        "dealId": deal_id,
                        "direction": direction,
                        "size": close_size,
                        "orderType": "MARKET",
                    },
                )
                
                if close_response.status_code in [200, 201]:
                    return OrderResult(
                        order_id=deal_id,
                        status=OrderStatus.FILLED,
                        symbol=symbol,
                        direction="close",
                        requested_size=close_size,
                        filled_size=close_size,
                        fill_time=datetime.now(),
                    )
        
        return OrderResult(
            order_id="",
            status=OrderStatus.REJECTED,
            symbol=symbol,
            direction="close",
            requested_size=0,
            message="Position not found",
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
        """Convert NEXUS symbol to IG epic."""
        return self.EPIC_MAP.get(nexus_symbol, nexus_symbol)
    
    def convert_symbol_from_broker(self, epic: str) -> str:
        """Convert IG epic to NEXUS symbol."""
        for nexus, ig_epic in self.EPIC_MAP.items():
            if ig_epic == epic:
                return nexus
        return epic
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert NEXUS timeframe to IG resolution."""
        mapping = {
            "1m": "MINUTE",
            "5m": "MINUTE_5",
            "15m": "MINUTE_15",
            "30m": "MINUTE_30",
            "1h": "HOUR",
            "4h": "HOUR_4",
            "1d": "DAY",
            "1w": "WEEK",
        }
        return mapping.get(timeframe, "HOUR")
