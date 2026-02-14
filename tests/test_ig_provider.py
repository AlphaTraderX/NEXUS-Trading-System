"""Tests for IG Markets spread betting provider."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pandas as pd

from nexus.data.ig import IGProvider
from nexus.data.base import (
    Quote,
    AccountInfo,
    Position,
    Order,
    OrderResult,
)


# ---------------------------------------------------------------------------
# Symbol Conversion
# ---------------------------------------------------------------------------

class TestSymbolConversion:
    """Test NEXUS <-> IG epic mapping."""

    def test_convert_forex_symbol(self):
        provider = IGProvider()
        assert provider.convert_symbol("EUR/USD") == "CS.D.EURUSD.CFD.IP"
        assert provider.convert_symbol("GBP/USD") == "CS.D.GBPUSD.CFD.IP"
        assert provider.convert_symbol("USD/JPY") == "CS.D.USDJPY.CFD.IP"

    def test_convert_index_symbol(self):
        provider = IGProvider()
        assert provider.convert_symbol("US500") == "IX.D.SPTRD.DAILY.IP"
        assert provider.convert_symbol("US100") == "IX.D.NASDAQ.DAILY.IP"
        assert provider.convert_symbol("UK100") == "IX.D.FTSE.DAILY.IP"
        assert provider.convert_symbol("DE40") == "IX.D.DAX.DAILY.IP"

    def test_convert_commodity_symbol(self):
        provider = IGProvider()
        assert provider.convert_symbol("GOLD") == "CS.D.USCGC.TODAY.IP"
        assert provider.convert_symbol("OIL") == "CC.D.CL.UNC.IP"

    def test_unknown_symbol_passthrough(self):
        provider = IGProvider()
        assert provider.convert_symbol("UNKNOWN_EPIC") == "UNKNOWN_EPIC"

    def test_convert_from_broker_forex(self):
        provider = IGProvider()
        assert provider.convert_symbol_from_broker("CS.D.EURUSD.CFD.IP") == "EUR/USD"
        assert provider.convert_symbol_from_broker("CS.D.GBPUSD.CFD.IP") == "GBP/USD"

    def test_convert_from_broker_index(self):
        provider = IGProvider()
        assert provider.convert_symbol_from_broker("IX.D.SPTRD.DAILY.IP") == "US500"
        assert provider.convert_symbol_from_broker("IX.D.FTSE.DAILY.IP") == "UK100"

    def test_convert_from_broker_unknown(self):
        provider = IGProvider()
        assert provider.convert_symbol_from_broker("XX.UNKNOWN") == "XX.UNKNOWN"

    def test_roundtrip_all_symbols(self):
        """Every mapped symbol should roundtrip correctly."""
        provider = IGProvider()
        for nexus_sym, epic in IGProvider.EPIC_MAP.items():
            assert provider.convert_symbol_from_broker(
                provider.convert_symbol(nexus_sym)
            ) == nexus_sym


# ---------------------------------------------------------------------------
# Timeframe Conversion
# ---------------------------------------------------------------------------

class TestTimeframeConversion:
    """Test NEXUS timeframe -> IG resolution mapping."""

    def test_minute_timeframes(self):
        provider = IGProvider()
        assert provider._convert_timeframe("1m") == "MINUTE"
        assert provider._convert_timeframe("5m") == "MINUTE_5"
        assert provider._convert_timeframe("15m") == "MINUTE_15"
        assert provider._convert_timeframe("30m") == "MINUTE_30"

    def test_hour_timeframes(self):
        provider = IGProvider()
        assert provider._convert_timeframe("1h") == "HOUR"
        assert provider._convert_timeframe("4h") == "HOUR_4"

    def test_daily_weekly(self):
        provider = IGProvider()
        assert provider._convert_timeframe("1d") == "DAY"
        assert provider._convert_timeframe("1w") == "WEEK"

    def test_unknown_defaults_to_hour(self):
        provider = IGProvider()
        assert provider._convert_timeframe("3d") == "HOUR"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnection:
    """Test IG authentication flow."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "CST": "test-cst-token",
            "X-SECURITY-TOKEN": "test-security-token",
        }
        mock_response.json.return_value = {
            "currentAccountId": "ABC123",
            "lightstreamerEndpoint": "https://push.lightstreamer.com",
        }

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.connect()

        assert result is True
        assert provider._connected is True
        assert provider._cst == "test-cst-token"
        assert provider._security_token == "test-security-token"
        assert provider._account_id == "ABC123"

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Invalid credentials"

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.connect()

        assert result is False
        assert provider._connected is False

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        provider = IGProvider()

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(side_effect=Exception("Network error"))

        result = await provider.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        provider = IGProvider()
        provider._connected = True
        provider._cst = "token"
        provider._security_token = "sec-token"

        mock_response = MagicMock()
        mock_response.status_code = 200

        provider._client = AsyncMock()
        provider._client.delete = AsyncMock(return_value=mock_response)

        await provider.disconnect()

        assert provider._connected is False
        assert provider._cst is None
        assert provider._security_token is None


# ---------------------------------------------------------------------------
# Auth Headers
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    """Test authentication header construction."""

    def test_auth_headers_with_tokens(self):
        provider = IGProvider()
        provider._cst = "my-cst"
        provider._security_token = "my-sec"

        headers = provider._auth_headers()

        assert headers["CST"] == "my-cst"
        assert headers["X-SECURITY-TOKEN"] == "my-sec"
        assert "X-IG-API-KEY" in headers

    def test_auth_headers_no_tokens(self):
        provider = IGProvider()
        headers = provider._auth_headers()

        assert headers["CST"] == ""
        assert headers["X-SECURITY-TOKEN"] == ""


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

class TestQuotes:
    """Test quote retrieval."""

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "snapshot": {
                "bid": 1.08500,
                "offer": 1.08520,
            }
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        quote = await provider.get_quote("EUR/USD")

        assert isinstance(quote, Quote)
        assert quote.symbol == "EUR/USD"
        assert quote.bid == 1.08500
        assert quote.ask == 1.08520

    @pytest.mark.asyncio
    async def test_get_quote_failure(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Market not found"

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception, match="Failed to get quote"):
            await provider.get_quote("INVALID")


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

class TestBars:
    """Test OHLCV bar retrieval."""

    @pytest.mark.asyncio
    async def test_get_bars_success(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [
                {
                    "snapshotTime": "2025/01/15 10:00:00",
                    "openPrice": {"bid": 100.0},
                    "highPrice": {"bid": 102.0},
                    "lowPrice": {"bid": 99.0},
                    "closePrice": {"bid": 101.0},
                    "lastTradedVolume": 1000,
                },
                {
                    "snapshotTime": "2025/01/15 11:00:00",
                    "openPrice": {"bid": 101.0},
                    "highPrice": {"bid": 103.0},
                    "lowPrice": {"bid": 100.5},
                    "closePrice": {"bid": 102.5},
                    "lastTradedVolume": 1500,
                },
            ]
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        bars = await provider.get_bars("US500", "1h", limit=2)

        assert isinstance(bars, pd.DataFrame)
        assert len(bars) == 2
        assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert bars["close"].iloc[0] == 101.0
        assert bars["volume"].iloc[1] == 1500

    @pytest.mark.asyncio
    async def test_get_bars_empty(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prices": []}

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        bars = await provider.get_bars("US500", "1h")

        assert isinstance(bars, pd.DataFrame)
        assert len(bars) == 0

    @pytest.mark.asyncio
    async def test_get_bars_api_error(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        bars = await provider.get_bars("US500", "1h")

        assert isinstance(bars, pd.DataFrame)
        assert len(bars) == 0


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class TestAccount:
    """Test account info retrieval."""

    @pytest.mark.asyncio
    async def test_get_account_success(self):
        provider = IGProvider()
        provider._account_id = "ABC123"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accounts": [
                {
                    "accountId": "ABC123",
                    "accountType": "SPREADBET",
                    "currency": "GBP",
                    "balance": {
                        "balance": 10000.0,
                        "deposit": 1500.0,
                        "profitLoss": 250.0,
                        "available": 8500.0,
                    },
                }
            ]
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        account = await provider.get_account()

        assert isinstance(account, AccountInfo)
        assert account.balance == 10000.0
        assert account.equity == 10250.0  # balance + profitLoss
        assert account.margin_used == 1500.0
        assert account.margin_available == 8500.0
        assert account.currency == "GBP"
        assert account.unrealized_pnl == 250.0

    @pytest.mark.asyncio
    async def test_get_account_failure(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception, match="Failed to get account"):
            await provider.get_account()


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestPositions:
    """Test position retrieval and P&L calculation."""

    @pytest.mark.asyncio
    async def test_get_positions_long(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "positions": [
                {
                    "position": {
                        "direction": "BUY",
                        "size": 10,
                        "openLevel": 100.0,
                        "dealId": "DEAL1",
                        "margin": 500.0,
                    },
                    "market": {
                        "epic": "IX.D.SPTRD.DAILY.IP",
                        "bid": 105.0,
                        "offer": 105.5,
                    },
                }
            ]
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        positions = await provider.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert isinstance(pos, Position)
        assert pos.symbol == "US500"
        assert pos.direction == "long"
        assert pos.size == 10
        assert pos.entry_price == 100.0
        assert pos.unrealized_pnl == (105.0 - 100.0) * 10  # 50.0
        assert pos.margin_used == 500.0

    @pytest.mark.asyncio
    async def test_get_positions_short(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "positions": [
                {
                    "position": {
                        "direction": "SELL",
                        "size": 5,
                        "openLevel": 1.0850,
                        "dealId": "DEAL2",
                        "margin": 200.0,
                    },
                    "market": {
                        "epic": "CS.D.EURUSD.CFD.IP",
                        "bid": 1.0820,
                        "offer": 1.0822,
                    },
                }
            ]
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        positions = await provider.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "EUR/USD"
        assert pos.direction == "short"
        assert pos.unrealized_pnl == pytest.approx((1.0850 - 1.0822) * 5, abs=0.001)

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"positions": []}

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        positions = await provider.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_api_error(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        positions = await provider.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_position_found(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "positions": [
                {
                    "position": {
                        "direction": "BUY",
                        "size": 2,
                        "openLevel": 2000.0,
                        "dealId": "D3",
                        "margin": 100.0,
                    },
                    "market": {
                        "epic": "CS.D.USCGC.TODAY.IP",
                        "bid": 2050.0,
                        "offer": 2051.0,
                    },
                }
            ]
        }

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        pos = await provider.get_position("GOLD")
        assert pos is not None
        assert pos.symbol == "GOLD"

    @pytest.mark.asyncio
    async def test_get_position_not_found(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"positions": []}

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        pos = await provider.get_position("GOLD")
        assert pos is None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class TestOrders:
    """Test order placement and management."""

    @pytest.mark.asyncio
    async def test_place_market_order_accepted(self):
        provider = IGProvider()

        # Mock the initial order response
        order_response = MagicMock()
        order_response.status_code = 200
        order_response.json.return_value = {"dealReference": "REF123"}

        # Mock the confirmation response
        confirm_response = MagicMock()
        confirm_response.status_code = 200
        confirm_response.json.return_value = {
            "dealId": "DEAL456",
            "dealStatus": "ACCEPTED",
            "level": 1.0850,
            "reason": "SUCCESS",
        }

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=order_response)
        provider._client.get = AsyncMock(return_value=confirm_response)

        order = Order(
            symbol="EUR/USD",
            direction="long",
            size=1.0,
            order_type="market",
            stop_loss=1.0800,
            take_profit=1.0950,
        )

        result = await provider.place_order(order)

        assert isinstance(result, OrderResult)
        assert result.order_id == "DEAL456"
        assert result.status == "filled"
        assert result.fill_price == 1.0850
        assert result.filled_size == 1.0
        assert result.fill_time is not None

    @pytest.mark.asyncio
    async def test_place_order_rejected_by_api(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Insufficient margin"

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        order = Order(
            symbol="US500",
            direction="long",
            size=100,
            order_type="market",
        )

        result = await provider.place_order(order)

        assert result.status == "rejected"
        assert "Insufficient margin" in result.message

    @pytest.mark.asyncio
    async def test_place_order_rejected_at_confirmation(self):
        provider = IGProvider()

        order_response = MagicMock()
        order_response.status_code = 200
        order_response.json.return_value = {"dealReference": "REF999"}

        confirm_response = MagicMock()
        confirm_response.status_code = 200
        confirm_response.json.return_value = {
            "dealId": "DEAL999",
            "dealStatus": "REJECTED",
            "reason": "MARKET_CLOSED",
        }

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=order_response)
        provider._client.get = AsyncMock(return_value=confirm_response)

        order = Order(
            symbol="EUR/USD",
            direction="short",
            size=2.0,
            order_type="market",
        )

        result = await provider.place_order(order)

        assert result.status == "rejected"
        assert result.filled_size == 0
        assert "MARKET_CLOSED" in result.message

    @pytest.mark.asyncio
    async def test_place_order_pending(self):
        provider = IGProvider()

        order_response = MagicMock()
        order_response.status_code = 200
        order_response.json.return_value = {"dealReference": "REF_PEND"}

        confirm_response = MagicMock()
        confirm_response.status_code = 500  # Confirmation failed

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=order_response)
        provider._client.get = AsyncMock(return_value=confirm_response)

        order = Order(
            symbol="EUR/USD",
            direction="long",
            size=1.0,
            order_type="market",
        )

        result = await provider.place_order(order)

        assert result.status == "pending"
        assert result.order_id == "REF_PEND"

    @pytest.mark.asyncio
    async def test_place_limit_order(self):
        provider = IGProvider()

        order_response = MagicMock()
        order_response.status_code = 200
        order_response.json.return_value = {"dealReference": "LIM_REF"}

        confirm_response = MagicMock()
        confirm_response.status_code = 200
        confirm_response.json.return_value = {
            "dealId": "LIM_DEAL",
            "dealStatus": "ACCEPTED",
            "level": 1.0800,
            "reason": "SUCCESS",
        }

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=order_response)
        provider._client.get = AsyncMock(return_value=confirm_response)

        order = Order(
            symbol="EUR/USD",
            direction="long",
            size=1.0,
            order_type="limit",
            limit_price=1.0800,
        )

        result = await provider.place_order(order)

        assert result.status == "filled"

        # Verify the payload sent
        call_args = provider._client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["orderType"] == "LIMIT"
        assert payload["level"] == 1.0800

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        provider._client = AsyncMock()
        provider._client.delete = AsyncMock(return_value=mock_response)

        result = await provider.cancel_order("ORDER123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 404

        provider._client = AsyncMock()
        provider._client.delete = AsyncMock(return_value=mock_response)

        result = await provider.cancel_order("INVALID")
        assert result is False


# ---------------------------------------------------------------------------
# Close Position
# ---------------------------------------------------------------------------

class TestClosePosition:
    """Test position closing."""

    @pytest.mark.asyncio
    async def test_close_position_success(self):
        provider = IGProvider()

        # Mock get positions
        get_response = MagicMock()
        get_response.status_code = 200
        get_response.json.return_value = {
            "positions": [
                {
                    "position": {
                        "direction": "BUY",
                        "size": 5,
                        "dealId": "DEAL_CLOSE",
                    },
                    "market": {
                        "epic": "IX.D.SPTRD.DAILY.IP",
                    },
                }
            ]
        }

        # Mock close response
        close_response = MagicMock()
        close_response.status_code = 200

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=get_response)
        provider._client.post = AsyncMock(return_value=close_response)

        result = await provider.close_position("US500")

        assert result.status == "filled"
        assert result.order_id == "DEAL_CLOSE"
        assert result.filled_size == 5

    @pytest.mark.asyncio
    async def test_close_position_partial(self):
        provider = IGProvider()

        get_response = MagicMock()
        get_response.status_code = 200
        get_response.json.return_value = {
            "positions": [
                {
                    "position": {
                        "direction": "SELL",
                        "size": 10,
                        "dealId": "DEAL_PART",
                    },
                    "market": {"epic": "CS.D.EURUSD.CFD.IP"},
                }
            ]
        }

        close_response = MagicMock()
        close_response.status_code = 200

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=get_response)
        provider._client.post = AsyncMock(return_value=close_response)

        result = await provider.close_position("EUR/USD", size=3.0)

        assert result.status == "filled"
        assert result.filled_size == 3.0

    @pytest.mark.asyncio
    async def test_close_position_not_found(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"positions": []}

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        result = await provider.close_position("GOLD")

        assert result.status == "rejected"
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_close_position_api_error(self):
        provider = IGProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        result = await provider.close_position("US500")

        assert result.status == "rejected"


# ---------------------------------------------------------------------------
# Epic Map Coverage
# ---------------------------------------------------------------------------

class TestEpicMap:
    """Test EPIC_MAP completeness."""

    def test_all_forex_pairs(self):
        expected_forex = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EUR/GBP"]
        for pair in expected_forex:
            assert pair in IGProvider.EPIC_MAP

    def test_all_indices(self):
        expected_indices = ["US500", "US100", "UK100", "DE40"]
        for idx in expected_indices:
            assert idx in IGProvider.EPIC_MAP

    def test_all_commodities(self):
        expected = ["GOLD", "OIL"]
        for comm in expected:
            assert comm in IGProvider.EPIC_MAP

    def test_epic_format(self):
        """All epics should follow IG's dot-separated format."""
        for epic in IGProvider.EPIC_MAP.values():
            parts = epic.split(".")
            assert len(parts) >= 4, f"Epic {epic} has unexpected format"


# ---------------------------------------------------------------------------
# Provider Properties
# ---------------------------------------------------------------------------

class TestProviderProperties:
    """Test provider configuration."""

    def test_demo_url(self):
        provider = IGProvider()
        # Default is demo mode
        assert "demo" in provider._base_url

    def test_live_url(self):
        provider = IGProvider()
        assert IGProvider.LIVE_URL == "https://api.ig.com/gateway/deal"
        assert IGProvider.DEMO_URL == "https://demo-api.ig.com/gateway/deal"

    def test_inherits_base_broker(self):
        from nexus.data.base import BaseBroker, ReconnectionMixin
        assert issubclass(IGProvider, BaseBroker)
        assert issubclass(IGProvider, ReconnectionMixin)

    def test_has_reconnection(self):
        provider = IGProvider()
        assert hasattr(provider, 'ensure_connected')
        assert hasattr(provider, '_reconnect_attempts')


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

class TestSubscriptions:
    """Test subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe(self):
        provider = IGProvider()
        callback = MagicMock()

        result = await provider.subscribe(["EUR/USD", "US500"], callback)
        assert result is True
        assert "EUR/USD" in provider._subscriptions
        assert "US500" in provider._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        provider = IGProvider()
        provider._subscriptions = {"EUR/USD": MagicMock(), "US500": MagicMock()}

        await provider.unsubscribe(["EUR/USD"])
        assert "EUR/USD" not in provider._subscriptions
        assert "US500" in provider._subscriptions


# ---------------------------------------------------------------------------
# Module Export
# ---------------------------------------------------------------------------

class TestModuleExport:
    """Test IGProvider is properly exported."""

    def test_import_from_data(self):
        from nexus.data import IGProvider as IP
        assert IP is IGProvider

    def test_in_all(self):
        import nexus.data as data_mod
        assert "IGProvider" in data_mod.__all__
