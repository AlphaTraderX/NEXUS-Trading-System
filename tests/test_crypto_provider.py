"""Tests for crypto data providers (Binance, Kraken)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.data.crypto import BinanceProvider, KrakenProvider


# =============================================================================
# Binance Provider
# =============================================================================


class TestBinanceSymbolConversion:
    """Test Binance symbol conversion."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    def test_nexus_to_binance(self, provider):
        assert provider._convert_symbol("BTC_USD") == "BTCUSDT"
        assert provider._convert_symbol("ETH_USD") == "ETHUSDT"
        assert provider._convert_symbol("SOL_USD") == "SOLUSDT"
        assert provider._convert_symbol("DOGE_USD") == "DOGEUSDT"

    def test_nexus_to_binance_non_usd(self, provider):
        assert provider._convert_symbol("ETH_BTC") == "ETHBTC"

    def test_already_binance_format(self, provider):
        assert provider._convert_symbol("BTCUSDT") == "BTCUSDT"

    def test_binance_to_nexus(self, provider):
        assert provider._convert_symbol_to_nexus("BTCUSDT") == "BTC_USD"
        assert provider._convert_symbol_to_nexus("ETHUSDT") == "ETH_USD"
        assert provider._convert_symbol_to_nexus("SOLUSDT") == "SOL_USD"

    def test_binance_to_nexus_btc_pair(self, provider):
        assert provider._convert_symbol_to_nexus("ETHBTC") == "ETH_BTC"


class TestBinanceTimeframeMap:
    """Test Binance timeframe mapping."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    def test_standard_timeframes(self, provider):
        assert provider.TIMEFRAME_MAP["5m"] == "5m"
        assert provider.TIMEFRAME_MAP["15m"] == "15m"
        assert provider.TIMEFRAME_MAP["1h"] == "1h"
        assert provider.TIMEFRAME_MAP["4h"] == "4h"
        assert provider.TIMEFRAME_MAP["1d"] == "1d"

    def test_uppercase_daily(self, provider):
        assert provider.TIMEFRAME_MAP["1D"] == "1d"


class TestBinanceConnect:
    """Test Binance connection."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    @pytest.mark.asyncio
    async def test_connect_success(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200)

            result = await provider.connect()

            assert result is True
            assert provider._connected is True
            assert provider._last_ping is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=500)

            result = await provider.connect()

            assert result is False
            assert provider._connected is False

    @pytest.mark.asyncio
    async def test_connect_exception(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await provider.connect()

            assert result is False


class TestBinanceGetQuote:
    """Test Binance quote retrieval."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    @pytest.mark.asyncio
    async def test_get_quote_success(self, provider):
        mock_response = MagicMock(
            status_code=200,
            json=lambda: {"bidPrice": "50000.00", "askPrice": "50010.00"},
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            quote = await provider.get_quote("BTC_USD")

            assert quote is not None
            assert quote.symbol == "BTC_USD"
            assert quote.bid == 50000.0
            assert quote.ask == 50010.0
            assert quote.last == 50005.0  # Mid price

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=400)

            quote = await provider.get_quote("BTC_USD")
            assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_exception(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("timeout")

            quote = await provider.get_quote("BTC_USD")
            assert quote is None


class TestBinanceGetBars:
    """Test Binance bar retrieval."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    @pytest.fixture
    def mock_kline_data(self):
        return [
            [
                1609459200000, "50000.00", "51000.00", "49000.00", "50500.00",
                "1000.0", 1609459260000, "50500000.00", 1000, "500.0",
                "25250000.00", "0",
            ],
            [
                1609459260000, "50500.00", "52000.00", "50000.00", "51500.00",
                "1200.0", 1609459320000, "61800000.00", 1200, "600.0",
                "30900000.00", "0",
            ],
        ]

    @pytest.mark.asyncio
    async def test_get_bars_success(self, provider, mock_kline_data):
        mock_response = MagicMock(
            status_code=200,
            json=lambda: mock_kline_data,
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            bars = await provider.get_bars("BTC_USD", "1h", 100)

            assert bars is not None
            assert len(bars) == 2
            assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
            assert bars["open"].iloc[0] == 50000.0
            assert bars["close"].iloc[1] == 51500.0

    @pytest.mark.asyncio
    async def test_get_bars_api_error(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=429)

            bars = await provider.get_bars("BTC_USD")
            assert bars is None

    @pytest.mark.asyncio
    async def test_get_bars_empty(self, provider):
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: [])

            bars = await provider.get_bars("BTC_USD")
            assert bars is None

    @pytest.mark.asyncio
    async def test_get_bars_limit_capped(self, provider, mock_kline_data):
        """Limit should be capped at 1000."""
        mock_response = MagicMock(status_code=200, json=lambda: mock_kline_data)

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            await provider.get_bars("BTC_USD", "1h", 5000)

            # Verify limit was capped
            call_params = mock_get.call_args[1]["params"]
            assert call_params["limit"] == 1000

    @pytest.mark.asyncio
    async def test_get_bars_with_end_date(self, provider, mock_kline_data):
        mock_response = MagicMock(status_code=200, json=lambda: mock_kline_data)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            await provider.get_bars("BTC_USD", "1h", 100, end_date=end)

            call_params = mock_get.call_args[1]["params"]
            assert "endTime" in call_params


class TestBinance24hStats:
    """Test Binance 24h statistics."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    @pytest.mark.asyncio
    async def test_get_24h_stats(self, provider):
        mock_response = MagicMock(
            status_code=200,
            json=lambda: {
                "priceChange": "1000.00",
                "priceChangePercent": "2.0",
                "highPrice": "52000.00",
                "lowPrice": "49000.00",
                "volume": "10000.0",
                "quoteVolume": "500000000.00",
                "count": 100000,
            },
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            stats = await provider.get_24h_stats("BTC_USD")

            assert stats is not None
            assert stats["price_change_pct"] == 2.0
            assert stats["high_24h"] == 52000.0
            assert stats["volume_24h"] == 10000.0
            assert stats["trades_24h"] == 100000


class TestBinanceAllTickers:
    """Test Binance all tickers."""

    @pytest.fixture
    def provider(self):
        return BinanceProvider()

    @pytest.mark.asyncio
    async def test_get_all_tickers(self, provider):
        mock_response = MagicMock(
            status_code=200,
            json=lambda: [
                {"symbol": "BTCUSDT", "price": "50000.00"},
                {"symbol": "ETHUSDT", "price": "3000.00"},
                {"symbol": "ETHBTC", "price": "0.06"},  # Should be filtered out
            ],
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            tickers = await provider.get_all_tickers()

            assert len(tickers) == 2  # Only USDT pairs
            assert tickers[0]["symbol"] == "BTC_USD"
            assert tickers[0]["price"] == 50000.0


# =============================================================================
# Kraken Provider
# =============================================================================


class TestKrakenSymbolMap:
    """Test Kraken symbol mapping."""

    @pytest.fixture
    def provider(self):
        return KrakenProvider()

    def test_symbol_map_btc(self, provider):
        assert provider.SYMBOL_MAP["BTC_USD"] == "XXBTZUSD"

    def test_symbol_map_eth(self, provider):
        assert provider.SYMBOL_MAP["ETH_USD"] == "XETHZUSD"

    def test_convert_known_symbol(self, provider):
        assert provider._convert_symbol("BTC_USD") == "XXBTZUSD"

    def test_convert_unknown_symbol(self, provider):
        assert provider._convert_symbol("UNKNOWN_USD") == "UNKNOWN_USD"


class TestKrakenTimeframeMap:
    """Test Kraken timeframe mapping."""

    @pytest.fixture
    def provider(self):
        return KrakenProvider()

    def test_standard_timeframes(self, provider):
        assert provider.TIMEFRAME_MAP["1m"] == 1
        assert provider.TIMEFRAME_MAP["5m"] == 5
        assert provider.TIMEFRAME_MAP["1h"] == 60
        assert provider.TIMEFRAME_MAP["4h"] == 240
        assert provider.TIMEFRAME_MAP["1d"] == 1440


class TestKrakenConnect:
    """Test Kraken connection."""

    @pytest.fixture
    def provider(self):
        return KrakenProvider()

    @pytest.mark.asyncio
    async def test_connect_success(self, provider):
        mock_response = MagicMock(
            json=lambda: {"error": [], "result": {"unixtime": 1234567890}},
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await provider.connect()

            assert result is True
            assert provider._connected is True

    @pytest.mark.asyncio
    async def test_connect_with_error(self, provider):
        mock_response = MagicMock(
            json=lambda: {"error": ["EAPI:Invalid nonce"]},
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await provider.connect()
            assert result is False


class TestKrakenGetQuote:
    """Test Kraken quote retrieval."""

    @pytest.fixture
    def provider(self):
        return KrakenProvider()

    @pytest.mark.asyncio
    async def test_get_quote_success(self, provider):
        mock_response = MagicMock(
            json=lambda: {
                "error": [],
                "result": {
                    "XXBTZUSD": {
                        "b": ["50000.00", "1"],
                        "a": ["50010.00", "1"],
                        "c": ["50005.00", "0.1"],
                        "v": ["1000.0", "5000.0"],
                    }
                },
            },
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            quote = await provider.get_quote("BTC_USD")

            assert quote is not None
            assert quote.bid == 50000.0
            assert quote.ask == 50010.0
            assert quote.last == 50005.0

    @pytest.mark.asyncio
    async def test_get_quote_error(self, provider):
        mock_response = MagicMock(
            json=lambda: {"error": ["EQuery:Unknown asset pair"], "result": {}},
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            quote = await provider.get_quote("INVALID_USD")
            assert quote is None


class TestKrakenGetBars:
    """Test Kraken bar retrieval."""

    @pytest.fixture
    def provider(self):
        return KrakenProvider()

    @pytest.mark.asyncio
    async def test_get_bars_success(self, provider):
        mock_response = MagicMock(
            json=lambda: {
                "error": [],
                "result": {
                    "XXBTZUSD": [
                        [1609459200, "50000.00", "51000.00", "49000.00", "50500.00", "50250.00", "100.0", 500],
                        [1609462800, "50500.00", "52000.00", "50000.00", "51500.00", "50750.00", "120.0", 600],
                    ],
                    "last": 1609462800,
                },
            },
        )

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            bars = await provider.get_bars("BTC_USD", "1h", 100)

            assert bars is not None
            assert len(bars) == 2
            assert "open" in bars.columns
            assert "close" in bars.columns
            assert bars["open"].iloc[0] == 50000.0


# =============================================================================
# Weekend Mode / Orchestrator Integration
# =============================================================================


class TestWeekendDetection:
    """Test weekend detection in orchestrator."""

    def test_weekend_saturday(self):
        from nexus.scanners.orchestrator import ScannerOrchestrator

        provider = MagicMock()
        orch = ScannerOrchestrator(data_provider=provider)

        # Saturday Feb 14 2026 is a Saturday (weekday() == 5)
        saturday = MagicMock()
        saturday.weekday.return_value = 5

        with patch("nexus.scanners.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            assert orch.is_weekend() is True
            assert orch.is_crypto_only_mode() is True

    def test_weekday_not_weekend(self):
        from nexus.scanners.orchestrator import ScannerOrchestrator

        provider = MagicMock()
        orch = ScannerOrchestrator(data_provider=provider)

        monday = MagicMock()
        monday.weekday.return_value = 0

        with patch("nexus.scanners.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = monday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            assert orch.is_weekend() is False
            assert orch.is_crypto_only_mode() is False

    def test_crypto_instruments_from_registry(self):
        from nexus.scanners.orchestrator import ScannerOrchestrator

        provider = MagicMock()
        orch = ScannerOrchestrator(data_provider=provider)

        instruments = orch.get_crypto_instruments()
        assert len(instruments) >= 10
        assert "BTC_USD" in instruments
        assert "ETH_USD" in instruments

    @pytest.mark.asyncio
    async def test_scan_crypto_only_no_provider(self):
        """Without crypto provider, returns empty list."""
        from nexus.scanners.orchestrator import ScannerOrchestrator

        provider = MagicMock()
        orch = ScannerOrchestrator(data_provider=provider)
        # No crypto provider set

        result = await orch.scan_crypto_only()
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_crypto_only_with_provider(self):
        """With crypto provider, runs HF scanners."""
        from nexus.scanners.orchestrator import ScannerOrchestrator

        provider = MagicMock()
        crypto_provider = MagicMock()
        crypto_provider.get_bars = AsyncMock(return_value=None)
        crypto_provider.get_quote = AsyncMock(return_value=None)

        orch = ScannerOrchestrator(data_provider=provider, crypto_provider=crypto_provider)

        result = await orch.scan_crypto_only()
        assert isinstance(result, list)


# =============================================================================
# Crypto Instruments in Registry
# =============================================================================


class TestCryptoInstruments:
    """Test crypto instruments in registry."""

    def test_crypto_in_registry(self):
        from nexus.data.instruments import get_instrument_registry, InstrumentType

        registry = get_instrument_registry()
        crypto = registry.get_by_type(InstrumentType.CRYPTO)

        assert len(crypto) >= 10
        symbols = [c.symbol for c in crypto]
        assert "BTC_USD" in symbols
        assert "ETH_USD" in symbols
        assert "SOL_USD" in symbols

    def test_crypto_24_7(self):
        from nexus.data.instruments import get_instrument_registry

        registry = get_instrument_registry()
        weekend = registry.get_weekend_tradeable()

        assert len(weekend) >= 10
        for inst in weekend:
            assert inst.instrument_type.value == "crypto"

    def test_crypto_uses_binance_provider(self):
        from nexus.data.instruments import get_instrument_registry, DataProvider

        registry = get_instrument_registry()
        crypto = registry.get_by_provider(DataProvider.BINANCE)

        assert len(crypto) >= 10
        for inst in crypto:
            assert inst.instrument_type.value == "crypto"
