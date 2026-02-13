"""Tests for global instrument registry."""

import pytest
from datetime import datetime, timezone, time

from nexus.data.instruments import (
    InstrumentRegistry,
    InstrumentType,
    Region,
    DataProvider,
    TradingSession,
    get_instrument_registry,
)


class TestTradingSession:
    """Test trading session logic."""

    def test_normal_session_open(self):
        """Test normal trading hours (not overnight)."""
        session = TradingSession(
            open_time=time(8, 0),
            close_time=time(16, 0),
            timezone="UTC",
            days=[0, 1, 2, 3, 4],
        )

        # Monday 10:00 UTC - should be open
        dt = datetime(2026, 2, 16, 10, 0, tzinfo=timezone.utc)  # Monday
        assert session.is_open(dt) is True

        # Monday 20:00 UTC - should be closed
        dt = datetime(2026, 2, 16, 20, 0, tzinfo=timezone.utc)
        assert session.is_open(dt) is False

    def test_weekend_closed(self):
        """Test markets closed on weekends."""
        session = TradingSession(
            open_time=time(8, 0),
            close_time=time(16, 0),
            timezone="UTC",
            days=[0, 1, 2, 3, 4],  # Mon-Fri
        )

        # Saturday - should be closed
        dt = datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc)  # Saturday
        assert session.is_open(dt) is False

    def test_24_7_session(self):
        """Test 24/7 crypto session."""
        session = TradingSession(
            open_time=time(0, 0),
            close_time=time(23, 59),
            timezone="UTC",
            days=[0, 1, 2, 3, 4, 5, 6],  # All week
        )

        # Saturday 3am - should be open (crypto)
        dt = datetime(2026, 2, 21, 3, 0, tzinfo=timezone.utc)
        assert session.is_open(dt) is True

    def test_overnight_session(self):
        """Test overnight session (open > close means wraps midnight)."""
        session = TradingSession(
            open_time=time(22, 0),
            close_time=time(6, 0),
            timezone="UTC",
            days=[0, 1, 2, 3, 4],
        )

        # Monday 23:00 - should be open
        dt = datetime(2026, 2, 16, 23, 0, tzinfo=timezone.utc)
        assert session.is_open(dt) is True

        # Monday 3:00 - should be open (after midnight)
        dt = datetime(2026, 2, 16, 3, 0, tzinfo=timezone.utc)
        assert session.is_open(dt) is True

        # Monday 12:00 - should be closed
        dt = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
        assert session.is_open(dt) is False


class TestInstrumentRegistry:
    """Test instrument registry."""

    @pytest.fixture
    def registry(self):
        return InstrumentRegistry()

    def test_loads_instruments(self, registry):
        """Test registry loads instruments."""
        assert registry.total_count > 0

    def test_minimum_count(self, registry):
        """Test we have substantial coverage."""
        assert registry.total_count >= 400

    def test_has_us_stocks(self, registry):
        """Test US stocks loaded."""
        us_stocks = registry.get_by_type(InstrumentType.STOCK)
        us_region = [s for s in us_stocks if s.region == Region.US]
        assert len(us_region) >= 100

    def test_has_uk_stocks(self, registry):
        """Test UK stocks loaded."""
        uk = [i for i in registry.get_by_type(InstrumentType.STOCK) if i.region == Region.UK]
        assert len(uk) >= 50

    def test_has_europe_stocks(self, registry):
        """Test European stocks loaded."""
        eu = [i for i in registry.get_by_type(InstrumentType.STOCK) if i.region == Region.EUROPE]
        assert len(eu) >= 40

    def test_has_asia_stocks(self, registry):
        """Test Asian stocks loaded."""
        asia = [
            i for i in registry.get_by_type(InstrumentType.STOCK)
            if i.region in (Region.ASIA_JAPAN, Region.ASIA_HK, Region.ASIA_AU)
        ]
        assert len(asia) >= 40

    def test_has_forex(self, registry):
        """Test forex pairs loaded."""
        forex = registry.get_by_type(InstrumentType.FOREX)
        assert len(forex) >= 20

    def test_has_crypto(self, registry):
        """Test crypto loaded."""
        crypto = registry.get_by_type(InstrumentType.CRYPTO)
        assert len(crypto) >= 10

    def test_has_indices(self, registry):
        """Test indices loaded."""
        indices = registry.get_by_type(InstrumentType.INDEX)
        assert len(indices) >= 10

    def test_has_commodities(self, registry):
        """Test commodities loaded."""
        commodities = registry.get_by_type(InstrumentType.COMMODITY)
        assert len(commodities) >= 10

    def test_get_by_provider(self, registry):
        """Test filtering by provider."""
        polygon = registry.get_by_provider(DataProvider.POLYGON)
        assert len(polygon) > 0
        assert all(i.provider == DataProvider.POLYGON for i in polygon)

        ig = registry.get_by_provider(DataProvider.IG)
        assert len(ig) > 0

        oanda = registry.get_by_provider(DataProvider.OANDA)
        assert len(oanda) > 0

        binance = registry.get_by_provider(DataProvider.BINANCE)
        assert len(binance) > 0

    def test_get_weekend_tradeable(self, registry):
        """Test weekend trading (crypto only)."""
        weekend = registry.get_weekend_tradeable()
        assert len(weekend) > 0
        assert all(i.instrument_type == InstrumentType.CRYPTO for i in weekend)

    def test_search(self, registry):
        """Test search functionality."""
        results = registry.search("AAPL")
        assert len(results) >= 1
        assert any(i.symbol == "AAPL" for i in results)

    def test_search_by_name(self, registry):
        """Test search by company name."""
        results = registry.search("APPLE")
        assert any(i.symbol == "AAPL" for i in results)

    def test_get_high_leverage(self, registry):
        """Test high leverage filter."""
        high_lev = registry.get_high_leverage(min_leverage=20.0)
        assert all(i.leverage_available >= 20.0 for i in high_lev)

    def test_get_low_spread(self, registry):
        """Test low spread filter."""
        low_spread = registry.get_low_spread(max_spread=0.05)
        assert all(i.typical_spread_pct <= 0.05 for i in low_spread)

    def test_get_by_sector(self, registry):
        """Test sector filter."""
        tech = registry.get_by_sector("Technology")
        assert len(tech) > 0
        assert all(i.sector == "Technology" for i in tech)

    def test_summary(self, registry):
        """Test summary statistics."""
        summary = registry.summary()

        assert "total" in summary
        assert "by_type" in summary
        assert "by_region" in summary
        assert "by_provider" in summary
        assert summary["total"] > 0

    def test_get_all(self, registry):
        """Test get_all returns all instruments."""
        all_instruments = registry.get_all()
        assert len(all_instruments) == registry.total_count


class TestSingleton:
    """Test singleton pattern."""

    def test_singleton(self):
        # Reset singleton for clean test
        import nexus.data.instruments as mod
        mod._registry = None

        r1 = get_instrument_registry()
        r2 = get_instrument_registry()
        assert r1 is r2

        # Clean up
        mod._registry = None


class TestSpecificInstruments:
    """Test specific important instruments exist."""

    @pytest.fixture
    def registry(self):
        return InstrumentRegistry()

    def test_major_us_stocks(self, registry):
        """Test major US stocks exist."""
        majors = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
        for symbol in majors:
            inst = registry.get(symbol)
            assert inst is not None, f"Missing {symbol}"
            assert inst.provider == DataProvider.POLYGON
            assert inst.region == Region.US

    def test_major_forex_pairs(self, registry):
        """Test major forex pairs exist."""
        majors = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
        for symbol in majors:
            inst = registry.get(symbol)
            assert inst is not None, f"Missing {symbol}"
            assert inst.provider == DataProvider.OANDA

    def test_major_crypto(self, registry):
        """Test major crypto exists."""
        majors = ["BTC_USD", "ETH_USD", "SOL_USD"]
        for symbol in majors:
            inst = registry.get(symbol)
            assert inst is not None, f"Missing {symbol}"
            assert inst.provider == DataProvider.BINANCE

    def test_major_indices(self, registry):
        """Test major indices exist."""
        majors = ["US500", "US100", "UK100", "DE40"]
        for symbol in majors:
            inst = registry.get(symbol)
            assert inst is not None, f"Missing {symbol}"
            assert inst.instrument_type == InstrumentType.INDEX

    def test_gold(self, registry):
        """Test gold exists."""
        gold = registry.get("XAUUSD")
        assert gold is not None
        assert gold.instrument_type == InstrumentType.COMMODITY

    def test_uk_stock_currency(self, registry):
        """Test UK stocks have GBP currency."""
        azn = registry.get("AZN.L")
        assert azn is not None
        assert azn.currency == "GBP"
        assert azn.region == Region.UK

    def test_japan_stock_currency(self, registry):
        """Test Japan stocks have JPY currency."""
        toyota = registry.get("7203.T")
        assert toyota is not None
        assert toyota.currency == "JPY"
        assert toyota.region == Region.ASIA_JAPAN

    def test_hk_stock_currency(self, registry):
        """Test HK stocks have HKD currency."""
        tencent = registry.get("0700.HK")
        assert tencent is not None
        assert tencent.currency == "HKD"
        assert tencent.region == Region.ASIA_HK

    def test_au_stock_currency(self, registry):
        """Test AU stocks have AUD currency."""
        bhp = registry.get("BHP.AX")
        assert bhp is not None
        assert bhp.currency == "AUD"
        assert bhp.region == Region.ASIA_AU

    def test_no_duplicate_symbols(self, registry):
        """Test no duplicate symbols (dict keys enforce this, but verify counts)."""
        all_instruments = registry.get_all()
        symbols = [i.symbol for i in all_instruments]
        assert len(symbols) == len(set(symbols))
