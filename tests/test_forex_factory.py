"""Tests for ForexFactory economic calendar client."""

import pytest
from datetime import datetime, date, timedelta, timezone

from nexus.data.forex_factory import (
    ForexFactoryClient,
    EconomicEvent,
    EventImpact,
    NoTradeWindow,
    get_forex_factory_client,
)


# ---------------------------------------------------------------------------
# EconomicEvent
# ---------------------------------------------------------------------------

class TestEconomicEvent:
    """Test EconomicEvent dataclass."""

    def test_is_high_impact(self):
        high = EconomicEvent(
            event_time=datetime.now(),
            currency="USD",
            impact=EventImpact.HIGH,
            event_name="Non-Farm Payrolls",
        )
        assert high.is_high_impact is True

        low = EconomicEvent(
            event_time=datetime.now(),
            currency="USD",
            impact=EventImpact.LOW,
            event_name="Some Report",
        )
        assert low.is_high_impact is False

    def test_is_high_impact_medium(self):
        med = EconomicEvent(
            event_time=datetime.now(),
            currency="USD",
            impact=EventImpact.MEDIUM,
            event_name="Retail Sales",
        )
        assert med.is_high_impact is False

    def test_minutes_until_future(self):
        future = EconomicEvent(
            event_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            currency="USD",
            impact=EventImpact.HIGH,
            event_name="Test",
        )
        assert 28 <= future.minutes_until <= 31

    def test_minutes_until_past(self):
        past = EconomicEvent(
            event_time=datetime.now(timezone.utc) - timedelta(minutes=15),
            currency="USD",
            impact=EventImpact.HIGH,
            event_name="Test",
        )
        assert past.minutes_until < 0

    def test_affects_symbol_direct_currency(self):
        event = EconomicEvent(
            event_time=datetime.now(),
            currency="EUR",
            impact=EventImpact.HIGH,
            event_name="ECB Rate",
        )
        assert event.affects_symbol("EUR/USD") is True
        assert event.affects_symbol("EURUSD") is True
        assert event.affects_symbol("GBP/USD") is False

    def test_affects_symbol_usd_affects_everything(self):
        event = EconomicEvent(
            event_time=datetime.now(),
            currency="USD",
            impact=EventImpact.HIGH,
            event_name="NFP",
        )
        assert event.affects_symbol("EUR/USD") is True
        assert event.affects_symbol("SPY") is True
        assert event.affects_symbol("AAPL") is True
        assert event.affects_symbol("GBP/JPY") is True  # USD affects all

    def test_affects_symbol_indirect_mapping(self):
        event = EconomicEvent(
            event_time=datetime.now(),
            currency="GBP",
            impact=EventImpact.HIGH,
            event_name="BOE Rate",
        )
        assert event.affects_symbol("FTSE") is True
        assert event.affects_symbol("UK100") is True
        assert event.affects_symbol("GBP/USD") is True
        assert event.affects_symbol("EUR/USD") is False

    def test_affects_symbol_case_insensitive(self):
        event = EconomicEvent(
            event_time=datetime.now(),
            currency="eur",
            impact=EventImpact.HIGH,
            event_name="ECB Rate",
        )
        assert event.affects_symbol("eur/usd") is True
        assert event.affects_symbol("EUR/USD") is True

    def test_affects_symbol_cad_oil(self):
        event = EconomicEvent(
            event_time=datetime.now(),
            currency="CAD",
            impact=EventImpact.HIGH,
            event_name="BOC Rate",
        )
        assert event.affects_symbol("OIL") is True
        assert event.affects_symbol("USOIL") is True
        assert event.affects_symbol("USD/CAD") is True


# ---------------------------------------------------------------------------
# ForexFactoryClient._parse_event
# ---------------------------------------------------------------------------

class TestParseEvent:
    """Test JSON event parsing."""

    @pytest.fixture
    def client(self):
        return ForexFactoryClient()

    def test_parse_high_impact(self, client):
        item = {
            "date": "2025-06-06",
            "time": "13:30",
            "country": "USD",
            "impact": "High",
            "title": "Non-Farm Employment Change",
            "forecast": "180K",
            "previous": "175K",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.event_name == "Non-Farm Employment Change"
        assert event.impact == EventImpact.HIGH
        assert event.currency == "USD"
        assert event.event_time == datetime(2025, 6, 6, 13, 30)
        assert event.forecast == "180K"
        assert event.previous == "175K"

    def test_parse_medium_impact(self, client):
        item = {
            "date": "2025-06-05",
            "time": "10:00",
            "country": "EUR",
            "impact": "Medium",
            "title": "Retail Sales m/m",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.impact == EventImpact.MEDIUM

    def test_parse_low_impact(self, client):
        item = {
            "date": "2025-06-05",
            "time": "08:00",
            "country": "GBP",
            "impact": "Low",
            "title": "BRC Retail Sales",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.impact == EventImpact.LOW

    def test_parse_red_impact_alias(self, client):
        """ForexFactory sometimes uses 'red' for high impact."""
        item = {
            "date": "2025-06-06",
            "time": "13:30",
            "country": "USD",
            "impact": "red",
            "title": "CPI m/m",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.impact == EventImpact.HIGH

    def test_parse_orange_impact_alias(self, client):
        item = {
            "date": "2025-06-06",
            "time": "10:00",
            "country": "USD",
            "impact": "orange",
            "title": "PMI",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.impact == EventImpact.MEDIUM

    def test_parse_holiday(self, client):
        item = {
            "date": "2025-07-04",
            "time": "All Day",
            "country": "USD",
            "impact": "Holiday",
            "title": "Independence Day",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.impact == EventImpact.HOLIDAY
        assert event.event_time == datetime(2025, 7, 4, 0, 0)

    def test_parse_tentative_time(self, client):
        item = {
            "date": "2025-06-10",
            "time": "Tentative",
            "country": "GBP",
            "impact": "High",
            "title": "BOE Interest Rate Decision",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.event_time == datetime(2025, 6, 10, 0, 0)

    def test_parse_empty_time(self, client):
        item = {
            "date": "2025-06-10",
            "time": "",
            "country": "USD",
            "impact": "Low",
            "title": "Some Report",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.event_time == datetime(2025, 6, 10, 0, 0)

    def test_parse_missing_date(self, client):
        item = {
            "time": "10:00",
            "country": "USD",
            "impact": "High",
            "title": "NFP",
        }
        assert client._parse_event(item) is None

    def test_parse_uses_currency_fallback(self, client):
        """Should fall back to 'currency' key if 'country' missing."""
        item = {
            "date": "2025-06-06",
            "time": "13:30",
            "currency": "usd",
            "impact": "High",
            "title": "CPI",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.currency == "USD"

    def test_parse_uses_event_fallback(self, client):
        """Should fall back to 'event' key if 'title' missing."""
        item = {
            "date": "2025-06-06",
            "time": "13:30",
            "country": "USD",
            "impact": "High",
            "event": "FOMC Statement",
        }
        event = client._parse_event(item)
        assert event is not None
        assert event.event_name == "FOMC Statement"


# ---------------------------------------------------------------------------
# ForexFactoryClient.get_no_trade_windows
# ---------------------------------------------------------------------------

class TestNoTradeWindows:
    """Test no-trade window generation."""

    @pytest.fixture
    def client(self):
        return ForexFactoryClient()

    def test_high_impact_creates_30min_window(self, client):
        event_time = datetime(2025, 6, 6, 13, 30)
        events = [
            EconomicEvent(
                event_time=event_time,
                currency="USD",
                impact=EventImpact.HIGH,
                event_name="FOMC",
            )
        ]

        windows = client.get_no_trade_windows(events, "EUR/USD")

        assert len(windows) == 1
        assert windows[0].reduce_size is False  # No trades allowed
        assert windows[0].start == event_time - timedelta(minutes=30)
        assert windows[0].end == event_time + timedelta(minutes=30)
        assert "HIGH impact" in windows[0].reason
        assert "FOMC" in windows[0].reason

    def test_medium_impact_creates_15min_window(self, client):
        event_time = datetime(2025, 6, 6, 10, 0)
        events = [
            EconomicEvent(
                event_time=event_time,
                currency="USD",
                impact=EventImpact.MEDIUM,
                event_name="Retail Sales",
            )
        ]

        windows = client.get_no_trade_windows(events, "SPY")

        assert len(windows) == 1
        assert windows[0].reduce_size is True
        assert windows[0].start == event_time - timedelta(minutes=15)
        assert windows[0].end == event_time + timedelta(minutes=15)

    def test_low_impact_no_window(self, client):
        events = [
            EconomicEvent(
                event_time=datetime(2025, 6, 6, 10, 0),
                currency="USD",
                impact=EventImpact.LOW,
                event_name="Some Report",
            )
        ]

        windows = client.get_no_trade_windows(events, "EUR/USD")
        assert len(windows) == 0

    def test_unrelated_currency_no_window(self, client):
        events = [
            EconomicEvent(
                event_time=datetime(2025, 6, 6, 10, 0),
                currency="JPY",
                impact=EventImpact.HIGH,
                event_name="BOJ Rate",
            )
        ]

        windows = client.get_no_trade_windows(events, "EUR/USD")
        assert len(windows) == 0

    def test_multiple_events_multiple_windows(self, client):
        events = [
            EconomicEvent(
                event_time=datetime(2025, 6, 6, 13, 30),
                currency="USD",
                impact=EventImpact.HIGH,
                event_name="NFP",
            ),
            EconomicEvent(
                event_time=datetime(2025, 6, 6, 15, 0),
                currency="USD",
                impact=EventImpact.MEDIUM,
                event_name="ISM",
            ),
        ]

        windows = client.get_no_trade_windows(events, "SPY")
        assert len(windows) == 2
        assert windows[0].reduce_size is False  # HIGH = no trade
        assert windows[1].reduce_size is True   # MEDIUM = reduce

    def test_holiday_no_window(self, client):
        events = [
            EconomicEvent(
                event_time=datetime(2025, 7, 4, 0, 0),
                currency="USD",
                impact=EventImpact.HOLIDAY,
                event_name="Independence Day",
            )
        ]

        windows = client.get_no_trade_windows(events, "SPY")
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# EconomicCalendar integration
# ---------------------------------------------------------------------------

class TestEconomicCalendarIntegration:
    """Test EconomicCalendar wrapper uses ForexFactory types."""

    def test_convert_ff_event(self):
        from nexus.data.economic_calendar import _convert_ff_event

        ff_event = EconomicEvent(
            event_time=datetime(2025, 6, 6, 13, 30),
            currency="USD",
            impact=EventImpact.HIGH,
            event_name="NFP",
            forecast="180K",
            previous="175K",
        )

        cal_event = _convert_ff_event(ff_event)
        assert cal_event.time == datetime(2025, 6, 6, 13, 30)
        assert cal_event.currency == "USD"
        assert cal_event.event == "NFP"
        assert cal_event.forecast == "180K"
        assert cal_event.is_high_impact is True

    def test_known_events_fallback_nfp_friday(self):
        """Should produce NFP events on first Friday of month."""
        from nexus.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        # First Friday of June 2025 is June 6
        events = cal._get_known_events(date(2025, 6, 6))
        assert len(events) == 2
        names = {e.event for e in events}
        assert "Non-Farm Payrolls" in names
        assert "Unemployment Rate" in names

    def test_known_events_non_nfp_day(self):
        """Should return empty on non-NFP days."""
        from nexus.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        events = cal._get_known_events(date(2025, 6, 10))  # Tuesday
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    """Test singleton pattern."""

    def test_singleton(self):
        import nexus.data.forex_factory as mod
        mod._ff_client = None

        c1 = get_forex_factory_client()
        c2 = get_forex_factory_client()
        assert c1 is c2

        mod._ff_client = None
