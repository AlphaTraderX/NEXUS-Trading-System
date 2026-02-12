"""
ForexFactory Economic Calendar Client

Fetches economic events for risk filtering.
Used to avoid trading into high-impact news events.

Data source: nfs.faireconomy.media (ForexFactory mirror with JSON API).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class EventImpact(str, Enum):
    """News event impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"


@dataclass
class EconomicEvent:
    """Single economic calendar event."""
    event_time: datetime
    currency: str
    impact: EventImpact
    event_name: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    @property
    def is_high_impact(self) -> bool:
        return self.impact == EventImpact.HIGH

    @property
    def minutes_until(self) -> float:
        """Minutes until event (negative if passed)."""
        now = datetime.now(timezone.utc)
        event_utc = (
            self.event_time.replace(tzinfo=timezone.utc)
            if self.event_time.tzinfo is None
            else self.event_time
        )
        delta = event_utc - now
        return delta.total_seconds() / 60

    def affects_symbol(self, symbol: str) -> bool:
        """Check if event currency affects a symbol."""
        symbol_upper = symbol.upper()
        currency_upper = self.currency.upper()

        # Direct currency match
        if currency_upper in symbol_upper:
            return True

        # USD affects everything
        if currency_upper == "USD":
            return True

        # Map currencies to affected symbols/keywords
        CURRENCY_SYMBOLS: Dict[str, List[str]] = {
            "EUR": ["EUR", "DAX", "CAC", "STOXX"],
            "GBP": ["GBP", "FTSE", "UK100"],
            "JPY": ["JPY", "NIKKEI", "JP225"],
            "AUD": ["AUD"],
            "CAD": ["CAD", "OIL", "USOIL"],
            "CHF": ["CHF"],
            "NZD": ["NZD"],
            "CNY": ["CNY", "CHINA", "HSI"],
        }

        affected = CURRENCY_SYMBOLS.get(currency_upper, [currency_upper])
        return any(a in symbol_upper for a in affected)


@dataclass
class NoTradeWindow:
    """Period when trading should be avoided or reduced."""
    start: datetime
    end: datetime
    reason: str
    reduce_size: bool  # True = reduce 50%, False = no new trades
    event: EconomicEvent


class ForexFactoryClient:
    """
    Client for ForexFactory economic calendar.

    Usage:
        client = ForexFactoryClient()
        events = await client.get_events_for_week()
        windows = client.get_no_trade_windows(events, "EUR/USD")
    """

    # ForexFactory calendar JSON endpoint (public mirror)
    CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        self._cache: Dict[str, Tuple[datetime, List[EconomicEvent]]] = {}
        self._cache_ttl = timedelta(hours=1)

    async def get_events_for_week(self) -> List[EconomicEvent]:
        """Get all events for current week."""
        cache_key = "week"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        events: List[EconomicEvent] = []

        try:
            response = await self._client.get(self.CALENDAR_URL)

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    try:
                        event = self._parse_event(item)
                        if event:
                            events.append(event)
                    except Exception as e:
                        logger.debug(f"Failed to parse event: {e}")
                        continue
            else:
                logger.warning(f"ForexFactory returned {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to fetch ForexFactory calendar: {e}")

        # Cache results
        self._cache[cache_key] = (datetime.now(), events)

        logger.info(f"Fetched {len(events)} economic events")
        return events

    def _parse_event(self, item: dict) -> Optional[EconomicEvent]:
        """Parse a single event from JSON."""
        try:
            date_str = item.get("date", "")
            time_str = item.get("time", "")

            if not date_str:
                return None

            # Handle "All Day" / "Tentative" / empty time
            if time_str in ("All Day", "Tentative", ""):
                time_str = "00:00"

            dt_str = f"{date_str} {time_str}"

            # Try multiple formats
            event_dt: Optional[datetime] = None
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %I:%M%p", "%m-%d-%Y %H:%M"]:
                try:
                    event_dt = datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    continue

            if event_dt is None:
                # Date only fallback
                try:
                    event_dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    return None

            # Parse impact
            impact_str = item.get("impact", "").lower()
            if "high" in impact_str or impact_str == "red":
                impact = EventImpact.HIGH
            elif "medium" in impact_str or impact_str == "orange":
                impact = EventImpact.MEDIUM
            elif "holiday" in impact_str:
                impact = EventImpact.HOLIDAY
            else:
                impact = EventImpact.LOW

            return EconomicEvent(
                event_time=event_dt,
                currency=item.get("country", item.get("currency", "")).upper(),
                impact=impact,
                event_name=item.get("title", item.get("event", "")),
                forecast=item.get("forecast"),
                previous=item.get("previous"),
                actual=item.get("actual"),
            )

        except Exception as e:
            logger.debug(f"Error parsing event: {e}")
            return None

    async def get_events_for_date(self, target_date: date) -> List[EconomicEvent]:
        """Get events for a specific date."""
        all_events = await self.get_events_for_week()
        return [e for e in all_events if e.event_time.date() == target_date]

    async def get_todays_events(self) -> List[EconomicEvent]:
        """Get all events for today."""
        return await self.get_events_for_date(date.today())

    async def get_high_impact_events(self, hours: int = 24) -> List[EconomicEvent]:
        """Get high-impact events in next N hours."""
        all_events = await self.get_events_for_week()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        return [
            e for e in all_events
            if e.is_high_impact
            and now <= e.event_time <= cutoff
        ]

    async def get_upcoming_events(self, hours: int = 4) -> List[EconomicEvent]:
        """Get all events in next N hours."""
        all_events = await self.get_events_for_week()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        return [
            e for e in all_events
            if now <= e.event_time <= cutoff
        ]

    def get_no_trade_windows(
        self,
        events: List[EconomicEvent],
        symbol: str,
    ) -> List[NoTradeWindow]:
        """
        Get no-trade windows for a symbol based on events.

        Rules:
        - HIGH impact: No trades 30 min before/after
        - MEDIUM impact: Reduce size 15 min before/after
        """
        windows: List[NoTradeWindow] = []

        for event in events:
            if not event.affects_symbol(symbol):
                continue

            if event.is_high_impact:
                windows.append(NoTradeWindow(
                    start=event.event_time - timedelta(minutes=30),
                    end=event.event_time + timedelta(minutes=30),
                    reason=f"HIGH impact: {event.event_name}",
                    reduce_size=False,
                    event=event,
                ))
            elif event.impact == EventImpact.MEDIUM:
                windows.append(NoTradeWindow(
                    start=event.event_time - timedelta(minutes=15),
                    end=event.event_time + timedelta(minutes=15),
                    reason=f"MEDIUM impact: {event.event_name}",
                    reduce_size=True,
                    event=event,
                ))

        return windows

    async def is_safe_to_trade(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict:
        """
        Check if it's safe to trade a symbol right now.

        Returns:
            {
                "safe": bool,
                "reduce_size": bool,
                "reason": str,
                "next_event": Optional[EconomicEvent],
                "minutes_until": Optional[float]
            }
        """
        if timestamp is None:
            timestamp = datetime.now()

        events = await self.get_upcoming_events(hours=2)
        windows = self.get_no_trade_windows(events, symbol)

        for window in windows:
            if window.start <= timestamp <= window.end:
                return {
                    "safe": window.reduce_size,  # Can trade at reduced size
                    "reduce_size": window.reduce_size,
                    "reason": window.reason,
                    "next_event": window.event,
                    "minutes_until": window.event.minutes_until,
                }

        # Check for upcoming relevant events
        relevant_events = [e for e in events if e.affects_symbol(symbol)]
        if relevant_events:
            next_event = min(relevant_events, key=lambda e: e.event_time)
            return {
                "safe": True,
                "reduce_size": False,
                "reason": f"Next event: {next_event.event_name} in {next_event.minutes_until:.0f} min",
                "next_event": next_event,
                "minutes_until": next_event.minutes_until,
            }

        return {
            "safe": True,
            "reduce_size": False,
            "reason": "No upcoming events",
            "next_event": None,
            "minutes_until": None,
        }

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Singleton
_ff_client: Optional[ForexFactoryClient] = None


def get_forex_factory_client() -> ForexFactoryClient:
    """Get global ForexFactory client."""
    global _ff_client
    if _ff_client is None:
        _ff_client = ForexFactoryClient()
    return _ff_client
