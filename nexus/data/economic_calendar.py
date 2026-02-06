"""
NEXUS Economic Calendar

Fetches economic events for risk filtering.
Primary purpose: DON'T TRADE INTO HIGH-IMPACT NEWS.

Sources: ForexFactory (scraped), Investing.com calendar
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class EventImpact(str, Enum):
    """Economic event impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    time: datetime
    currency: str  # USD, EUR, GBP, etc.
    impact: EventImpact
    event: str  # "Non-Farm Payrolls"
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    
    @property
    def is_high_impact(self) -> bool:
        return self.impact == EventImpact.HIGH
    
    @property
    def minutes_until(self) -> int:
        """Minutes until this event."""
        delta = self.time - datetime.now()
        return int(delta.total_seconds() / 60)


@dataclass
class TimeWindow:
    """A time window (e.g., no-trade period)."""
    start: datetime
    end: datetime
    reason: str
    
    def contains(self, dt: datetime) -> bool:
        return self.start <= dt <= self.end


# Currency to affected markets mapping
CURRENCY_MARKETS = {
    "USD": ["US_STOCKS", "US_FUTURES", "FOREX_MAJORS"],
    "EUR": ["EU_STOCKS", "FOREX_MAJORS", "FOREX_CROSSES"],
    "GBP": ["UK_STOCKS", "FOREX_MAJORS", "FOREX_CROSSES"],
    "JPY": ["FOREX_MAJORS", "FOREX_CROSSES"],
    "AUD": ["FOREX_MAJORS", "FOREX_CROSSES"],
    "CAD": ["FOREX_MAJORS", "FOREX_CROSSES"],
    "CHF": ["FOREX_CROSSES"],
    "NZD": ["FOREX_CROSSES"],
}

# High impact events we always watch
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "NFP",
    "FOMC",
    "Federal Funds Rate",
    "Fed Interest Rate Decision",
    "CPI",
    "Consumer Price Index",
    "GDP",
    "Gross Domestic Product",
    "ECB Interest Rate",
    "ECB Rate Decision",
    "BOE Interest Rate",
    "BOE Rate Decision",
    "Unemployment Rate",
    "Retail Sales",
    "PMI",
    "ISM Manufacturing",
    "ISM Services",
]


class EconomicCalendar:
    """
    Economic calendar for risk filtering.
    
    Primary purpose: RISK MANAGEMENT (not signal generation)
    
    Rules:
    - 30 min before/after HIGH impact = NO NEW TRADES
    - 15 min before/after MEDIUM impact = REDUCE SIZE
    - Check calendar at start of each session
    """
    
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cache: List[EconomicEvent] = []
        self._cache_date: Optional[date] = None
    
    async def fetch_calendar(self, target_date: Optional[date] = None) -> List[EconomicEvent]:
        """
        Fetch economic calendar for a date.
        
        Note: In production, this would scrape ForexFactory or use a paid API.
        For now, returns known scheduled events.
        """
        target = target_date or date.today()
        
        # Check cache
        if self._cache_date == target and self._cache:
            return self._cache
        
        events = []
        
        # Try to fetch from ForexFactory (simplified)
        try:
            events = await self._fetch_forex_factory(target)
        except Exception as e:
            logger.warning(f"Failed to fetch ForexFactory: {e}")
        
        # Add known recurring events if none fetched
        if not events:
            events = self._get_known_events(target)
        
        self._cache = events
        self._cache_date = target
        
        return events
    
    async def _fetch_forex_factory(self, target: date) -> List[EconomicEvent]:
        """
        Fetch from ForexFactory.
        
        Note: This is a simplified version. Full implementation would
        parse the ForexFactory HTML or use their calendar feed.
        """
        # ForexFactory uses a specific URL format
        url = f"https://www.forexfactory.com/calendar?day={target.strftime('%b%d.%Y').lower()}"
        
        # For now, return empty - would need proper HTML parsing
        # In production, use BeautifulSoup or similar
        return []
    
    def _get_known_events(self, target: date) -> List[EconomicEvent]:
        """Get known scheduled events (fallback)."""
        events = []
        
        # FOMC meetings (roughly every 6 weeks)
        # First Wednesday after first Monday of months with meetings
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        if target.month in fomc_months:
            # Simplified - would need actual FOMC calendar
            pass
        
        # NFP - First Friday of every month
        first_day = target.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        if target == first_friday:
            events.append(EconomicEvent(
                time=datetime.combine(target, time(13, 30)),  # 8:30 AM ET = 13:30 UK
                currency="USD",
                impact=EventImpact.HIGH,
                event="Non-Farm Payrolls",
            ))
            events.append(EconomicEvent(
                time=datetime.combine(target, time(13, 30)),
                currency="USD",
                impact=EventImpact.HIGH,
                event="Unemployment Rate",
            ))
        
        return events
    
    async def get_todays_events(self) -> List[EconomicEvent]:
        """Get all events for today."""
        return await self.fetch_calendar(date.today())
    
    async def get_upcoming_events(self, hours: int = 4) -> List[EconomicEvent]:
        """Get events in the next N hours."""
        events = await self.get_todays_events()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        
        return [e for e in events if now <= e.time <= cutoff]
    
    async def get_high_impact_events(self, hours: int = 4) -> List[EconomicEvent]:
        """Get only HIGH impact events in next N hours."""
        upcoming = await self.get_upcoming_events(hours)
        return [e for e in upcoming if e.is_high_impact]
    
    async def is_safe_to_trade(
        self, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Check if it's safe to trade given upcoming news.
        
        Returns:
            {
                "safe": bool,
                "reduce_size": bool,
                "reason": str,
                "next_event": Optional[EconomicEvent],
                "minutes_until": int
            }
        """
        now = timestamp or datetime.now()
        events = await self.get_upcoming_events(hours=2)
        
        # Determine which currencies affect this symbol
        affected_currencies = self._get_affected_currencies(symbol)
        
        for event in events:
            # Skip if currency doesn't affect this symbol
            if event.currency not in affected_currencies:
                continue
            
            minutes_until = int((event.time - now).total_seconds() / 60)
            minutes_after = -minutes_until if minutes_until < 0 else 0
            
            # HIGH impact: 30 min before/after = NO TRADE
            if event.impact == EventImpact.HIGH:
                if -30 <= minutes_until <= 30:
                    return {
                        "safe": False,
                        "reduce_size": False,
                        "reason": f"High-impact event in {minutes_until}min: {event.event}",
                        "next_event": event,
                        "minutes_until": minutes_until,
                    }
                elif 30 < minutes_until <= 60:
                    return {
                        "safe": True,
                        "reduce_size": True,
                        "reason": f"High-impact event in {minutes_until}min",
                        "next_event": event,
                        "minutes_until": minutes_until,
                    }
            
            # MEDIUM impact: 15 min before/after = REDUCE SIZE
            elif event.impact == EventImpact.MEDIUM:
                if -15 <= minutes_until <= 15:
                    return {
                        "safe": True,
                        "reduce_size": True,
                        "reason": f"Medium-impact event in {minutes_until}min: {event.event}",
                        "next_event": event,
                        "minutes_until": minutes_until,
                    }
        
        return {
            "safe": True,
            "reduce_size": False,
            "reason": "No impactful events nearby",
            "next_event": None,
            "minutes_until": 999,
        }
    
    async def get_no_trade_windows(self) -> List[TimeWindow]:
        """Get time windows where no new trades should open."""
        events = await self.get_todays_events()
        windows = []
        
        for event in events:
            if event.impact == EventImpact.HIGH:
                windows.append(TimeWindow(
                    start=event.time - timedelta(minutes=30),
                    end=event.time + timedelta(minutes=30),
                    reason=f"HIGH: {event.event} ({event.currency})",
                ))
        
        return windows
    
    async def get_reduced_size_windows(self) -> List[TimeWindow]:
        """Get time windows where position size should be reduced."""
        events = await self.get_todays_events()
        windows = []
        
        for event in events:
            if event.impact == EventImpact.HIGH:
                # 60-30 min before
                windows.append(TimeWindow(
                    start=event.time - timedelta(minutes=60),
                    end=event.time - timedelta(minutes=30),
                    reason=f"Approaching HIGH: {event.event}",
                ))
            elif event.impact == EventImpact.MEDIUM:
                # 30-15 min before
                windows.append(TimeWindow(
                    start=event.time - timedelta(minutes=30),
                    end=event.time - timedelta(minutes=15),
                    reason=f"Approaching MEDIUM: {event.event}",
                ))
        
        return windows
    
    def _get_affected_currencies(self, symbol: str) -> List[str]:
        """Get currencies that affect a symbol."""
        symbol_upper = symbol.upper()
        
        # Forex pair
        if "/" in symbol:
            base, quote = symbol.split("/")
            return [base, quote]
        
        # Stock/ETF
        us_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", 
                      "META", "GOOGL", "AMZN", "XLF", "XLE", "XLK", "XLV"]
        if symbol_upper in us_symbols:
            return ["USD"]
        
        # Futures
        if symbol_upper in ["ES", "NQ", "RTY"]:
            return ["USD"]
        if symbol_upper in ["CL", "GC"]:
            return ["USD"]
        
        # Default - USD affects most things
        return ["USD"]
