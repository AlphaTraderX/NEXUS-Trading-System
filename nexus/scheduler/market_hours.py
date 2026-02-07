"""
Market Hours Module - Knows when markets and sessions are active.

All times in UTC for consistency.
"""

from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


class MarketHours:
    """
    Handles all market session timing logic.
    
    Markets:
    - Forex: Sunday 22:00 UTC - Friday 22:00 UTC (24/5)
    - US Stocks: Mon-Fri 14:30-21:00 UTC
    - US Pre-Market: Mon-Fri 09:00-14:30 UTC
    - UK Stocks: Mon-Fri 08:00-16:30 UTC
    
    Sessions (for scanners):
    - Asian: 00:00-07:00 UTC
    - London Open: 07:00-09:00 UTC
    - European: 08:00-14:30 UTC
    - US Pre-Market: 12:00-14:30 UTC
    - US Open: 14:30-15:30 UTC
    - US Session: 14:30-21:00 UTC
    - Power Hour: 20:00-21:00 UTC
    """
    
    # Market hours (UTC)
    FOREX_OPEN = time(22, 0)   # Sunday
    FOREX_CLOSE = time(22, 0)  # Friday
    
    US_STOCKS_OPEN = time(14, 30)
    US_STOCKS_CLOSE = time(21, 0)
    
    US_PREMARKET_OPEN = time(9, 0)
    US_PREMARKET_CLOSE = time(14, 30)
    
    UK_STOCKS_OPEN = time(8, 0)
    UK_STOCKS_CLOSE = time(16, 30)
    
    # Session windows (UTC)
    SESSIONS = {
        "asian": (time(0, 0), time(7, 0)),
        "london_open": (time(7, 0), time(9, 0)),
        "european": (time(8, 0), time(14, 30)),
        "us_premarket": (time(12, 0), time(14, 30)),
        "us_open": (time(14, 30), time(15, 30)),
        "us_session": (time(14, 30), time(21, 0)),
        "power_hour": (time(20, 0), time(21, 0)),
    }
    
    @classmethod
    def now_utc(cls) -> datetime:
        """Get current time in UTC."""
        return datetime.now(UTC)
    
    @classmethod
    def is_weekday(cls, dt: Optional[datetime] = None) -> bool:
        """Check if it's a weekday (Mon=0 to Fri=4)."""
        dt = dt or cls.now_utc()
        return dt.weekday() < 5
    
    @classmethod
    def is_forex_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if forex market is open.
        
        Forex is open from Sunday 22:00 UTC to Friday 22:00 UTC.
        """
        dt = dt or cls.now_utc()
        weekday = dt.weekday()  # Monday=0, Sunday=6
        current_time = dt.time()
        
        # Saturday: always closed
        if weekday == 5:
            return False
        
        # Sunday: open after 22:00 UTC
        if weekday == 6:
            return current_time >= cls.FOREX_OPEN
        
        # Friday: open until 22:00 UTC
        if weekday == 4:
            return current_time < cls.FOREX_CLOSE
        
        # Monday-Thursday: always open
        return True
    
    @classmethod
    def is_us_stocks_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if US stock market is open.
        
        US stocks: Mon-Fri 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)
        """
        dt = dt or cls.now_utc()
        
        if not cls.is_weekday(dt):
            return False
        
        current_time = dt.time()
        return cls.US_STOCKS_OPEN <= current_time < cls.US_STOCKS_CLOSE
    
    @classmethod
    def is_us_premarket_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if US pre-market is open.
        
        Pre-market: Mon-Fri 09:00-14:30 UTC (4:00 AM - 9:30 AM ET)
        """
        dt = dt or cls.now_utc()
        
        if not cls.is_weekday(dt):
            return False
        
        current_time = dt.time()
        return cls.US_PREMARKET_OPEN <= current_time < cls.US_PREMARKET_CLOSE
    
    @classmethod
    def is_uk_stocks_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if UK stock market is open.
        
        UK stocks: Mon-Fri 08:00-16:30 UTC
        """
        dt = dt or cls.now_utc()
        
        if not cls.is_weekday(dt):
            return False
        
        current_time = dt.time()
        return cls.UK_STOCKS_OPEN <= current_time < cls.UK_STOCKS_CLOSE
    
    @classmethod
    def is_session_active(cls, session: str, dt: Optional[datetime] = None) -> bool:
        """
        Check if a specific trading session is active.
        
        Sessions: asian, london_open, european, us_premarket, us_open, us_session, power_hour
        """
        dt = dt or cls.now_utc()
        
        if session not in cls.SESSIONS:
            return False
        
        # Sessions only active on weekdays (except asian which can span weekend edge)
        if session != "asian" and not cls.is_weekday(dt):
            return False
        
        # Asian session: also check forex is open
        if session == "asian" and not cls.is_forex_open(dt):
            return False
        
        start_time, end_time = cls.SESSIONS[session]
        current_time = dt.time()
        
        # Handle sessions that cross midnight (none currently, but future-proof)
        if start_time <= end_time:
            return start_time <= current_time < end_time
        else:
            return current_time >= start_time or current_time < end_time
    
    @classmethod
    def get_active_sessions(cls, dt: Optional[datetime] = None) -> list[str]:
        """Get list of currently active sessions."""
        dt = dt or cls.now_utc()
        return [session for session in cls.SESSIONS if cls.is_session_active(session, dt)]
    
    @classmethod
    def get_market_status(cls, dt: Optional[datetime] = None) -> dict:
        """
        Get comprehensive market status.
        
        Returns dict with all market/session states.
        """
        dt = dt or cls.now_utc()
        
        return {
            "timestamp": dt.isoformat(),
            "weekday": dt.strftime("%A"),
            "markets": {
                "forex": cls.is_forex_open(dt),
                "us_stocks": cls.is_us_stocks_open(dt),
                "us_premarket": cls.is_us_premarket_open(dt),
                "uk_stocks": cls.is_uk_stocks_open(dt),
            },
            "active_sessions": cls.get_active_sessions(dt),
        }
    
    @classmethod
    def get_next_market_open(cls, dt: Optional[datetime] = None) -> dict:
        """
        Get info about next market opening.
        
        Useful for knowing when to start scanning.
        """
        dt = dt or cls.now_utc()
        
        status = cls.get_market_status(dt)
        
        # If any market is open, return that
        for market, is_open in status["markets"].items():
            if is_open:
                return {
                    "market": market,
                    "status": "OPEN",
                    "message": f"{market} is currently open"
                }
        
        # Calculate time until next opening
        weekday = dt.weekday()
        current_time = dt.time()
        
        # Saturday - forex opens Sunday 22:00
        if weekday == 5:
            hours_until = 24 + (22 - current_time.hour)
            return {
                "market": "forex",
                "status": "CLOSED",
                "message": f"Forex opens in ~{hours_until} hours (Sunday 22:00 UTC)"
            }
        
        # Sunday before 22:00 - forex opens at 22:00
        if weekday == 6 and current_time < time(22, 0):
            hours_until = 22 - current_time.hour
            return {
                "market": "forex",
                "status": "CLOSED",
                "message": f"Forex opens in ~{hours_until} hours (22:00 UTC)"
            }
        
        # Weekday after hours - US stocks open tomorrow 14:30
        if weekday < 4 and current_time >= time(21, 0):
            return {
                "market": "us_stocks",
                "status": "CLOSED",
                "message": "US stocks open tomorrow at 14:30 UTC"
            }
        
        # Friday after forex close - wait until Sunday
        if weekday == 4 and current_time >= time(22, 0):
            return {
                "market": "forex",
                "status": "CLOSED",
                "message": "Markets closed for weekend. Forex opens Sunday 22:00 UTC"
            }
        
        return {
            "market": "unknown",
            "status": "CLOSED",
            "message": "Calculating next market open..."
        }
