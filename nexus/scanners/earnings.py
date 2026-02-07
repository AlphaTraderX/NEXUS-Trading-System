"""
Earnings Drift (PEAD) Scanner.

Detects post-earnings announcement drift opportunities in small/mid-cap stocks.
"""

import pytz
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd

from .base import BaseScanner
from nexus.core.enums import EdgeType, Market, Direction
from nexus.core.models import Opportunity


class EarningsDriftScanner(BaseScanner):
    """
    Post-Earnings Announcement Drift (PEAD) scanner.

    EDGE: Stocks continue drifting in the direction of earnings surprise
    SOURCE: Academic research shows 2.1% monthly abnormal returns

    CONDITION: Only works in small/mid-cap stocks (large-cap arbitraged away)

    TIMING: 1-5 days after earnings release

    LOGIC:
    1. Detect recent earnings releases
    2. Filter for surprise > 5%
    3. Filter for small/mid cap (market cap < $10B)
    4. Trade in direction of initial reaction

    NOTE: Requires earnings data provider for production.
    This implementation uses placeholder logic for Phase 2.
    """

    def __init__(self, data_provider=None, settings=None, earnings_provider=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.EARNINGS_DRIFT
        self.markets = [Market.US_STOCKS]
        self.instruments = []  # Dynamically populated from earnings calendar
        self.earnings_provider = earnings_provider

        # Edge parameters
        self.min_surprise_pct = 5.0      # Minimum earnings surprise to trigger
        self.drift_window_days = 5        # Days to hold for drift
        self.max_market_cap_billions = 10  # Only small/mid cap
        self.min_price = 5.0              # Avoid penny stocks

    def is_active(self, timestamp: datetime) -> bool:
        """
        Always active during market hours - earnings can release any day.
        The real filter is whether there are recent earnings to trade.
        """
        eastern = pytz.timezone('US/Eastern')
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        local = timestamp.astimezone(eastern)

        # Active during US market hours (9:30 AM - 4:00 PM ET)
        if local.weekday() >= 5:  # Weekend
            return False

        hour = local.hour
        minute = local.minute
        market_open = (hour == 9 and minute >= 30) or hour > 9
        market_close = hour < 16

        return market_open and market_close

    async def _get_recent_earnings(self) -> List[Dict[str, Any]]:
        """
        Get stocks with recent earnings releases.

        In production: Use earnings_provider to fetch real data.
        For Phase 2: Return placeholder data for testing.
        """
        if self.earnings_provider is not None:
            return await self.earnings_provider.get_recent_earnings(days=5)

        # Placeholder earnings data for testing
        # In production, this comes from Yahoo Finance, Earnings Whispers, etc.
        return [
            {
                "symbol": "CRWD",
                "report_date": datetime.now(pytz.UTC) - timedelta(days=2),
                "surprise_pct": 8.5,
                "reaction_direction": "up",  # Stock went up after earnings
                "market_cap_billions": 5.2
            },
            {
                "symbol": "ZS",
                "report_date": datetime.now(pytz.UTC) - timedelta(days=1),
                "surprise_pct": -6.2,
                "reaction_direction": "down",
                "market_cap_billions": 4.8
            },
            {
                "symbol": "AAPL",  # Large cap - should be filtered out
                "report_date": datetime.now(pytz.UTC) - timedelta(days=3),
                "surprise_pct": 7.0,
                "reaction_direction": "up",
                "market_cap_billions": 2800  # Too large
            }
        ]

    def _filter_candidates(self, earnings: List[Dict]) -> List[Dict]:
        """
        Filter earnings for tradeable drift candidates.

        Filters:
        - Surprise > min_surprise_pct (5%)
        - Market cap < max (10B) - small/mid only
        - Within drift window (5 days)
        """
        now = datetime.now(pytz.UTC)
        filtered = []

        for e in earnings:
            # Check surprise threshold
            if abs(e.get("surprise_pct", 0)) < self.min_surprise_pct:
                continue

            # Check market cap (large caps are arbitraged - no edge)
            if e.get("market_cap_billions", 0) > self.max_market_cap_billions:
                continue

            # Check within drift window
            report_date = e.get("report_date")
            if report_date:
                days_since = (now - report_date).days
                if days_since > self.drift_window_days:
                    continue

            filtered.append(e)

        return filtered

    async def scan(self) -> List[Opportunity]:
        """
        Scan for post-earnings drift opportunities.
        """
        now = datetime.now(pytz.UTC)

        if not self.is_active(now):
            return []

        opportunities = []

        # Get recent earnings
        earnings = await self._get_recent_earnings()

        # Filter for valid drift candidates
        candidates = self._filter_candidates(earnings)

        for candidate in candidates:
            symbol = candidate["symbol"]

            try:
                # Get current price and bars
                current_price = await self.get_current_price(symbol)

                if current_price < self.min_price:
                    continue

                bars = await self.get_bars(symbol, "1D", 20)
                if bars is None or len(bars) < 10:
                    continue

                atr = self.calculate_atr(bars, 14)

                # Direction based on earnings reaction
                reaction = candidate.get("reaction_direction", "up")
                direction = Direction.LONG if reaction == "up" else Direction.SHORT

                # Calculate entry, stop, target
                entry, stop, target = self.calculate_entry_stop_target(
                    current_price=current_price,
                    direction=direction,
                    atr=atr,
                    stop_multiplier=2.0,    # Wider stop for multi-day hold
                    target_multiplier=3.0   # Larger target for drift
                )

                # Calculate days remaining in drift window
                report_date = candidate.get("report_date")
                days_since = (now - report_date).days if report_date else 0
                days_remaining = max(0, self.drift_window_days - days_since)

                opp = self.create_opportunity(
                    symbol=symbol,
                    market=Market.US_STOCKS,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    edge_data={
                        "earnings_surprise_pct": candidate.get("surprise_pct"),
                        "report_date": str(report_date.date()) if report_date else None,
                        "days_since_earnings": days_since,
                        "drift_days_remaining": days_remaining,
                        "reaction_direction": reaction,
                        "market_cap_billions": candidate.get("market_cap_billions"),
                        "atr": atr
                    },
                    valid_until=now + timedelta(days=days_remaining)
                )
                opportunities.append(opp)

            except Exception as e:
                continue

        return opportunities

    def get_earnings_drift_status(self) -> Dict[str, Any]:
        """Get current status of earnings drift scanner."""
        now = datetime.now(pytz.UTC)

        return {
            "is_active": self.is_active(now),
            "min_surprise_threshold": self.min_surprise_pct,
            "drift_window_days": self.drift_window_days,
            "max_market_cap_billions": self.max_market_cap_billions,
            "note": "Requires earnings_provider for live data"
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        scanner = EarningsDriftScanner()
        print("=" * 50)
        print("EARNINGS DRIFT SCANNER")
        print("=" * 50)
        print(f"Status: {scanner.get_earnings_drift_status()}")

        opps = await scanner.scan()
        print(f"\nOpportunities found: {len(opps)}")
        for opp in opps:
            print(f"  {opp.symbol}: {opp.direction.value.upper()}")
            print(f"    Entry={opp.entry_price:.2f}, Stop={opp.stop_loss:.2f}, Target={opp.take_profit:.2f}")
            print(f"    Edge data: {opp.edge_data}")

    asyncio.run(test())
