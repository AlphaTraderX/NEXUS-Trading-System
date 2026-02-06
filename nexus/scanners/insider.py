"""
Insider cluster scanner (2.1% monthly abnormal returns).
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd

from scanners.base import BaseScanner
from core.enums import EdgeType, Market, Direction
from core.models import Opportunity

logger = logging.getLogger(__name__)


@dataclass
class InsiderTransaction:
    """Single insider transaction from SEC Form 4."""
    filing_date: datetime
    symbol: str
    company: str
    insider_name: str
    insider_title: str
    transaction_type: str  # "P" = Purchase, "S" = Sale
    shares: int
    price: float
    value: float


@dataclass
class InsiderCluster:
    """Cluster of insider buys for a single stock."""
    symbol: str
    company: str
    insider_count: int
    total_shares: int
    total_value: float
    avg_price: float
    transactions: List[InsiderTransaction]
    days_span: int
    score: int


class InsiderScanner(BaseScanner):
    """
    Insider Cluster Buy scanner.

    EDGE: STRONGEST - 2.1% monthly abnormal returns (Alldredge & Blank, 2019)
    SIGNAL: 3+ insiders buying within 14 days = strong buy signal

    Why it works:
    - Insiders have asymmetric information about their company
    - Multiple insiders buying simultaneously = high conviction
    - Academic research shows this predicts future outperformance

    Note: This scanner would normally fetch from SEC EDGAR or OpenInsider.
    For now, it uses mock data structure that can be replaced with real API.
    """

    def __init__(self, data_provider=None, settings=None, insider_data_provider=None):
        super().__init__(data_provider, settings)
        self.edge_type = EdgeType.INSIDER_CLUSTER
        self.markets = [Market.US_STOCKS]
        self.instruments = []
        self.insider_provider = insider_data_provider

        # Cluster parameters
        self.min_insiders = 3  # Minimum 3 insiders buying
        self.lookback_days = 14  # Within 14-day window
        self.min_transaction_value = 10000  # Minimum $10k per transaction
        self.min_cluster_value = 100000  # Minimum $100k total cluster value

    def is_active(self, timestamp: Optional[datetime] = None) -> bool:
        """Insider scanner runs daily (filings come in throughout day)."""
        return True

    async def scan(self) -> List[Opportunity]:
        """
        Scan for insider cluster buy opportunities.

        Flow:
        1. Fetch recent insider transactions (last 14 days)
        2. Group by symbol
        3. Find clusters (3+ insiders buying)
        4. Score and create opportunities
        """
        opportunities = []

        try:
            # Get insider clusters
            clusters = await self._find_insider_clusters()

            for cluster in clusters:
                try:
                    opp = await self._create_opportunity_from_cluster(cluster)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.warning(f"Failed to create opportunity for {cluster.symbol}: {e}")

        except Exception as e:
            logger.error(f"Insider scan failed: {e}")

        logger.info(f"Insider scan complete: {len(opportunities)} opportunities")
        return opportunities

    async def _find_insider_clusters(self) -> List[InsiderCluster]:
        """
        Find stocks with insider cluster buying activity.

        In production, this would fetch from SEC EDGAR or OpenInsider API.
        Returns list of InsiderCluster objects.
        """

        # If we have a real insider data provider, use it
        if self.insider_provider:
            try:
                return await self.insider_provider.get_cluster_buys(
                    min_insiders=self.min_insiders,
                    lookback_days=self.lookback_days
                )
            except Exception as e:
                logger.warning(f"Insider provider failed: {e}")

        # Return empty list if no provider (will be populated in production)
        # Mock data for testing can be injected via insider_provider
        logger.debug("No insider data provider - returning empty clusters")
        return []

    async def _create_opportunity_from_cluster(self, cluster: InsiderCluster) -> Optional[Opportunity]:
        """Create trading opportunity from insider cluster.
        REQUIRED: 3+ different insiders buying within 14 days (cluster signal = strongest edge).
        """
        # Cluster must have at least 3 different insiders (not just 3 transactions)
        unique_insiders = len(set(t.insider_name for t in cluster.transactions))
        if unique_insiders < self.min_insiders:
            logger.debug(
                f"Cluster {cluster.symbol}: only {unique_insiders} unique insiders "
                f"(need {self.min_insiders})"
            )
            return None

        if cluster.insider_count < self.min_insiders:
            return None

        # Cluster must be within 14-day window (validated edge requirement)
        if cluster.days_span > self.lookback_days:
            logger.debug(
                f"Cluster {cluster.symbol}: days_span {cluster.days_span} > {self.lookback_days}"
            )
            return None

        if cluster.total_value < self.min_cluster_value:
            logger.debug(f"Cluster {cluster.symbol} value ${cluster.total_value:,.0f} below minimum")
            return None

        # Get current price data
        bars = await self.get_bars_safe(cluster.symbol, "1D", 30)

        if bars is None or len(bars) < 10:
            return None

        current_price = bars['close'].iloc[-1]

        # Calculate ATR for stops
        atr = self.calculate_atr(bars, 14)
        if atr is None or atr <= 0:
            atr = current_price * 0.02  # 2% fallback

        # Score the cluster
        score = self._score_cluster(cluster, current_price)

        # Insider clusters are ALWAYS long (we're following their buys)
        entry = current_price
        stop = current_price - (atr * 2.0)  # Wider stop for position trades
        target = current_price + (atr * 4.0)  # Larger target (holding for drift)

        # Check if current price is still near insider buy prices
        price_vs_insider = (current_price - cluster.avg_price) / cluster.avg_price * 100

        if price_vs_insider > 15:
            logger.debug(f"Cluster {cluster.symbol}: Price {price_vs_insider:.1f}% above insider avg - may have missed move")
            # Still create opportunity but note the gap

        opp = self.create_opportunity(
            symbol=cluster.symbol,
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            edge_data={
                "insider_count": cluster.insider_count,
                "total_shares": cluster.total_shares,
                "total_value": round(cluster.total_value, 2),
                "avg_insider_price": round(cluster.avg_price, 4),
                "current_price": round(current_price, 4),
                "price_vs_insider_pct": round(price_vs_insider, 2),
                "days_span": cluster.days_span,
                "cluster_score": score,
                "insiders": [t.insider_name for t in cluster.transactions[:5]],  # Top 5 names
                "atr": round(atr, 4),
            }
        )

        logger.info(
            f"Insider Cluster: {cluster.symbol} LONG | "
            f"{cluster.insider_count} insiders | ${cluster.total_value:,.0f} total | "
            f"Score: {score}"
        )

        return opp

    def _score_cluster(self, cluster: InsiderCluster, current_price: float) -> int:
        """
        Score insider cluster quality (0-100).

        Factors:
        - Number of insiders (more = better)
        - Total value (larger = more conviction)
        - Insider titles (CEO/CFO worth more than directors)
        - Recency (more recent = better)
        - Price proximity (closer to insider price = better)
        """
        score = 0

        # Number of insiders (up to 30 points)
        insider_points = min(cluster.insider_count * 8, 30)
        score += insider_points

        # Total value (up to 25 points)
        if cluster.total_value >= 1000000:
            score += 25
        elif cluster.total_value >= 500000:
            score += 20
        elif cluster.total_value >= 250000:
            score += 15
        elif cluster.total_value >= 100000:
            score += 10

        # Insider titles (up to 20 points)
        title_score = 0
        for txn in cluster.transactions:
            title = txn.insider_title.upper()
            if 'CEO' in title or 'CHIEF EXECUTIVE' in title:
                title_score += 10
            elif 'CFO' in title or 'CHIEF FINANCIAL' in title:
                title_score += 8
            elif 'COO' in title or 'PRESIDENT' in title:
                title_score += 6
            elif 'VP' in title or 'VICE PRESIDENT' in title:
                title_score += 4
            elif 'DIRECTOR' in title:
                title_score += 2
        score += min(title_score, 20)

        # Recency (up to 15 points) - newer clusters score higher
        if cluster.days_span <= 7:
            score += 15
        elif cluster.days_span <= 10:
            score += 10
        elif cluster.days_span <= 14:
            score += 5

        # Price proximity (up to 10 points)
        price_diff_pct = abs(current_price - cluster.avg_price) / cluster.avg_price * 100
        if price_diff_pct <= 5:
            score += 10
        elif price_diff_pct <= 10:
            score += 7
        elif price_diff_pct <= 15:
            score += 4

        return min(score, 100)


class MockInsiderProvider:
    """
    Mock insider data provider for testing.
    In production, replace with SEC EDGAR or OpenInsider API.
    """

    def __init__(self, clusters: List[InsiderCluster] = None):
        self.clusters = clusters or []

    async def get_cluster_buys(self, min_insiders: int = 3, lookback_days: int = 14) -> List[InsiderCluster]:
        """Return mock clusters for testing."""
        return [c for c in self.clusters if c.insider_count >= min_insiders]
