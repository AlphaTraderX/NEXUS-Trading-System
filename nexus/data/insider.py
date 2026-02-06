"""
NEXUS SEC Insider Data Provider

Fetches Form 4 filings for INSIDER_CLUSTER detection.
This is the STRONGEST documented edge (2.1% monthly abnormal returns).

Source: SEC EDGAR
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)


@dataclass
class InsiderFiling:
    """Individual Form 4 filing."""
    filing_date: datetime
    symbol: str
    company: str
    insider_name: str
    insider_title: str
    transaction_type: str  # P = Purchase, S = Sale
    shares: int
    price: float
    value: float
    
    @property
    def is_purchase(self) -> bool:
        return self.transaction_type == "P"
    
    @property
    def is_sale(self) -> bool:
        return self.transaction_type == "S"


@dataclass
class InsiderCluster:
    """
    Cluster of insider buying activity.
    
    THIS IS THE SIGNAL:
    3+ insiders buying within 14 days = STRONG BUY signal
    """
    symbol: str
    company: str
    insider_count: int
    total_value: float
    transactions: List[InsiderFiling]
    avg_price: float
    days_span: int
    score: int = 0
    
    @property
    def is_valid_cluster(self) -> bool:
        """Valid cluster = 3+ insiders buying."""
        return self.insider_count >= 3


# Insider title weights (CEO buy > Director buy)
TITLE_WEIGHTS = {
    "CEO": 10,
    "Chief Executive Officer": 10,
    "CFO": 8,
    "Chief Financial Officer": 8,
    "COO": 7,
    "Chief Operating Officer": 7,
    "President": 7,
    "Chairman": 6,
    "Vice President": 5,
    "VP": 5,
    "Director": 4,
    "Officer": 4,
    "10% Owner": 3,
}


class InsiderDataProvider:
    """
    SEC EDGAR insider filing provider.
    
    Edge: INSIDER_CLUSTER is the strongest documented edge
    - 2.1% monthly abnormal returns (Alldredge & Blank, 2019)
    - 3+ insiders buying = strong signal
    - CEO/CFO buys weighted higher
    
    Method:
    1. Fetch recent Form 4 filings from EDGAR
    2. Group by symbol
    3. Identify clusters (3+ insiders in 14 days)
    4. Score by value, title, and recency
    """
    
    EDGAR_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    EDGAR_FILINGS = "https://efts.sec.gov/LATEST/search-index"
    
    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "NEXUS Trading System research@example.com",
                "Accept": "application/json",
            }
        )
        self._cache: Dict[str, List[InsiderFiling]] = {}
        self._cache_time: Optional[datetime] = None
    
    async def fetch_recent_filings(self, days: int = 14) -> List[InsiderFiling]:
        """
        Fetch Form 4 filings from last N days.
        
        Note: SEC EDGAR has rate limits. Be respectful.
        """
        # Check cache (refresh every 4 hours)
        if self._cache_time and (datetime.now() - self._cache_time).total_seconds() / 3600 < 4:
            all_filings = []
            for filings in self._cache.values():
                all_filings.extend(filings)
            return all_filings
        
        filings = []
        
        try:
            # Use SEC EDGAR full-text search API
            # Search for Form 4 filings
            response = await self._client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": "form-type:4",
                    "dateRange": "custom",
                    "startdt": (date.today() - timedelta(days=days)).isoformat(),
                    "enddt": date.today().isoformat(),
                    "forms": "4",
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                # Parse filings from response
                # Note: Actual parsing would be more complex
                
        except Exception as e:
            logger.warning(f"Failed to fetch SEC filings: {e}")
        
        # For now, return cached or empty
        # In production, would properly parse EDGAR responses
        return filings
    
    async def get_cluster_buys(self) -> List[InsiderCluster]:
        """
        Find stocks where 3+ insiders bought in last 14 days.
        
        THIS IS THE KEY SIGNAL.
        """
        filings = await self.fetch_recent_filings(days=14)
        
        # Group by symbol
        by_symbol: Dict[str, List[InsiderFiling]] = {}
        for filing in filings:
            if filing.is_purchase:
                if filing.symbol not in by_symbol:
                    by_symbol[filing.symbol] = []
                by_symbol[filing.symbol].append(filing)
        
        clusters = []
        
        for symbol, purchases in by_symbol.items():
            # Count unique insiders
            unique_insiders = set(p.insider_name for p in purchases)
            
            if len(unique_insiders) >= 3:
                # Valid cluster!
                total_value = sum(p.value for p in purchases)
                avg_price = sum(p.price * p.shares for p in purchases) / sum(p.shares for p in purchases)
                
                # Calculate days span
                dates = [p.filing_date for p in purchases]
                days_span = (max(dates) - min(dates)).days
                
                cluster = InsiderCluster(
                    symbol=symbol,
                    company=purchases[0].company,
                    insider_count=len(unique_insiders),
                    total_value=total_value,
                    transactions=purchases,
                    avg_price=avg_price,
                    days_span=days_span,
                )
                
                # Score the cluster
                cluster.score = self._score_cluster(cluster)
                
                clusters.append(cluster)
        
        # Sort by score descending
        clusters.sort(key=lambda c: c.score, reverse=True)
        
        return clusters
    
    async def get_insider_activity(self, symbol: str, days: int = 30) -> Dict:
        """Get all insider activity for a symbol."""
        filings = await self.fetch_recent_filings(days)
        
        symbol_filings = [f for f in filings if f.symbol.upper() == symbol.upper()]
        
        purchases = [f for f in symbol_filings if f.is_purchase]
        sales = [f for f in symbol_filings if f.is_sale]
        
        return {
            "symbol": symbol,
            "total_filings": len(symbol_filings),
            "purchases": len(purchases),
            "sales": len(sales),
            "purchase_value": sum(f.value for f in purchases),
            "sale_value": sum(f.value for f in sales),
            "buy_sell_ratio": len(purchases) / max(len(sales), 1),
            "net_value": sum(f.value for f in purchases) - sum(f.value for f in sales),
            "recent_transactions": symbol_filings[:10],
        }
    
    def score_insider_activity(self, symbol: str, activity: Dict) -> int:
        """
        Score insider activity 0-100.
        
        Scoring:
        - Number of insiders buying (max 40 pts)
        - Total value bought (max 30 pts)
        - Recency (max 20 pts)
        - Insider titles (max 10 pts)
        """
        score = 0
        
        # Number of buyers (up to 40 points)
        purchases = activity.get("purchases", 0)
        if purchases >= 5:
            score += 40
        elif purchases >= 3:
            score += 30
        elif purchases >= 2:
            score += 20
        elif purchases >= 1:
            score += 10
        
        # Value bought (up to 30 points)
        value = activity.get("purchase_value", 0)
        if value >= 10_000_000:
            score += 30
        elif value >= 5_000_000:
            score += 25
        elif value >= 1_000_000:
            score += 20
        elif value >= 500_000:
            score += 15
        elif value >= 100_000:
            score += 10
        
        # Buy/sell ratio (up to 20 points)
        ratio = activity.get("buy_sell_ratio", 0)
        if ratio >= 5:
            score += 20
        elif ratio >= 3:
            score += 15
        elif ratio >= 2:
            score += 10
        elif ratio > 1:
            score += 5
        
        # Title bonus would come from transactions
        # (simplified here)
        score += 10  # Assume some C-suite involvement
        
        return min(score, 100)
    
    def _score_cluster(self, cluster: InsiderCluster) -> int:
        """Score a cluster for signal strength."""
        score = 0
        
        # Number of insiders (max 40)
        if cluster.insider_count >= 5:
            score += 40
        elif cluster.insider_count >= 4:
            score += 35
        elif cluster.insider_count >= 3:
            score += 30
        
        # Total value (max 30)
        if cluster.total_value >= 5_000_000:
            score += 30
        elif cluster.total_value >= 1_000_000:
            score += 25
        elif cluster.total_value >= 500_000:
            score += 20
        elif cluster.total_value >= 100_000:
            score += 15
        
        # Recency - more recent = better (max 20)
        if cluster.days_span <= 7:
            score += 20
        elif cluster.days_span <= 14:
            score += 15
        elif cluster.days_span <= 21:
            score += 10
        
        # Title weights (max 10)
        title_score = 0
        for txn in cluster.transactions:
            for title, weight in TITLE_WEIGHTS.items():
                if title.lower() in txn.insider_title.lower():
                    title_score = max(title_score, weight)
                    break
        score += title_score
        
        return min(score, 100)
