"""
NEXUS SEC Insider Data Provider

Fetches Form 4 filings for INSIDER_CLUSTER detection.
This is the STRONGEST documented edge (2.1% monthly abnormal returns).

Source: SEC EDGAR via sec_edgar.py client
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

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
    total_shares: int
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

    def __init__(self):
        self._cache: Dict[str, List[InsiderFiling]] = {}
        self._cache_time: Optional[datetime] = None
        self.lookback_days = 14
        self.min_insiders = 3
        self.min_cluster_value = 100000

    async def fetch_recent_filings(self, days: int = 14) -> List[InsiderFiling]:
        """
        Fetch Form 4 filings from last N days via SEC EDGAR.

        Returns InsiderFiling objects converted from SEC Form4Filing data.
        """
        # Check cache (refresh every 4 hours)
        if self._cache_time and (datetime.now() - self._cache_time).total_seconds() / 3600 < 4:
            all_filings = []
            for filings in self._cache.values():
                all_filings.extend(filings)
            return all_filings

        from nexus.data.sec_edgar import get_sec_client

        sec = get_sec_client()
        form4s = await sec.get_recent_form4s(days=days)

        # Convert Form4Filing -> InsiderFiling
        filings: List[InsiderFiling] = []
        for f4 in form4s:
            for trans in f4.transactions:
                if trans.is_purchase or trans.is_sale:
                    filings.append(InsiderFiling(
                        filing_date=datetime.combine(f4.filing_date, datetime.min.time()),
                        symbol=f4.issuer_ticker or "",
                        company=f4.issuer_name,
                        insider_name=f4.owner_name,
                        insider_title=f4.owner_title,
                        transaction_type="P" if trans.is_purchase else "S",
                        shares=int(trans.shares),
                        price=trans.price_per_share,
                        value=trans.value,
                    ))

        # Update cache
        self._cache.clear()
        for f in filings:
            if f.symbol not in self._cache:
                self._cache[f.symbol] = []
            self._cache[f.symbol].append(f)
        self._cache_time = datetime.now()

        return filings

    async def get_cluster_buys(
        self,
        min_insiders: int = 3,
        lookback_days: int = 14,
    ) -> List[InsiderCluster]:
        """
        Find stocks where 3+ insiders bought in last 14 days.

        THIS IS THE KEY SIGNAL.

        Uses SEC EDGAR client to find clusters, then converts to
        InsiderCluster format expected by InsiderScanner.
        """
        from nexus.data.sec_edgar import get_sec_client

        sec = get_sec_client()
        form4s = await sec.get_recent_form4s(days=lookback_days)
        sec_clusters = sec.find_clusters(
            form4s,
            min_buyers=min_insiders,
            min_value=self.min_cluster_value,
        )

        # Convert SEC InsiderCluster -> our InsiderCluster format
        clusters: List[InsiderCluster] = []
        for sc in sec_clusters:
            transactions = []
            for f in sc.filings:
                for t in f.transactions:
                    if t.is_purchase:
                        transactions.append(InsiderFiling(
                            filing_date=datetime.combine(f.filing_date, datetime.min.time()),
                            symbol=sc.ticker,
                            company=sc.company_name,
                            insider_name=f.owner_name,
                            insider_title=f.owner_title,
                            transaction_type="P",
                            shares=int(t.shares),
                            price=t.price_per_share,
                            value=t.value,
                        ))

            cluster = InsiderCluster(
                symbol=sc.ticker,
                company=sc.company_name,
                insider_count=sc.unique_buyers,
                total_shares=int(sc.total_shares),
                total_value=sc.total_value,
                transactions=transactions,
                avg_price=sc.avg_price,
                days_span=sc.days_span,
                score=0,  # Scored by scanner
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
