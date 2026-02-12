"""
SEC EDGAR Form 4 Client

Fetches insider trading filings (Form 4) from SEC EDGAR.
This is the data source for INSIDER_CLUSTER edge detection.

Rate Limits:
- SEC requires max 10 requests/second
- Use User-Agent header with contact info
- Cache results to minimize requests

API Docs: https://www.sec.gov/developer
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Form4Transaction:
    """Individual transaction within Form 4."""
    transaction_date: date
    transaction_code: str  # P=Purchase, S=Sale, A=Award, etc.
    shares: float
    price_per_share: float
    shares_after: float  # Holdings after transaction
    direct_or_indirect: str  # D=Direct, I=Indirect

    @property
    def value(self) -> float:
        return abs(self.shares * self.price_per_share)

    @property
    def is_purchase(self) -> bool:
        return self.transaction_code == "P"

    @property
    def is_sale(self) -> bool:
        return self.transaction_code == "S"


@dataclass
class Form4Filing:
    """Parsed Form 4 insider filing."""
    accession_number: str
    filing_date: date
    issuer_cik: str
    issuer_name: str
    issuer_ticker: Optional[str]
    owner_cik: str
    owner_name: str
    owner_title: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    transactions: List[Form4Transaction] = field(default_factory=list)

    @property
    def total_bought(self) -> float:
        """Total value of purchases."""
        return sum(t.value for t in self.transactions if t.is_purchase)

    @property
    def total_sold(self) -> float:
        """Total value of sales."""
        return sum(t.value for t in self.transactions if t.is_sale)

    @property
    def net_bought(self) -> float:
        """Net purchase value (positive = buying)."""
        return self.total_bought - self.total_sold

    @property
    def shares_bought(self) -> float:
        """Total shares purchased."""
        return sum(t.shares for t in self.transactions if t.is_purchase)


@dataclass
class InsiderCluster:
    """Cluster of insider buying activity."""
    ticker: str
    company_name: str
    filings: List[Form4Filing]

    @property
    def unique_buyers(self) -> int:
        """Count of unique insiders who bought."""
        buyers = set()
        for f in self.filings:
            if f.total_bought > 0:
                buyers.add(f.owner_cik)
        return len(buyers)

    @property
    def total_value(self) -> float:
        """Total purchase value."""
        return sum(f.total_bought for f in self.filings)

    @property
    def total_shares(self) -> float:
        """Total shares purchased."""
        return sum(f.shares_bought for f in self.filings)

    @property
    def avg_price(self) -> float:
        """Average purchase price."""
        total_shares = 0.0
        total_value = 0.0
        for f in self.filings:
            for t in f.transactions:
                if t.is_purchase:
                    total_shares += t.shares
                    total_value += t.value
        return total_value / total_shares if total_shares > 0 else 0

    @property
    def days_span(self) -> int:
        """Days between first and last filing."""
        if not self.filings:
            return 0
        dates = [f.filing_date for f in self.filings]
        return (max(dates) - min(dates)).days

    def is_valid_cluster(self, min_buyers: int = 3, min_value: float = 100000) -> bool:
        """Check if this is a valid insider cluster signal."""
        return self.unique_buyers >= min_buyers and self.total_value >= min_value


class SECEdgarClient:
    """
    Client for SEC EDGAR Form 4 filings.

    Usage:
        client = SECEdgarClient()
        filings = await client.get_recent_form4s(days=14)
        clusters = client.find_clusters(filings)
    """

    BASE_URL = "https://www.sec.gov"
    EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

    # SEC requires identifying User-Agent
    USER_AGENT = "NEXUS Trading System (contact@example.com)"

    # Rate limiting: max 10 requests/second
    REQUEST_DELAY = 0.15  # 150ms between requests

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "application/json, application/xml, text/html",
            },
            follow_redirects=True,
        )
        self._last_request_time = 0.0
        self._cache: Dict[str, Tuple[datetime, List[Form4Filing]]] = {}
        self._cache_ttl = timedelta(hours=1)

    async def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    async def get_recent_form4s(
        self,
        days: int = 14,
        tickers: Optional[List[str]] = None,
    ) -> List[Form4Filing]:
        """
        Fetch recent Form 4 filings.

        Args:
            days: Look back this many days
            tickers: Optional list of tickers to filter (None = all)

        Returns:
            List of parsed Form4Filing objects
        """
        cache_key = f"form4s_{days}_{hash(tuple(tickers) if tickers else ())}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        filings: List[Form4Filing] = []

        try:
            # Use SEC full-text search API
            await self._rate_limit()

            start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date = date.today().strftime("%Y-%m-%d")

            params = {
                "q": "*",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "forms": "4",
                "from": 0,
                "size": 200,
            }

            response = await self._client.get(self.EFTS_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                hits = data.get("hits", {}).get("hits", [])

                for hit in hits:
                    try:
                        filing = await self._parse_filing_from_search(hit)
                        if filing:
                            if tickers is None or (
                                filing.issuer_ticker
                                and filing.issuer_ticker.upper()
                                in [t.upper() for t in tickers]
                            ):
                                filings.append(filing)
                    except Exception as e:
                        logger.debug(f"Failed to parse filing: {e}")
                        continue
            else:
                logger.warning(f"SEC EDGAR search returned {response.status_code}")
                filings = await self._get_form4s_from_rss(days)

        except Exception as e:
            logger.error(f"Failed to fetch Form 4 filings: {e}")
            filings = await self._get_form4s_from_rss(days)

        # Cache results
        self._cache[cache_key] = (datetime.now(), filings)

        logger.info(f"Fetched {len(filings)} Form 4 filings from last {days} days")
        return filings

    async def _get_form4s_from_rss(self, days: int) -> List[Form4Filing]:
        """Fallback: Get Form 4s from SEC RSS feed."""
        filings: List[Form4Filing] = []

        try:
            await self._rate_limit()

            rss_url = (
                "https://www.sec.gov/cgi-bin/browse-edgar"
                "?action=getcurrent&type=4&company=&dateb=&owner=only"
                "&count=100&output=atom"
            )

            response = await self._client.get(rss_url)

            if response.status_code == 200:
                root = ElementTree.fromstring(response.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                cutoff_date = date.today() - timedelta(days=days)

                for entry in root.findall("atom:entry", ns):
                    try:
                        link_elem = entry.find("atom:link", ns)
                        if link_elem is None:
                            continue

                        filing_url = link_elem.get("href", "")

                        updated = entry.find("atom:updated", ns)
                        if updated is not None and updated.text:
                            filing_date = datetime.fromisoformat(
                                updated.text.replace("Z", "+00:00")
                            ).date()
                            if filing_date < cutoff_date:
                                continue

                        filing = await self._fetch_and_parse_form4(filing_url)
                        if filing:
                            filings.append(filing)

                    except Exception as e:
                        logger.debug(f"Failed to parse RSS entry: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to fetch Form 4 RSS: {e}")

        return filings

    async def _parse_filing_from_search(self, hit: dict) -> Optional[Form4Filing]:
        """Parse a filing from search results."""
        try:
            source = hit.get("_source", {})

            accession = source.get("adsh", "").replace("-", "")
            cik = source.get("ciks", [""])[0] if source.get("ciks") else ""

            if not accession or not cik:
                return None

            filing_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession}"
            return await self._fetch_and_parse_form4(filing_url)

        except Exception as e:
            logger.debug(f"Error parsing search hit: {e}")
            return None

    async def _fetch_and_parse_form4(self, filing_url: str) -> Optional[Form4Filing]:
        """Fetch and parse a Form 4 XML filing."""
        try:
            await self._rate_limit()

            if not filing_url.endswith(".xml"):
                index_url = filing_url if filing_url.endswith("/") else filing_url + "/"
                response = await self._client.get(index_url)

                if response.status_code != 200:
                    return None

                xml_match = re.search(
                    r'href="([^"]*(?:form4|primary_doc)[^"]*\.xml)"',
                    response.text,
                    re.I,
                )
                if xml_match:
                    xml_file = xml_match.group(1)
                    filing_url = index_url + xml_file
                else:
                    return None

            await self._rate_limit()
            response = await self._client.get(filing_url)

            if response.status_code != 200:
                return None

            return self._parse_form4_xml(response.text)

        except Exception as e:
            logger.debug(f"Error fetching Form 4: {e}")
            return None

    def _parse_form4_xml(self, xml_content: str) -> Optional[Form4Filing]:
        """Parse Form 4 XML content into a Form4Filing."""
        try:
            root = ElementTree.fromstring(xml_content)

            # Issuer info
            issuer = root.find(".//issuer")
            if issuer is None:
                issuer = root.find(".//issuerInfo")
            if issuer is None:
                return None

            issuer_cik = self._get_text(issuer, "issuerCik")
            issuer_name = self._get_text(issuer, "issuerName")
            issuer_ticker = self._get_text(issuer, "issuerTradingSymbol")

            # Owner info
            owner = root.find(".//reportingOwner")
            if owner is None:
                return None

            owner_id = owner.find(".//reportingOwnerId")
            owner_rel = owner.find(".//reportingOwnerRelationship")

            owner_cik = self._get_text(owner_id, "rptOwnerCik") if owner_id else ""
            owner_name = self._get_text(owner_id, "rptOwnerName") if owner_id else ""

            is_director = (
                self._get_text(owner_rel, "isDirector") == "1" if owner_rel else False
            )
            is_officer = (
                self._get_text(owner_rel, "isOfficer") == "1" if owner_rel else False
            )
            is_ten_pct = (
                self._get_text(owner_rel, "isTenPercentOwner") == "1"
                if owner_rel
                else False
            )
            owner_title = self._get_text(owner_rel, "officerTitle") if owner_rel else ""

            # Filing date
            period = root.find(".//periodOfReport")
            if period is not None and period.text:
                filing_date = datetime.strptime(period.text, "%Y-%m-%d").date()
            else:
                filing_date = date.today()

            # Accession number
            accession = ""
            acc_elem = root.find(".//accessionNumber")
            if acc_elem is not None and acc_elem.text:
                accession = acc_elem.text.strip()

            # Non-derivative transactions
            transactions: List[Form4Transaction] = []

            for trans in root.findall(".//nonDerivativeTransaction"):
                try:
                    t = self._parse_transaction(trans)
                    if t:
                        transactions.append(t)
                except Exception:
                    continue

            # Derivative transactions
            for trans in root.findall(".//derivativeTransaction"):
                try:
                    t = self._parse_transaction(trans)
                    if t:
                        transactions.append(t)
                except Exception:
                    continue

            return Form4Filing(
                accession_number=accession,
                filing_date=filing_date,
                issuer_cik=issuer_cik,
                issuer_name=issuer_name,
                issuer_ticker=issuer_ticker,
                owner_cik=owner_cik,
                owner_name=owner_name,
                owner_title=owner_title,
                is_director=is_director,
                is_officer=is_officer,
                is_ten_percent_owner=is_ten_pct,
                transactions=transactions,
            )

        except Exception as e:
            logger.debug(f"Error parsing Form 4 XML: {e}")
            return None

    def _parse_transaction(self, trans_elem) -> Optional[Form4Transaction]:
        """Parse a single transaction element from Form 4 XML."""
        try:
            date_elem = trans_elem.find(".//transactionDate/value")
            if date_elem is None or not date_elem.text:
                return None
            trans_date = datetime.strptime(date_elem.text, "%Y-%m-%d").date()

            code_elem = trans_elem.find(".//transactionCoding/transactionCode")
            trans_code = code_elem.text if code_elem is not None else ""

            shares_elem = trans_elem.find(
                ".//transactionAmounts/transactionShares/value"
            )
            shares = (
                float(shares_elem.text)
                if shares_elem is not None and shares_elem.text
                else 0
            )

            price_elem = trans_elem.find(
                ".//transactionAmounts/transactionPricePerShare/value"
            )
            price = (
                float(price_elem.text)
                if price_elem is not None and price_elem.text
                else 0
            )

            after_elem = trans_elem.find(
                ".//postTransactionAmounts/sharesOwnedFollowingTransaction/value"
            )
            shares_after = (
                float(after_elem.text)
                if after_elem is not None and after_elem.text
                else 0
            )

            ownership_elem = trans_elem.find(
                ".//ownershipNature/directOrIndirectOwnership/value"
            )
            direct_indirect = (
                ownership_elem.text if ownership_elem is not None else "D"
            )

            return Form4Transaction(
                transaction_date=trans_date,
                transaction_code=trans_code,
                shares=shares,
                price_per_share=price,
                shares_after=shares_after,
                direct_or_indirect=direct_indirect,
            )

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")
            return None

    def _get_text(self, elem, tag: str) -> str:
        """Safely get text from child element."""
        if elem is None:
            return ""
        child = elem.find(f".//{tag}")
        if child is None:
            child = elem.find(tag)
        return child.text.strip() if child is not None and child.text else ""

    def find_clusters(
        self,
        filings: List[Form4Filing],
        min_buyers: int = 3,
        min_value: float = 100000,
    ) -> List[InsiderCluster]:
        """
        Find insider buying clusters from filings.

        Args:
            filings: List of Form 4 filings
            min_buyers: Minimum unique buyers for cluster
            min_value: Minimum total purchase value

        Returns:
            List of valid InsiderCluster objects
        """
        # Group by ticker
        by_ticker: Dict[str, List[Form4Filing]] = {}

        for f in filings:
            if not f.issuer_ticker:
                continue
            ticker = f.issuer_ticker.upper()
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(f)

        clusters = []

        for ticker, ticker_filings in by_ticker.items():
            cluster = InsiderCluster(
                ticker=ticker,
                company_name=ticker_filings[0].issuer_name if ticker_filings else "",
                filings=ticker_filings,
            )

            if cluster.is_valid_cluster(min_buyers, min_value):
                clusters.append(cluster)
                logger.info(
                    f"Found insider cluster: {ticker} - "
                    f"{cluster.unique_buyers} buyers, ${cluster.total_value:,.0f}"
                )

        return clusters

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# Singleton instance
_sec_client: Optional[SECEdgarClient] = None


def get_sec_client() -> SECEdgarClient:
    """Get the global SEC EDGAR client."""
    global _sec_client
    if _sec_client is None:
        _sec_client = SECEdgarClient()
    return _sec_client
