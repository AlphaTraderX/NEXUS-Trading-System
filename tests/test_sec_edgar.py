"""Tests for SEC EDGAR Form 4 client."""

import pytest
from datetime import date, timedelta

from nexus.data.sec_edgar import (
    SECEdgarClient,
    Form4Filing,
    Form4Transaction,
    InsiderCluster,
    get_sec_client,
)


# ---------------------------------------------------------------------------
# Form4Transaction
# ---------------------------------------------------------------------------

class TestForm4Transaction:
    """Test Form4Transaction properties."""

    def test_value(self):
        t = Form4Transaction(
            transaction_date=date.today(),
            transaction_code="P",
            shares=100,
            price_per_share=50.0,
            shares_after=100,
            direct_or_indirect="D",
        )
        assert t.value == 5000.0

    def test_is_purchase(self):
        t = Form4Transaction(
            transaction_date=date.today(),
            transaction_code="P",
            shares=100,
            price_per_share=50.0,
            shares_after=100,
            direct_or_indirect="D",
        )
        assert t.is_purchase is True
        assert t.is_sale is False

    def test_is_sale(self):
        t = Form4Transaction(
            transaction_date=date.today(),
            transaction_code="S",
            shares=100,
            price_per_share=50.0,
            shares_after=0,
            direct_or_indirect="D",
        )
        assert t.is_purchase is False
        assert t.is_sale is True


# ---------------------------------------------------------------------------
# Form4Filing
# ---------------------------------------------------------------------------

class TestForm4Filing:
    """Test Form4Filing dataclass."""

    def _make_filing(self, transactions):
        return Form4Filing(
            accession_number="123",
            filing_date=date.today(),
            issuer_cik="0001234",
            issuer_name="Test Corp",
            issuer_ticker="TEST",
            owner_cik="0005678",
            owner_name="John Doe",
            owner_title="CEO",
            is_director=False,
            is_officer=True,
            is_ten_percent_owner=False,
            transactions=transactions,
        )

    def test_total_bought(self):
        """Should sum purchase transactions."""
        filing = self._make_filing([
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="P",
                shares=1000,
                price_per_share=50.0,
                shares_after=1000,
                direct_or_indirect="D",
            ),
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="P",
                shares=500,
                price_per_share=51.0,
                shares_after=1500,
                direct_or_indirect="D",
            ),
        ])

        assert filing.total_bought == 75500.0  # 50000 + 25500
        assert filing.total_sold == 0
        assert filing.net_bought == 75500.0

    def test_net_bought_with_sales(self):
        """Should calculate net correctly with buys and sells."""
        filing = self._make_filing([
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="P",
                shares=1000,
                price_per_share=50.0,
                shares_after=1000,
                direct_or_indirect="D",
            ),
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="S",
                shares=200,
                price_per_share=52.0,
                shares_after=800,
                direct_or_indirect="D",
            ),
        ])

        assert filing.total_bought == 50000.0
        assert filing.total_sold == 10400.0
        assert filing.net_bought == 39600.0

    def test_shares_bought(self):
        """Should sum only purchase shares."""
        filing = self._make_filing([
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="P",
                shares=1000,
                price_per_share=50.0,
                shares_after=1000,
                direct_or_indirect="D",
            ),
            Form4Transaction(
                transaction_date=date.today(),
                transaction_code="S",
                shares=200,
                price_per_share=52.0,
                shares_after=800,
                direct_or_indirect="D",
            ),
        ])

        assert filing.shares_bought == 1000

    def test_empty_transactions(self):
        filing = self._make_filing([])
        assert filing.total_bought == 0
        assert filing.total_sold == 0
        assert filing.net_bought == 0
        assert filing.shares_bought == 0


# ---------------------------------------------------------------------------
# InsiderCluster
# ---------------------------------------------------------------------------

class TestInsiderCluster:
    """Test InsiderCluster."""

    def _make_filing(self, ticker, owner_cik, bought, filing_date=None):
        """Helper to create a test filing with a purchase."""
        fd = filing_date or date.today()
        transactions = []
        if bought > 0:
            transactions.append(Form4Transaction(
                transaction_date=fd,
                transaction_code="P",
                shares=bought / 50,
                price_per_share=50.0,
                shares_after=bought / 50,
                direct_or_indirect="D",
            ))
        return Form4Filing(
            accession_number="123",
            filing_date=fd,
            issuer_cik="0001234",
            issuer_name=f"{ticker} Corp",
            issuer_ticker=ticker,
            owner_cik=owner_cik,
            owner_name=f"Insider {owner_cik}",
            owner_title="Director",
            is_director=True,
            is_officer=False,
            is_ten_percent_owner=False,
            transactions=transactions,
        )

    def test_unique_buyers(self):
        """Should count unique insiders."""
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),
                self._make_filing("TEST", "002", 30000),
                self._make_filing("TEST", "001", 20000),  # Same owner
                self._make_filing("TEST", "003", 40000),
            ],
        )
        assert cluster.unique_buyers == 3

    def test_total_value(self):
        """Should sum all purchases."""
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),
                self._make_filing("TEST", "002", 30000),
                self._make_filing("TEST", "003", 40000),
            ],
        )
        assert cluster.total_value == 120000.0

    def test_total_shares(self):
        """Should sum all purchased shares."""
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),  # 1000 shares
                self._make_filing("TEST", "002", 25000),  # 500 shares
            ],
        )
        assert cluster.total_shares == 1500.0

    def test_avg_price(self):
        """Should compute weighted average price."""
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),
            ],
        )
        assert cluster.avg_price == 50.0

    def test_avg_price_empty(self):
        """Should return 0 for empty cluster."""
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[],
        )
        assert cluster.avg_price == 0

    def test_days_span(self):
        """Should compute filing date range."""
        today = date.today()
        cluster = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000, today - timedelta(days=10)),
                self._make_filing("TEST", "002", 30000, today - timedelta(days=3)),
                self._make_filing("TEST", "003", 40000, today),
            ],
        )
        assert cluster.days_span == 10

    def test_days_span_empty(self):
        cluster = InsiderCluster(ticker="TEST", company_name="Test Corp", filings=[])
        assert cluster.days_span == 0

    def test_is_valid_cluster(self):
        """Should validate cluster criteria."""
        valid = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),
                self._make_filing("TEST", "002", 30000),
                self._make_filing("TEST", "003", 40000),
            ],
        )
        assert valid.is_valid_cluster(min_buyers=3, min_value=100000) is True

    def test_invalid_cluster_too_few_buyers(self):
        invalid = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 50000),
                self._make_filing("TEST", "002", 70000),
            ],
        )
        assert invalid.is_valid_cluster(min_buyers=3, min_value=100000) is False

    def test_invalid_cluster_value_too_low(self):
        invalid = InsiderCluster(
            ticker="TEST",
            company_name="Test Corp",
            filings=[
                self._make_filing("TEST", "001", 20000),
                self._make_filing("TEST", "002", 20000),
                self._make_filing("TEST", "003", 20000),
            ],
        )
        assert invalid.is_valid_cluster(min_buyers=3, min_value=100000) is False


# ---------------------------------------------------------------------------
# SECEdgarClient.find_clusters
# ---------------------------------------------------------------------------

class TestSECEdgarClientFindClusters:
    """Test cluster detection logic (no network)."""

    def test_find_clusters(self):
        """Should find valid clusters from filings."""
        client = SECEdgarClient()

        filings = [
            Form4Filing(
                accession_number="1",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Alpha Corp",
                issuer_ticker="ALPH",
                owner_cik="100",
                owner_name="Alice",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=1000,
                    price_per_share=50.0,
                    shares_after=1000,
                    direct_or_indirect="D",
                )],
            ),
            Form4Filing(
                accession_number="2",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Alpha Corp",
                issuer_ticker="ALPH",
                owner_cik="101",
                owner_name="Bob",
                owner_title="CFO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=600,
                    price_per_share=51.0,
                    shares_after=600,
                    direct_or_indirect="D",
                )],
            ),
            Form4Filing(
                accession_number="3",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Alpha Corp",
                issuer_ticker="ALPH",
                owner_cik="102",
                owner_name="Carol",
                owner_title="Director",
                is_director=True,
                is_officer=False,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=500,
                    price_per_share=49.0,
                    shares_after=500,
                    direct_or_indirect="D",
                )],
            ),
            # Different company — only 1 buyer, shouldn't cluster
            Form4Filing(
                accession_number="4",
                filing_date=date.today(),
                issuer_cik="002",
                issuer_name="Beta Inc",
                issuer_ticker="BETA",
                owner_cik="200",
                owner_name="Dave",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=2000,
                    price_per_share=100.0,
                    shares_after=2000,
                    direct_or_indirect="D",
                )],
            ),
        ]

        clusters = client.find_clusters(filings, min_buyers=3, min_value=50000)

        assert len(clusters) == 1
        assert clusters[0].ticker == "ALPH"
        assert clusters[0].unique_buyers == 3
        assert clusters[0].total_value == 105100.0  # 50000 + 30600 + 24500

    def test_find_clusters_no_ticker(self):
        """Should skip filings without ticker."""
        client = SECEdgarClient()

        filings = [
            Form4Filing(
                accession_number="1",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Unknown Corp",
                issuer_ticker=None,
                owner_cik="100",
                owner_name="Alice",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=1000,
                    price_per_share=50.0,
                    shares_after=1000,
                    direct_or_indirect="D",
                )],
            ),
        ]

        clusters = client.find_clusters(filings, min_buyers=1, min_value=0)
        assert len(clusters) == 0

    def test_find_clusters_case_insensitive_ticker(self):
        """Should group by uppercase ticker."""
        client = SECEdgarClient()

        filings = [
            Form4Filing(
                accession_number="1",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Alpha Corp",
                issuer_ticker="alph",
                owner_cik="100",
                owner_name="Alice",
                owner_title="CEO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=1000,
                    price_per_share=50.0,
                    shares_after=1000,
                    direct_or_indirect="D",
                )],
            ),
            Form4Filing(
                accession_number="2",
                filing_date=date.today(),
                issuer_cik="001",
                issuer_name="Alpha Corp",
                issuer_ticker="ALPH",
                owner_cik="101",
                owner_name="Bob",
                owner_title="CFO",
                is_director=False,
                is_officer=True,
                is_ten_percent_owner=False,
                transactions=[Form4Transaction(
                    transaction_date=date.today(),
                    transaction_code="P",
                    shares=600,
                    price_per_share=50.0,
                    shares_after=600,
                    direct_or_indirect="D",
                )],
            ),
        ]

        clusters = client.find_clusters(filings, min_buyers=2, min_value=0)
        assert len(clusters) == 1
        assert clusters[0].ticker == "ALPH"

    def test_find_clusters_empty(self):
        """Should return empty list for no filings."""
        client = SECEdgarClient()
        clusters = client.find_clusters([], min_buyers=3, min_value=100000)
        assert clusters == []


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------

SAMPLE_FORM4_XML = """\
<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0306</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2025-01-15</periodOfReport>
  <issuer>
    <issuerCik>0001234567</issuerCik>
    <issuerName>Acme Corp</issuerName>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0009876543</rptOwnerCik>
      <rptOwnerName>Jane Smith</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2025-01-15</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>25.50</value></transactionPricePerShare>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>15000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2025-01-14</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>26.00</value></transactionPricePerShare>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>10000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class TestParseForm4XML:
    """Test XML parsing of Form 4 filings."""

    def test_parse_valid_xml(self):
        """Should correctly parse a complete Form 4 XML."""
        client = SECEdgarClient()
        filing = client._parse_form4_xml(SAMPLE_FORM4_XML)

        assert filing is not None
        assert filing.issuer_cik == "0001234567"
        assert filing.issuer_name == "Acme Corp"
        assert filing.issuer_ticker == "ACME"
        assert filing.owner_cik == "0009876543"
        assert filing.owner_name == "Jane Smith"
        assert filing.owner_title == "Chief Executive Officer"
        assert filing.is_director is False
        assert filing.is_officer is True
        assert filing.is_ten_percent_owner is False
        assert filing.filing_date == date(2025, 1, 15)

    def test_parse_transactions(self):
        """Should parse both purchase and sale transactions."""
        client = SECEdgarClient()
        filing = client._parse_form4_xml(SAMPLE_FORM4_XML)

        assert filing is not None
        assert len(filing.transactions) == 2

        purchase = filing.transactions[0]
        assert purchase.transaction_code == "P"
        assert purchase.shares == 5000
        assert purchase.price_per_share == 25.50
        assert purchase.shares_after == 15000
        assert purchase.direct_or_indirect == "D"
        assert purchase.is_purchase is True
        assert purchase.value == 127500.0

        sale = filing.transactions[1]
        assert sale.transaction_code == "S"
        assert sale.shares == 1000
        assert sale.price_per_share == 26.00
        assert sale.is_sale is True
        assert sale.value == 26000.0

    def test_parse_totals(self):
        """Should compute correct totals from parsed XML."""
        client = SECEdgarClient()
        filing = client._parse_form4_xml(SAMPLE_FORM4_XML)

        assert filing is not None
        assert filing.total_bought == 127500.0
        assert filing.total_sold == 26000.0
        assert filing.net_bought == 101500.0
        assert filing.shares_bought == 5000.0

    def test_parse_invalid_xml(self):
        """Should return None for invalid XML."""
        client = SECEdgarClient()
        assert client._parse_form4_xml("<bad>xml</bad>") is None
        assert client._parse_form4_xml("not xml at all") is None

    def test_parse_missing_issuer(self):
        """Should return None when issuer element is missing."""
        client = SECEdgarClient()
        xml = """\
<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>001</rptOwnerCik>
      <rptOwnerName>Test</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
</ownershipDocument>"""
        assert client._parse_form4_xml(xml) is None

    def test_parse_missing_owner(self):
        """Should return None when reportingOwner is missing."""
        client = SECEdgarClient()
        xml = """\
<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>001</issuerCik>
    <issuerName>Test</issuerName>
    <issuerTradingSymbol>TST</issuerTradingSymbol>
  </issuer>
</ownershipDocument>"""
        assert client._parse_form4_xml(xml) is None

    def test_parse_no_transactions(self):
        """Should parse filing with zero transactions."""
        client = SECEdgarClient()
        xml = """\
<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2025-01-15</periodOfReport>
  <issuer>
    <issuerCik>001</issuerCik>
    <issuerName>Test Corp</issuerName>
    <issuerTradingSymbol>TST</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>100</rptOwnerCik>
      <rptOwnerName>John</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
</ownershipDocument>"""
        filing = client._parse_form4_xml(xml)
        assert filing is not None
        assert len(filing.transactions) == 0
        assert filing.issuer_ticker == "TST"
        assert filing.is_director is True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSECClientSingleton:
    """Test singleton pattern."""

    def test_singleton(self):
        """Should return same instance."""
        # Reset global
        import nexus.data.sec_edgar as mod
        mod._sec_client = None

        client1 = get_sec_client()
        client2 = get_sec_client()
        assert client1 is client2

        # Cleanup
        mod._sec_client = None
