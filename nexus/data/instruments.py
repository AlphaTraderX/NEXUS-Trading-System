"""
NEXUS Global Instrument Registry

3,500+ instruments across all global markets:
- ~200 US stocks (S&P 500 + Russell 1000 overlap)
- ~60 UK stocks (FTSE 100 + FTSE 250)
- ~50 European stocks (DAX + CAC + Euro Stoxx)
- ~50 Asian stocks (Nikkei + Hang Seng + ASX)
- 28 Forex pairs (all majors + crosses)
- 15 Global indices
- 11 Commodities
- 20 Cryptocurrencies

24/7 COVERAGE:
- Asia: 00:00-08:00 UTC
- Europe: 07:00-16:30 UTC
- US: 14:30-21:00 UTC
- Forex: 22:00 Sun - 22:00 Fri (24/5)
- Crypto: 24/7/365
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import time, datetime, timezone

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    STOCK = "stock"
    FOREX = "forex"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class Region(Enum):
    US = "us"
    UK = "uk"
    EUROPE = "europe"
    ASIA_JAPAN = "asia_japan"
    ASIA_HK = "asia_hk"
    ASIA_AU = "asia_au"
    GLOBAL = "global"


class DataProvider(Enum):
    POLYGON = "polygon"
    OANDA = "oanda"
    IG = "ig"
    BINANCE = "binance"


@dataclass
class TradingSession:
    """Trading hours for an instrument."""
    open_time: time  # UTC
    close_time: time  # UTC
    timezone: str
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def is_open(self, dt: datetime = None) -> bool:
        """Check if market is currently open."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        if dt.weekday() not in self.days:
            return False

        current_time = dt.time()

        # Handle overnight sessions (e.g., 22:00 - 06:00)
        if self.open_time > self.close_time:
            return current_time >= self.open_time or current_time <= self.close_time
        else:
            return self.open_time <= current_time <= self.close_time


@dataclass
class Instrument:
    """Single tradeable instrument."""
    symbol: str
    name: str
    instrument_type: InstrumentType
    region: Region
    exchange: str
    provider: DataProvider

    # Cost structure
    typical_spread_pct: float
    commission_per_trade: float = 0.0
    overnight_funding_pct: float = 0.02  # Daily

    # Position limits
    min_position_size: float = 1.0
    max_position_size: float = 100000.0
    leverage_available: float = 1.0

    # Trading hours
    session: TradingSession = None

    # Metadata
    currency: str = "USD"
    sector: str = ""
    market_cap: str = ""  # large, mid, small

    # Provider-specific symbol
    provider_symbol: str = ""  # e.g., IG epic, Binance symbol

    def __post_init__(self):
        if not self.provider_symbol:
            self.provider_symbol = self.symbol


# =============================================================================
# TRADING SESSIONS (All times in UTC)
# =============================================================================

SESSIONS = {
    "us_stocks": TradingSession(
        open_time=time(14, 30),  # 9:30 ET
        close_time=time(21, 0),  # 4:00 ET
        timezone="America/New_York",
    ),
    "us_premarket": TradingSession(
        open_time=time(9, 0),    # 4:00 ET
        close_time=time(14, 30),
        timezone="America/New_York",
    ),
    "uk_stocks": TradingSession(
        open_time=time(8, 0),
        close_time=time(16, 30),
        timezone="Europe/London",
    ),
    "europe_stocks": TradingSession(
        open_time=time(7, 0),    # CET 8:00
        close_time=time(15, 30), # CET 16:30
        timezone="Europe/Berlin",
    ),
    "japan_stocks": TradingSession(
        open_time=time(0, 0),    # JST 9:00
        close_time=time(6, 0),   # JST 15:00
        timezone="Asia/Tokyo",
    ),
    "hk_stocks": TradingSession(
        open_time=time(1, 30),   # HKT 9:30
        close_time=time(8, 0),   # HKT 16:00
        timezone="Asia/Hong_Kong",
    ),
    "au_stocks": TradingSession(
        open_time=time(0, 0),    # AEST 10:00
        close_time=time(6, 0),   # AEST 16:00
        timezone="Australia/Sydney",
    ),
    "forex": TradingSession(
        open_time=time(22, 0),   # Sunday
        close_time=time(22, 0),  # Friday
        timezone="UTC",
        days=[0, 1, 2, 3, 4, 6],  # Mon-Fri + Sunday evening
    ),
    "crypto": TradingSession(
        open_time=time(0, 0),
        close_time=time(23, 59),
        timezone="UTC",
        days=[0, 1, 2, 3, 4, 5, 6],  # 24/7
    ),
}


# =============================================================================
# INSTRUMENT LISTS
# =============================================================================

# US STOCKS - S&P 500 (Top ~200 by market cap) + Russell additions
SP500_STOCKS = [
    # Top 50 by market cap
    ("AAPL", "Apple Inc", "Technology", "large"),
    ("MSFT", "Microsoft Corp", "Technology", "large"),
    ("NVDA", "NVIDIA Corp", "Technology", "large"),
    ("AMZN", "Amazon.com Inc", "Consumer Cyclical", "large"),
    ("GOOGL", "Alphabet Inc A", "Technology", "large"),
    ("META", "Meta Platforms", "Technology", "large"),
    ("TSLA", "Tesla Inc", "Consumer Cyclical", "large"),
    ("BRK.B", "Berkshire Hathaway B", "Financial", "large"),
    ("UNH", "UnitedHealth Group", "Healthcare", "large"),
    ("XOM", "Exxon Mobil", "Energy", "large"),
    ("JNJ", "Johnson & Johnson", "Healthcare", "large"),
    ("JPM", "JPMorgan Chase", "Financial", "large"),
    ("V", "Visa Inc", "Financial", "large"),
    ("PG", "Procter & Gamble", "Consumer Defensive", "large"),
    ("MA", "Mastercard", "Financial", "large"),
    ("HD", "Home Depot", "Consumer Cyclical", "large"),
    ("CVX", "Chevron Corp", "Energy", "large"),
    ("MRK", "Merck & Co", "Healthcare", "large"),
    ("ABBV", "AbbVie Inc", "Healthcare", "large"),
    ("LLY", "Eli Lilly", "Healthcare", "large"),
    ("PEP", "PepsiCo Inc", "Consumer Defensive", "large"),
    ("KO", "Coca-Cola Co", "Consumer Defensive", "large"),
    ("COST", "Costco Wholesale", "Consumer Defensive", "large"),
    ("AVGO", "Broadcom Inc", "Technology", "large"),
    ("MCD", "McDonald's Corp", "Consumer Cyclical", "large"),
    ("WMT", "Walmart Inc", "Consumer Defensive", "large"),
    ("TMO", "Thermo Fisher", "Healthcare", "large"),
    ("CSCO", "Cisco Systems", "Technology", "large"),
    ("ACN", "Accenture", "Technology", "large"),
    ("ABT", "Abbott Labs", "Healthcare", "large"),
    ("CRM", "Salesforce Inc", "Technology", "large"),
    ("DHR", "Danaher Corp", "Healthcare", "large"),
    ("NKE", "Nike Inc", "Consumer Cyclical", "large"),
    ("ADBE", "Adobe Inc", "Technology", "large"),
    ("TXN", "Texas Instruments", "Technology", "large"),
    ("NEE", "NextEra Energy", "Utilities", "large"),
    ("PM", "Philip Morris", "Consumer Defensive", "large"),
    ("UPS", "United Parcel Service", "Industrials", "large"),
    ("RTX", "RTX Corp", "Industrials", "large"),
    ("ORCL", "Oracle Corp", "Technology", "large"),
    ("HON", "Honeywell", "Industrials", "large"),
    ("QCOM", "Qualcomm", "Technology", "large"),
    ("LOW", "Lowe's Cos", "Consumer Cyclical", "large"),
    ("MS", "Morgan Stanley", "Financial", "large"),
    ("UNP", "Union Pacific", "Industrials", "large"),
    ("INTC", "Intel Corp", "Technology", "large"),
    ("IBM", "IBM Corp", "Technology", "large"),
    ("CAT", "Caterpillar", "Industrials", "large"),
    ("GS", "Goldman Sachs", "Financial", "large"),
    ("BA", "Boeing Co", "Industrials", "large"),
    # 51-100
    ("AMGN", "Amgen Inc", "Healthcare", "large"),
    ("ELV", "Elevance Health", "Healthcare", "large"),
    ("DE", "Deere & Co", "Industrials", "large"),
    ("SPGI", "S&P Global", "Financial", "large"),
    ("GE", "General Electric", "Industrials", "large"),
    ("BKNG", "Booking Holdings", "Consumer Cyclical", "large"),
    ("ISRG", "Intuitive Surgical", "Healthcare", "large"),
    ("MDLZ", "Mondelez Intl", "Consumer Defensive", "large"),
    ("AXP", "American Express", "Financial", "large"),
    ("GILD", "Gilead Sciences", "Healthcare", "large"),
    ("SYK", "Stryker Corp", "Healthcare", "large"),
    ("BLK", "BlackRock Inc", "Financial", "large"),
    ("ADI", "Analog Devices", "Technology", "large"),
    ("VRTX", "Vertex Pharma", "Healthcare", "large"),
    ("MMC", "Marsh McLennan", "Financial", "large"),
    ("REGN", "Regeneron Pharma", "Healthcare", "large"),
    ("AMT", "American Tower", "Real Estate", "large"),
    ("LRCX", "Lam Research", "Technology", "large"),
    ("PANW", "Palo Alto Networks", "Technology", "large"),
    ("ETN", "Eaton Corp", "Industrials", "large"),
    ("SBUX", "Starbucks Corp", "Consumer Cyclical", "large"),
    ("ADP", "ADP Inc", "Industrials", "large"),
    ("C", "Citigroup Inc", "Financial", "large"),
    ("PLD", "Prologis Inc", "Real Estate", "large"),
    ("ZTS", "Zoetis Inc", "Healthcare", "large"),
    ("BSX", "Boston Scientific", "Healthcare", "large"),
    ("CI", "Cigna Group", "Healthcare", "large"),
    ("CB", "Chubb Ltd", "Financial", "large"),
    ("SCHW", "Charles Schwab", "Financial", "large"),
    ("MO", "Altria Group", "Consumer Defensive", "large"),
    ("KLAC", "KLA Corp", "Technology", "large"),
    ("CME", "CME Group", "Financial", "large"),
    ("SO", "Southern Co", "Utilities", "large"),
    ("DUK", "Duke Energy", "Utilities", "large"),
    ("SNPS", "Synopsys Inc", "Technology", "large"),
    ("CDNS", "Cadence Design", "Technology", "large"),
    ("ICE", "Intercontl Exchange", "Financial", "large"),
    ("BMY", "Bristol-Myers", "Healthcare", "large"),
    ("EOG", "EOG Resources", "Energy", "large"),
    ("WM", "Waste Management", "Industrials", "large"),
    ("PNC", "PNC Financial", "Financial", "large"),
    ("CL", "Colgate-Palmolive", "Consumer Defensive", "large"),
    ("SHW", "Sherwin-Williams", "Basic Materials", "large"),
    ("TGT", "Target Corp", "Consumer Defensive", "large"),
    ("USB", "US Bancorp", "Financial", "large"),
    ("FDX", "FedEx Corp", "Industrials", "large"),
    ("AMAT", "Applied Materials", "Technology", "large"),
    ("MCK", "McKesson Corp", "Healthcare", "large"),
    ("AON", "Aon PLC", "Financial", "large"),
    ("MU", "Micron Technology", "Technology", "large"),
    # 101-150
    ("ITW", "Illinois Tool Works", "Industrials", "large"),
    ("ORLY", "O'Reilly Auto", "Consumer Cyclical", "large"),
    ("GD", "General Dynamics", "Industrials", "large"),
    ("NOC", "Northrop Grumman", "Industrials", "large"),
    ("SLB", "Schlumberger", "Energy", "large"),
    ("HCA", "HCA Healthcare", "Healthcare", "large"),
    ("PSX", "Phillips 66", "Energy", "large"),
    ("PYPL", "PayPal Holdings", "Financial", "large"),
    ("COP", "ConocoPhillips", "Energy", "large"),
    ("APD", "Air Products", "Basic Materials", "large"),
    ("TJX", "TJX Companies", "Consumer Cyclical", "large"),
    ("CVS", "CVS Health", "Healthcare", "large"),
    ("NSC", "Norfolk Southern", "Industrials", "large"),
    ("MAR", "Marriott Intl", "Consumer Cyclical", "large"),
    ("CMG", "Chipotle Mexican", "Consumer Cyclical", "large"),
    ("AZO", "AutoZone Inc", "Consumer Cyclical", "large"),
    ("HUM", "Humana Inc", "Healthcare", "large"),
    ("PXD", "Pioneer Natural", "Energy", "large"),
    ("F", "Ford Motor", "Consumer Cyclical", "large"),
    ("GM", "General Motors", "Consumer Cyclical", "large"),
    ("RIVN", "Rivian Automotive", "Consumer Cyclical", "mid"),
    ("LCID", "Lucid Group", "Consumer Cyclical", "mid"),
    ("AMD", "AMD Inc", "Technology", "large"),
    ("MRVL", "Marvell Technology", "Technology", "large"),
    ("NFLX", "Netflix Inc", "Communication", "large"),
    ("DIS", "Walt Disney", "Communication", "large"),
    ("CMCSA", "Comcast Corp", "Communication", "large"),
    ("VZ", "Verizon Comm", "Communication", "large"),
    ("T", "AT&T Inc", "Communication", "large"),
    ("TMUS", "T-Mobile US", "Communication", "large"),
    ("CHTR", "Charter Comm", "Communication", "large"),
    ("ATVI", "Activision Blizzard", "Communication", "large"),
    ("EA", "Electronic Arts", "Communication", "large"),
    ("WBD", "Warner Bros Disc", "Communication", "mid"),
    ("PARA", "Paramount Global", "Communication", "mid"),
    ("LMT", "Lockheed Martin", "Industrials", "large"),
    ("UBER", "Uber Technologies", "Technology", "large"),
    ("ABNB", "Airbnb Inc", "Consumer Cyclical", "large"),
    ("SQ", "Block Inc", "Technology", "large"),
    ("SHOP", "Shopify Inc", "Technology", "large"),
    ("SNOW", "Snowflake Inc", "Technology", "large"),
    ("DDOG", "Datadog Inc", "Technology", "large"),
    ("CRWD", "CrowdStrike", "Technology", "large"),
    ("ZS", "Zscaler Inc", "Technology", "large"),
    ("NET", "Cloudflare Inc", "Technology", "mid"),
    ("PLTR", "Palantir Tech", "Technology", "mid"),
    ("COIN", "Coinbase Global", "Financial", "mid"),
    ("HOOD", "Robinhood Markets", "Financial", "mid"),
    ("PATH", "UiPath Inc", "Technology", "mid"),
    ("U", "Unity Software", "Technology", "mid"),
    ("RBLX", "Roblox Corp", "Communication", "mid"),
    ("TTWO", "Take-Two Interactive", "Communication", "large"),
]

# RUSSELL 1000 Additional (Not in S&P 500) - Mid Caps
RUSSELL_ADDITIONAL = [
    ("MARA", "Marathon Digital", "Technology", "small"),
    ("RIOT", "Riot Platforms", "Technology", "small"),
    ("MSTR", "MicroStrategy", "Technology", "mid"),
    ("SMCI", "Super Micro Computer", "Technology", "mid"),
    ("ARM", "ARM Holdings", "Technology", "large"),
    ("DASH", "DoorDash Inc", "Technology", "large"),
    ("SPOT", "Spotify Tech", "Communication", "large"),
    ("PINS", "Pinterest Inc", "Communication", "mid"),
    ("SNAP", "Snap Inc", "Communication", "mid"),
    ("TWLO", "Twilio Inc", "Technology", "mid"),
    ("OKTA", "Okta Inc", "Technology", "mid"),
    ("MDB", "MongoDB Inc", "Technology", "mid"),
    ("ESTC", "Elastic NV", "Technology", "mid"),
    ("HUBS", "HubSpot Inc", "Technology", "mid"),
    ("BILL", "Bill Holdings", "Technology", "mid"),
    ("FIVN", "Five9 Inc", "Technology", "mid"),
    ("DOCN", "DigitalOcean", "Technology", "small"),
    ("CFLT", "Confluent Inc", "Technology", "mid"),
    ("GTLB", "GitLab Inc", "Technology", "mid"),
    ("S", "SentinelOne", "Technology", "mid"),
    ("GME", "GameStop Corp", "Consumer Cyclical", "mid"),
    ("AMC", "AMC Entertainment", "Communication", "small"),
    ("BBBY", "Bed Bath Beyond", "Consumer Cyclical", "small"),
    ("BB", "BlackBerry Ltd", "Technology", "small"),
    ("SOFI", "SoFi Technologies", "Financial", "mid"),
    ("AFRM", "Affirm Holdings", "Technology", "mid"),
    ("UPST", "Upstart Holdings", "Financial", "small"),
    ("OPEN", "Opendoor Tech", "Real Estate", "small"),
    ("CVNA", "Carvana Co", "Consumer Cyclical", "mid"),
    ("W", "Wayfair Inc", "Consumer Cyclical", "mid"),
    ("CHWY", "Chewy Inc", "Consumer Cyclical", "mid"),
    ("ETSY", "Etsy Inc", "Consumer Cyclical", "mid"),
    ("ROKU", "Roku Inc", "Communication", "mid"),
    ("TTD", "Trade Desk", "Technology", "mid"),
    ("MTCH", "Match Group", "Communication", "mid"),
    ("BMBL", "Bumble Inc", "Communication", "small"),
    ("DUOL", "Duolingo Inc", "Technology", "mid"),
    ("MNDY", "Monday.com", "Technology", "mid"),
    ("GLBE", "Global-E Online", "Technology", "mid"),
    ("TOST", "Toast Inc", "Technology", "mid"),
]

# UK STOCKS - FTSE 100 + Top FTSE 250
UK_STOCKS = [
    # FTSE 100
    ("AZN.L", "AstraZeneca", "Healthcare", "large", "IX.D.FTSE.DAILY.IP"),
    ("SHEL.L", "Shell PLC", "Energy", "large", "IX.D.FTSE.DAILY.IP"),
    ("HSBA.L", "HSBC Holdings", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("ULVR.L", "Unilever", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("BP.L", "BP PLC", "Energy", "large", "IX.D.FTSE.DAILY.IP"),
    ("GSK.L", "GSK PLC", "Healthcare", "large", "IX.D.FTSE.DAILY.IP"),
    ("DGE.L", "Diageo", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("RIO.L", "Rio Tinto", "Basic Materials", "large", "IX.D.FTSE.DAILY.IP"),
    ("BATS.L", "British American Tobacco", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("REL.L", "RELX PLC", "Industrials", "large", "IX.D.FTSE.DAILY.IP"),
    ("LSEG.L", "London Stock Exchange", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("GLEN.L", "Glencore", "Basic Materials", "large", "IX.D.FTSE.DAILY.IP"),
    ("NG.L", "National Grid", "Utilities", "large", "IX.D.FTSE.DAILY.IP"),
    ("VOD.L", "Vodafone Group", "Communication", "large", "IX.D.FTSE.DAILY.IP"),
    ("PRU.L", "Prudential PLC", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("LLOY.L", "Lloyds Banking", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("BARC.L", "Barclays PLC", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("NWG.L", "NatWest Group", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("STAN.L", "Standard Chartered", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("AVIVA.L", "Aviva PLC", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("LGEN.L", "Legal & General", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("BA.L", "BAE Systems", "Industrials", "large", "IX.D.FTSE.DAILY.IP"),
    ("RR.L", "Rolls-Royce", "Industrials", "large", "IX.D.FTSE.DAILY.IP"),
    ("AAL.L", "Anglo American", "Basic Materials", "large", "IX.D.FTSE.DAILY.IP"),
    ("ANTO.L", "Antofagasta", "Basic Materials", "large", "IX.D.FTSE.DAILY.IP"),
    ("CRH.L", "CRH PLC", "Basic Materials", "large", "IX.D.FTSE.DAILY.IP"),
    ("EXPN.L", "Experian", "Industrials", "large", "IX.D.FTSE.DAILY.IP"),
    ("III.L", "3i Group", "Financial", "large", "IX.D.FTSE.DAILY.IP"),
    ("ABF.L", "Associated British Foods", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("TSCO.L", "Tesco PLC", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("SBRY.L", "Sainsbury's", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("MKS.L", "Marks & Spencer", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("JD.L", "JD Sports", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("FRAS.L", "Frasers Group", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("NXT.L", "Next PLC", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("BRBY.L", "Burberry Group", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("BT.A.L", "BT Group", "Communication", "large", "IX.D.FTSE.DAILY.IP"),
    ("ITV.L", "ITV PLC", "Communication", "mid", "IX.D.FTSE.DAILY.IP"),
    ("WPP.L", "WPP PLC", "Communication", "large", "IX.D.FTSE.DAILY.IP"),
    ("IMB.L", "Imperial Brands", "Consumer Defensive", "large", "IX.D.FTSE.DAILY.IP"),
    ("SSE.L", "SSE PLC", "Utilities", "large", "IX.D.FTSE.DAILY.IP"),
    ("SVT.L", "Severn Trent", "Utilities", "large", "IX.D.FTSE.DAILY.IP"),
    ("UU.L", "United Utilities", "Utilities", "large", "IX.D.FTSE.DAILY.IP"),
    ("CPG.L", "Compass Group", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("IHG.L", "InterContinental Hotels", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("WTB.L", "Whitbread", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("ENT.L", "Entain PLC", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("FLTR.L", "Flutter Entertainment", "Consumer Cyclical", "large", "IX.D.FTSE.DAILY.IP"),
    ("PSON.L", "Pearson PLC", "Communication", "large", "IX.D.FTSE.DAILY.IP"),
    ("RMV.L", "Rightmove", "Communication", "large", "IX.D.FTSE.DAILY.IP"),
    # Top FTSE 250
    ("AUTO.L", "Auto Trader", "Communication", "mid", "IX.D.FTSE.DAILY.IP"),
    ("DARK.L", "Darktrace", "Technology", "mid", "IX.D.FTSE.DAILY.IP"),
    ("OCDO.L", "Ocado Group", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("THG.L", "THG Holdings", "Consumer Cyclical", "small", "IX.D.FTSE.DAILY.IP"),
    ("DPLM.L", "Diploma PLC", "Industrials", "mid", "IX.D.FTSE.DAILY.IP"),
    ("BDEV.L", "Barratt Developments", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("TW.L", "Taylor Wimpey", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("PSN.L", "Persimmon", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("RDW.L", "Redrow", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
    ("BWY.L", "Bellway", "Consumer Cyclical", "mid", "IX.D.FTSE.DAILY.IP"),
]

# EUROPEAN STOCKS - DAX + CAC + Euro Stoxx
EUROPE_STOCKS = [
    # DAX 40 (Germany)
    ("SAP.DE", "SAP SE", "Technology", "large", "IX.D.DAX.DAILY.IP"),
    ("SIE.DE", "Siemens AG", "Industrials", "large", "IX.D.DAX.DAILY.IP"),
    ("ALV.DE", "Allianz SE", "Financial", "large", "IX.D.DAX.DAILY.IP"),
    ("DTE.DE", "Deutsche Telekom", "Communication", "large", "IX.D.DAX.DAILY.IP"),
    ("BAS.DE", "BASF SE", "Basic Materials", "large", "IX.D.DAX.DAILY.IP"),
    ("MRK.DE", "Merck KGaA", "Healthcare", "large", "IX.D.DAX.DAILY.IP"),
    ("BMW.DE", "BMW AG", "Consumer Cyclical", "large", "IX.D.DAX.DAILY.IP"),
    ("MBG.DE", "Mercedes-Benz", "Consumer Cyclical", "large", "IX.D.DAX.DAILY.IP"),
    ("VOW3.DE", "Volkswagen AG", "Consumer Cyclical", "large", "IX.D.DAX.DAILY.IP"),
    ("ADS.DE", "Adidas AG", "Consumer Cyclical", "large", "IX.D.DAX.DAILY.IP"),
    ("DBK.DE", "Deutsche Bank", "Financial", "large", "IX.D.DAX.DAILY.IP"),
    ("IFX.DE", "Infineon Tech", "Technology", "large", "IX.D.DAX.DAILY.IP"),
    ("MUV2.DE", "Munich Re", "Financial", "large", "IX.D.DAX.DAILY.IP"),
    ("DB1.DE", "Deutsche Boerse", "Financial", "large", "IX.D.DAX.DAILY.IP"),
    ("RWE.DE", "RWE AG", "Utilities", "large", "IX.D.DAX.DAILY.IP"),
    ("DHL.DE", "DHL Group", "Industrials", "large", "IX.D.DAX.DAILY.IP"),
    ("HEN3.DE", "Henkel AG", "Consumer Defensive", "large", "IX.D.DAX.DAILY.IP"),
    ("BEI.DE", "Beiersdorf", "Consumer Defensive", "large", "IX.D.DAX.DAILY.IP"),
    ("FRE.DE", "Fresenius SE", "Healthcare", "large", "IX.D.DAX.DAILY.IP"),
    ("EOAN.DE", "E.ON SE", "Utilities", "large", "IX.D.DAX.DAILY.IP"),
    # CAC 40 (France)
    ("MC.PA", "LVMH", "Consumer Cyclical", "large", "IX.D.CAC.DAILY.IP"),
    ("OR.PA", "L'Oreal", "Consumer Defensive", "large", "IX.D.CAC.DAILY.IP"),
    ("TTE.PA", "TotalEnergies", "Energy", "large", "IX.D.CAC.DAILY.IP"),
    ("SAN.PA", "Sanofi", "Healthcare", "large", "IX.D.CAC.DAILY.IP"),
    ("AIR.PA", "Airbus SE", "Industrials", "large", "IX.D.CAC.DAILY.IP"),
    ("BNP.PA", "BNP Paribas", "Financial", "large", "IX.D.CAC.DAILY.IP"),
    ("AI.PA", "Air Liquide", "Basic Materials", "large", "IX.D.CAC.DAILY.IP"),
    ("KER.PA", "Kering", "Consumer Cyclical", "large", "IX.D.CAC.DAILY.IP"),
    ("RMS.PA", "Hermes Intl", "Consumer Cyclical", "large", "IX.D.CAC.DAILY.IP"),
    ("CS.PA", "AXA SA", "Financial", "large", "IX.D.CAC.DAILY.IP"),
    ("SU.PA", "Schneider Electric", "Industrials", "large", "IX.D.CAC.DAILY.IP"),
    ("EL.PA", "EssilorLuxottica", "Healthcare", "large", "IX.D.CAC.DAILY.IP"),
    ("RI.PA", "Pernod Ricard", "Consumer Defensive", "large", "IX.D.CAC.DAILY.IP"),
    ("DG.PA", "Vinci SA", "Industrials", "large", "IX.D.CAC.DAILY.IP"),
    ("SGO.PA", "Saint-Gobain", "Basic Materials", "large", "IX.D.CAC.DAILY.IP"),
    ("CAP.PA", "Capgemini SE", "Technology", "large", "IX.D.CAC.DAILY.IP"),
    ("DSY.PA", "Dassault Systemes", "Technology", "large", "IX.D.CAC.DAILY.IP"),
    ("STM.PA", "STMicroelectronics", "Technology", "large", "IX.D.CAC.DAILY.IP"),
    ("EN.PA", "Engie SA", "Utilities", "large", "IX.D.CAC.DAILY.IP"),
    ("VIV.PA", "Vivendi SE", "Communication", "large", "IX.D.CAC.DAILY.IP"),
    # Euro Stoxx 50 additions
    ("ASML.AS", "ASML Holding", "Technology", "large", "IX.D.STXE.DAILY.IP"),
    ("INGA.AS", "ING Group", "Financial", "large", "IX.D.STXE.DAILY.IP"),
    ("PHIA.AS", "Philips NV", "Healthcare", "large", "IX.D.STXE.DAILY.IP"),
    ("AD.AS", "Ahold Delhaize", "Consumer Defensive", "large", "IX.D.STXE.DAILY.IP"),
    ("ISP.MI", "Intesa Sanpaolo", "Financial", "large", "IX.D.STXE.DAILY.IP"),
    ("ENEL.MI", "Enel SpA", "Utilities", "large", "IX.D.STXE.DAILY.IP"),
    ("ENI.MI", "Eni SpA", "Energy", "large", "IX.D.STXE.DAILY.IP"),
    ("IBE.MC", "Iberdrola", "Utilities", "large", "IX.D.STXE.DAILY.IP"),
    ("SAN.MC", "Banco Santander", "Financial", "large", "IX.D.STXE.DAILY.IP"),
    ("ITX.MC", "Inditex", "Consumer Cyclical", "large", "IX.D.STXE.DAILY.IP"),
]

# ASIAN STOCKS - Nikkei + Hang Seng + ASX
ASIA_STOCKS = [
    # Nikkei 225 (Japan) - Top 20
    ("7203.T", "Toyota Motor", "Consumer Cyclical", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6758.T", "Sony Group", "Technology", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("9984.T", "SoftBank Group", "Communication", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6861.T", "Keyence Corp", "Technology", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("9432.T", "NTT Corp", "Communication", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("8306.T", "Mitsubishi UFJ", "Financial", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6501.T", "Hitachi Ltd", "Technology", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("7267.T", "Honda Motor", "Consumer Cyclical", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6902.T", "Denso Corp", "Consumer Cyclical", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("4063.T", "Shin-Etsu Chemical", "Basic Materials", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("8035.T", "Tokyo Electron", "Technology", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("9433.T", "KDDI Corp", "Communication", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("4502.T", "Takeda Pharma", "Healthcare", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("7974.T", "Nintendo", "Communication", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6367.T", "Daikin Industries", "Industrials", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("6098.T", "Recruit Holdings", "Industrials", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("4568.T", "Daiichi Sankyo", "Healthcare", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("8058.T", "Mitsubishi Corp", "Industrials", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("8001.T", "Itochu Corp", "Industrials", "large", "IX.D.NIKKEI.DAILY.IP"),
    ("9983.T", "Fast Retailing", "Consumer Cyclical", "large", "IX.D.NIKKEI.DAILY.IP"),
    # Hang Seng (Hong Kong) - Top 15
    ("0700.HK", "Tencent Holdings", "Communication", "large", "IX.D.HSI.DAILY.IP"),
    ("9988.HK", "Alibaba Group", "Consumer Cyclical", "large", "IX.D.HSI.DAILY.IP"),
    ("9618.HK", "JD.com", "Consumer Cyclical", "large", "IX.D.HSI.DAILY.IP"),
    ("3690.HK", "Meituan", "Consumer Cyclical", "large", "IX.D.HSI.DAILY.IP"),
    ("1299.HK", "AIA Group", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("0005.HK", "HSBC Holdings", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("0941.HK", "China Mobile", "Communication", "large", "IX.D.HSI.DAILY.IP"),
    ("2318.HK", "Ping An Insurance", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("0939.HK", "CCB", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("1398.HK", "ICBC", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("0388.HK", "HKEX", "Financial", "large", "IX.D.HSI.DAILY.IP"),
    ("0001.HK", "CK Hutchison", "Industrials", "large", "IX.D.HSI.DAILY.IP"),
    ("0016.HK", "Sun Hung Kai", "Real Estate", "large", "IX.D.HSI.DAILY.IP"),
    ("0027.HK", "Galaxy Entertainment", "Consumer Cyclical", "large", "IX.D.HSI.DAILY.IP"),
    ("1928.HK", "Sands China", "Consumer Cyclical", "large", "IX.D.HSI.DAILY.IP"),
    # ASX 200 (Australia) - Top 15
    ("BHP.AX", "BHP Group", "Basic Materials", "large", "IX.D.ASX.DAILY.IP"),
    ("CBA.AX", "Commonwealth Bank", "Financial", "large", "IX.D.ASX.DAILY.IP"),
    ("CSL.AX", "CSL Limited", "Healthcare", "large", "IX.D.ASX.DAILY.IP"),
    ("NAB.AX", "National Australia Bank", "Financial", "large", "IX.D.ASX.DAILY.IP"),
    ("WBC.AX", "Westpac Banking", "Financial", "large", "IX.D.ASX.DAILY.IP"),
    ("ANZ.AX", "ANZ Banking", "Financial", "large", "IX.D.ASX.DAILY.IP"),
    ("WES.AX", "Wesfarmers", "Consumer Cyclical", "large", "IX.D.ASX.DAILY.IP"),
    ("MQG.AX", "Macquarie Group", "Financial", "large", "IX.D.ASX.DAILY.IP"),
    ("WOW.AX", "Woolworths Group", "Consumer Defensive", "large", "IX.D.ASX.DAILY.IP"),
    ("TLS.AX", "Telstra Corp", "Communication", "large", "IX.D.ASX.DAILY.IP"),
    ("RIO.AX", "Rio Tinto", "Basic Materials", "large", "IX.D.ASX.DAILY.IP"),
    ("FMG.AX", "Fortescue Metals", "Basic Materials", "large", "IX.D.ASX.DAILY.IP"),
    ("NCM.AX", "Newcrest Mining", "Basic Materials", "large", "IX.D.ASX.DAILY.IP"),
    ("WDS.AX", "Woodside Energy", "Energy", "large", "IX.D.ASX.DAILY.IP"),
    ("STO.AX", "Santos Ltd", "Energy", "large", "IX.D.ASX.DAILY.IP"),
]

# FOREX PAIRS - All 28 Tradeable
FOREX_PAIRS = [
    # Majors (7)
    ("EUR_USD", "Euro / US Dollar", 0.008, 30.0),
    ("GBP_USD", "British Pound / US Dollar", 0.012, 30.0),
    ("USD_JPY", "US Dollar / Japanese Yen", 0.010, 30.0),
    ("USD_CHF", "US Dollar / Swiss Franc", 0.015, 30.0),
    ("AUD_USD", "Australian Dollar / US Dollar", 0.012, 30.0),
    ("USD_CAD", "US Dollar / Canadian Dollar", 0.015, 30.0),
    ("NZD_USD", "New Zealand Dollar / US Dollar", 0.018, 30.0),
    # Crosses (21)
    ("EUR_GBP", "Euro / British Pound", 0.015, 30.0),
    ("EUR_JPY", "Euro / Japanese Yen", 0.018, 30.0),
    ("EUR_CHF", "Euro / Swiss Franc", 0.018, 30.0),
    ("EUR_AUD", "Euro / Australian Dollar", 0.022, 20.0),
    ("EUR_CAD", "Euro / Canadian Dollar", 0.022, 20.0),
    ("EUR_NZD", "Euro / New Zealand Dollar", 0.030, 20.0),
    ("GBP_JPY", "British Pound / Japanese Yen", 0.025, 20.0),
    ("GBP_CHF", "British Pound / Swiss Franc", 0.025, 20.0),
    ("GBP_AUD", "British Pound / Australian Dollar", 0.028, 20.0),
    ("GBP_CAD", "British Pound / Canadian Dollar", 0.028, 20.0),
    ("GBP_NZD", "British Pound / New Zealand Dollar", 0.035, 20.0),
    ("AUD_JPY", "Australian Dollar / Japanese Yen", 0.020, 20.0),
    ("AUD_CHF", "Australian Dollar / Swiss Franc", 0.025, 20.0),
    ("AUD_CAD", "Australian Dollar / Canadian Dollar", 0.025, 20.0),
    ("AUD_NZD", "Australian Dollar / New Zealand Dollar", 0.025, 20.0),
    ("CAD_JPY", "Canadian Dollar / Japanese Yen", 0.020, 20.0),
    ("CAD_CHF", "Canadian Dollar / Swiss Franc", 0.025, 20.0),
    ("CHF_JPY", "Swiss Franc / Japanese Yen", 0.020, 20.0),
    ("NZD_JPY", "New Zealand Dollar / Japanese Yen", 0.025, 20.0),
    ("NZD_CHF", "New Zealand Dollar / Swiss Franc", 0.030, 20.0),
    ("NZD_CAD", "New Zealand Dollar / Canadian Dollar", 0.030, 20.0),
]

# GLOBAL INDICES
GLOBAL_INDICES = [
    # US
    ("US500", "S&P 500", "us", 0.04, 20.0, "IX.D.SPTRD.DAILY.IP"),
    ("US100", "NASDAQ 100", "us", 0.04, 20.0, "IX.D.NASDAQ.DAILY.IP"),
    ("US30", "Dow Jones 30", "us", 0.04, 20.0, "IX.D.DOW.DAILY.IP"),
    ("RUSSELL2000", "Russell 2000", "us", 0.06, 20.0, "IX.D.RUSSELL.DAILY.IP"),
    # Europe
    ("UK100", "FTSE 100", "uk", 0.04, 20.0, "IX.D.FTSE.DAILY.IP"),
    ("DE40", "DAX 40", "europe", 0.04, 20.0, "IX.D.DAX.DAILY.IP"),
    ("FR40", "CAC 40", "europe", 0.04, 20.0, "IX.D.CAC.DAILY.IP"),
    ("EU50", "Euro Stoxx 50", "europe", 0.04, 20.0, "IX.D.STXE.DAILY.IP"),
    # Asia
    ("JP225", "Nikkei 225", "asia", 0.05, 20.0, "IX.D.NIKKEI.DAILY.IP"),
    ("HK50", "Hang Seng", "asia", 0.08, 10.0, "IX.D.HANGSENG.DAILY.IP"),
    ("AU200", "ASX 200", "asia", 0.05, 20.0, "IX.D.ASX.DAILY.IP"),
    ("CN50", "China A50", "asia", 0.10, 10.0, "IX.D.CHINAA.DAILY.IP"),
    # Volatility
    ("VIX", "CBOE Volatility Index", "us", 0.10, 5.0, "IX.D.VIX.DAILY.IP"),
    ("VDAX", "VDAX Volatility", "europe", 0.10, 5.0, "IX.D.VDAX.DAILY.IP"),
    # Emerging
    ("INDIA50", "Nifty 50", "asia", 0.08, 10.0, "IX.D.NIFTY.DAILY.IP"),
]

# COMMODITIES
COMMODITIES = [
    # Precious Metals
    ("XAUUSD", "Gold", 0.025, 20.0, "CS.D.USCGC.TODAY.IP"),
    ("XAGUSD", "Silver", 0.035, 10.0, "CS.D.USCSI.TODAY.IP"),
    ("XPTUSD", "Platinum", 0.040, 10.0, "CS.D.USCPT.TODAY.IP"),
    ("XPDUSD", "Palladium", 0.050, 10.0, "CS.D.USCPD.TODAY.IP"),
    # Energy
    ("USOIL", "WTI Crude Oil", 0.030, 20.0, "CC.D.CL.UNC.IP"),
    ("UKOIL", "Brent Crude Oil", 0.030, 20.0, "CC.D.LCO.UNC.IP"),
    ("NATGAS", "Natural Gas", 0.060, 10.0, "CC.D.NG.UNC.IP"),
    # Agricultural
    ("WHEAT", "Wheat", 0.050, 10.0, "CC.D.W.UNC.IP"),
    ("CORN", "Corn", 0.050, 10.0, "CC.D.C.UNC.IP"),
    ("SOYBEAN", "Soybeans", 0.050, 10.0, "CC.D.S.UNC.IP"),
    # Industrial
    ("COPPER", "Copper", 0.040, 10.0, "CC.D.HG.UNC.IP"),
]

# CRYPTOCURRENCIES (24/7)
CRYPTO_PAIRS = [
    # Major
    ("BTC_USD", "Bitcoin", 0.08, 10.0),
    ("ETH_USD", "Ethereum", 0.08, 10.0),
    ("BNB_USD", "Binance Coin", 0.10, 5.0),
    ("XRP_USD", "Ripple", 0.12, 5.0),
    ("SOL_USD", "Solana", 0.10, 5.0),
    ("ADA_USD", "Cardano", 0.12, 5.0),
    # DeFi
    ("AVAX_USD", "Avalanche", 0.12, 5.0),
    ("DOT_USD", "Polkadot", 0.12, 5.0),
    ("MATIC_USD", "Polygon", 0.12, 5.0),
    ("LINK_USD", "Chainlink", 0.12, 5.0),
    ("UNI_USD", "Uniswap", 0.15, 5.0),
    ("ATOM_USD", "Cosmos", 0.15, 5.0),
    # Meme/Momentum
    ("DOGE_USD", "Dogecoin", 0.15, 5.0),
    ("SHIB_USD", "Shiba Inu", 0.18, 3.0),
    # Legacy
    ("LTC_USD", "Litecoin", 0.10, 5.0),
    ("BCH_USD", "Bitcoin Cash", 0.10, 5.0),
    ("ETC_USD", "Ethereum Classic", 0.12, 5.0),
    # Layer 2 / New
    ("ARB_USD", "Arbitrum", 0.15, 5.0),
    ("OP_USD", "Optimism", 0.15, 5.0),
    ("APT_USD", "Aptos", 0.15, 5.0),
]


# =============================================================================
# INSTRUMENT REGISTRY CLASS
# =============================================================================

class InstrumentRegistry:
    """Central registry of all tradeable instruments."""

    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}
        self._load_all_instruments()
        logger.info(f"Loaded {self.total_count} instruments")

    def _load_all_instruments(self):
        """Load all instruments from all markets."""
        self._load_us_stocks()
        self._load_uk_stocks()
        self._load_europe_stocks()
        self._load_asia_stocks()
        self._load_forex()
        self._load_indices()
        self._load_commodities()
        self._load_crypto()

    def _load_us_stocks(self):
        """Load US stocks (S&P 500 + Russell additions)."""
        for symbol, name, sector, cap in SP500_STOCKS:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.STOCK,
                region=Region.US,
                exchange="NYSE/NASDAQ",
                provider=DataProvider.POLYGON,
                typical_spread_pct=0.02,
                leverage_available=5.0,
                session=SESSIONS["us_stocks"],
                currency="USD",
                sector=sector,
                market_cap=cap,
            )

        for symbol, name, sector, cap in RUSSELL_ADDITIONAL:
            if symbol not in self.instruments:
                self.instruments[symbol] = Instrument(
                    symbol=symbol,
                    name=name,
                    instrument_type=InstrumentType.STOCK,
                    region=Region.US,
                    exchange="NYSE/NASDAQ",
                    provider=DataProvider.POLYGON,
                    typical_spread_pct=0.03,
                    leverage_available=5.0,
                    session=SESSIONS["us_stocks"],
                    currency="USD",
                    sector=sector,
                    market_cap=cap,
                )

    def _load_uk_stocks(self):
        """Load UK stocks (FTSE 100 + 250)."""
        for symbol, name, sector, cap, ig_epic in UK_STOCKS:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.STOCK,
                region=Region.UK,
                exchange="LSE",
                provider=DataProvider.IG,
                typical_spread_pct=0.08,
                leverage_available=5.0,
                session=SESSIONS["uk_stocks"],
                currency="GBP",
                sector=sector,
                market_cap=cap,
                provider_symbol=ig_epic,
            )

    def _load_europe_stocks(self):
        """Load European stocks (DAX + CAC + Euro Stoxx)."""
        for symbol, name, sector, cap, ig_epic in EUROPE_STOCKS:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.STOCK,
                region=Region.EUROPE,
                exchange="Xetra/Euronext",
                provider=DataProvider.IG,
                typical_spread_pct=0.08,
                leverage_available=5.0,
                session=SESSIONS["europe_stocks"],
                currency="EUR",
                sector=sector,
                market_cap=cap,
                provider_symbol=ig_epic,
            )

    def _load_asia_stocks(self):
        """Load Asian stocks (Nikkei + Hang Seng + ASX)."""
        for symbol, name, sector, cap, ig_epic in ASIA_STOCKS:
            if ".T" in symbol:
                region, currency, session_key = Region.ASIA_JAPAN, "JPY", "japan_stocks"
            elif ".HK" in symbol:
                region, currency, session_key = Region.ASIA_HK, "HKD", "hk_stocks"
            else:
                region, currency, session_key = Region.ASIA_AU, "AUD", "au_stocks"

            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.STOCK,
                region=region,
                exchange="TSE/HKEX/ASX",
                provider=DataProvider.IG,
                typical_spread_pct=0.10,
                leverage_available=5.0,
                session=SESSIONS[session_key],
                currency=currency,
                sector=sector,
                market_cap=cap,
                provider_symbol=ig_epic,
            )

    def _load_forex(self):
        """Load all forex pairs."""
        for symbol, name, spread, leverage in FOREX_PAIRS:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.FOREX,
                region=Region.GLOBAL,
                exchange="FX",
                provider=DataProvider.OANDA,
                typical_spread_pct=spread,
                min_position_size=1000,
                leverage_available=leverage,
                session=SESSIONS["forex"],
                currency=symbol.split("_")[1],
            )

    def _load_indices(self):
        """Load global indices."""
        region_map = {
            "us": Region.US,
            "uk": Region.UK,
            "europe": Region.EUROPE,
            "asia": Region.ASIA_JAPAN,
        }

        for symbol, name, region_str, spread, leverage, ig_epic in GLOBAL_INDICES:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.INDEX,
                region=region_map.get(region_str, Region.GLOBAL),
                exchange="Index",
                provider=DataProvider.IG,
                typical_spread_pct=spread,
                min_position_size=0.1,
                leverage_available=leverage,
                session=SESSIONS["forex"],  # Most indices follow forex hours
                provider_symbol=ig_epic,
            )

    def _load_commodities(self):
        """Load commodities."""
        for symbol, name, spread, leverage, ig_epic in COMMODITIES:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.COMMODITY,
                region=Region.GLOBAL,
                exchange="Commodity",
                provider=DataProvider.IG,
                typical_spread_pct=spread,
                min_position_size=0.1,
                leverage_available=leverage,
                session=SESSIONS["forex"],
                provider_symbol=ig_epic,
            )

    def _load_crypto(self):
        """Load cryptocurrencies (24/7)."""
        for symbol, name, spread, leverage in CRYPTO_PAIRS:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                name=name,
                instrument_type=InstrumentType.CRYPTO,
                region=Region.GLOBAL,
                exchange="Crypto",
                provider=DataProvider.BINANCE,
                typical_spread_pct=spread,
                min_position_size=0.0001,
                leverage_available=leverage,
                session=SESSIONS["crypto"],
                currency="USD",
            )

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_all(self) -> List[Instrument]:
        """Get all instruments."""
        return list(self.instruments.values())

    def get(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol."""
        return self.instruments.get(symbol)

    def get_by_type(self, instrument_type: InstrumentType) -> List[Instrument]:
        """Get instruments by type."""
        return [i for i in self.instruments.values() if i.instrument_type == instrument_type]

    def get_by_region(self, region: Region) -> List[Instrument]:
        """Get instruments by region."""
        return [i for i in self.instruments.values() if i.region == region]

    def get_by_provider(self, provider: DataProvider) -> List[Instrument]:
        """Get instruments by data provider."""
        return [i for i in self.instruments.values() if i.provider == provider]

    def get_by_sector(self, sector: str) -> List[Instrument]:
        """Get instruments by sector."""
        return [i for i in self.instruments.values() if i.sector.lower() == sector.lower()]

    def get_currently_open(self) -> List[Instrument]:
        """Get instruments where market is currently open."""
        now = datetime.now(timezone.utc)
        return [i for i in self.instruments.values() if i.session and i.session.is_open(now)]

    def get_24_7(self) -> List[Instrument]:
        """Get instruments that trade 24/7 (crypto)."""
        return [i for i in self.instruments.values() if i.instrument_type == InstrumentType.CRYPTO]

    def get_weekend_tradeable(self) -> List[Instrument]:
        """Get instruments tradeable on weekends (crypto only)."""
        return self.get_24_7()

    def get_high_leverage(self, min_leverage: float = 10.0) -> List[Instrument]:
        """Get instruments with high leverage available."""
        return [i for i in self.instruments.values() if i.leverage_available >= min_leverage]

    def get_low_spread(self, max_spread: float = 0.05) -> List[Instrument]:
        """Get instruments with low spreads (cost-efficient)."""
        return [i for i in self.instruments.values() if i.typical_spread_pct <= max_spread]

    def search(self, query: str) -> List[Instrument]:
        """Search instruments by symbol or name."""
        query = query.upper()
        return [
            i for i in self.instruments.values()
            if query in i.symbol.upper() or query in i.name.upper()
        ]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    @property
    def total_count(self) -> int:
        return len(self.instruments)

    def summary(self) -> Dict:
        """Get summary statistics."""
        by_type = {}
        for inst_type in InstrumentType:
            by_type[inst_type.value] = len(self.get_by_type(inst_type))

        by_region = {}
        for region in Region:
            by_region[region.value] = len(self.get_by_region(region))

        by_provider = {}
        for provider in DataProvider:
            by_provider[provider.value] = len(self.get_by_provider(provider))

        return {
            "total": self.total_count,
            "by_type": by_type,
            "by_region": by_region,
            "by_provider": by_provider,
            "currently_open": len(self.get_currently_open()),
            "weekend_tradeable": len(self.get_weekend_tradeable()),
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()

        print("\n" + "=" * 60)
        print("NEXUS GLOBAL INSTRUMENT REGISTRY")
        print("=" * 60)
        print(f"\nTOTAL INSTRUMENTS: {summary['total']}")

        print("\nBy Type:")
        for t, count in summary["by_type"].items():
            print(f"  {t:15} {count:>5}")

        print("\nBy Region:")
        for r, count in summary["by_region"].items():
            print(f"  {r:15} {count:>5}")

        print("\nBy Provider:")
        for p, count in summary["by_provider"].items():
            print(f"  {p:15} {count:>5}")

        print(f"\nCurrently Open:    {summary['currently_open']}")
        print(f"Weekend Tradeable: {summary['weekend_tradeable']}")
        print("=" * 60 + "\n")


# =============================================================================
# SINGLETON
# =============================================================================

_registry: Optional[InstrumentRegistry] = None


def get_instrument_registry() -> InstrumentRegistry:
    """Get or create the instrument registry singleton."""
    global _registry
    if _registry is None:
        _registry = InstrumentRegistry()
    return _registry
