"""
GOD MODE Data Orchestrator - FULL REGISTRY VERSION

Routes all scanners to correct data providers using the complete
InstrumentRegistry (426 instruments across all global markets).

Each Instrument already knows its DataProvider (POLYGON, OANDA, IG, BINANCE).
The orchestrator connects providers at startup and routes requests by matching
the instrument's provider field to a live connection.

Session schedule determines WHICH instruments are active at any given time,
giving us true 24/5 (weekday) + 24/7 (crypto) coverage.
"""

import logging
from datetime import datetime, time, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from nexus.config.settings import get_settings
from nexus.core.enums import EdgeType, Market, Timeframe
from nexus.data.base import BaseDataProvider, Quote
from nexus.data.instruments import (
    DataProvider,
    Instrument,
    InstrumentRegistry,
    InstrumentType,
    Region,
    get_instrument_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

class TradingSession(Enum):
    """Global trading sessions."""
    ASIA = "asia"
    LONDON_OPEN = "london_open"
    EUROPEAN = "european"
    US_PREMARKET = "us_premarket"
    US_OPEN = "us_open"
    US_SESSION = "us_session"
    POWER_HOUR = "power_hour"
    OVERNIGHT = "overnight"
    WEEKEND = "weekend"


@dataclass
class SessionConfig:
    """What instruments and edges are active during a session."""
    session: TradingSession
    start_time: time  # UTC
    end_time: time
    instrument_filter: Dict[str, Any]  # keys: types, regions (matched via OR)
    priority: str  # HIGH, SCANNING, LOW
    description: str


@dataclass
class ProviderStatus:
    """Runtime status of a connected provider."""
    name: str
    provider_enum: DataProvider
    connected: bool
    instrument_count: int
    error_count: int = 0
    last_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class DataOrchestrator:
    """
    Master orchestrator for all data providers.

    Uses the FULL InstrumentRegistry (426 instruments).
    Each Instrument's ``provider`` field determines which connection to use.
    """

    # Session schedule (UTC).  instrument_filter selects which instruments
    # are active.  ``types`` and ``regions`` are combined with OR logic:
    # an instrument matches if its type OR region appears in the filter.
    # ``always`` lists types that are active in every weekday session.
    SESSIONS: List[SessionConfig] = [
        SessionConfig(
            session=TradingSession.ASIA,
            start_time=time(22, 0),   # prev day
            end_time=time(7, 0),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.CRYPTO],
                "regions": [Region.ASIA_JAPAN, Region.ASIA_HK, Region.ASIA_AU],
            },
            priority="SCANNING",
            description="Asian session - Asia stocks, Forex, Crypto",
        ),
        SessionConfig(
            session=TradingSession.LONDON_OPEN,
            start_time=time(7, 0),
            end_time=time(9, 0),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.INDEX],
                "regions": [Region.UK, Region.EUROPE],
            },
            priority="HIGH",
            description="London Open - Forex, UK/EU stocks",
        ),
        SessionConfig(
            session=TradingSession.EUROPEAN,
            start_time=time(9, 0),
            end_time=time(14, 30),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.INDEX, InstrumentType.COMMODITY],
                "regions": [Region.UK, Region.EUROPE],
            },
            priority="SCANNING",
            description="European session",
        ),
        SessionConfig(
            session=TradingSession.US_PREMARKET,
            start_time=time(12, 0),
            end_time=time(14, 30),
            instrument_filter={
                "types": [],
                "regions": [Region.US],
            },
            priority="SCANNING",
            description="US Pre-market - Gap scanning",
        ),
        SessionConfig(
            session=TradingSession.US_OPEN,
            start_time=time(14, 30),
            end_time=time(15, 30),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.INDEX, InstrumentType.COMMODITY],
                "regions": [Region.US],
            },
            priority="HIGH",
            description="US Open - Maximum opportunity",
        ),
        SessionConfig(
            session=TradingSession.US_SESSION,
            start_time=time(15, 30),
            end_time=time(20, 0),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.INDEX, InstrumentType.COMMODITY],
                "regions": [Region.US],
            },
            priority="SCANNING",
            description="US Session",
        ),
        SessionConfig(
            session=TradingSession.POWER_HOUR,
            start_time=time(20, 0),
            end_time=time(21, 0),
            instrument_filter={
                "types": [InstrumentType.INDEX],
                "regions": [Region.US],
            },
            priority="HIGH",
            description="Power Hour - End of day momentum",
        ),
    ]

    # Session -> active edge types
    SESSION_EDGES: Dict[TradingSession, List[EdgeType]] = {
        TradingSession.ASIA: [
            EdgeType.ASIAN_RANGE,
            EdgeType.RSI_EXTREME,
            EdgeType.VWAP_DEVIATION,
        ],
        TradingSession.LONDON_OPEN: [
            EdgeType.LONDON_OPEN,
            EdgeType.ASIAN_RANGE,
            EdgeType.GAP_FILL,
            EdgeType.RSI_EXTREME,
        ],
        TradingSession.EUROPEAN: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.BOLLINGER_TOUCH,
        ],
        TradingSession.US_PREMARKET: [
            EdgeType.GAP_FILL,
            EdgeType.OVERNIGHT_PREMIUM,
        ],
        TradingSession.US_OPEN: [
            EdgeType.GAP_FILL,
            EdgeType.ORB,
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.TURN_OF_MONTH,
            EdgeType.INSIDER_CLUSTER,
        ],
        TradingSession.US_SESSION: [
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
            EdgeType.BOLLINGER_TOUCH,
            EdgeType.TURN_OF_MONTH,
        ],
        TradingSession.POWER_HOUR: [
            EdgeType.POWER_HOUR,
            EdgeType.VWAP_DEVIATION,
            EdgeType.RSI_EXTREME,
        ],
    }

    # DataProvider enum  ->  provider instance key
    _PROVIDER_KEY = {
        DataProvider.POLYGON: "polygon",
        DataProvider.OANDA: "oanda",
        DataProvider.IG: "ig",
        DataProvider.BINANCE: "crypto",
    }

    def __init__(self):
        self.settings = get_settings()
        self.registry: InstrumentRegistry = get_instrument_registry()

        # provider key -> live BaseDataProvider instance
        self._providers: Dict[str, BaseDataProvider] = {}
        self._provider_status: Dict[str, ProviderStatus] = {}
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to every provider that has credentials configured."""
        results: Dict[str, bool] = {}

        for prov_enum, key, factory in self._provider_factories():
            try:
                provider = factory()
                if provider is None:
                    logger.info("  [SKIP] %s (no config)", key.upper())
                    results[key] = False
                    continue

                connected = await provider.connect()
                self._providers[key] = provider

                inst_count = len(self.registry.get_by_provider(prov_enum))
                self._provider_status[key] = ProviderStatus(
                    name=key,
                    provider_enum=prov_enum,
                    connected=connected,
                    instrument_count=inst_count,
                )
                results[key] = connected

                tag = "OK" if connected else "FAIL"
                logger.info("  [%s] %s  (%d instruments)", tag, key.upper(), inst_count)

            except Exception as e:
                logger.error("  [ERR] %s: %s", key.upper(), e)
                results[key] = False
                self._provider_status[key] = ProviderStatus(
                    name=key,
                    provider_enum=prov_enum,
                    connected=False,
                    instrument_count=0,
                    error_count=1,
                    last_error=str(e),
                )

        self._connected = any(results.values())
        ok = sum(1 for v in results.values() if v)
        logger.info("Providers: %d/%d connected", ok, len(results))
        return results

    def _provider_factories(self):
        """Yield (DataProvider, key, factory_fn) for each provider."""
        s = self.settings

        # Polygon
        if s.polygon_api_key:
            def _polygon():
                from nexus.data.polygon import PolygonProvider
                return PolygonProvider()
            yield DataProvider.POLYGON, "polygon", _polygon

        # OANDA
        if s.oanda_api_key and s.oanda_account_id:
            def _oanda():
                from nexus.data.oanda import OANDAProvider
                return OANDAProvider(
                    api_key=s.oanda_api_key,
                    account_id=s.oanda_account_id,
                    practice=True,
                )
            yield DataProvider.OANDA, "oanda", _oanda

        # IG
        if s.ig_api_key:
            def _ig():
                from nexus.data.ig import IGProvider
                return IGProvider()
            yield DataProvider.IG, "ig", _ig

        # IBKR (no key; connects to TWS)
        def _ibkr():
            from nexus.data.ibkr import IBKRProvider
            return IBKRProvider()
        yield DataProvider.POLYGON, "ibkr", _ibkr  # fallback for US too

        # Binance (public API, no key)
        def _crypto():
            from nexus.data.crypto import BinanceProvider
            return BinanceProvider()
        yield DataProvider.BINANCE, "crypto", _crypto

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    def get_current_session(self) -> SessionConfig:
        """Determine current session from UTC clock."""
        now = datetime.now(timezone.utc)

        if now.weekday() >= 5:
            return SessionConfig(
                session=TradingSession.WEEKEND,
                start_time=time(0, 0),
                end_time=time(23, 59),
                instrument_filter={"types": [InstrumentType.CRYPTO], "regions": []},
                priority="SCANNING",
                description="Weekend - Crypto only",
            )

        utc_time = now.time()

        for cfg in self.SESSIONS:
            if cfg.start_time > cfg.end_time:
                # crosses midnight
                if utc_time >= cfg.start_time or utc_time < cfg.end_time:
                    return cfg
            else:
                if cfg.start_time <= utc_time < cfg.end_time:
                    return cfg

        return SessionConfig(
            session=TradingSession.OVERNIGHT,
            start_time=time(21, 0),
            end_time=time(22, 0),
            instrument_filter={
                "types": [InstrumentType.FOREX, InstrumentType.CRYPTO],
                "regions": [],
            },
            priority="LOW",
            description="Overnight",
        )

    # ------------------------------------------------------------------
    # Instrument queries (registry-backed)
    # ------------------------------------------------------------------

    def get_instruments_for_session(
        self, session: Optional[TradingSession] = None
    ) -> List[Instrument]:
        """
        Get ALL instruments active during a session from the full registry.

        An instrument matches if:
        - its type is in the filter's ``types`` list, OR
        - its region is in the filter's ``regions`` list
        """
        if session is None:
            cfg = self.get_current_session()
        else:
            cfg = next(
                (s for s in self.SESSIONS if s.session == session),
                self.get_current_session(),
            )
        return self._filter_instruments(cfg.instrument_filter)

    def _filter_instruments(self, filt: Dict[str, Any]) -> List[Instrument]:
        types: List[InstrumentType] = filt.get("types", [])
        regions: List[Region] = filt.get("regions", [])

        matched: Dict[str, Instrument] = {}
        for t in types:
            for inst in self.registry.get_by_type(t):
                matched[inst.symbol] = inst
        for r in regions:
            for inst in self.registry.get_by_region(r):
                matched[inst.symbol] = inst
        return list(matched.values())

    def get_instruments_by_provider(self, provider_key: str) -> List[Instrument]:
        """Get all instruments served by a provider key (polygon/oanda/ig/crypto)."""
        for prov_enum, key in self._PROVIDER_KEY.items():
            if key == provider_key:
                return self.registry.get_by_provider(prov_enum)
        return []

    def get_connected_instruments(self) -> List[Instrument]:
        """Get instruments whose provider is actually connected."""
        result: List[Instrument] = []
        for inst in self.registry.get_all():
            key = self._PROVIDER_KEY.get(inst.provider)
            if key and key in self._providers:
                status = self._provider_status.get(key)
                if status and status.connected:
                    result.append(inst)
        return result

    # ------------------------------------------------------------------
    # Provider routing
    # ------------------------------------------------------------------

    def get_provider_for_instrument(self, instrument: Instrument) -> Optional[BaseDataProvider]:
        """Get the live provider for an instrument based on its DataProvider field."""
        key = self._PROVIDER_KEY.get(instrument.provider)
        if key and key in self._providers:
            status = self._provider_status.get(key)
            if status and status.connected:
                return self._providers[key]
        return None

    def get_provider_for_symbol(self, symbol: str) -> Optional[BaseDataProvider]:
        """Lookup instrument by symbol, then route to its provider."""
        inst = self.registry.get(symbol)
        if inst:
            return self.get_provider_for_instrument(inst)
        return None

    # ------------------------------------------------------------------
    # Unified data access (pass-through to correct provider)
    # ------------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get quote, auto-routing to the correct provider."""
        inst = self.registry.get(symbol)
        if inst is None:
            return None
        provider = self.get_provider_for_instrument(inst)
        if provider is None:
            return None
        try:
            return await provider.get_quote(symbol)
        except Exception as e:
            logger.warning("Quote error %s: %s", symbol, e)
            self._record_error(inst.provider, str(e))
            return None

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Get bars, auto-routing to the correct provider."""
        inst = self.registry.get(symbol)
        if inst is None:
            return None
        provider = self.get_provider_for_instrument(inst)
        if provider is None:
            return None
        try:
            return await provider.get_bars(symbol, timeframe, limit, end_date)
        except Exception as e:
            logger.warning("Bars error %s: %s", symbol, e)
            self._record_error(inst.provider, str(e))
            return None

    def _record_error(self, prov_enum: DataProvider, error: str) -> None:
        key = self._PROVIDER_KEY.get(prov_enum)
        if key and key in self._provider_status:
            self._provider_status[key].error_count += 1
            self._provider_status[key].last_error = error

    # ------------------------------------------------------------------
    # Edge / scanner helpers
    # ------------------------------------------------------------------

    def get_edges_for_session(self) -> List[EdgeType]:
        """Get which edge scanners should run for the current session."""
        session = self.get_current_session()
        return self.SESSION_EDGES.get(session.session, [])

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Full orchestrator status report."""
        session = self.get_current_session()
        session_instruments = self.get_instruments_for_session()
        summary = self.registry.summary()

        return {
            "connected": self._connected,
            "current_session": session.session.value,
            "session_priority": session.priority,
            "session_description": session.description,
            "session_instruments": len(session_instruments),
            "total_registry": summary["total"],
            "active_edges": [e.value for e in self.get_edges_for_session()],
            "providers": {
                name: {
                    "connected": st.connected,
                    "instruments": st.instrument_count,
                    "errors": st.error_count,
                }
                for name, st in self._provider_status.items()
            },
            "by_type": summary["by_type"],
            "by_provider": summary["by_provider"],
            "currently_open": summary["currently_open"],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect_all(self) -> None:
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
                logger.info("Disconnected %s", name)
            except Exception as e:
                logger.error("Disconnect error %s: %s", name, e)
        self._connected = False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[DataOrchestrator] = None


def get_orchestrator() -> DataOrchestrator:
    """Get or create the global DataOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DataOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset singleton (for testing)."""
    global _orchestrator
    _orchestrator = None
