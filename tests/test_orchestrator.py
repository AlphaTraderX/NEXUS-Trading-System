"""Tests for the full-registry DataOrchestrator."""

import pytest
from datetime import time

from nexus.data.orchestrator import (
    DataOrchestrator,
    TradingSession,
    SessionConfig,
    get_orchestrator,
    reset_orchestrator,
)
from nexus.data.instruments import InstrumentType, Region, DataProvider
from nexus.core.enums import EdgeType


class TestRegistryCounts:
    """Verify the orchestrator loads the full 426-instrument registry."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_total_instruments_400_plus(self):
        total = self.orch.registry.total_count
        assert total >= 400, f"Only {total} instruments, expected 400+"

    def test_us_stocks(self):
        us = self.orch.registry.get_by_region(Region.US)
        assert len(us) >= 150, f"US stocks: {len(us)}"

    def test_uk_stocks(self):
        uk = self.orch.registry.get_by_region(Region.UK)
        assert len(uk) >= 50, f"UK stocks: {len(uk)}"

    def test_europe_stocks(self):
        eu = self.orch.registry.get_by_region(Region.EUROPE)
        assert len(eu) >= 30, f"EU stocks: {len(eu)}"

    def test_asia_stocks(self):
        asia = (
            self.orch.registry.get_by_region(Region.ASIA_JAPAN)
            + self.orch.registry.get_by_region(Region.ASIA_HK)
            + self.orch.registry.get_by_region(Region.ASIA_AU)
        )
        assert len(asia) >= 40, f"Asia stocks: {len(asia)}"

    def test_forex_pairs(self):
        fx = self.orch.registry.get_by_type(InstrumentType.FOREX)
        assert len(fx) >= 20, f"Forex: {len(fx)}"

    def test_indices(self):
        idx = self.orch.registry.get_by_type(InstrumentType.INDEX)
        assert len(idx) >= 10, f"Indices: {len(idx)}"

    def test_commodities(self):
        com = self.orch.registry.get_by_type(InstrumentType.COMMODITY)
        assert len(com) >= 10, f"Commodities: {len(com)}"

    def test_crypto(self):
        cry = self.orch.registry.get_by_type(InstrumentType.CRYPTO)
        assert len(cry) >= 15, f"Crypto: {len(cry)}"


class TestProviderMapping:
    """Verify instruments map to correct providers."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_polygon_serves_us_stocks(self):
        polygon = self.orch.registry.get_by_provider(DataProvider.POLYGON)
        assert len(polygon) >= 150
        # Spot check
        aapl = self.orch.registry.get("AAPL")
        assert aapl is not None
        assert aapl.provider == DataProvider.POLYGON

    def test_oanda_serves_forex(self):
        oanda = self.orch.registry.get_by_provider(DataProvider.OANDA)
        assert len(oanda) >= 20
        eur = self.orch.registry.get("EUR_USD")
        assert eur is not None
        assert eur.provider == DataProvider.OANDA

    def test_ig_serves_uk_eu_asia(self):
        ig = self.orch.registry.get_by_provider(DataProvider.IG)
        assert len(ig) >= 100  # UK + EU + Asia + Indices + Commodities
        azn = self.orch.registry.get("AZN.L")
        assert azn is not None
        assert azn.provider == DataProvider.IG

    def test_binance_serves_crypto(self):
        binance = self.orch.registry.get_by_provider(DataProvider.BINANCE)
        assert len(binance) >= 15
        btc = self.orch.registry.get("BTC_USD")
        assert btc is not None
        assert btc.provider == DataProvider.BINANCE


class TestSessionDetection:
    """Verify session determination."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_sessions_defined(self):
        sessions = [s.session for s in self.orch.SESSIONS]
        assert TradingSession.LONDON_OPEN in sessions
        assert TradingSession.US_OPEN in sessions
        assert TradingSession.POWER_HOUR in sessions
        assert TradingSession.ASIA in sessions

    def test_current_session_valid(self):
        cfg = self.orch.get_current_session()
        assert isinstance(cfg, SessionConfig)
        assert cfg.session in TradingSession

    def test_session_has_instruments(self):
        """Every non-overnight session should have instruments."""
        for sess_cfg in self.orch.SESSIONS:
            instruments = self.orch.get_instruments_for_session(sess_cfg.session)
            assert len(instruments) > 0, f"{sess_cfg.session.value} has 0 instruments"


class TestSessionInstrumentCounts:
    """Verify session-specific instrument lists are substantial."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_asia_includes_asia_and_forex(self):
        insts = self.orch.get_instruments_for_session(TradingSession.ASIA)
        types = {i.instrument_type for i in insts}
        assert InstrumentType.FOREX in types
        regions = {i.region for i in insts}
        assert Region.ASIA_JAPAN in regions or Region.ASIA_HK in regions

    def test_london_open_includes_uk_forex(self):
        insts = self.orch.get_instruments_for_session(TradingSession.LONDON_OPEN)
        types = {i.instrument_type for i in insts}
        regions = {i.region for i in insts}
        assert InstrumentType.FOREX in types
        assert Region.UK in regions

    def test_us_open_includes_us_stocks(self):
        insts = self.orch.get_instruments_for_session(TradingSession.US_OPEN)
        regions = {i.region for i in insts}
        assert Region.US in regions
        # Should have 190+ US stocks + forex + indices + commodities
        assert len(insts) >= 200, f"US Open: only {len(insts)} instruments"

    def test_us_open_has_most_instruments(self):
        us_open = self.orch.get_instruments_for_session(TradingSession.US_OPEN)
        asia = self.orch.get_instruments_for_session(TradingSession.ASIA)
        assert len(us_open) > len(asia)


class TestEdgeMapping:
    """Verify edges are mapped to sessions."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_edges_for_us_open(self):
        edges = self.orch.SESSION_EDGES[TradingSession.US_OPEN]
        assert EdgeType.GAP_FILL in edges
        assert EdgeType.ORB in edges
        assert EdgeType.VWAP_DEVIATION in edges

    def test_edges_for_london(self):
        edges = self.orch.SESSION_EDGES[TradingSession.LONDON_OPEN]
        assert EdgeType.LONDON_OPEN in edges

    def test_edges_for_power_hour(self):
        edges = self.orch.SESSION_EDGES[TradingSession.POWER_HOUR]
        assert EdgeType.POWER_HOUR in edges

    def test_get_edges_returns_list(self):
        edges = self.orch.get_edges_for_session()
        assert isinstance(edges, list)
        for e in edges:
            assert isinstance(e, EdgeType)


class TestStatusReport:
    """Verify status report completeness."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_status_keys(self):
        status = self.orch.get_status()
        assert "connected" in status
        assert "current_session" in status
        assert "session_instruments" in status
        assert "total_registry" in status
        assert "by_type" in status
        assert "by_provider" in status
        assert "active_edges" in status

    def test_total_registry_matches(self):
        status = self.orch.get_status()
        assert status["total_registry"] == self.orch.registry.total_count


class TestSymbolLookup:
    """Verify symbol-based provider routing."""

    def setup_method(self):
        reset_orchestrator()
        self.orch = DataOrchestrator()

    def test_known_symbols_in_registry(self):
        for symbol in ["AAPL", "MSFT", "EUR_USD", "BTC_USD", "AZN.L", "UK100"]:
            inst = self.orch.registry.get(symbol)
            assert inst is not None, f"{symbol} not in registry"

    def test_provider_for_symbol_without_connection(self):
        """Without connected providers, should return None."""
        provider = self.orch.get_provider_for_symbol("AAPL")
        assert provider is None  # Nothing connected yet
