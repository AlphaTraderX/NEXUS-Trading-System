"""Tests for StorageService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from nexus.storage.service import StorageService, get_storage_service
from nexus.core.enums import SignalStatus, EdgeType, Direction, Market

UTC = timezone.utc


class TestStorageService:
    """Test StorageService functionality."""

    @pytest.fixture
    def storage(self):
        return StorageService()

    @pytest.fixture
    def sample_signal(self):
        from nexus.core.models import NexusSignal
        from nexus.intelligence.cost_engine import CostBreakdown

        return NexusSignal(
            signal_id="test-123",
            created_at=datetime.now(UTC),
            opportunity_id="opp-123",
            symbol="AAPL",
            market=Market.US_STOCKS,
            direction=Direction.LONG,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            position_size=10,
            position_value=1500.0,
            risk_amount=50.0,
            risk_percent=1.0,
            primary_edge=EdgeType.VWAP_DEVIATION,
            secondary_edges=[],
            edge_score=75,
            tier="B",
            gross_expected=0.15,
            costs=CostBreakdown(),
            net_expected=0.10,
            cost_ratio=33.0,
            ai_reasoning="Test reasoning",
            confluence_factors=["Test factor"],
            risk_factors=[],
            market_context="Test context",
            session="US",
            valid_until=datetime.now(UTC),
            status=SignalStatus.PENDING,
        )

    def test_singleton(self):
        """Test singleton pattern."""
        s1 = get_storage_service()
        s2 = get_storage_service()
        assert s1 is s2

    @pytest.mark.asyncio
    async def test_not_initialized_by_default(self, storage):
        """Test storage is not initialized by default."""
        assert storage._initialized is False

    @pytest.mark.asyncio
    @patch("nexus.storage.service.init_db_async")
    async def test_initialize_success(self, mock_init, storage):
        """Test successful initialization."""
        mock_init.return_value = None

        result = await storage.initialize()

        assert result is True
        assert storage._initialized is True
        mock_init.assert_called_once()

    @pytest.mark.asyncio
    @patch("nexus.storage.service.init_db_async")
    async def test_initialize_failure(self, mock_init, storage):
        """Test initialization failure."""
        mock_init.side_effect = Exception("DB connection failed")

        result = await storage.initialize()

        assert result is False
        assert storage._initialized is False
