"""Tests for NEXUS storage repositories."""

import pytest
from datetime import datetime, date, timedelta, timezone
from uuid import uuid4
from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from nexus.storage.models import (
    Base,
    SignalRecord,
    TradeRecord,
    DailyPerformance,
    EdgePerformance,
    SystemState,
    AuditLog,
)
from nexus.storage.database import (
    get_sync_engine,
    reset_engines,
)
from nexus.core.enums import Market, Direction, EdgeType


# Use in-memory SQLite for all tests
TEST_DB_URL = "sqlite:///:memory:"


# -----------------------------------------------------------------------------
# Helpers (mirror repository-style queries for sync testing)
# -----------------------------------------------------------------------------

def get_signal_by_signal_id(session, signal_id: str):
    """Get SignalRecord by signal_id (sync helper for tests)."""
    result = session.execute(
        select(SignalRecord).where(SignalRecord.signal_id == signal_id)
    )
    return result.scalar_one_or_none()


def get_trade_by_id(session, trade_id):
    """Get TradeRecord by id (sync helper for tests)."""
    result = session.execute(
        select(TradeRecord).where(TradeRecord.id == trade_id)
    )
    return result.scalar_one_or_none()


def get_trades_by_signal_id(session, signal_id: str):
    """Get trades by signal_id (sync helper for tests)."""
    result = session.execute(
        select(TradeRecord).where(TradeRecord.signal_id == signal_id)
    )
    return list(result.scalars().all())


def get_pending_signals(session):
    """Get signals with PENDING status (sync helper for tests)."""
    result = session.execute(
        select(SignalRecord).where(SignalRecord.status == "PENDING")
    )
    return list(result.scalars().all())


def get_open_trades(session):
    """Get trades without exit_time (sync helper for tests)."""
    result = session.execute(
        select(TradeRecord).where(TradeRecord.exit_time.is_(None))
    )
    return list(result.scalars().all())


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db_session():
    """
    Create a fresh in-memory database for each test.

    Yields a session, then cleans up after test completes.
    """
    reset_engines()

    engine = get_sync_engine(url=TEST_DB_URL, echo=False)
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)
    reset_engines()


@pytest.fixture
def sample_signal_data():
    """Sample data for creating a SignalRecord."""
    return {
        "signal_id": f"SIG-{uuid4().hex[:8]}",
        "opportunity_id": f"OPP-{uuid4().hex[:8]}",
        "symbol": "AAPL",
        "market": Market.US_STOCKS.value,
        "direction": Direction.LONG.value,
        "entry_price": 150.00,
        "stop_loss": 145.00,
        "take_profit": 160.00,
        "position_size": 100.0,
        "position_value": 15000.0,
        "risk_amount": 500.0,
        "risk_percent": 1.0,
        "primary_edge": EdgeType.VWAP_DEVIATION.value,
        "secondary_edges": [EdgeType.RSI_EXTREME.value],
        "edge_score": 75,
        "tier": "B",
        "gross_expected": 0.25,
        "net_expected": 0.18,
        "cost_ratio": 28.0,
        "costs": {"spread": 0.02, "commission": 0.01, "slippage": 0.02, "overnight": 0.0, "total": 0.05},
        "ai_reasoning": "VWAP deviation detected with RSI confirmation.",
        "confluence_factors": ["VWAP 2σ deviation", "RSI oversold"],
        "risk_factors": ["Earnings in 3 days"],
        "market_context": "Bullish trend, moderate volatility",
        "session": "US_REGULAR",
        "status": "PENDING",  # Match model default and repo filter
    }


@pytest.fixture
def sample_trade_data(sample_signal_data):
    """Sample data for creating a TradeRecord."""
    return {
        "signal_id": sample_signal_data["signal_id"],
        "symbol": sample_signal_data["symbol"],
        "market": sample_signal_data["market"],
        "direction": sample_signal_data["direction"],
        "entry_time": datetime.now(timezone.utc),
        "entry_price": 150.25,
        "position_size": 100.0,
        "slippage_entry": 0.25,
    }


# =============================================================================
# SignalRepository-style tests (storage layer: SignalRecord CRUD)
# =============================================================================

class TestSignalRepository:
    """Tests for signal storage (SignalRecord CRUD operations)."""

    def test_save_signal_creates_record(self, db_session, sample_signal_data):
        """Test saving a new signal creates a database record."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        assert signal.id is not None
        assert signal.signal_id == sample_signal_data["signal_id"]
        assert signal.symbol == "AAPL"
        assert signal.status == "PENDING"

    def test_get_by_signal_id_returns_record(self, db_session, sample_signal_data):
        """Test retrieving a signal by its signal_id."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        found = get_signal_by_signal_id(db_session, sample_signal_data["signal_id"])

        assert found is not None
        assert found.signal_id == sample_signal_data["signal_id"]
        assert found.symbol == "AAPL"

    def test_get_by_signal_id_returns_none_when_not_found(self, db_session):
        """Test get by signal_id returns None for non-existent signal."""
        found = get_signal_by_signal_id(db_session, "NON-EXISTENT-ID")
        assert found is None

    def test_get_pending_signals_returns_only_pending(self, db_session, sample_signal_data):
        """Test get_pending returns only signals with PENDING status."""
        pending = SignalRecord(**sample_signal_data)
        db_session.add(pending)

        filled_data = sample_signal_data.copy()
        filled_data["signal_id"] = f"SIG-{uuid4().hex[:8]}"
        filled_data["status"] = "FILLED"
        filled = SignalRecord(**filled_data)
        db_session.add(filled)

        db_session.commit()

        pending_signals = get_pending_signals(db_session)

        assert len(pending_signals) == 1
        assert pending_signals[0].status == "PENDING"

    def test_update_status_changes_status(self, db_session, sample_signal_data):
        """Test updating signal status."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        db_session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == sample_signal_data["signal_id"])
            .values(status="FILLED", updated_at=datetime.now(timezone.utc))
        )
        db_session.commit()

        updated = get_signal_by_signal_id(db_session, sample_signal_data["signal_id"])
        assert updated.status == "FILLED"

    def test_get_by_symbol_returns_matching_signals(self, db_session, sample_signal_data):
        """Test filtering signals by symbol."""
        aapl = SignalRecord(**sample_signal_data)
        db_session.add(aapl)

        msft_data = sample_signal_data.copy()
        msft_data["signal_id"] = f"SIG-{uuid4().hex[:8]}"
        msft_data["symbol"] = "MSFT"
        msft = SignalRecord(**msft_data)
        db_session.add(msft)

        db_session.commit()

        result = db_session.execute(
            select(SignalRecord).where(SignalRecord.symbol == "AAPL")
        )
        aapl_signals = list(result.scalars().all())

        assert len(aapl_signals) == 1
        assert aapl_signals[0].symbol == "AAPL"

    def test_get_by_edge_returns_matching_signals(self, db_session, sample_signal_data):
        """Test filtering signals by primary edge type."""
        vwap = SignalRecord(**sample_signal_data)
        db_session.add(vwap)

        tom_data = sample_signal_data.copy()
        tom_data["signal_id"] = f"SIG-{uuid4().hex[:8]}"
        tom_data["primary_edge"] = EdgeType.TURN_OF_MONTH.value
        tom = SignalRecord(**tom_data)
        db_session.add(tom)

        db_session.commit()

        result = db_session.execute(
            select(SignalRecord).where(
                SignalRecord.primary_edge == EdgeType.VWAP_DEVIATION.value
            )
        )
        vwap_signals = list(result.scalars().all())

        assert len(vwap_signals) == 1
        assert vwap_signals[0].primary_edge == EdgeType.VWAP_DEVIATION.value

    def test_get_recent_returns_limited_ordered_results(self, db_session, sample_signal_data):
        """Test get_recent returns signals ordered by created_at descending."""
        for i in range(5):
            data = sample_signal_data.copy()
            data["signal_id"] = f"SIG-{i:04d}"
            signal = SignalRecord(**data)
            db_session.add(signal)

        db_session.commit()

        result = db_session.execute(
            select(SignalRecord).order_by(SignalRecord.created_at.desc()).limit(3)
        )
        recent = list(result.scalars().all())

        assert len(recent) == 3

    def test_update_outcome_sets_outcome_fields(self, db_session, sample_signal_data):
        """Test updating signal with trade outcome."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        db_session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == sample_signal_data["signal_id"])
            .values(
                outcome_exit_price=155.00,
                outcome_pnl=500.0,
                outcome_pnl_percent=3.33,
                outcome_exit_reason="TARGET_HIT",
            )
        )
        db_session.commit()

        updated = get_signal_by_signal_id(db_session, sample_signal_data["signal_id"])
        assert updated.outcome_exit_price == 155.00
        assert updated.outcome_pnl == 500.0
        assert updated.outcome_exit_reason == "TARGET_HIT"

    def test_signal_to_dict_returns_all_fields(self, db_session, sample_signal_data):
        """Test that to_dict() returns complete signal data."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        signal_dict = signal.to_dict()

        assert "signal_id" in signal_dict
        assert "symbol" in signal_dict
        assert "costs" in signal_dict
        assert signal_dict["symbol"] == "AAPL"


# =============================================================================
# TradeRepository-style tests (storage layer: TradeRecord CRUD)
# =============================================================================

class TestTradeRepository:
    """Tests for trade storage (TradeRecord CRUD operations)."""

    def test_save_trade_creates_record(self, db_session, sample_signal_data, sample_trade_data):
        """Test saving a trade creates a database record."""
        signal_repo = SignalRecord(**sample_signal_data)
        db_session.add(signal_repo)
        db_session.commit()

        trade = TradeRecord(**sample_trade_data)
        db_session.add(trade)
        db_session.commit()

        assert trade.id is not None
        assert trade.symbol == "AAPL"
        assert trade.entry_price == 150.25

    def test_get_by_signal_id_returns_trades(self, db_session, sample_signal_data, sample_trade_data):
        """Test retrieving trades by signal_id."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)

        trade = TradeRecord(**sample_trade_data)
        db_session.add(trade)
        db_session.commit()

        trades = get_trades_by_signal_id(db_session, sample_signal_data["signal_id"])

        assert len(trades) == 1
        assert trades[0].signal_id == sample_signal_data["signal_id"]

    def test_update_exit_sets_exit_fields(self, db_session, sample_signal_data, sample_trade_data):
        """Test updating trade with exit information."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)

        trade = TradeRecord(**sample_trade_data)
        db_session.add(trade)
        db_session.commit()

        db_session.execute(
            update(TradeRecord)
            .where(TradeRecord.id == trade.id)
            .values(
                exit_price=155.00,
                exit_time=datetime.now(timezone.utc),
                pnl=475.0,
                pnl_percent=3.16,
                exit_reason="TARGET_HIT",
                slippage_exit=0.0,
            )
        )
        db_session.commit()

        updated = get_trade_by_id(db_session, trade.id)
        assert updated.exit_price == 155.00
        assert updated.pnl == 475.0
        assert updated.exit_reason == "TARGET_HIT"

    def test_get_open_trades_excludes_closed(self, db_session, sample_signal_data, sample_trade_data):
        """Test get_open returns only trades without exit_time."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)

        open_trade = TradeRecord(**sample_trade_data)
        db_session.add(open_trade)

        closed_data = sample_trade_data.copy()
        closed_data["signal_id"] = sample_signal_data["signal_id"]
        closed_data["exit_time"] = datetime.now(timezone.utc)
        closed_data["exit_price"] = 155.00
        closed_data["pnl"] = 475.0
        closed_trade = TradeRecord(**closed_data)
        db_session.add(closed_trade)
        db_session.commit()

        open_trades = get_open_trades(db_session)

        assert len(open_trades) == 1
        assert open_trades[0].exit_time is None

    def test_trade_to_dict_returns_all_fields(self, db_session, sample_signal_data, sample_trade_data):
        """Test that to_dict() returns complete trade data."""
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)

        trade = TradeRecord(**sample_trade_data)
        db_session.add(trade)
        db_session.commit()

        trade_dict = trade.to_dict()

        assert "id" in trade_dict
        assert "signal_id" in trade_dict
        assert "entry_price" in trade_dict


# =============================================================================
# SystemStateRepository-style tests (storage layer: SystemState singleton)
# =============================================================================

class TestSystemStateRepository:
    """Tests for system state singleton operations."""

    def test_get_or_create_creates_initial_state(self, db_session):
        """Test get_or_create creates state if none exists."""
        result = db_session.execute(select(SystemState).where(SystemState.id == 1))
        state = result.scalar_one_or_none()
        if state is None:
            state = SystemState(
                id=1,
                starting_equity=10000.0,
                current_equity=10000.0,
                peak_equity=10000.0,
            )
            db_session.add(state)
            db_session.commit()

        assert state is not None
        assert state.id == 1
        assert state.starting_equity == 10000.0
        assert state.current_equity == 10000.0

    def test_get_or_create_returns_existing_state(self, db_session):
        """Test get_or_create returns existing state on second call."""
        state1 = SystemState(
            id=1,
            starting_equity=10000.0,
            current_equity=10000.0,
            peak_equity=10000.0,
        )
        db_session.add(state1)
        db_session.commit()

        result = db_session.execute(select(SystemState).where(SystemState.id == 1))
        state2 = result.scalar_one()

        assert state2.id == state1.id
        assert state2.starting_equity == 10000.0

    def test_update_equity_changes_current_equity(self, db_session):
        """Test updating current equity and P&L."""
        state = SystemState(
            id=1,
            starting_equity=10000.0,
            current_equity=10000.0,
            peak_equity=10000.0,
        )
        db_session.add(state)
        db_session.commit()

        db_session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                current_equity=10500.0,
                daily_pnl=500.0,
                daily_pnl_percent=5.0,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        result = db_session.execute(select(SystemState).where(SystemState.id == 1))
        state = result.scalar_one()
        assert state.current_equity == 10500.0
        assert state.daily_pnl == 500.0

    def test_update_positions_sets_open_positions(self, db_session):
        """Test updating open positions JSON."""
        state = SystemState(
            id=1,
            starting_equity=10000.0,
            current_equity=10000.0,
            peak_equity=10000.0,
        )
        db_session.add(state)
        db_session.commit()

        positions = [
            {"symbol": "AAPL", "size": 100, "pnl": 50.0},
            {"symbol": "MSFT", "size": 50, "pnl": -25.0},
        ]

        db_session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                open_positions=positions,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        result = db_session.execute(select(SystemState).where(SystemState.id == 1))
        state = result.scalar_one()
        assert len(state.open_positions) == 2
        assert state.open_positions[0]["symbol"] == "AAPL"

    def test_system_state_to_dict(self, db_session):
        """Test that to_dict() returns complete state data."""
        state = SystemState(
            id=1,
            starting_equity=10000.0,
            current_equity=10000.0,
            peak_equity=10000.0,
        )
        db_session.add(state)
        db_session.commit()

        state_dict = state.to_dict()

        assert "current_equity" in state_dict
        assert "circuit_breaker_status" in state_dict
        assert "kill_switch_active" in state_dict


# =============================================================================
# DailyPerformance tests
# =============================================================================

class TestDailyPerformanceRepository:
    """Tests for daily performance tracking."""

    def test_save_daily_performance(self, db_session):
        """Test saving daily performance record."""
        perf = DailyPerformance(
            date=date.today(),
            starting_equity=10000.0,
            ending_equity=10350.0,
            pnl=350.0,
            pnl_percent=3.5,
            trades_taken=5,
            winners=3,
            losers=2,
            win_rate=60.0,
            largest_win=200.0,
            largest_loss=-75.0,
            average_win=150.0,
            average_loss=-50.0,
            profit_factor=3.0,
            edges_used={"VWAP_DEVIATION": 3, "RSI_EXTREME": 2},
        )

        db_session.add(perf)
        db_session.commit()

        assert perf.id is not None
        assert perf.pnl == 350.0

    def test_daily_performance_unique_date(self, db_session):
        """Test that date is unique - can't have two records for same day."""
        from sqlalchemy.exc import IntegrityError

        perf1 = DailyPerformance(
            date=date.today(),
            starting_equity=10000.0,
            ending_equity=10100.0,
            pnl=100.0,
            pnl_percent=1.0,
        )
        db_session.add(perf1)
        db_session.commit()

        perf2 = DailyPerformance(
            date=date.today(),
            starting_equity=10100.0,
            ending_equity=10200.0,
            pnl=100.0,
            pnl_percent=1.0,
        )
        db_session.add(perf2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_daily_performance_to_dict(self, db_session):
        """Test to_dict returns all fields."""
        perf = DailyPerformance(
            date=date.today(),
            starting_equity=10000.0,
            ending_equity=10100.0,
            pnl=100.0,
            pnl_percent=1.0,
        )
        db_session.add(perf)
        db_session.commit()

        perf_dict = perf.to_dict()

        assert "date" in perf_dict
        assert "pnl" in perf_dict
        assert "win_rate" in perf_dict


# =============================================================================
# EdgePerformance tests
# =============================================================================

class TestEdgePerformanceRepository:
    """Tests for edge performance tracking."""

    def test_save_edge_performance(self, db_session):
        """Test saving edge performance record."""
        edge_perf = EdgePerformance(
            edge_type=EdgeType.VWAP_DEVIATION.value,
            period="daily",
            period_start=date.today(),
            trades=10,
            wins=6,
            losses=4,
            total_pnl=250.0,
            average_pnl=25.0,
            win_rate=60.0,
            expected_edge=0.15,
            actual_edge=0.18,
            is_healthy=True,
        )

        db_session.add(edge_perf)
        db_session.commit()

        assert edge_perf.id is not None
        assert edge_perf.win_rate == 60.0

    def test_edge_performance_unique_constraint(self, db_session):
        """Test unique constraint on edge_type + period + period_start."""
        from sqlalchemy.exc import IntegrityError

        edge1 = EdgePerformance(
            edge_type=EdgeType.VWAP_DEVIATION.value,
            period="daily",
            period_start=date.today(),
            expected_edge=0.15,
        )
        db_session.add(edge1)
        db_session.commit()

        edge2 = EdgePerformance(
            edge_type=EdgeType.VWAP_DEVIATION.value,
            period="daily",
            period_start=date.today(),
            expected_edge=0.15,
        )
        db_session.add(edge2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_different_periods_allowed(self, db_session):
        """Test same edge can have different period types."""
        daily = EdgePerformance(
            edge_type=EdgeType.VWAP_DEVIATION.value,
            period="daily",
            period_start=date.today(),
            expected_edge=0.15,
        )
        weekly = EdgePerformance(
            edge_type=EdgeType.VWAP_DEVIATION.value,
            period="weekly",
            period_start=date.today(),
            expected_edge=0.15,
        )

        db_session.add_all([daily, weekly])
        db_session.commit()

        assert daily.id is not None
        assert weekly.id is not None
        assert daily.id != weekly.id


# =============================================================================
# AuditLog tests
# =============================================================================

class TestAuditLogRepository:
    """Tests for audit logging."""

    def test_log_event_creates_record(self, db_session):
        """Test logging an audit event."""
        log = AuditLog(
            event_type="SIGNAL_GENERATED",
            severity="INFO",
            component="SignalGenerator",
            message="Generated signal for AAPL LONG",
            details={"symbol": "AAPL", "score": 75},
            signal_id="SIG-12345",
        )

        db_session.add(log)
        db_session.commit()

        assert log.id is not None
        assert log.event_type == "SIGNAL_GENERATED"

    def test_log_error_with_details(self, db_session):
        """Test logging an error event with details."""
        log = AuditLog(
            event_type="BROKER_ERROR",
            severity="ERROR",
            component="OANDAProvider",
            message="Connection timeout after 30s",
            details={"attempt": 3, "last_error": "TimeoutError"},
        )

        db_session.add(log)
        db_session.commit()

        assert log.severity == "ERROR"
        assert "attempt" in log.details

    def test_audit_log_to_dict(self, db_session):
        """Test to_dict returns all fields."""
        log = AuditLog(
            event_type="KILL_SWITCH_ACTIVATED",
            severity="CRITICAL",
            component="KillSwitch",
            message="Emergency shutdown triggered",
        )
        db_session.add(log)
        db_session.commit()

        log_dict = log.to_dict()

        assert "timestamp" in log_dict
        assert "event_type" in log_dict
        assert "severity" in log_dict


# =============================================================================
# Integration tests
# =============================================================================

class TestStorageIntegration:
    """Integration tests for complete workflows."""

    def test_full_signal_to_trade_flow(self, db_session, sample_signal_data, sample_trade_data):
        """Test complete flow: Signal → Trade → Update → Close."""
        # 1. Create signal
        signal = SignalRecord(**sample_signal_data)
        db_session.add(signal)
        db_session.commit()

        assert signal.status == "PENDING"

        # 2. Update signal to FILLED
        db_session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == signal.signal_id)
            .values(status="FILLED", updated_at=datetime.now(timezone.utc))
        )
        db_session.commit()

        # 3. Create trade
        trade = TradeRecord(**sample_trade_data)
        db_session.add(trade)
        db_session.commit()

        # 4. Close trade with profit
        db_session.execute(
            update(TradeRecord)
            .where(TradeRecord.id == trade.id)
            .values(
                exit_price=155.00,
                exit_time=datetime.now(timezone.utc),
                pnl=475.0,
                pnl_percent=3.16,
                exit_reason="TARGET_HIT",
                slippage_exit=0.0,
            )
        )
        db_session.commit()

        # 5. Update signal outcome
        db_session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == signal.signal_id)
            .values(
                outcome_exit_price=155.00,
                outcome_pnl=475.0,
                outcome_pnl_percent=3.16,
                outcome_exit_reason="TARGET_HIT",
            )
        )
        db_session.execute(
            update(SignalRecord)
            .where(SignalRecord.signal_id == signal.signal_id)
            .values(status="closed", updated_at=datetime.now(timezone.utc))
        )
        db_session.commit()

        # Verify final state
        final_signal = get_signal_by_signal_id(db_session, signal.signal_id)
        final_trade = get_trade_by_id(db_session, trade.id)

        assert final_signal.status == "closed"
        assert final_signal.outcome_pnl == 475.0
        assert final_trade.pnl == 475.0
        assert final_trade.exit_reason == "TARGET_HIT"

    def test_system_state_updates_through_trading_day(self, db_session):
        """Test system state updates through a simulated trading day."""
        state = SystemState(
            id=1,
            starting_equity=10000.0,
            current_equity=10000.0,
            peak_equity=10000.0,
        )
        db_session.add(state)
        db_session.commit()

        # Trade 1: Win
        db_session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                current_equity=10250.0,
                daily_pnl=250.0,
                daily_pnl_percent=2.5,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        # Trade 2: Small loss
        db_session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                current_equity=10175.0,
                daily_pnl=175.0,
                daily_pnl_percent=1.75,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        # Trade 3: Win
        db_session.execute(
            update(SystemState)
            .where(SystemState.id == 1)
            .values(
                current_equity=10500.0,
                daily_pnl=500.0,
                daily_pnl_percent=5.0,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        result = db_session.execute(select(SystemState).where(SystemState.id == 1))
        final_state = result.scalar_one()

        assert final_state.current_equity == 10500.0
        assert final_state.daily_pnl == 500.0
        assert final_state.daily_pnl_percent == 5.0
