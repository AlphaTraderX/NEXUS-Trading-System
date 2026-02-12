"""
NEXUS Execution Layer

Coordinates signal generation, order management, position tracking,
trade execution, and reconciliation.
"""

from nexus.execution.signal_generator import SignalGenerator
from nexus.execution.cooldown_manager import CooldownManager, get_cooldown_manager
from nexus.execution.position_manager import (
    PositionManager,
    Position,
    PositionStatus,
    PortfolioMetrics,
)
from nexus.execution.order_manager import (
    OrderManager,
    Order,
    OrderFill,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderPurpose,
    SlippageStats,
)
from nexus.execution.trade_executor import (
    TradeExecutor,
    BaseBrokerExecutor,
    PaperBrokerExecutor,
    BrokerConfig,
    BrokerType,
    ExecutionResult,
    BrokerPosition,
    BrokerAccount,
    create_paper_executor,
)
from nexus.execution.reconciliation import (
    ReconciliationEngine,
    ReconciliationReport,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ReconciliationAction,
)

__all__ = [
    # Signal Generator
    "SignalGenerator",
    # Cooldown Manager
    "CooldownManager",
    "get_cooldown_manager",
    # Position Manager
    "PositionManager",
    "Position",
    "PositionStatus",
    "PortfolioMetrics",
    # Order Manager
    "OrderManager",
    "Order",
    "OrderFill",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "OrderPurpose",
    "SlippageStats",
    # Trade Executor
    "TradeExecutor",
    "BaseBrokerExecutor",
    "PaperBrokerExecutor",
    "BrokerConfig",
    "BrokerType",
    "ExecutionResult",
    "BrokerPosition",
    "BrokerAccount",
    "create_paper_executor",
    # Reconciliation
    "ReconciliationEngine",
    "ReconciliationReport",
    "Discrepancy",
    "DiscrepancyType",
    "DiscrepancySeverity",
    "ReconciliationAction",
]
