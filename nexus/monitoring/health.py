"""
NEXUS System Health Checks

Checks all critical components and returns unified health status.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


class HealthChecker:
    """
    Unified health checker for all NEXUS components.

    Checks:
    - Database connectivity
    - Broker connections (IBKR, IG, OANDA)
    - Data feed freshness
    - Risk system status
    - Alert delivery channels
    """

    def __init__(self):
        self._results: Dict[str, ComponentHealth] = {}
        self._check_interval = 60  # seconds
        self._last_full_check: Optional[datetime] = None

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        try:
            from nexus.storage.database import check_database_health_async
            start = time.perf_counter()
            result = await check_database_health_async()
            latency = (time.perf_counter() - start) * 1000

            if result.get("healthy"):
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="Connected",
                )
            else:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message=result.get("error", "Unknown error"),
                )
        except Exception as e:
            logger.exception("Database health check failed")
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_broker(self, broker_name: str, broker_instance: Any) -> ComponentHealth:
        """Check broker connectivity."""
        try:
            start = time.perf_counter()

            if hasattr(broker_instance, "is_connected"):
                connected = (
                    await broker_instance.is_connected()
                    if asyncio.iscoroutinefunction(broker_instance.is_connected)
                    else broker_instance.is_connected()
                )
            else:
                connected = False

            latency = (time.perf_counter() - start) * 1000

            if connected:
                return ComponentHealth(
                    name=f"broker_{broker_name}",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="Connected",
                )
            else:
                return ComponentHealth(
                    name=f"broker_{broker_name}",
                    status=HealthStatus.UNHEALTHY,
                    message="Not connected",
                )
        except Exception as e:
            logger.exception("Broker health check failed for %s", broker_name)
            return ComponentHealth(
                name=f"broker_{broker_name}",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_data_feed(
        self, provider_name: str, last_quote_time: Optional[datetime]
    ) -> ComponentHealth:
        """Check data feed freshness."""
        if last_quote_time is None:
            return ComponentHealth(
                name=f"feed_{provider_name}",
                status=HealthStatus.UNKNOWN,
                message="No data received yet",
            )

        age = datetime.utcnow() - last_quote_time

        if age < timedelta(seconds=30):
            status = HealthStatus.HEALTHY
            message = f"Fresh ({age.seconds}s old)"
        elif age < timedelta(minutes=5):
            status = HealthStatus.DEGRADED
            message = f"Stale ({age.seconds}s old)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Very stale ({age.seconds}s old)"

        return ComponentHealth(
            name=f"feed_{provider_name}",
            status=status,
            message=message,
            details={"age_seconds": age.total_seconds()},
        )

    async def check_risk_system(
        self, circuit_breaker: Any, kill_switch: Any
    ) -> ComponentHealth:
        """Check risk management system status."""
        details: Dict[str, Any] = {}
        status = HealthStatus.HEALTHY
        messages: List[str] = []

        if circuit_breaker:
            cb_status = getattr(circuit_breaker, "current_status", {})
            details["circuit_breaker"] = cb_status.get("status", "unknown")
            if cb_status.get("status") == "STOPPED":
                status = HealthStatus.DEGRADED
                messages.append("Circuit breaker triggered")

        if kill_switch:
            ks_active = getattr(kill_switch, "is_active", False)
            details["kill_switch_active"] = ks_active
            if ks_active:
                status = HealthStatus.UNHEALTHY
                messages.append("Kill switch active")

        return ComponentHealth(
            name="risk_system",
            status=status,
            message="; ".join(messages) if messages else "Operational",
            details=details,
        )

    async def check_alert_channels(self, alert_manager: Any) -> ComponentHealth:
        """Check alert delivery channels."""
        if not alert_manager:
            return ComponentHealth(
                name="alerts",
                status=HealthStatus.UNKNOWN,
                message="Alert manager not configured",
            )

        try:
            channels = getattr(alert_manager, "_channels", {})
            active = [
                name
                for name, ch in channels.items()
                if getattr(ch, "is_configured", getattr(ch, "enabled", True))
            ]

            if len(active) >= 2:
                status = HealthStatus.HEALTHY
            elif len(active) == 1:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            return ComponentHealth(
                name="alerts",
                status=status,
                message=f"{len(active)} channels active",
                details={"active_channels": active},
            )
        except Exception as e:
            logger.warning("Alert channels check failed: %s", e)
            return ComponentHealth(
                name="alerts",
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )

    async def check_all(
        self,
        brokers: Optional[Dict[str, Any]] = None,
        data_feeds: Optional[Dict[str, datetime]] = None,
        circuit_breaker: Any = None,
        kill_switch: Any = None,
        alert_manager: Any = None,
    ) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            {
                "status": "healthy|degraded|unhealthy",
                "timestamp": "ISO timestamp",
                "components": {component_name: ComponentHealth.to_dict()},
                "summary": "Human readable summary"
            }
        """
        results: Dict[str, ComponentHealth] = {}

        # Database
        results["database"] = await self.check_database()

        # Brokers
        if brokers:
            for name, broker in brokers.items():
                results[f"broker_{name}"] = await self.check_broker(name, broker)

        # Data feeds
        if data_feeds:
            for name, last_time in data_feeds.items():
                results[f"feed_{name}"] = await self.check_data_feed(name, last_time)

        # Risk system
        results["risk_system"] = await self.check_risk_system(
            circuit_breaker, kill_switch
        )

        # Alerts
        results["alerts"] = await self.check_alert_channels(alert_manager)

        # Store results
        self._results = results
        self._last_full_check = datetime.utcnow()

        # Determine overall status
        statuses = [r.status for r in results.values()]
        if HealthStatus.UNHEALTHY in statuses:
            overall = "unhealthy"
        elif HealthStatus.DEGRADED in statuses:
            overall = "degraded"
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = "healthy"
        else:
            overall = "unknown"

        unhealthy = [
            r.name for r in results.values() if r.status == HealthStatus.UNHEALTHY
        ]
        degraded = [
            r.name for r in results.values() if r.status == HealthStatus.DEGRADED
        ]

        summary_parts: List[str] = []
        if unhealthy:
            summary_parts.append(f"Unhealthy: {', '.join(unhealthy)}")
        if degraded:
            summary_parts.append(f"Degraded: {', '.join(degraded)}")
        if not summary_parts:
            summary_parts.append("All systems operational")

        return {
            "status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {name: r.to_dict() for name, r in results.items()},
            "summary": "; ".join(summary_parts),
        }

    def get_cached_results(self) -> Optional[Dict[str, Any]]:
        """Get cached health check results."""
        if not self._results:
            return None

        return {
            "status": "cached",
            "last_check": (
                self._last_full_check.isoformat() if self._last_full_check else None
            ),
            "components": {name: r.to_dict() for name, r in self._results.items()},
        }


# Convenience function matching original stub signature
async def check_health() -> Dict[str, Any]:
    """Run health checks; return dict of component -> status."""
    checker = HealthChecker()
    result = await checker.check_all()
    return {
        name: comp["status"]
        for name, comp in result.get("components", {}).items()
    }
