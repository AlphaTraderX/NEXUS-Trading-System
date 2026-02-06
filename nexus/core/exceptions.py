"""
NEXUS custom exceptions.
"""


class NexusError(Exception):
    """Base exception for NEXUS."""

    pass


class NexusConfigError(NexusError):
    """Configuration error."""

    pass


class NexusDataError(NexusError):
    """Data or feed error."""

    pass


class NexusRiskError(NexusError):
    """Risk limit or circuit breaker triggered."""

    pass


class NexusExecutionError(NexusError):
    """Order or execution error."""

    pass
