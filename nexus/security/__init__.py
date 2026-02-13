"""
NEXUS Security Module

Provides:
- Comprehensive audit logging for trading activity
- Encrypted secrets storage for API keys
- Hardware-bound license management
"""

from .audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    audit,
    get_audit_logger,
)
from .license import (
    LicenseInfo,
    LicenseManager,
    check_license,
)
from .secrets import (
    SecureSecretsManager,
    get_secret,
)

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "audit",
    "get_audit_logger",
    "LicenseInfo",
    "LicenseManager",
    "check_license",
    "SecureSecretsManager",
    "get_secret",
]
