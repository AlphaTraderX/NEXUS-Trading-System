"""
NEXUS License Protection

Binds the software to a specific machine to prevent unauthorized copying.

Limitations (be aware):
- Anyone with source code access can bypass this check.
- Hardware changes (new NIC, VM migration) will invalidate the license.
- This is a deterrent, not a security boundary. Real IP protection
  comes from legal agreements, access control, and private repos.
"""

import base64
import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class LicenseInfo:
    """License validation result."""

    owner: str
    machine_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    features: List[str]
    is_valid: bool
    validation_message: str


class LicenseManager:
    """
    Hardware-bound license manager.

    Generates an encrypted license file that is tied to the current
    machine's fingerprint (MAC address, hostname, OS, user).
    """

    LICENSE_FILE = ".nexus_license"
    _KDF_SALT = b"nexus_license_kdf_salt_v1"

    def __init__(self, owner_name: str = "STUGE"):
        self.owner = owner_name
        self._machine_id = self._get_machine_fingerprint()
        self._cipher = self._create_cipher()

    def _create_cipher(self) -> Fernet:
        """Create cipher keyed to this machine's fingerprint."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._KDF_SALT,
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._machine_id.encode()))
        return Fernet(key)

    def _get_machine_fingerprint(self) -> str:
        """Generate a stable machine fingerprint."""
        components = [
            str(uuid.getnode()),
            platform.node(),
            platform.system(),
            platform.machine(),
        ]
        try:
            components.append(os.getlogin())
        except OSError:
            components.append(os.environ.get("USERNAME", "unknown"))

        raw = "|".join(components)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def generate_license(
        self,
        expires_days: Optional[int] = None,
        features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate and save a license file for this machine.

        Returns:
            Path to the license file.
        """
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(days=expires_days)) if expires_days else None

        license_data = {
            "owner": self.owner,
            "machine_id": self._machine_id,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat() if expires else None,
            "features": features or ["full_access"],
            "version": "1.0",
        }

        encrypted = self._cipher.encrypt(json.dumps(license_data).encode())

        license_path = Path.home() / self.LICENSE_FILE
        license_path.write_bytes(encrypted)
        return str(license_path)

    def validate_license(self) -> LicenseInfo:
        """Validate the license file on this machine."""
        now = datetime.now(timezone.utc)

        # Search for license file
        license_path = Path.home() / self.LICENSE_FILE
        if not license_path.exists():
            license_path = Path(".") / self.LICENSE_FILE
        if not license_path.exists():
            return LicenseInfo(
                owner="UNLICENSED",
                machine_id="",
                created_at=now,
                expires_at=None,
                features=[],
                is_valid=False,
                validation_message="No license file found. Run: nexus --generate-license",
            )

        # Decrypt
        try:
            encrypted = license_path.read_bytes()
            decrypted = self._cipher.decrypt(encrypted)
            data = json.loads(decrypted)
        except (InvalidToken, json.JSONDecodeError, OSError) as e:
            return LicenseInfo(
                owner="INVALID",
                machine_id="",
                created_at=now,
                expires_at=None,
                features=[],
                is_valid=False,
                validation_message=f"License decryption failed (wrong machine?): {type(e).__name__}",
            )

        # Verify machine binding
        if data["machine_id"] != self._machine_id:
            return LicenseInfo(
                owner=data["owner"],
                machine_id=data["machine_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=(
                    datetime.fromisoformat(data["expires_at"])
                    if data["expires_at"]
                    else None
                ),
                features=data["features"],
                is_valid=False,
                validation_message="License not valid for this machine",
            )

        # Check expiration
        expires_at = None
        if data["expires_at"]:
            expires_at = datetime.fromisoformat(data["expires_at"])
            if now > expires_at:
                return LicenseInfo(
                    owner=data["owner"],
                    machine_id=data["machine_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    expires_at=expires_at,
                    features=data["features"],
                    is_valid=False,
                    validation_message=f"License expired on {expires_at.date()}",
                )

        return LicenseInfo(
            owner=data["owner"],
            machine_id=data["machine_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=expires_at,
            features=data["features"],
            is_valid=True,
            validation_message="License valid",
        )

    def require_valid_license(self) -> LicenseInfo:
        """Validate and return license info, raising on failure."""
        info = self.validate_license()
        if not info.is_valid:
            raise PermissionError(f"NEXUS License Error: {info.validation_message}")
        return info


def check_license() -> bool:
    """Quick license check. Returns True if valid."""
    try:
        manager = LicenseManager()
        info = manager.validate_license()
        return info.is_valid
    except Exception:
        return False
