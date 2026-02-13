"""Tests for NEXUS security modules."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.security.audit import AuditEventType, AuditLogger
from nexus.security.license import LicenseManager
from nexus.security.secrets import SecureSecretsManager


# ── Audit Logger ───────────────────────────────────────────────────────────


class TestAuditLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = AuditLogger(audit_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_creates_file(self):
        """Logging an event should create a daily JSONL file."""
        self.logger.log(
            AuditEventType.SYSTEM_START, "test", "initialize", {"foo": "bar"}
        )
        files = list(Path(self.tmpdir).glob("audit_*.jsonl"))
        assert len(files) == 1

    def test_log_returns_event_id(self):
        event_id = self.logger.log(
            AuditEventType.SIGNAL_GENERATED, "scanner", "vwap_deviation"
        )
        assert event_id  # non-empty string

    def test_log_writes_valid_json(self):
        self.logger.log(AuditEventType.ORDER_PLACED, "executor", "buy_AAPL")

        log_file = next(Path(self.tmpdir).glob("audit_*.jsonl"))
        line = log_file.read_text(encoding="utf-8").strip()
        data = json.loads(line)

        assert data["event_type"] == "order_placed"
        assert data["component"] == "executor"
        assert data["action"] == "buy_AAPL"
        assert "checksum" in data

    def test_integrity_verification_passes(self):
        """Clean logs should pass integrity check."""
        for i in range(5):
            self.logger.log(AuditEventType.SIGNAL_GENERATED, "scanner", f"signal_{i}")

        result = self.logger.verify_integrity()
        assert result["verified"] is True
        assert result["events_checked"] == 5

    def test_integrity_detects_tampering(self):
        """Modified checksums should fail integrity check."""
        self.logger.log(AuditEventType.ORDER_PLACED, "exec", "buy")
        self.logger.log(AuditEventType.ORDER_FILLED, "exec", "filled")

        # Tamper with first event
        log_file = next(Path(self.tmpdir).glob("audit_*.jsonl"))
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        data = json.loads(lines[0])
        data["action"] = "TAMPERED"
        lines[0] = json.dumps(data)
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = self.logger.verify_integrity()
        assert result["verified"] is False
        assert len(result["issues"]) > 0

    def test_query_filters_by_type(self):
        self.logger.log(AuditEventType.ORDER_PLACED, "exec", "buy")
        self.logger.log(AuditEventType.ERROR, "system", "timeout")
        self.logger.log(AuditEventType.ORDER_FILLED, "exec", "filled")

        results = self.logger.query(event_types=[AuditEventType.ERROR])
        assert len(results) == 1
        assert results[0]["event_type"] == "error"

    def test_query_filters_by_component(self):
        self.logger.log(AuditEventType.SIGNAL_GENERATED, "scanner", "sig1")
        self.logger.log(AuditEventType.ORDER_PLACED, "executor", "buy")

        results = self.logger.query(component="scanner")
        assert len(results) == 1
        assert results[0]["component"] == "scanner"


# ── License Manager ────────────────────────────────────────────────────────


class TestLicenseManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.license_file = Path(self.tmpdir) / ".nexus_license"

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generate_and_validate(self):
        """License generated on this machine should validate."""
        manager = LicenseManager(owner_name="TestUser")

        # Patch LICENSE_FILE to use temp dir
        with patch.object(LicenseManager, "LICENSE_FILE", ".nexus_license"):
            with patch("pathlib.Path.home", return_value=Path(self.tmpdir)):
                path = manager.generate_license()
                assert Path(path).exists()

                info = manager.validate_license()
                assert info.is_valid
                assert info.owner == "TestUser"
                assert "full_access" in info.features

    def test_no_license_file_invalid(self):
        """Missing license file should return invalid."""
        manager = LicenseManager()

        with patch("pathlib.Path.home", return_value=Path(self.tmpdir)):
            with patch("pathlib.Path.exists", return_value=False):
                info = manager.validate_license()
                assert not info.is_valid
                assert "No license file" in info.validation_message

    def test_machine_fingerprint_stable(self):
        """Same machine should produce same fingerprint."""
        m1 = LicenseManager()
        m2 = LicenseManager()
        assert m1._machine_id == m2._machine_id

    def test_check_license_returns_bool(self):
        """check_license() should return bool without raising."""
        from nexus.security.license import check_license

        result = check_license()
        assert isinstance(result, bool)


# ── Secrets Manager ────────────────────────────────────────────────────────


class TestSecretsManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_dir = os.getcwd()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self.orig_dir)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_store_and_retrieve(self):
        """Stored secret should be retrievable."""
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("API_KEY", "sk-abc123")

        value = manager.get_secret("API_KEY", purpose="test")
        assert value == "sk-abc123"

    def test_env_var_takes_priority(self):
        """Environment variable should override stored secret."""
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("TEST_SECRET", "from-file")

        with patch.dict(os.environ, {"TEST_SECRET": "from-env"}):
            value = manager.get_secret("TEST_SECRET")
            assert value == "from-env"

    def test_missing_secret_returns_none(self):
        manager = SecureSecretsManager(master_password="test-pass-123")
        assert manager.get_secret("NONEXISTENT") is None

    def test_delete_secret(self):
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("TO_DELETE", "value")
        assert manager.delete_secret("TO_DELETE")
        assert manager.get_secret("TO_DELETE") is None

    def test_list_secrets(self):
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("KEY_A", "val_a")
        manager.store_secret("KEY_B", "val_b")
        names = manager.list_secrets()
        assert "KEY_A" in names
        assert "KEY_B" in names

    def test_wrong_password_cannot_read(self):
        """Different password should not decrypt secrets."""
        manager1 = SecureSecretsManager(master_password="correct-password")
        manager1.store_secret("SENSITIVE", "top-secret")

        manager2 = SecureSecretsManager(master_password="wrong-password")
        # Wrong password -> _load_secrets returns {} (decryption fails)
        assert manager2.get_secret("SENSITIVE") is None

    def test_no_password_raises(self):
        """Missing master password should raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove NEXUS_MASTER_KEY if present
            os.environ.pop("NEXUS_MASTER_KEY", None)
            with pytest.raises(ValueError, match="Master password required"):
                SecureSecretsManager()

    def test_rotation_increments_count(self):
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("ROTATING", "v1")
        manager.rotate_secret("ROTATING", "v2")

        secrets = manager._load_secrets()
        assert secrets["ROTATING"]["value"] == "v2"
        assert secrets["ROTATING"]["rotated_count"] == 2

    def test_audit_log_created(self):
        """Accessing a secret should create an audit log entry."""
        manager = SecureSecretsManager(master_password="test-pass-123")
        manager.store_secret("AUDITED", "value")
        manager.get_secret("AUDITED", purpose="unit-test")

        audit_path = Path(SecureSecretsManager.AUDIT_FILE)
        assert audit_path.exists()
        content = audit_path.read_text(encoding="utf-8")
        assert "AUDITED" in content
        assert "unit-test" in content
