"""
Initialize NEXUS security components.

Run this ONCE when setting up the system on a new machine.
"""

import os
import sys
import getpass
from pathlib import Path


def main():
    print("=" * 60)
    print("  NEXUS Security Initialization")
    print("=" * 60)
    print()
    
    # Check if already initialized
    if Path(".nexus_license").exists():
        response = input("Security already initialized. Re-initialize? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    # Get owner name
    owner = input("Enter owner name (default: STUGE): ").strip() or "STUGE"
    
    # Generate license
    print("\n1. Generating hardware-bound license...")
    
    try:
        from nexus.security.license import LicenseManager
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Make sure you're in the NEXUS-Project directory")
        return
    
    license_mgr = LicenseManager(owner)
    license_path = license_mgr.generate_license(
        features=["full_access", "live_trading", "god_mode"]
    )
    print(f"   ✅ License created: {license_path}")
    
    # Verify license
    info = license_mgr.validate_license()
    print(f"   ✅ License valid for: {info.owner}")
    print(f"   ✅ Machine ID: {info.machine_id[:16]}...")
    
    # Set up master password for secrets
    print("\n2. Setting up secrets encryption...")
    print("   Enter a master password for encrypting API keys.")
    print("   IMPORTANT: Remember this password! You'll need it to start NEXUS.")
    print()
    
    password = getpass.getpass("   Master password: ")
    confirm = getpass.getpass("   Confirm password: ")
    
    if password != confirm:
        print("   ❌ Passwords don't match!")
        return
    
    if len(password) < 8:
        print("   ❌ Password must be at least 8 characters!")
        return
    
    # Save master key instruction
    print(f"\n   ✅ Master password set.")
    
    # Store any existing API keys
    from nexus.security.secrets import SecureSecretsManager
    
    # Set the env var temporarily
    os.environ["NEXUS_MASTER_KEY"] = password
    secrets_mgr = SecureSecretsManager()
    
    print("\n3. Migrating existing API keys from .env...")
    env_file = Path(".env")
    if env_file.exists():
        migrated = 0
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value and ("API" in key.upper() or "KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper()):
                    secrets_mgr.store_secret(key, value)
                    migrated += 1
                    print(f"      → Migrated: {key}")
        print(f"   ✅ Migrated {migrated} secrets to encrypted storage")
    else:
        print("   ⚠️  No .env file found - skipping migration")
    
    # Initialize audit logging
    print("\n4. Initializing audit logging...")
    from nexus.security.audit import get_audit_logger, AuditEventType
    
    audit = get_audit_logger()
    audit.log(
        event_type=AuditEventType.SYSTEM_START,
        component="security",
        action="initialization_complete",
        details={"owner": owner},
    )
    print("   ✅ Audit logging initialized")
    
    # Create the PowerShell setup script
    ps_script = f'''# NEXUS Environment Setup
# Run this in PowerShell before starting NEXUS

$env:NEXUS_MASTER_KEY = "{password}"

Write-Host "✅ NEXUS environment configured" -ForegroundColor Green
Write-Host "Master key set for this session" -ForegroundColor Cyan
'''
    
    setup_script = Path("setup_nexus_env.ps1")
    setup_script.write_text(ps_script)
    
    print("\n" + "=" * 60)
    print("  SECURITY INITIALIZATION COMPLETE")
    print("=" * 60)
    print(f'''
Next steps:

1. Before running NEXUS, set the master password:
   
   PowerShell (run each session):
   $env:NEXUS_MASTER_KEY = "your-password"
   
   OR run the generated script:
   .\\setup_nexus_env.ps1

2. NEVER commit these files to git:
   ✗ .nexus_license
   ✗ .nexus_secrets.enc
   ✗ setup_nexus_env.ps1
   ✗ .env
   ✗ audit_logs/

3. Your system is now protected with:
   ✅ Hardware-bound license (Machine: {info.machine_id[:16]}...)
   ✅ Encrypted secrets storage (AES-256)
   ✅ Tamper-evident audit logging

⚠️  BACKUP WARNING: Save your master password securely!
    If you lose it, you'll need to re-enter all API keys.
''')


if __name__ == "__main__":
    main()
