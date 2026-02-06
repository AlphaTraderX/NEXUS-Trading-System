"""
Setup script: DB migrations, Redis, initial config.
"""

import asyncio
import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def main() -> None:
    """Run setup (migrations, etc.)."""
    print("NEXUS setup: run Alembic migrations and checks.")
    # asyncio.run(run_migrations())
    print("Done.")


if __name__ == "__main__":
    main()
