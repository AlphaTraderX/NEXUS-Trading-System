"""
Verify NEXUS project setup is correct.
"""

import pytest
from pathlib import Path


def test_project_structure_exists():
    """Verify all required directories exist."""
    root = Path(__file__).parent.parent

    required_dirs = [
        "config",
        "core",
        "data",
        "scanners",
        "intelligence",
        "risk",
        "execution",
        "delivery",
        "storage",
        "monitoring",
        "api",
        "api/routes",
        "tests",
        "scripts",
    ]

    for dir_name in required_dirs:
        dir_path = root / dir_name
        assert dir_path.exists(), f"Missing directory: {dir_name}"
        assert (dir_path / "__init__.py").exists(), f"Missing __init__.py in {dir_name}"


def test_requirements_file_exists():
    """Verify requirements.txt exists and has content."""
    root = Path(__file__).parent.parent
    req_file = root / "requirements.txt"

    assert req_file.exists(), "Missing requirements.txt"

    content = req_file.read_text()
    assert "fastapi" in content
    assert "sqlalchemy" in content
    assert "pydantic" in content


def test_env_example_exists():
    """Verify .env.example exists with required keys."""
    root = Path(__file__).parent.parent
    env_file = root / ".env.example"

    assert env_file.exists(), "Missing .env.example"

    content = env_file.read_text()
    assert "NEXUS_MODE" in content
    assert "NEXUS_DATABASE_URL" in content


def test_main_entry_point():
    """Verify main.py can be imported."""
    import main
    assert hasattr(main, "main")
