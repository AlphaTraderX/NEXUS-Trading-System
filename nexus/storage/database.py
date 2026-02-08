"""
NEXUS Database Connection and Session Management

SQLAlchemy 2.0 with async support.
- Production: PostgreSQL
- Testing: SQLite (in-memory or file-based)
- Both sync and async session support.
"""

import os
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Generator, AsyncGenerator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool

from nexus.storage.models import Base
from nexus.config.settings import settings


# =============================================================================
# DATABASE URL
# =============================================================================

def get_database_url(async_mode: bool = False) -> str:
    """
    Get database URL from settings or environment.

    Supports:
    - PostgreSQL: postgresql://user:pass@host:port/db
    - PostgreSQL Async: postgresql+asyncpg://user:pass@host:port/db
    - SQLite: sqlite:///path/to/db.sqlite
    - SQLite Async: sqlite+aiosqlite:///path/to/db.sqlite
    - SQLite Memory: sqlite:///:memory:

    Args:
        async_mode: If True, return async-compatible URL

    Returns:
        Database URL string
    """
    # Check environment first, then settings
    url = os.getenv("NEXUS_DATABASE_URL") or getattr(settings, "database_url", None)

    if not url:
        # Default to SQLite for development
        url = "sqlite:///nexus.db"

    if async_mode:
        # Convert to async driver
        if url.startswith("postgresql://") and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("sqlite://"):
            url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    else:
        # Convert to sync driver (in case stored URL is async)
        if "postgresql+asyncpg://" in url:
            url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        elif "sqlite+aiosqlite://" in url:
            url = url.replace("sqlite+aiosqlite://", "sqlite://", 1)

    return url


# =============================================================================
# ENGINE CREATION
# =============================================================================

# Module-level engine storage
_sync_engine = None
_async_engine = None


def get_sync_engine(url: Optional[str] = None, echo: bool = False):
    """
    Get or create synchronous database engine.

    Args:
        url: Optional database URL (uses default if not provided)
        echo: If True, log all SQL statements

    Returns:
        SQLAlchemy Engine
    """
    global _sync_engine

    if _sync_engine is None or url is not None:
        if _sync_engine is not None and url is not None:
            _sync_engine.dispose()
            _sync_engine = None
        db_url = url or get_database_url(async_mode=False)

        # Configure pool based on database type
        if "sqlite" in db_url:
            # SQLite needs special handling for threading
            _sync_engine = create_engine(
                db_url,
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool if ":memory:" in db_url else QueuePool,
            )
            # Enable foreign keys for SQLite
            @event.listens_for(_sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # PostgreSQL with connection pooling
            _sync_engine = create_engine(
                db_url,
                echo=echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections after 1 hour
            )

    return _sync_engine


async def get_async_engine(url: Optional[str] = None, echo: bool = False):
    """
    Get or create asynchronous database engine.

    Args:
        url: Optional database URL (uses default if not provided)
        echo: If True, log all SQL statements

    Returns:
        SQLAlchemy AsyncEngine
    """
    global _async_engine

    if _async_engine is None or url is not None:
        if _async_engine is not None and url is not None:
            await _async_engine.dispose()
            _async_engine = None
        db_url = url or get_database_url(async_mode=True)

        if "sqlite" in db_url:
            _async_engine = create_async_engine(
                db_url,
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool if ":memory:" in db_url else None,
            )
        else:
            _async_engine = create_async_engine(
                db_url,
                echo=echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

    return _async_engine


# =============================================================================
# SESSION FACTORIES
# =============================================================================

# Session factories (created lazily)
_sync_session_factory = None
_async_session_factory = None


def get_sync_session_factory() -> sessionmaker:
    """Get or create synchronous session factory."""
    global _sync_session_factory

    if _sync_session_factory is None:
        engine = get_sync_engine()
        _sync_session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    return _sync_session_factory


def get_async_session_factory() -> async_sessionmaker:
    """Get or create asynchronous session factory."""
    global _async_session_factory

    if _async_session_factory is None:
        import asyncio
        # Need to get engine in async context
        loop = asyncio.get_event_loop()
        engine = loop.run_until_complete(get_async_engine())
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    return _async_session_factory


# =============================================================================
# SESSION CONTEXT MANAGERS
# =============================================================================

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for synchronous database sessions.

    Usage:
        with get_session() as session:
            session.add(record)
            session.commit()

    Automatically handles:
    - Session creation
    - Commit on success
    - Rollback on exception
    - Session cleanup
    """
    factory = get_sync_session_factory()
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_async_session() as session:
            session.add(record)
            await session.commit()

    Automatically handles:
    - Session creation
    - Commit on success
    - Rollback on exception
    - Session cleanup
    """
    engine = await get_async_engine()

    # Create factory if needed
    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    session = factory()

    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db(url: Optional[str] = None, echo: bool = False) -> None:
    """
    Initialize database - create all tables.

    Args:
        url: Optional database URL
        echo: If True, log SQL statements

    Call this at application startup to ensure tables exist.
    """
    engine = get_sync_engine(url=url, echo=echo)
    Base.metadata.create_all(bind=engine)


async def init_db_async(url: Optional[str] = None, echo: bool = False) -> None:
    """
    Initialize database asynchronously - create all tables.

    Args:
        url: Optional database URL
        echo: If True, log SQL statements
    """
    engine = await get_async_engine(url=url, echo=echo)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def drop_all_tables(url: Optional[str] = None) -> None:
    """
    Drop all tables - USE WITH CAUTION!

    Only for testing/development. Never call in production.
    """
    engine = get_sync_engine(url=url)
    Base.metadata.drop_all(bind=engine)


async def drop_all_tables_async(url: Optional[str] = None) -> None:
    """Drop all tables asynchronously - USE WITH CAUTION!"""
    engine = await get_async_engine(url=url)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# =============================================================================
# HEALTH CHECK
# =============================================================================

def check_database_health() -> dict:
    """
    Check database connection health.

    Returns:
        dict with:
        - healthy: bool
        - latency_ms: float (query time)
        - error: str or None
    """
    import time

    try:
        engine = get_sync_engine()
        start = time.perf_counter()

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        latency = (time.perf_counter() - start) * 1000

        return {
            "healthy": True,
            "latency_ms": round(latency, 2),
            "error": None,
            "database_url": get_database_url()[:30] + "..."  # Truncate for security
        }
    except Exception as e:
        return {
            "healthy": False,
            "latency_ms": None,
            "error": str(e),
            "database_url": None
        }


async def check_database_health_async() -> dict:
    """Check database connection health asynchronously."""
    import time

    try:
        engine = await get_async_engine()
        start = time.perf_counter()

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        latency = (time.perf_counter() - start) * 1000

        return {
            "healthy": True,
            "latency_ms": round(latency, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "healthy": False,
            "latency_ms": None,
            "error": str(e),
        }


# =============================================================================
# CLEANUP
# =============================================================================

def close_database() -> None:
    """Close database connections and cleanup."""
    global _sync_engine, _sync_session_factory

    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None

    _sync_session_factory = None


async def close_database_async() -> None:
    """Close async database connections and cleanup."""
    global _async_engine, _async_session_factory

    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None

    _async_session_factory = None


def reset_engines() -> None:
    """
    Reset all engines - useful for testing.

    Clears both sync and async engines so they get recreated
    with fresh settings on next use.
    """
    global _sync_engine, _async_engine, _sync_session_factory, _async_session_factory

    if _sync_engine:
        _sync_engine.dispose()
    if _async_engine:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_engine.dispose())
            else:
                loop.run_until_complete(_async_engine.dispose())
        except Exception:
            pass

    _sync_engine = None
    _async_engine = None
    _sync_session_factory = None
    _async_session_factory = None
