"""
NEXUS Database Connection

Async PostgreSQL connection using SQLAlchemy 2.0 and asyncpg.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from nexus.config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE ENGINE
# =============================================================================

# Create async engine
# NullPool recommended for async to avoid connection issues
engine = create_async_engine(
    settings.database_url,
    echo=False,  # Set True for SQL debugging
    poolclass=NullPool,
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for all models
Base = declarative_base()


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI endpoints.

    Usage in FastAPI:
        @app.get("/signals")
        async def get_signals(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


# =============================================================================
# DATABASE LIFECYCLE
# =============================================================================

async def init_db() -> None:
    """
    Initialize database - create all tables.

    Call this on application startup.
    """
    async with engine.begin() as conn:
        # Import models to register them with Base
        from storage import models  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def close_db() -> None:
    """
    Close database connections.

    Call this on application shutdown.
    """
    await engine.dispose()
    logger.info("Database connections closed")


async def check_connection() -> bool:
    """
    Check if database is reachable.

    Returns True if connection successful, False otherwise.
    """
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
