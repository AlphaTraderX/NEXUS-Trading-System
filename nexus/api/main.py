"""
NEXUS Trading API

FastAPI application with CORS, middleware, lifecycle management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nexus.api.routes import signals, positions, performance, settings
from nexus.storage.database import init_db_async, close_database_async, check_database_health_async
from nexus.storage.service import get_storage_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifecycle management."""
    logger.info("Starting NEXUS API...")

    try:
        await init_db_async()
        logger.info("Database initialized")

        storage = get_storage_service()
        await storage.initialize()
        logger.info("Storage service initialized")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down NEXUS API...")
    await close_database_async()
    logger.info("Database connections closed")


app = FastAPI(
    title="NEXUS Trading API",
    version="2.1.0",
    description="Multi-Asset Automated Trading System | 13 Validated Edges",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://localhost:3000",      # React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        # Add production domain when deployed:
        # "https://nexus.yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


app.include_router(signals.router)
app.include_router(positions.router)
app.include_router(performance.router)
app.include_router(settings.router)


@app.get("/", tags=["root"])
async def root():
    return {"service": "NEXUS Trading API", "version": "2.1.0", "status": "operational"}


@app.get("/health", tags=["health"])
async def health():
    db_health = await check_database_health_async()
    return {
        "status": "healthy" if db_health.get("healthy") else "degraded",
        "service": "nexus",
        "database": db_health,
    }


@app.get("/health/detailed", tags=["health"])
async def health_detailed():
    from nexus.monitoring.health import HealthChecker
    checker = HealthChecker()
    return await checker.check_all()


# WebSocket endpoint
from nexus.api.websocket import websocket_handler


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)
