"""
FastAPI application entry for NEXUS API.
"""

from fastapi import FastAPI

from api.routes import signals, positions, performance, settings

app = FastAPI(
    title="NEXUS Trading API",
    version="2.1",
    description="Multi-Asset Automated Trading | 13 Validated Edges",
)

app.include_router(signals.router)
app.include_router(positions.router)
app.include_router(performance.router)
app.include_router(settings.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "nexus"}
