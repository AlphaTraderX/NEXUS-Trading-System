"""
NEXUS configuration - loaded from environment (NEXUS_*).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment."""

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Market data / brokers
    polygon_api_key: str = ""
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    paper_trading: bool = True

    # Database (for storage layer)
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/nexus"

    # Optional: IG, IBKR, etc.
    ig_api_key: str = ""
    ig_username: str = ""
    ig_password: str = ""
    ig_demo: bool = True
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_timeout: float = 60.0
    order_timeout_seconds: float = 30.0


settings = Settings()
