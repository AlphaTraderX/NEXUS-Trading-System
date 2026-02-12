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

    # AI / LLM
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-70b-versatile"

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

    # Trading mode and account
    mode: str = "standard"  # conservative | standard | aggressive | maximum
    starting_balance: float = 100_000.0
    currency: str = "USD"
    max_positions: int = 8
    min_score_to_trade: float = 50.0

    # Risk / position sizing
    base_risk_pct: float = 1.0
    max_risk_pct: float = 2.0
    base_heat_limit: float = 25.0
    max_heat_limit: float = 35.0
    min_heat_limit: float = 15.0
    max_per_market: float = 10.0
    max_correlated: int = 3

    # Correlation limits (hidden concentration detector)
    max_sector_positions: int = 3
    max_same_direction_per_market: int = 3
    correlation_warning_threshold: float = 0.7
    max_effective_risk_multiplier: float = 2.0  # Effective risk can't exceed 2x nominal

    # Circuit breaker thresholds (loss-based only; never caps profits)
    daily_loss_warning: float = -1.5  # Warning level
    daily_loss_reduce: float = -2.0  # Reduce position size
    daily_loss_stop: float = -3.0  # Stop trading for day
    weekly_loss_stop: float = -6.0  # Stop trading for week
    max_drawdown: float = -10.0  # Full stop - manual review required

    # Delivery / alerts
    discord_webhook_url: str = ""
    discord_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = True
    alert_retry_attempts: int = 3
    alert_retry_delay_seconds: float = 5.0
    alert_timeout_seconds: float = 30.0

    # Kill switch (emergency shutdown)
    kill_switch_enabled: bool = True
    connection_timeout_seconds: int = 300  # 5 minutes
    data_stale_threshold_seconds: int = 30
    kill_switch_cooldown_minutes: int = 60  # Minimum time before reset allowed

    def get_mode_multipliers(self) -> dict:
        """Return risk/heat multipliers for current mode (for API display)."""
        from nexus.config.settings import MODE_MULTIPLIERS
        return MODE_MULTIPLIERS.get(self.mode, MODE_MULTIPLIERS["standard"])


settings = Settings()


def get_settings() -> Settings:
    """Return application settings (for dependency injection / heat manager)."""
    return settings


# Mode multipliers for risk/heat scaling (used by API and risk layer)
MODE_MULTIPLIERS = {
    "conservative": {"risk": 0.7, "heat": 0.8},
    "standard": {"risk": 1.0, "heat": 1.0},
    "aggressive": {"risk": 1.3, "heat": 1.2},
    "maximum": {"risk": 1.5, "heat": 1.4},
}
