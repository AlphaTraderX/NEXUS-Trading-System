"""
NEXUS Delivery Layer

Alert and signal delivery to Discord, Telegram, and other channels.
"""

from nexus.delivery.formatter import SignalFormatter, formatter
from nexus.delivery.discord import (
    DiscordDelivery,
    DeliveryResult,
    create_discord_delivery,
)
from nexus.delivery.telegram import (
    TelegramDelivery,
    create_telegram_delivery,
)
from nexus.delivery.alert_manager import (
    AlertManager,
    DeliveryRecord,
    DeliveryStatus,
    QueuedAlert,
    create_alert_manager,
)

__all__ = [
    # Formatter
    "SignalFormatter",
    "formatter",
    # Discord
    "DiscordDelivery",
    "DeliveryResult",
    "create_discord_delivery",
    # Telegram
    "TelegramDelivery",
    "create_telegram_delivery",
    # Alert Manager
    "AlertManager",
    "DeliveryRecord",
    "DeliveryStatus",
    "QueuedAlert",
    "create_alert_manager",
]
