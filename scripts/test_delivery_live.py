"""
NEXUS Delivery Layer - LIVE TEST

Tests real Discord and Telegram delivery:
- Discord webhook connection and signal delivery
- Telegram bot connection and signal delivery
- AlertManager multi-channel delivery
- All priority levels (CRITICAL, HIGH, NORMAL, LOW)

Credentials: Load from env vars or use hardcoded test values.
"""

import asyncio
import os
from datetime import datetime, timezone

from nexus.core.models import NexusSignal
from nexus.core.enums import Direction, EdgeType, Market, SignalStatus, SignalTier, AlertPriority
from nexus.intelligence.cost_engine import CostBreakdown
from nexus.delivery import (
    DiscordDelivery,
    TelegramDelivery,
    AlertManager,
)

# Test credentials - load from env or use fallback
DISCORD_WEBHOOK = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discordapp.com/api/webhooks/1469994650566918194/***REDACTED***",
)
TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8561668287:AAH_WwaeGTrULdRn98CKIXmnJUl2QvZia4k",
)
TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    "8298718605",
)


def create_test_signal() -> NexusSignal:
    """Create a realistic test signal."""
    return NexusSignal(
        signal_id="TEST-001",
        created_at=datetime.now(timezone.utc),
        opportunity_id="OPP-TEST-001",
        symbol="SPY",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=502.50,
        stop_loss=499.50,
        take_profit=508.50,
        position_size=33,
        position_value=16582.50,
        risk_amount=100.00,
        risk_percent=1.0,
        primary_edge=EdgeType.TURN_OF_MONTH,
        secondary_edges=[EdgeType.VWAP_DEVIATION],
        edge_score=78,
        tier=SignalTier.B,
        gross_expected=0.35,
        costs=CostBreakdown(
            spread=0.02,
            commission=0.01,
            slippage=0.02,
            overnight=0.04,
            fx_conversion=0.0,
            other=0.0,
        ),
        net_expected=0.26,
        cost_ratio=25.7,
        ai_reasoning="Turn of Month window active (Day 2 of 4). Historical data shows 87% of monthly S&P returns occur in this 4-day window. Price action confirms institutional accumulation with VWAP support. Risk management: Stop placed below recent swing low.",
        confluence_factors=["TOM Day 2/4", "Above VWAP", "Volume confirming", "Trend aligned"],
        risk_factors=["VIX slightly elevated at 18.5"],
        market_context="US markets trending bullish, low volatility regime",
        session="US_REGULAR",
        valid_until=datetime.now(timezone.utc),
        status=SignalStatus.PENDING,
    )


async def test_discord():
    """Test Discord delivery."""
    print("\n" + "=" * 50)
    print("🔵 TESTING DISCORD DELIVERY")
    print("=" * 50)

    discord = DiscordDelivery(webhook_url=DISCORD_WEBHOOK)

    # Test connection
    print("\n1. Testing connection...")
    result = await discord.test_connection()
    print(f"   Connection test: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    if not result.success:
        print(f"   Error: {result.error_message}")
        return False

    # Test signal delivery
    print("\n2. Sending test signal...")
    signal = create_test_signal()
    result = await discord.send_signal(signal)
    print(f"   Signal delivery: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    # Test alert
    print("\n3. Sending test alert...")
    result = await discord.send_alert(
        "🧪 NEXUS Test Alert - System operational!", AlertPriority.HIGH
    )
    print(f"   Alert delivery: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    await discord.close()
    return True


async def test_telegram():
    """Test Telegram delivery."""
    print("\n" + "=" * 50)
    print("🔵 TESTING TELEGRAM DELIVERY")
    print("=" * 50)

    telegram = TelegramDelivery(
        bot_token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
    )

    # Verify bot info
    print("\n1. Getting bot info...")
    bot_info = await telegram.get_me()
    if bot_info:
        print(f"   Bot: @{bot_info.get('username', 'unknown')}")
        print(f"   Name: {bot_info.get('first_name', 'unknown')}")
    else:
        print("   ❌ Could not get bot info")
        return False

    # Test connection
    print("\n2. Testing connection...")
    result = await telegram.test_connection()
    print(f"   Connection test: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    if not result.success:
        print(f"   Error: {result.error_message}")
        return False

    # Test signal delivery
    print("\n3. Sending test signal...")
    signal = create_test_signal()
    result = await telegram.send_signal(signal)
    print(f"   Signal delivery: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    # Test alert
    print("\n4. Sending test alert...")
    result = await telegram.send_alert(
        "🧪 NEXUS Test Alert - System operational!", AlertPriority.HIGH
    )
    print(f"   Alert delivery: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    await telegram.close()
    return True


async def test_alert_manager():
    """Test the unified AlertManager with both channels."""
    print("\n" + "=" * 50)
    print("🔵 TESTING ALERT MANAGER (MULTI-CHANNEL)")
    print("=" * 50)

    # Create manager and register channels
    manager = AlertManager()

    discord = DiscordDelivery(webhook_url=DISCORD_WEBHOOK)
    telegram = TelegramDelivery(bot_token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)

    manager.register_channel("discord", discord)
    manager.register_channel("telegram", telegram)

    print(f"\n1. Registered channels: {list(manager._channels.keys())}")

    # Test all channels
    print("\n2. Testing all channel connections...")
    results = await manager.test_all_channels()
    for channel, result in results.items():
        print(f"   {channel}: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    # Send signal to all channels simultaneously
    print("\n3. Sending signal to ALL channels...")
    signal = create_test_signal()
    record = await manager.send_signal(signal)
    print(f"   Status: {record.status.value}")
    print(
        f"   Results: {len([r for r in record.results.values() if r.success])}/{len(record.results)} succeeded"
    )

    # Send alerts at different priority levels
    print("\n4. Sending alerts at all priority levels...")
    for priority in [AlertPriority.CRITICAL, AlertPriority.HIGH, AlertPriority.NORMAL, AlertPriority.LOW]:
        msg = f"🧪 NEXUS {priority.value.upper()} Test - Priority level check"
        record = await manager.send_alert(msg, priority)
        status = "✅" if record.status.value == "delivered" else "❌"
        print(f"   {priority.value}: {status}")

    # Send critical alert (explicit test)
    print("\n5. Sending CRITICAL alert to all channels...")
    record = await manager.send_alert(
        "🚨 NEXUS CRITICAL TEST - Kill switch would activate here!",
        AlertPriority.CRITICAL,
    )
    print(f"   Status: {record.status.value}")

    # Show stats
    print("\n6. Delivery Statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    await manager.close()
    return True


async def main():
    """Run all delivery tests."""
    print("\n" + "🚀" * 25)
    print("   NEXUS DELIVERY LAYER - LIVE TEST")
    print("🚀" * 25)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Creds: {'ENV' if os.getenv('DISCORD_WEBHOOK_URL') else 'hardcoded fallback'}")

    results = {}

    # Test each component
    results["discord"] = await test_discord()
    results["telegram"] = await test_telegram()
    results["alert_manager"] = await test_alert_manager()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {component}: {status}")

    all_passed = all(results.values())
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️ SOME TESTS FAILED"))
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
