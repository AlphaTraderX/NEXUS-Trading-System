"""
NEXUS Telegram Delivery
Send signals to Telegram for mobile notifications.

FEATURES:
- Clean, readable format for mobile
- Markdown formatting
- Quick action buttons (optional)
- Multiple chat support
- Silent mode for off-hours
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional telegram import
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from core.enums import Direction, EdgeType
from core.models import NexusSignal


@dataclass
class TelegramResult:
    """Result of Telegram delivery attempt."""
    success: bool
    message_id: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramDelivery:
    """
    Send signals to Telegram via Bot API.

    Creates clean, mobile-friendly messages.
    """

    BASE_URL = "https://api.telegram.org/bot{token}"

    # Tier indicators (text-safe)
    TIER_INDICATORS = {
        "A": "[A] ELITE",
        "B": "[B] STRONG",
        "C": "[C] STANDARD",
        "D": "[D] CAUTION",
        "F": "[F] AVOID",
    }

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
        quiet_hours: tuple = None,  # (start_hour, end_hour) e.g., (22, 7)
    ):
        """
        Initialize Telegram delivery.

        Args:
            bot_token: Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)
            chat_id: Default chat ID (or set TELEGRAM_CHAT_ID env var)
            parse_mode: Message format (Markdown or HTML)
            disable_notification: Silent messages by default
            quiet_hours: Tuple of (start, end) hours for silent mode
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.quiet_hours = quiet_hours

        # Statistics
        self.messages_sent = 0
        self.messages_failed = 0

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is configured."""
        return self.bot_token is not None and self.chat_id is not None and HTTPX_AVAILABLE

    @property
    def api_url(self) -> str:
        """Get API URL with token."""
        return self.BASE_URL.format(token=self.bot_token)

    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        if not self.quiet_hours:
            return False

        start, end = self.quiet_hours
        current_hour = datetime.now().hour

        if start < end:
            # Same day range (e.g., 9-17)
            return start <= current_hour < end
        else:
            # Overnight range (e.g., 22-7)
            return current_hour >= start or current_hour < end

    def send_signal(self, signal: NexusSignal, chat_id: str = None) -> TelegramResult:
        """
        Send a signal to Telegram.

        Args:
            signal: The NexusSignal to send
            chat_id: Override default chat ID

        Returns:
            TelegramResult with success status
        """
        if not self.is_configured:
            return TelegramResult(
                success=False,
                error="Telegram not configured (missing bot token, chat ID, or httpx)"
            )

        try:
            # Build the message
            message = self._format_signal(signal)

            # Send
            return self._send_message(
                text=message,
                chat_id=chat_id or self.chat_id,
                silent=self._is_quiet_hours(),
            )

        except Exception as e:
            self.messages_failed += 1
            return TelegramResult(
                success=False,
                error=str(e),
            )

    def _format_signal(self, signal: NexusSignal) -> str:
        """Format signal as Telegram message."""

        # Get string values
        direction = signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction)
        tier = signal.tier.value if hasattr(signal.tier, 'value') else str(signal.tier)
        edge = signal.primary_edge.value if hasattr(signal.primary_edge, 'value') else str(signal.primary_edge)

        # Direction indicator
        dir_emoji = "LONG" if direction == "long" else "SHORT"

        # Tier indicator
        tier_text = self.TIER_INDICATORS.get(tier, f"[{tier}]")

        # Build message
        lines = [
            f"*NEXUS SIGNAL*",
            f"*{signal.symbol}* | {tier_text}",
            "",
            f"Direction: *{dir_emoji}*",
            f"Entry: `${signal.entry_price:.2f}`",
            f"Stop: `${signal.stop_loss:.2f}`",
            f"Target: `${signal.take_profit:.2f}`",
            "",
            f"Score: *{signal.edge_score}/100*",
            f"R:R: *{signal.risk_reward_ratio:.1f}:1*",
            f"Risk: `${signal.risk_amount:.2f}` ({signal.risk_percent:.1f}%)",
            "",
            f"Edge: _{edge.replace('_', ' ').title()}_",
            f"Net Edge: +{signal.net_expected:.2f}%",
        ]

        # Add secondary edges
        if signal.secondary_edges:
            secondary = [e.value if hasattr(e, 'value') else str(e) for e in signal.secondary_edges[:2]]
            secondary_str = ", ".join([e.replace('_', ' ').title() for e in secondary])
            lines.append(f"Also: _{secondary_str}_")

        # Add position
        lines.extend([
            "",
            f"Position: *{signal.position_size:.1f} units*",
            f"Value: `${signal.position_value:.2f}`",
        ])

        # Add AI reasoning (shortened)
        if signal.ai_reasoning:
            reasoning = signal.ai_reasoning[:200] + "..." if len(signal.ai_reasoning) > 200 else signal.ai_reasoning
            lines.extend([
                "",
                f"_{reasoning}_",
            ])

        # Add risk factors
        if signal.risk_factors:
            lines.extend([
                "",
                "Risks:",
            ])
            for risk in signal.risk_factors[:2]:
                lines.append(f"- {risk}")

        # Add footer
        valid_time = signal.valid_until.strftime('%H:%M') if signal.valid_until else "N/A"
        lines.extend([
            "",
            f"Valid until: {valid_time} | {signal.session}",
        ])

        return "\n".join(lines)

    def _send_message(
        self,
        text: str,
        chat_id: str,
        silent: bool = False,
    ) -> TelegramResult:
        """Send a message via Telegram API."""

        url = f"{self.api_url}/sendMessage"

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_notification": silent or self.disable_notification,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                data = response.json()

            if data.get("ok"):
                self.messages_sent += 1
                return TelegramResult(
                    success=True,
                    message_id=data.get("result", {}).get("message_id"),
                )
            else:
                self.messages_failed += 1
                return TelegramResult(
                    success=False,
                    error=data.get("description", "Unknown error"),
                )

        except Exception as e:
            self.messages_failed += 1
            return TelegramResult(
                success=False,
                error=str(e),
            )

    def send_alert(
        self,
        title: str,
        message: str,
        chat_id: str = None,
        silent: bool = None,
    ) -> TelegramResult:
        """
        Send a general alert.

        Args:
            title: Alert title
            message: Alert message
            chat_id: Override default chat ID
            silent: Override notification setting

        Returns:
            TelegramResult
        """
        if not self.is_configured:
            return TelegramResult(success=False, error="Telegram not configured")

        text = f"*{title}*\n\n{message}"

        return self._send_message(
            text=text,
            chat_id=chat_id or self.chat_id,
            silent=silent if silent is not None else self._is_quiet_hours(),
        )

    def send_daily_summary(
        self,
        date: datetime,
        trades_taken: int,
        winners: int,
        losers: int,
        total_pnl: float,
        total_pnl_pct: float,
        best_trade: Optional[str] = None,
        worst_trade: Optional[str] = None,
        chat_id: str = None,
    ) -> TelegramResult:
        """Send daily performance summary."""

        win_rate = (winners / trades_taken * 100) if trades_taken > 0 else 0

        # Status indicator
        if total_pnl_pct >= 1.0:
            status = "Excellent"
        elif total_pnl_pct >= 0:
            status = "Positive"
        elif total_pnl_pct >= -1.0:
            status = "Minor Loss"
        else:
            status = "Significant Loss"

        lines = [
            f"*NEXUS Daily Summary*",
            f"_{date.strftime('%d %B %Y')}_",
            "",
            f"Status: *{status}*",
            "",
            f"Trades: *{trades_taken}*",
            f"Winners: {winners} | Losers: {losers}",
            f"Win Rate: *{win_rate:.1f}%*",
            "",
            f"P&L: `${total_pnl:.2f}`",
            f"Return: *{total_pnl_pct:+.2f}%*",
        ]

        if best_trade:
            lines.extend(["", f"Best: _{best_trade}_"])

        if worst_trade:
            lines.extend([f"Worst: _{worst_trade}_"])

        return self.send_alert(
            title="",
            message="\n".join(lines),
            chat_id=chat_id,
        )

    def send_kill_switch_alert(
        self,
        reason: str,
        message: str,
        chat_id: str = None,
    ) -> TelegramResult:
        """Send kill switch activation alert (never silent)."""

        text = f"""*KILL SWITCH ACTIVATED*

*Reason:* {reason}

{message}

*MANUAL INTERVENTION REQUIRED*"""

        return self._send_message(
            text=text,
            chat_id=chat_id or self.chat_id,
            silent=False,  # Always notify for kill switch
        )

    def send_circuit_breaker_alert(
        self,
        status: str,
        reason: str,
        chat_id: str = None,
    ) -> TelegramResult:
        """Send circuit breaker alert."""

        if status.lower() == "warning":
            title = "Circuit Breaker WARNING"
        else:
            title = "Circuit Breaker TRIGGERED"

        return self.send_alert(
            title=title,
            message=reason,
            chat_id=chat_id,
            silent=False,  # Always notify for circuit breaker
        )

    def get_statistics(self) -> Dict:
        """Get delivery statistics."""
        total = self.messages_sent + self.messages_failed
        return {
            "configured": self.is_configured,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": (self.messages_sent / total * 100) if total > 0 else 0,
            "quiet_hours": self.quiet_hours,
        }


# Test the Telegram delivery (preview mode - doesn't actually send)
if __name__ == "__main__":
    from core.models import NexusSignal
    from intelligence.cost_engine import CostBreakdown
    from intelligence.scorer import SignalTier
    from core.enums import Market, SignalStatus

    print("=" * 60)
    print("NEXUS TELEGRAM DELIVERY TEST")
    print("=" * 60)

    delivery = TelegramDelivery(quiet_hours=(22, 7))

    print(f"\nConfigured: {delivery.is_configured}")
    print(f"Bot token set: {delivery.bot_token is not None}")
    print(f"Chat ID set: {delivery.chat_id is not None}")
    print(f"httpx available: {HTTPX_AVAILABLE}")
    print(f"Quiet hours: {delivery.quiet_hours}")
    print(f"Currently quiet: {delivery._is_quiet_hours()}")

    # Create a test signal
    print("\n--- Test 1: Format Signal (Preview) ---")

    test_signal = NexusSignal(
        signal_id="test-001",
        created_at=datetime.now(),
        opportunity_id="opp-001",
        symbol="AAPL",
        market=Market.US_STOCKS,
        direction=Direction.LONG,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=162.0,
        position_size=20.0,
        position_value=3000.0,
        risk_amount=100.0,
        risk_percent=1.0,
        primary_edge=EdgeType.INSIDER_CLUSTER,
        secondary_edges=[EdgeType.RSI_EXTREME],
        edge_score=85,
        tier=SignalTier.A,
        gross_expected=0.35,
        costs=CostBreakdown(spread=0.02, commission=0.01, slippage=0.02, overnight=0.02, fx_conversion=0.0, other=0.0),
        net_expected=0.28,
        cost_ratio=20.0,
        ai_reasoning="Insider Cluster setup on AAPL with strong internal confidence. Score 85/100 in bullish market.",
        confluence_factors=["High conviction", "Trend aligned"],
        risk_factors=["Earnings in 2 weeks", "VIX elevated"],
        market_context="Regime: trending_up",
        session="us_regular",
        valid_until=datetime.now(),
        status=SignalStatus.PENDING,
    )

    # Format signal (preview without sending)
    message = delivery._format_signal(test_signal)

    print("\nFormatted message:")
    print("-" * 40)
    print(message)
    print("-" * 40)

    # Test 2: Try to send (will fail without credentials)
    print("\n--- Test 2: Send Signal (Expected to fail) ---")
    result = delivery.send_signal(test_signal)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 3: Statistics
    print("\n--- Test 3: Delivery Statistics ---")
    stats = delivery.get_statistics()
    print(f"Stats: {stats}")

    print("\n" + "=" * 60)
    print("TELEGRAM DELIVERY TEST COMPLETE [OK]")
    print("=" * 60)
    print("\nTo enable Telegram delivery:")
    print("1. Create a Telegram bot via @BotFather")
    print("2. Set TELEGRAM_BOT_TOKEN environment variable")
    print("3. Set TELEGRAM_CHAT_ID environment variable")
    print("4. pip install httpx")
