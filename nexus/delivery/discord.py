"""
NEXUS Discord Delivery
Send rich, actionable signals to Discord via webhooks.

FEATURES:
- Rich embeds with all signal details
- Color-coded by direction (green=long, red=short)
- Tier badges (A/B/C/D)
- Cost analysis
- Risk warnings
- One-click copyable trade details
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional httpx import
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from nexus.core.enums import Direction, EdgeType
from nexus.core.models import NexusSignal


@dataclass
class DeliveryResult:
    """Result of delivery attempt."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DiscordDelivery:
    """
    Send signals to Discord via webhooks.

    Creates rich embeds with all trade details.
    """

    # Tier emojis and colors
    TIER_CONFIG = {
        "A": {"emoji": "🔥", "color": 0x00FF00, "label": "ELITE"},
        "B": {"emoji": "✅", "color": 0x90EE90, "label": "STRONG"},
        "C": {"emoji": "📊", "color": 0xFFFF00, "label": "STANDARD"},
        "D": {"emoji": "⚠️", "color": 0xFFA500, "label": "CAUTION"},
        "F": {"emoji": "❌", "color": 0xFF0000, "label": "AVOID"},
    }

    # Direction colors
    DIRECTION_COLORS = {
        "long": 0x00FF00,   # Green
        "short": 0xFF4444,  # Red
    }

    # Edge emojis
    EDGE_EMOJIS = {
        "insider_cluster": "👔",
        "vwap_deviation": "📈",
        "turn_of_month": "📅",
        "month_end": "📆",
        "gap_fill": "🕳️",
        "rsi_extreme": "📉",
        "power_hour": "⚡",
        "asian_range": "🌏",
        "orb": "🌅",
        "bollinger_touch": "〰️",
        "london_open": "🇬🇧",
        "ny_open": "🗽",
        "earnings_drift": "📊",
    }

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "NEXUS Trading",
        avatar_url: Optional[str] = None,
        mention_role_id: Optional[str] = None,
        mention_on_tier_a: bool = True,
    ):
        """
        Initialize Discord delivery.

        Args:
            webhook_url: Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)
            username: Bot username to display
            avatar_url: Bot avatar URL
            mention_role_id: Role ID to mention for alerts
            mention_on_tier_a: Whether to mention on Tier A signals
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.username = username
        self.avatar_url = avatar_url
        self.mention_role_id = mention_role_id
        self.mention_on_tier_a = mention_on_tier_a

        # Statistics
        self.messages_sent = 0
        self.messages_failed = 0

    @property
    def is_configured(self) -> bool:
        """Check if webhook is configured."""
        return self.webhook_url is not None and HTTPX_AVAILABLE

    def send_signal(self, signal: NexusSignal) -> DeliveryResult:
        """
        Send a signal to Discord.

        Args:
            signal: The NexusSignal to send

        Returns:
            DeliveryResult with success status
        """
        if not self.is_configured:
            return DeliveryResult(
                success=False,
                error="Discord not configured (missing webhook URL or httpx)"
            )

        try:
            # Build the embed
            embed = self._build_signal_embed(signal)

            # Build payload
            payload = {
                "username": self.username,
                "embeds": [embed],
            }

            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url

            # Add mention for Tier A
            tier = signal.tier.value if hasattr(signal.tier, 'value') else str(signal.tier)
            if self.mention_on_tier_a and tier == "A" and self.mention_role_id:
                payload["content"] = f"<@&{self.mention_role_id}> **TIER A SIGNAL**"

            # Send request
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                )
                response.raise_for_status()

            self.messages_sent += 1

            return DeliveryResult(
                success=True,
                message_id=None,  # Webhooks don't return message ID
            )

        except Exception as e:
            self.messages_failed += 1
            return DeliveryResult(
                success=False,
                error=str(e),
            )

    def _build_signal_embed(self, signal: NexusSignal) -> Dict:
        """Build Discord embed for a signal."""

        # Get direction and tier strings
        direction = signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction)
        tier = signal.tier.value if hasattr(signal.tier, 'value') else str(signal.tier)
        edge = signal.primary_edge.value if hasattr(signal.primary_edge, 'value') else str(signal.primary_edge)
        market = signal.market.value if hasattr(signal.market, 'value') else str(signal.market)

        # Get tier config
        tier_config = self.TIER_CONFIG.get(tier, self.TIER_CONFIG["C"])

        # Get edge emoji
        edge_emoji = self.EDGE_EMOJIS.get(edge, "📊")

        # Color based on direction
        color = self.DIRECTION_COLORS.get(direction, 0x808080)

        # Direction arrow
        direction_arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"

        # Build title
        title = f"{tier_config['emoji']} NEXUS SIGNAL | {signal.symbol} | Tier {tier}"

        # Build fields
        fields = [
            {
                "name": "📍 Direction",
                "value": f"**{direction_arrow}** @ ${signal.entry_price:.2f}",
                "inline": True,
            },
            {
                "name": "🎯 Target",
                "value": f"${signal.take_profit:.2f}",
                "inline": True,
            },
            {
                "name": "🛑 Stop",
                "value": f"${signal.stop_loss:.2f}",
                "inline": True,
            },
            {
                "name": "📊 Score",
                "value": f"**{signal.edge_score}/100** ({tier_config['label']})",
                "inline": True,
            },
            {
                "name": "⚖️ Risk:Reward",
                "value": f"**{signal.risk_reward_ratio:.1f}:1**",
                "inline": True,
            },
            {
                "name": "💰 Risk",
                "value": f"${signal.risk_amount:.2f} ({signal.risk_percent:.2f}%)",
                "inline": True,
            },
            {
                "name": f"{edge_emoji} Primary Edge",
                "value": f"**{edge.replace('_', ' ').title()}**",
                "inline": True,
            },
            {
                "name": "📈 Net Edge",
                "value": f"+{signal.net_expected:.2f}%",
                "inline": True,
            },
            {
                "name": "💸 Cost Ratio",
                "value": f"{signal.cost_ratio:.1f}%",
                "inline": True,
            },
        ]

        # Add secondary edges if present
        if signal.secondary_edges:
            secondary = [e.value if hasattr(e, 'value') else str(e) for e in signal.secondary_edges]
            secondary_str = ", ".join([e.replace('_', ' ').title() for e in secondary[:3]])
            fields.append({
                "name": "🔗 Secondary Edges",
                "value": secondary_str,
                "inline": False,
            })

        # Add position sizing
        fields.append({
            "name": "📦 Position",
            "value": f"**{signal.position_size:.2f} units** (${signal.position_value:.2f})",
            "inline": False,
        })

        # Add AI reasoning
        if signal.ai_reasoning:
            reasoning = signal.ai_reasoning[:500] + "..." if len(signal.ai_reasoning) > 500 else signal.ai_reasoning
            fields.append({
                "name": "🤖 Analysis",
                "value": reasoning,
                "inline": False,
            })

        # Add risk factors if present
        if signal.risk_factors:
            risk_str = "\n".join([f"⚠️ {r}" for r in signal.risk_factors[:3]])
            fields.append({
                "name": "⚠️ Risk Factors",
                "value": risk_str,
                "inline": False,
            })

        # Build embed
        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"Market: {market.replace('_', ' ').title()} | Session: {signal.session} | Valid until: {signal.valid_until.strftime('%H:%M %d/%m') if signal.valid_until else 'N/A'}",
            },
            "timestamp": signal.created_at.isoformat() if signal.created_at else datetime.now().isoformat(),
        }

        return embed

    def send_alert(
        self,
        title: str,
        message: str,
        color: int = 0xFFFF00,
        fields: List[Dict] = None,
    ) -> DeliveryResult:
        """
        Send a general alert to Discord.

        Args:
            title: Alert title
            message: Alert message
            color: Embed color
            fields: Optional additional fields

        Returns:
            DeliveryResult
        """
        if not self.is_configured:
            return DeliveryResult(
                success=False,
                error="Discord not configured"
            )

        try:
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
            }

            if fields:
                embed["fields"] = fields

            payload = {
                "username": self.username,
                "embeds": [embed],
            }

            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url

            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                )
                response.raise_for_status()

            self.messages_sent += 1
            return DeliveryResult(success=True)

        except Exception as e:
            self.messages_failed += 1
            return DeliveryResult(success=False, error=str(e))

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
    ) -> DeliveryResult:
        """Send daily performance summary."""

        win_rate = (winners / trades_taken * 100) if trades_taken > 0 else 0

        # Determine color based on P&L
        if total_pnl_pct >= 1.0:
            color = 0x00FF00  # Green
            emoji = "🚀"
        elif total_pnl_pct >= 0:
            color = 0x90EE90  # Light green
            emoji = "✅"
        elif total_pnl_pct >= -1.0:
            color = 0xFFA500  # Orange
            emoji = "⚠️"
        else:
            color = 0xFF0000  # Red
            emoji = "🔴"

        fields = [
            {"name": "📊 Trades", "value": str(trades_taken), "inline": True},
            {"name": "✅ Winners", "value": str(winners), "inline": True},
            {"name": "❌ Losers", "value": str(losers), "inline": True},
            {"name": "🎯 Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "💰 P&L", "value": f"${total_pnl:.2f}", "inline": True},
            {"name": "📈 Return", "value": f"{total_pnl_pct:+.2f}%", "inline": True},
        ]

        if best_trade:
            fields.append({"name": "🏆 Best Trade", "value": best_trade, "inline": False})

        if worst_trade:
            fields.append({"name": "📉 Worst Trade", "value": worst_trade, "inline": False})

        return self.send_alert(
            title=f"{emoji} NEXUS Daily Summary | {date.strftime('%d %B %Y')}",
            message=f"Daily performance report",
            color=color,
            fields=fields,
        )

    def send_kill_switch_alert(self, reason: str, message: str) -> DeliveryResult:
        """Send kill switch activation alert."""
        return self.send_alert(
            title="🚨🚨🚨 KILL SWITCH ACTIVATED 🚨🚨🚨",
            message=f"**Reason:** {reason}\n\n{message}\n\n⚠️ **MANUAL INTERVENTION REQUIRED**",
            color=0xFF0000,
        )

    def send_circuit_breaker_alert(self, status: str, reason: str) -> DeliveryResult:
        """Send circuit breaker alert."""
        color = 0xFFA500 if status == "warning" else 0xFF0000
        emoji = "⚠️" if status == "warning" else "🛑"

        return self.send_alert(
            title=f"{emoji} Circuit Breaker: {status.upper()}",
            message=reason,
            color=color,
        )

    def get_statistics(self) -> Dict:
        """Get delivery statistics."""
        return {
            "configured": self.is_configured,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": (self.messages_sent / (self.messages_sent + self.messages_failed) * 100) if (self.messages_sent + self.messages_failed) > 0 else 0,
        }


# Test the Discord delivery (preview mode - doesn't actually send)
if __name__ == "__main__":
    from nexus.core.models import NexusSignal
    from nexus.intelligence.cost_engine import CostBreakdown
    from nexus.intelligence.scorer import SignalTier
    from nexus.core.enums import Market, SignalStatus

    print("=" * 60)
    print("NEXUS DISCORD DELIVERY TEST")
    print("=" * 60)

    delivery = DiscordDelivery()

    print(f"\nConfigured: {delivery.is_configured}")
    print(f"Webhook URL set: {delivery.webhook_url is not None}")
    print(f"httpx available: {HTTPX_AVAILABLE}")

    # Create a test signal
    print("\n--- Test 1: Build Signal Embed (Preview) ---")

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
        secondary_edges=[EdgeType.RSI_EXTREME, EdgeType.VWAP_DEVIATION],
        edge_score=85,
        tier=SignalTier.A,
        gross_expected=0.35,
        costs=CostBreakdown(spread=0.02, commission=0.01, slippage=0.02, overnight=0.02, fx_conversion=0.0, other=0.0),
        net_expected=0.28,
        cost_ratio=20.0,
        ai_reasoning="Insider Cluster setup on AAPL. Multiple company insiders purchasing shares within a short timeframe, indicating strong internal confidence. Score 85/100 (Tier A) in bullish trending market. Net edge of 0.28% after costs with 2.4:1 reward-to-risk.",
        confluence_factors=["High conviction score", "Trend aligned", "Strong volume"],
        risk_factors=["Earnings in 2 weeks"],
        market_context="Regime: trending_up, VIX: 18.0",
        session="us_regular",
        valid_until=datetime.now(),
        status=SignalStatus.PENDING,
    )

    # Build embed (preview without sending)
    embed = delivery._build_signal_embed(test_signal)

    def _safe(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    print(f"Title: {_safe(embed['title'])}")
    print(f"Color: {hex(embed['color'])}")
    print(f"Fields: {len(embed['fields'])}")
    for field in embed['fields'][:5]:
        val = field['value']
        disp = _safe(val[:50] + "..." if len(val) > 50 else val)
        print(f"  - {_safe(field['name'])}: {disp}")

    # Test 2: Try to send (will fail without webhook)
    print("\n--- Test 2: Send Signal (Expected to fail without webhook) ---")
    result = delivery.send_signal(test_signal)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 3: Statistics
    print("\n--- Test 3: Delivery Statistics ---")
    stats = delivery.get_statistics()
    print(f"Stats: {stats}")

    # Test 4: Preview daily summary embed
    print("\n--- Test 4: Daily Summary Preview ---")
    # Just show we can build it
    print("Daily summary method available: send_daily_summary()")
    print("Kill switch alert method available: send_kill_switch_alert()")
    print("Circuit breaker alert method available: send_circuit_breaker_alert()")

    print("\n" + "=" * 60)
    print("DISCORD DELIVERY TEST COMPLETE [OK]")
    print("=" * 60)
    print("\nTo enable Discord delivery:")
    print("1. Create a Discord webhook in your server")
    print("2. Set DISCORD_WEBHOOK_URL environment variable")
    print("3. pip install httpx")
