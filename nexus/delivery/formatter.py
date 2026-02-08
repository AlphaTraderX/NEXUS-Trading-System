"""
Signal formatting engine.

Formats NexusSignal objects consistently for Discord, Telegram, and logs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from nexus.core.enums import AlertPriority, Direction, EdgeType, SignalTier
from nexus.core.models import NexusSignal


# Tier colors (Discord hex) and emojis
TIER_COLORS: Dict[str, int] = {
    "A": 0x00FF00,   # bright green
    "B": 0x32CD32,   # lime green
    "C": 0xFFD700,   # gold
    "D": 0xFFA500,   # orange
    "F": 0xFF4444,   # red
}

TIER_EMOJIS: Dict[str, str] = {
    "A": "🔥",
    "B": "✅",
    "C": "📊",
    "D": "⚠️",
    "F": "❌",
}

# Alert priority colors and emojis
ALERT_COLORS: Dict[str, int] = {
    "critical": 0xFF0000,   # red
    "high": 0xFFA500,       # orange
    "normal": 0x0099FF,     # blue
    "low": 0x808080,        # gray
}

ALERT_EMOJIS: Dict[str, str] = {
    "critical": "🚨",
    "high": "⚡",
    "normal": "📢",
    "low": "ℹ️",
}

UTC = timezone.utc


def _normalize_tier(tier: Any) -> str:
    """Return tier as single letter string A/B/C/D/F."""
    if tier is None:
        return "C"
    if hasattr(tier, "value"):
        return str(tier.value).upper() if len(str(tier.value)) == 1 else str(tier.value)
    s = str(tier).strip().upper()
    return s[0] if s else "C"


def _normalize_direction(direction: Any) -> str:
    """Return direction as 'long' or 'short'."""
    if direction is None:
        return "long"
    if hasattr(direction, "value"):
        return str(direction.value).lower()
    return str(direction).lower()


class SignalFormatter:
    """
    Formats NexusSignal for Discord, Telegram, and logs.
    """

    def _get_tier_color(self, tier: Any) -> int:
        """Return Discord embed color (int) for the given tier."""
        t = _normalize_tier(tier)
        return TIER_COLORS.get(t, TIER_COLORS["C"])

    def _get_tier_emoji(self, tier: Any) -> str:
        """Return emoji for the given tier."""
        t = _normalize_tier(tier)
        return TIER_EMOJIS.get(t, TIER_EMOJIS["C"])

    def _get_direction_emoji(self, direction: Any) -> str:
        """Return emoji for LONG (🟢) or SHORT (🔴)."""
        d = _normalize_direction(direction)
        return "🟢" if d == "long" else "🔴"

    def _format_edge_name(self, edge: Union[EdgeType, str]) -> str:
        """Convert EdgeType to display name, e.g. TURN_OF_MONTH -> 'Turn of Month'."""
        if hasattr(edge, "value"):
            raw = edge.value
        else:
            raw = str(edge)
        return raw.replace("_", " ").title()

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max_length; append '...' if truncated."""
        if not text:
            return ""
        text = text.strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def format_for_discord(self, signal: NexusSignal) -> Dict[str, Any]:
        """
        Return Discord embed structure for the signal.

        Returns:
            dict with "embeds" key containing a list of one embed dict.
        """
        tier = _normalize_tier(signal.tier)
        tier_emoji = self._get_tier_emoji(signal.tier)
        tier_color = self._get_tier_color(signal.tier)
        direction_str = _normalize_direction(signal.direction).upper()
        direction_emoji = self._get_direction_emoji(signal.direction)
        primary_edge = self._format_edge_name(signal.primary_edge)
        secondary = (
            ", ".join(self._format_edge_name(e) for e in signal.secondary_edges)
            if signal.secondary_edges
            else "None"
        )
        ratio = getattr(signal, "risk_reward_ratio", 0.0)
        levels_value = (
            f"Stop: {signal.stop_loss}\nTarget: {signal.take_profit}\nR:R: {ratio:.1f}:1"
        )
        reasoning_value = self._truncate(signal.ai_reasoning or "", 500)
        factors_value = "\n".join(signal.confluence_factors) if signal.confluence_factors else "—"
        valid_text = "N/A"
        if signal.valid_until:
            valid_text = signal.valid_until.strftime("%H:%M %d/%m") if hasattr(signal.valid_until, "strftime") else str(signal.valid_until)

        fields: List[Dict[str, Any]] = [
            {
                "name": "Direction",
                "value": f"{direction_emoji} {direction_str} @ {signal.entry_price}",
                "inline": True,
            },
            {
                "name": "Risk",
                "value": f"£{signal.risk_amount:.2f} ({signal.risk_percent:.2f}%)",
                "inline": True,
            },
            {
                "name": "Score",
                "value": f"{signal.edge_score}/100 (Tier {tier})",
                "inline": True,
            },
            {
                "name": "Levels",
                "value": levels_value,
                "inline": False,
            },
            {
                "name": "Edge",
                "value": primary_edge,
                "inline": True,
            },
            {
                "name": "Secondary",
                "value": secondary,
                "inline": True,
            },
            {
                "name": "AI Reasoning",
                "value": reasoning_value or "—",
                "inline": False,
            },
            {
                "name": "Confluence",
                "value": factors_value,
                "inline": False,
            },
        ]

        embed = {
            "title": f"{tier_emoji} NEXUS SIGNAL | {signal.symbol}",
            "color": tier_color,
            "fields": fields,
            "footer": {"text": f"Valid until: {valid_text}"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return {"embeds": [embed]}

    def format_for_telegram(self, signal: NexusSignal) -> str:
        """
        Return Telegram Markdown string with tree-style formatting.
        """
        tier_emoji = self._get_tier_emoji(signal.tier)
        direction_emoji = self._get_direction_emoji(signal.direction)
        tier = _normalize_tier(signal.tier)
        ratio = getattr(signal, "risk_reward_ratio", 0.0)
        primary_edge = self._format_edge_name(signal.primary_edge)
        secondary = (
            ", ".join(self._format_edge_name(e) for e in signal.secondary_edges)
            if signal.secondary_edges
            else "None"
        )
        reasoning = self._truncate(signal.ai_reasoning or "", 300)
        valid_text = "N/A"
        if signal.valid_until:
            valid_text = signal.valid_until.strftime("%H:%M") if hasattr(signal.valid_until, "strftime") else str(signal.valid_until)

        direction_label = _normalize_direction(signal.direction).upper()
        lines = [
            f"{tier_emoji} NEXUS SIGNAL | {signal.symbol}",
            f"{direction_emoji} {direction_label} @ {signal.entry_price}",
            "📊 Details",
            f"├ Risk: £{signal.risk_amount:.2f} ({signal.risk_percent:.2f}%)",
            f"├ Score: {signal.edge_score}/100 (Tier {tier})",
            f"└ R:R: {ratio:.1f}:1",
            "📍 Levels",
            f"├ Stop: {signal.stop_loss}",
            f"└ Target: {signal.take_profit}",
            "🎯 Edge",
            f"├ Primary: {primary_edge}",
            f"└ Secondary: {secondary}",
            "🤖 AI Reasoning",
            reasoning or "—",
            f"⏰ Valid until: {valid_text}",
        ]
        return "\n".join(lines)

    def format_summary(self, signal: NexusSignal) -> str:
        """One-liner summary: '[Tier B] 🟢 LONG SPY @ 502.50 | Score: 78 | Risk: £100'."""
        tier = _normalize_tier(signal.tier)
        direction_str = _normalize_direction(signal.direction).upper()
        direction_emoji = self._get_direction_emoji(signal.direction)
        return (
            f"[Tier {tier}] {direction_emoji} {direction_str} {signal.symbol} @ {signal.entry_price} | "
            f"Score: {signal.edge_score} | Risk: £{signal.risk_amount:.2f}"
        )

    def format_alert(self, message: str, priority: AlertPriority) -> Dict[str, Any]:
        """
        Discord embed for general alerts with priority-based color and emoji.
        """
        p = priority.value if hasattr(priority, "value") else str(priority).lower()
        color = ALERT_COLORS.get(p, ALERT_COLORS["normal"])
        emoji = ALERT_EMOJIS.get(p, ALERT_EMOJIS["normal"])
        title = f"{emoji} Alert"
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return {"embeds": [embed]}

    def format_alert_telegram(self, message: str, priority: AlertPriority) -> str:
        """Telegram markdown for alerts."""
        p = priority.value if hasattr(priority, "value") else str(priority).lower()
        emoji = ALERT_EMOJIS.get(p, ALERT_EMOJIS["normal"])
        return f"{emoji} *Alert*\n\n{message}"


# Singleton instance
formatter = SignalFormatter()
