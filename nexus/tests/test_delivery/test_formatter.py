"""Tests for nexus.delivery.formatter (SignalFormatter)."""

import pytest

from nexus.core.enums import AlertPriority, Direction, EdgeType, SignalTier
from nexus.core.models import NexusSignal
from nexus.delivery.formatter import (
    TIER_COLORS,
    TIER_EMOJIS,
    ALERT_COLORS,
    ALERT_EMOJIS,
    SignalFormatter,
    formatter,
)


class TestDiscordEmbedStructure:
    """Discord embed structure: title, color, fields."""

    def test_embed_has_title_color_fields(self, sample_signal):
        out = formatter.format_for_discord(sample_signal)
        assert "embeds" in out
        assert len(out["embeds"]) == 1
        embed = out["embeds"][0]
        assert "title" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "NEXUS SIGNAL" in embed["title"]
        assert sample_signal.symbol in embed["title"]

    def test_embed_title_includes_tier_emoji(self, sample_signal_tier_a):
        out = formatter.format_for_discord(sample_signal_tier_a)
        embed = out["embeds"][0]
        assert TIER_EMOJIS["A"] in embed["title"]

    def test_embed_has_expected_field_names(self, sample_signal):
        out = formatter.format_for_discord(sample_signal)
        embed = out["embeds"][0]
        names = {f["name"] for f in embed["fields"]}
        assert "Direction" in names
        assert "Risk" in names
        assert "Score" in names
        assert "Levels" in names
        assert "Edge" in names
        assert "Secondary" in names
        assert "AI Reasoning" in names
        assert "Confluence" in names


class TestTierColors:
    """All tier colors A through F."""

    @pytest.mark.parametrize("tier", ["A", "B", "C", "D", "F"])
    def test_tier_color_matches_constants(self, sample_signal, tier):
        sig = sample_signal.model_copy(update={"tier": tier})
        out = formatter.format_for_discord(sig)
        embed = out["embeds"][0]
        assert embed["color"] == TIER_COLORS[tier]

    def test_tier_a_color(self, sample_signal_tier_a):
        out = formatter.format_for_discord(sample_signal_tier_a)
        assert out["embeds"][0]["color"] == TIER_COLORS["A"]

    def test_tier_d_color(self, sample_signal_tier_d):
        out = formatter.format_for_discord(sample_signal_tier_d)
        assert out["embeds"][0]["color"] == TIER_COLORS["D"]


class TestDirectionEmojis:
    """Direction emojis: LONG (🟢) / SHORT (🔴)."""

    def test_long_emoji_in_discord(self, sample_signal):
        out = formatter.format_for_discord(sample_signal)
        embed = out["embeds"][0]
        direction_field = next(f for f in embed["fields"] if f["name"] == "Direction")
        assert "🟢" in direction_field["value"]
        assert "LONG" in direction_field["value"]

    def test_short_emoji_in_discord(self, sample_signal_short):
        out = formatter.format_for_discord(sample_signal_short)
        embed = out["embeds"][0]
        direction_field = next(f for f in embed["fields"] if f["name"] == "Direction")
        assert "🔴" in direction_field["value"]
        assert "SHORT" in direction_field["value"]

    def test_long_emoji_in_telegram(self, sample_signal):
        text = formatter.format_for_telegram(sample_signal)
        assert "🟢" in text
        assert "LONG" in text

    def test_short_emoji_in_telegram(self, sample_signal_short):
        text = formatter.format_for_telegram(sample_signal_short)
        assert "🔴" in text
        assert "SHORT" in text


class TestLevelFormatting:
    """Level formatting (stop, target, R:R)."""

    def test_levels_field_discord(self, sample_signal):
        out = formatter.format_for_discord(sample_signal)
        levels_field = next(f for f in out["embeds"][0]["fields"] if f["name"] == "Levels")
        val = levels_field["value"]
        assert "Stop:" in val
        assert str(sample_signal.stop_loss) in val
        assert "Target:" in val
        assert str(sample_signal.take_profit) in val
        assert "R:R:" in val

    def test_levels_telegram(self, sample_signal):
        text = formatter.format_for_telegram(sample_signal)
        assert "Stop:" in text
        assert "Target:" in text


class TestAIReasoningTruncation:
    """AI reasoning truncation: 500 for Discord, 300 for Telegram."""

    def test_discord_truncates_at_500(self, sample_signal):
        long_reasoning = "x" * 600
        sig = sample_signal.model_copy(update={"ai_reasoning": long_reasoning})
        out = formatter.format_for_discord(sig)
        reasoning_field = next(
            f for f in out["embeds"][0]["fields"] if f["name"] == "AI Reasoning"
        )
        assert len(reasoning_field["value"]) <= 503  # 500 + "..."
        assert reasoning_field["value"].endswith("...")

    def test_telegram_truncates_at_300(self, sample_signal):
        long_reasoning = "y" * 400
        sig = sample_signal.model_copy(update={"ai_reasoning": long_reasoning})
        text = formatter.format_for_telegram(sig)
        # Telegram output is multi-line; find the AI Reasoning line and following content
        assert "🤖 AI Reasoning" in text
        # Truncated content should be at most 303 chars (300 + "...")
        parts = text.split("🤖 AI Reasoning")
        reasoning_part = parts[-1].strip().split("\n")[0] if len(parts) > 1 else ""
        assert len(reasoning_part) <= 303 or "..." in reasoning_part

    def test_empty_reasoning_shows_placeholder(self, sample_signal):
        sig = sample_signal.model_copy(update={"ai_reasoning": ""})
        out = formatter.format_for_discord(sig)
        reasoning_field = next(
            f for f in out["embeds"][0]["fields"] if f["name"] == "AI Reasoning"
        )
        assert reasoning_field["value"] in ("—", "")


class TestTelegramMarkdownFormat:
    """Telegram Markdown format (tree-style)."""

    def test_telegram_contains_symbol_and_tier_emoji(self, sample_signal):
        text = formatter.format_for_telegram(sample_signal)
        assert sample_signal.symbol in text
        assert TIER_EMOJIS["B"] in text

    def test_telegram_has_details_levels_edge_sections(self, sample_signal):
        text = formatter.format_for_telegram(sample_signal)
        assert "📊 Details" in text
        assert "📍 Levels" in text
        assert "🎯 Edge" in text
        assert "🤖 AI Reasoning" in text
        assert "⏰ Valid until" in text

    def test_telegram_primary_secondary_edges(self, sample_signal):
        text = formatter.format_for_telegram(sample_signal)
        assert "Primary:" in text
        assert "Secondary:" in text


class TestSummaryOneLiner:
    """Summary one-liner format."""

    def test_summary_contains_tier_direction_symbol_score_risk(self, sample_signal):
        s = formatter.format_summary(sample_signal)
        assert "[Tier B]" in s or "B" in s
        assert "🟢" in s
        assert "LONG" in s
        assert sample_signal.symbol in s
        assert str(sample_signal.entry_price) in s
        assert "Score:" in s
        assert str(sample_signal.edge_score) in s
        assert "Risk:" in s
        assert "£" in s


class TestAlertFormatting:
    """Alert formatting with all priorities."""

    @pytest.mark.parametrize("priority", list(AlertPriority))
    def test_discord_alert_has_embed_and_color(self, priority):
        out = formatter.format_alert("Test message", priority)
        assert "embeds" in out
        assert len(out["embeds"]) == 1
        embed = out["embeds"][0]
        assert "title" in embed
        assert "description" in embed
        assert "color" in embed
        p_val = priority.value
        assert embed["color"] == ALERT_COLORS.get(p_val, ALERT_COLORS["normal"])

    @pytest.mark.parametrize("priority", list(AlertPriority))
    def test_telegram_alert_has_emoji_and_message(self, priority):
        text = formatter.format_alert_telegram("Hello", priority)
        assert "*Alert*" in text
        assert "Hello" in text
        p_val = priority.value
        assert ALERT_EMOJIS.get(p_val, ALERT_EMOJIS["normal"]) in text


class TestEdgeCases:
    """Edge cases: missing fields, zero risk, no secondary edges."""

    def test_missing_confluence_factors(self, sample_signal):
        sig = sample_signal.model_copy(update={"confluence_factors": []})
        out = formatter.format_for_discord(sig)
        confluence_field = next(
            f for f in out["embeds"][0]["fields"] if f["name"] == "Confluence"
        )
        assert confluence_field["value"] == "—"

    def test_zero_risk(self, sample_signal):
        sig = sample_signal.model_copy(
            update={"risk_amount": 0.0, "risk_percent": 0.0}
        )
        out = formatter.format_for_discord(sig)
        risk_field = next(f for f in out["embeds"][0]["fields"] if f["name"] == "Risk")
        assert "£0.00" in risk_field["value"]
        assert "0.00%" in risk_field["value"]

    def test_no_secondary_edges(self, sample_signal):
        sig = sample_signal.model_copy(update={"secondary_edges": []})
        out = formatter.format_for_discord(sig)
        secondary_field = next(
            f for f in out["embeds"][0]["fields"] if f["name"] == "Secondary"
        )
        assert secondary_field["value"] == "None"

    def test_no_secondary_telegram(self, sample_signal):
        sig = sample_signal.model_copy(update={"secondary_edges": []})
        text = formatter.format_for_telegram(sig)
        assert "Secondary:" in text
        assert "None" in text
