"""
Scanner Validation Script - test fixed scanners with real or placeholder data.

Run BEFORE enabling live execution to confirm:
1. Scanners produce signals (not empty)
2. Signals have correct direction per strategy
3. Edge data matches expected format (notional_pct, score, strategy)
4. Score/notional values are set correctly
5. Alert delivery works (optional)

Usage:
    python -m nexus.scripts.validate_scanners              # Use Polygon if API key set
    python -m nexus.scripts.validate_scanners --offline     # Use placeholder data only
    python -m nexus.scripts.validate_scanners --alerts      # Also test alert delivery
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nexus.core.enums import AlertPriority, Direction, EdgeType
from nexus.core.models import Opportunity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

# Expected configuration per scanner (from validated backtest parameters)
EXPECTED_CONFIG = {
    "GapScanner": {
        "edge_type": EdgeType.GAP_FILL,
        "symbols": ["SPY", "NVDA", "TSLA", "AAPL", "AMD", "COIN", "ROKU", "SHOP", "SQ", "MARA"],
        "directions": {Direction.LONG, Direction.SHORT},  # Gap up=LONG, gap down=SHORT
        "required_edge_keys": ["gap_pct", "volume_ratio", "notional_pct"],
        "expected_notional": 16,
        "min_score": 1,  # Dynamic scoring, but must be set
    },
    "OvernightPremiumScanner": {
        "edge_type": EdgeType.OVERNIGHT_PREMIUM,
        "symbols": ["SPY", "QQQ", "TSLA", "NVDA", "AMD", "AAPL", "GOOGL", "META", "NFLX", "CRM"],
        "directions": {Direction.LONG},  # Always LONG
        "required_edge_keys": ["regime_filter", "sma_200", "notional_pct"],
        "expected_notional": 20,
        "min_score": 60,
    },
    "VWAPScanner": {
        "edge_type": EdgeType.VWAP_DEVIATION,
        "symbols": ["SPY", "QQQ", "IWM"],
        "directions": {Direction.LONG, Direction.SHORT},  # Crossover both ways
        "required_edge_keys": ["vwap", "deviation_pct", "volume_ratio", "notional_pct"],
        "expected_notional": 16,
        "min_score": 65,
    },
    "RSIScanner": {
        "edge_type": EdgeType.RSI_EXTREME,
        "symbols": ["SPY", "QQQ"],
        "directions": {Direction.LONG, Direction.SHORT},  # Oversold=LONG, overbought=SHORT
        "required_edge_keys": ["rsi_value", "adx", "sma_200", "notional_pct"],
        "expected_notional": 16,
        "min_score": 75,
    },
}


def validate_opportunity(opp: Opportunity, scanner_name: str) -> List[str]:
    """Validate a single opportunity against expected configuration."""
    errors = []
    config = EXPECTED_CONFIG.get(scanner_name, {})

    # Check raw_score is set
    score = getattr(opp, "raw_score", None)
    if score is None or score == 0:
        errors.append(f"raw_score not set (got {score})")

    min_score = config.get("min_score", 0)
    if score is not None and score < min_score:
        errors.append(f"raw_score {score} < expected minimum {min_score}")

    # Check direction is valid for this scanner
    expected_dirs = config.get("directions", set())
    if expected_dirs and opp.direction not in expected_dirs:
        errors.append(
            f"direction {opp.direction} not in expected {expected_dirs}"
        )

    # Check symbol is from validated list
    expected_symbols = config.get("symbols", [])
    if expected_symbols and opp.symbol not in expected_symbols:
        errors.append(
            f"symbol {opp.symbol} not in validated list {expected_symbols}"
        )

    # Check required edge_data keys
    for key in config.get("required_edge_keys", []):
        if key not in opp.edge_data:
            errors.append(f"missing edge_data key: {key}")

    # Check notional_pct value
    expected_notional = config.get("expected_notional")
    actual_notional = opp.edge_data.get("notional_pct")
    if expected_notional and actual_notional != expected_notional:
        errors.append(
            f"notional_pct {actual_notional} != expected {expected_notional}"
        )

    # Basic sanity checks
    if opp.entry_price <= 0:
        errors.append(f"entry_price <= 0: {opp.entry_price}")
    if opp.stop_loss <= 0:
        errors.append(f"stop_loss <= 0: {opp.stop_loss}")

    # Direction-specific stop/target sanity
    if opp.direction == Direction.LONG:
        if opp.stop_loss >= opp.entry_price:
            errors.append(
                f"LONG stop {opp.stop_loss} >= entry {opp.entry_price}"
            )
        if opp.take_profit <= opp.entry_price:
            errors.append(
                f"LONG target {opp.take_profit} <= entry {opp.entry_price}"
            )
    elif opp.direction == Direction.SHORT:
        if opp.stop_loss <= opp.entry_price:
            errors.append(
                f"SHORT stop {opp.stop_loss} <= entry {opp.entry_price}"
            )
        if opp.take_profit >= opp.entry_price:
            errors.append(
                f"SHORT target {opp.take_profit} >= entry {opp.entry_price}"
            )

    return errors


async def validate_scanner(scanner: Any, name: str) -> Dict:
    """Run a scanner and validate its output."""
    result = {
        "name": name,
        "success": False,
        "signals": 0,
        "errors": [],
        "warnings": [],
        "sample_signals": [],
    }

    try:
        opportunities = await scanner.scan()
        result["signals"] = len(opportunities)

        for opp in opportunities:
            opp_errors = validate_opportunity(opp, name)
            if opp_errors:
                result["errors"].extend(
                    [f"{opp.symbol}: {e}" for e in opp_errors]
                )

            result["sample_signals"].append({
                "symbol": opp.symbol,
                "direction": opp.direction.value if hasattr(opp.direction, "value") else str(opp.direction),
                "entry": round(opp.entry_price, 2),
                "stop": round(opp.stop_loss, 2),
                "target": round(opp.take_profit, 2),
                "raw_score": getattr(opp, "raw_score", None),
                "edge_data_keys": list(opp.edge_data.keys()),
            })

        if not opportunities:
            result["warnings"].append(
                "No signals generated (may be normal depending on market conditions)"
            )

        result["success"] = len(result["errors"]) == 0

    except Exception as e:
        result["errors"].append(f"Scanner crashed: {e}")
        logger.error(f"{name} failed: {e}", exc_info=True)

    return result


async def validate_scanner_config(scanner: Any, name: str) -> List[str]:
    """Validate scanner static configuration without running scan()."""
    errors = []
    config = EXPECTED_CONFIG.get(name, {})

    # Check edge_type
    expected_edge = config.get("edge_type")
    if expected_edge and scanner.edge_type != expected_edge:
        errors.append(
            f"edge_type {scanner.edge_type} != expected {expected_edge}"
        )

    # Check INSTRUMENTS has validated symbols
    expected_symbols = config.get("symbols", [])
    if expected_symbols and hasattr(scanner, "INSTRUMENTS"):
        from nexus.core.enums import Market
        actual_symbols = []
        for market_symbols in scanner.INSTRUMENTS.values():
            actual_symbols.extend(market_symbols)
        if set(actual_symbols) != set(expected_symbols):
            errors.append(
                f"INSTRUMENTS {sorted(actual_symbols)} != expected {sorted(expected_symbols)}"
            )

    # Check score is set (for scanners that have it)
    if hasattr(scanner, "score"):
        min_score = config.get("min_score", 0)
        if scanner.score < min_score:
            errors.append(f"score {scanner.score} < expected {min_score}")

    return errors


def is_market_open() -> bool:
    """Check if US market is currently open (rough check)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


async def main() -> int:
    """Run validation on all fixed scanners."""
    parser = argparse.ArgumentParser(description="Validate NEXUS scanners")
    parser.add_argument(
        "--offline", action="store_true",
        help="Use placeholder data only (no Polygon API)",
    )
    parser.add_argument(
        "--alerts", action="store_true",
        help="Also test alert delivery via Discord/Telegram",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NEXUS SCANNER VALIDATION")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Market: {'OPEN' if is_market_open() else 'CLOSED'}")
    print()

    # Initialize data provider
    data_provider = None
    if not args.offline:
        try:
            from nexus.data.polygon import PolygonProvider
            provider = PolygonProvider()
            connected = await provider.connect()
            if connected:
                data_provider = provider
                print("Data: Polygon.io (live)")
            else:
                print("Data: Polygon connection failed, using placeholder data")
        except Exception as e:
            print(f"Data: Polygon unavailable ({e}), using placeholder data")

    if data_provider is None:
        print("Data: Placeholder (offline mode)")
    print()

    # Initialize scanners
    from nexus.scanners.gap import GapScanner
    from nexus.scanners.overnight import OvernightPremiumScanner
    from nexus.scanners.vwap import VWAPScanner
    from nexus.scanners.rsi import RSIScanner

    scanners = [
        (GapScanner(data_provider=data_provider), "GapScanner"),
        (OvernightPremiumScanner(data_provider=data_provider), "OvernightPremiumScanner"),
        (VWAPScanner(data_provider=data_provider), "VWAPScanner"),
        (RSIScanner(data_provider=data_provider), "RSIScanner"),
    ]

    all_results = []
    total_errors = 0

    for scanner, name in scanners:
        print(f"--- {name} ---")

        # Phase 1: Static config validation
        config_errors = await validate_scanner_config(scanner, name)
        if config_errors:
            print(f"  CONFIG ERRORS:")
            for err in config_errors:
                print(f"    - {err}")
            total_errors += len(config_errors)
        else:
            config = EXPECTED_CONFIG.get(name, {})
            print(f"  Config OK: edge={scanner.edge_type.value}, "
                  f"symbols={len(config.get('symbols', []))}, "
                  f"score={getattr(scanner, 'score', 'N/A')}")

        # Phase 2: Run scanner
        result = await validate_scanner(scanner, name)
        all_results.append(result)

        if result["signals"] > 0:
            print(f"  Signals: {result['signals']}")
            for sig in result["sample_signals"][:3]:  # Show up to 3
                print(
                    f"    {sig['symbol']} {sig['direction']} "
                    f"@ {sig['entry']:.2f} | "
                    f"Score={sig['raw_score']} | "
                    f"Keys={sig['edge_data_keys']}"
                )
        else:
            print(f"  Signals: 0 (no current opportunities)")

        if result["errors"]:
            print(f"  ERRORS:")
            for err in result["errors"]:
                print(f"    - {err}")
            total_errors += len(result["errors"])

        if result["warnings"]:
            for warn in result["warnings"]:
                print(f"  Warning: {warn}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_signals = sum(r["signals"] for r in all_results)
    scanners_with_errors = [r for r in all_results if r["errors"]]

    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['signals']} signals, {len(r['errors'])} errors")

    print(f"\nTotal signals: {total_signals}")
    print(f"Total errors: {total_errors}")

    if total_errors > 0:
        print(f"\nFAILED: {total_errors} error(s) found")
    else:
        print(f"\nPASSED: All scanners validated")

    # Optional alert delivery test
    if args.alerts:
        print("\n--- Alert Delivery Test ---")
        try:
            from nexus.delivery.alert_manager import create_alert_manager
            manager = create_alert_manager()

            if not manager._channels:
                print("  No delivery channels configured (check settings)")
            else:
                channels = list(manager._channels.keys())
                print(f"  Channels: {channels}")

                # Test connections
                test_results = await manager.test_all_channels()
                for ch_name, ch_result in test_results.items():
                    status = "OK" if ch_result.success else f"FAIL: {ch_result.error_message}"
                    print(f"  {ch_name}: {status}")

                # Send validation summary
                summary = (
                    f"**NEXUS Scanner Validation**\n\n"
                    f"Scanners: {len(all_results)}\n"
                    f"Signals: {total_signals}\n"
                    f"Errors: {total_errors}\n"
                    f"Status: {'PASSED' if total_errors == 0 else 'FAILED'}"
                )
                record = await manager.send_alert(summary, AlertPriority.NORMAL)
                if record.status.value == "sent":
                    print("  Validation summary sent successfully")
                else:
                    print(f"  Alert delivery: {record.status.value} ({record.error})")

                await manager.close()
        except Exception as e:
            print(f"  Alert test failed: {e}")

    # Disconnect data provider
    if data_provider is not None:
        await data_provider.disconnect()

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
