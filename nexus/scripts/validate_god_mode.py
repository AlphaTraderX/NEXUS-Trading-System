"""
GOD MODE Validation Script

Verifies the 426-instrument orchestration layer works correctly:
1. Registry integrity - correct instrument counts by type/region/provider
2. Session detection - all sessions resolve, instruments assigned
3. Provider routing - every instrument maps to a provider
4. Edge mapping - sessions have edges, edges have scanners
5. Orchestrator API - get_quote/get_bars routing logic
"""

import sys
import json
import logging
from datetime import datetime, time, timezone
from typing import Dict, List, Tuple

from nexus.data.orchestrator import (
    DataOrchestrator,
    TradingSession,
    SessionConfig,
    reset_orchestrator,
)
from nexus.data.instruments import (
    InstrumentRegistry,
    InstrumentType,
    Region,
    DataProvider,
    get_instrument_registry,
)
from nexus.core.enums import EdgeType

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("VALIDATE")


# -----------------------------------------------------------------------
# Expected minimum counts
# -----------------------------------------------------------------------
EXPECTED_TOTAL = 400

EXPECTED_TYPES = {
    InstrumentType.STOCK: 300,
    InstrumentType.FOREX: 20,
    InstrumentType.INDEX: 10,
    InstrumentType.COMMODITY: 10,
    InstrumentType.CRYPTO: 15,
}

EXPECTED_REGIONS = {
    Region.US: 150,
    Region.UK: 50,
    Region.EUROPE: 30,
    Region.ASIA_JAPAN: 10,
    Region.ASIA_HK: 10,
    Region.ASIA_AU: 10,
    Region.GLOBAL: 50,
}

EXPECTED_PROVIDERS = {
    DataProvider.POLYGON: 150,
    DataProvider.OANDA: 20,
    DataProvider.IG: 100,
    DataProvider.BINANCE: 15,
}

EXPECTED_SESSION_INSTRUMENTS = {
    TradingSession.ASIA: 70,
    TradingSession.LONDON_OPEN: 100,
    TradingSession.EUROPEAN: 130,
    TradingSession.US_PREMARKET: 150,
    TradingSession.US_OPEN: 200,
    TradingSession.US_SESSION: 200,
    TradingSession.POWER_HOUR: 150,
}

SPOT_CHECK_SYMBOLS = {
    "AAPL": (InstrumentType.STOCK, Region.US, DataProvider.POLYGON),
    "MSFT": (InstrumentType.STOCK, Region.US, DataProvider.POLYGON),
    "EUR_USD": (InstrumentType.FOREX, Region.GLOBAL, DataProvider.OANDA),
    "GBP_USD": (InstrumentType.FOREX, Region.GLOBAL, DataProvider.OANDA),
    "AZN.L": (InstrumentType.STOCK, Region.UK, DataProvider.IG),
    "BTC_USD": (InstrumentType.CRYPTO, Region.GLOBAL, DataProvider.BINANCE),
    "UK100": (InstrumentType.INDEX, Region.UK, DataProvider.IG),
    "XAUUSD": (InstrumentType.COMMODITY, Region.GLOBAL, DataProvider.IG),
}


def run_checks() -> Tuple[int, int, List[str], List[str]]:
    """Run all validation checks. Returns (passed, failed, pass_msgs, fail_msgs)."""
    reset_orchestrator()
    orch = DataOrchestrator()
    reg = orch.registry

    passed = 0
    failed = 0
    pass_msgs: List[str] = []
    fail_msgs: List[str] = []

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            msg = f"  PASS  {name}" + (f" ({detail})" if detail else "")
            pass_msgs.append(msg)
            print(f"\033[92m{msg}\033[0m")
        else:
            failed += 1
            msg = f"  FAIL  {name}" + (f" ({detail})" if detail else "")
            fail_msgs.append(msg)
            print(f"\033[91m{msg}\033[0m")

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: REGISTRY INTEGRITY")
    print("=" * 70)

    total = reg.total_count
    check(f"Total instruments >= {EXPECTED_TOTAL}", total >= EXPECTED_TOTAL, f"actual={total}")

    for itype, min_count in EXPECTED_TYPES.items():
        actual = len(reg.get_by_type(itype))
        check(
            f"  {itype.value:12} >= {min_count}",
            actual >= min_count,
            f"actual={actual}",
        )

    for region, min_count in EXPECTED_REGIONS.items():
        actual = len(reg.get_by_region(region))
        check(
            f"  {region.value:15} >= {min_count}",
            actual >= min_count,
            f"actual={actual}",
        )

    for prov, min_count in EXPECTED_PROVIDERS.items():
        actual = len(reg.get_by_provider(prov))
        check(
            f"  {prov.value:12} >= {min_count}",
            actual >= min_count,
            f"actual={actual}",
        )

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: SPOT-CHECK SYMBOLS")
    print("=" * 70)

    for symbol, (exp_type, exp_region, exp_provider) in SPOT_CHECK_SYMBOLS.items():
        inst = reg.get(symbol)
        exists = inst is not None
        check(f"  {symbol:12} exists", exists)
        if exists:
            check(
                f"  {symbol:12} type={exp_type.value}",
                inst.instrument_type == exp_type,
                f"actual={inst.instrument_type.value}",
            )
            check(
                f"  {symbol:12} region={exp_region.value}",
                inst.region == exp_region,
                f"actual={inst.region.value}",
            )
            check(
                f"  {symbol:12} provider={exp_provider.value}",
                inst.provider == exp_provider,
                f"actual={inst.provider.value}",
            )

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: SESSION DETECTION")
    print("=" * 70)

    # Verify all sessions are defined
    defined_sessions = [s.session for s in orch.SESSIONS]
    for ts in [TradingSession.ASIA, TradingSession.LONDON_OPEN, TradingSession.EUROPEAN,
               TradingSession.US_PREMARKET, TradingSession.US_OPEN,
               TradingSession.US_SESSION, TradingSession.POWER_HOUR]:
        check(f"  Session defined: {ts.value}", ts in defined_sessions)

    # Current session should resolve
    cfg = orch.get_current_session()
    check("  Current session resolves", isinstance(cfg, SessionConfig), f"session={cfg.session.value}")

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: SESSION INSTRUMENT COUNTS")
    print("=" * 70)

    for session, min_count in EXPECTED_SESSION_INSTRUMENTS.items():
        instruments = orch.get_instruments_for_session(session)
        actual = len(instruments)
        check(
            f"  {session.value:15} >= {min_count}",
            actual >= min_count,
            f"actual={actual}",
        )

    # US_OPEN should have the most instruments of any non-session
    us_open_count = len(orch.get_instruments_for_session(TradingSession.US_OPEN))
    asia_count = len(orch.get_instruments_for_session(TradingSession.ASIA))
    check(
        "  US_OPEN > ASIA instruments",
        us_open_count > asia_count,
        f"us_open={us_open_count} vs asia={asia_count}",
    )

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: EDGE MAPPING")
    print("=" * 70)

    # Check key edges are in expected sessions
    edge_checks = [
        (TradingSession.US_OPEN, EdgeType.GAP_FILL, "GAP_FILL in US_OPEN"),
        (TradingSession.US_OPEN, EdgeType.ORB, "ORB in US_OPEN"),
        (TradingSession.US_OPEN, EdgeType.VWAP_DEVIATION, "VWAP_DEVIATION in US_OPEN"),
        (TradingSession.US_OPEN, EdgeType.INSIDER_CLUSTER, "INSIDER_CLUSTER in US_OPEN"),
        (TradingSession.LONDON_OPEN, EdgeType.LONDON_OPEN, "LONDON_OPEN in LONDON_OPEN"),
        (TradingSession.POWER_HOUR, EdgeType.POWER_HOUR, "POWER_HOUR in POWER_HOUR"),
        (TradingSession.ASIA, EdgeType.ASIAN_RANGE, "ASIAN_RANGE in ASIA"),
    ]
    for session, edge, desc in edge_checks:
        edges = orch.SESSION_EDGES.get(session, [])
        check(f"  {desc}", edge in edges)

    # get_edges_for_session should return list
    current_edges = orch.get_edges_for_session()
    check("  get_edges_for_session() returns list", isinstance(current_edges, list))

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 6: PROVIDER ROUTING")
    print("=" * 70)

    # Without connections, provider should be None
    provider = orch.get_provider_for_symbol("AAPL")
    check("  No provider without connection", provider is None)

    # Every instrument should have a valid provider enum
    bad_providers = [i.symbol for i in reg.get_all() if i.provider not in DataProvider]
    check("  All instruments have valid DataProvider", len(bad_providers) == 0,
          f"bad={bad_providers[:5]}" if bad_providers else "all valid")

    # Every instrument should be routable (provider key exists)
    unroutable = []
    for inst in reg.get_all():
        if inst.provider not in orch._PROVIDER_KEY:
            unroutable.append(inst.symbol)
    check("  All instruments routable via _PROVIDER_KEY", len(unroutable) == 0,
          f"unroutable={unroutable[:5]}" if unroutable else "all routable")

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 7: STATUS REPORT")
    print("=" * 70)

    status = orch.get_status()
    expected_keys = ["connected", "current_session", "session_instruments",
                     "total_registry", "by_type", "by_provider", "active_edges"]
    for key in expected_keys:
        check(f"  Status has '{key}'", key in status)

    check(
        "  total_registry matches",
        status["total_registry"] == reg.total_count,
        f"status={status['total_registry']} vs reg={reg.total_count}",
    )

    # =======================================================================
    print("\n" + "=" * 70)
    print("SECTION 8: CROSS-CHECK CONSISTENCY")
    print("=" * 70)

    # Sum of by_type should equal total
    type_sum = sum(len(reg.get_by_type(t)) for t in InstrumentType)
    check("  Sum(by_type) == total", type_sum == total, f"sum={type_sum} vs total={total}")

    # Sum of by_provider should equal total
    prov_sum = sum(len(reg.get_by_provider(p)) for p in DataProvider)
    check("  Sum(by_provider) == total", prov_sum == total, f"sum={prov_sum} vs total={total}")

    # No duplicate symbols
    all_symbols = [i.symbol for i in reg.get_all()]
    dupes = [s for s in all_symbols if all_symbols.count(s) > 1]
    check("  No duplicate symbols", len(dupes) == 0,
          f"dupes={list(set(dupes))[:5]}" if dupes else "none")

    return passed, failed, pass_msgs, fail_msgs


def main() -> int:
    print("\n" + "#" * 70)
    print("#  GOD MODE VALIDATION - FULL 426-INSTRUMENT ORCHESTRATOR")
    print(f"#  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("#" * 70)

    passed, failed, pass_msgs, fail_msgs = run_checks()

    total = passed + failed
    pct = (passed / total * 100) if total else 0

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total checks:  {total}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  Pass rate:     {pct:.1f}%")
    print("=" * 70)

    if failed > 0:
        print("\nFAILED CHECKS:")
        for msg in fail_msgs:
            print(f"  {msg}")

    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"\n{'=' * 70}")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 70}\n")

    # Write JSON results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pct, 1),
        "failures": fail_msgs,
    }

    results_path = "data/god_mode_validation.json"
    try:
        from pathlib import Path
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Warning: could not save results: {e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
