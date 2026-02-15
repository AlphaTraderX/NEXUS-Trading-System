# NEXUS System Audit & Gap Analysis

## Executive Summary

**NEXUS is a well-architected trading system that is operationally limited to ~40% of its potential due to missing intraday data and execution infrastructure.**

| Area | Design | Implementation | Gap |
|------|--------|----------------|-----|
| Data Providers | 4 providers, 426 instruments | ✅ Complete | None |
| Daily Backtesting | Score-based, compounding, trailing stops | ✅ Complete | None |
| Intraday Backtesting | 5m/15m bars, session-aware | ❌ Not implemented | **Critical** |
| Paper Trading | Real-time scanning, execution | ❌ Not implemented | **Critical** |
| Live Trading | Broker order execution | ❌ Not implemented | Future |

---

## Section 1: What's Working Well

### 1.1 Data Infrastructure
- **Polygon**: 192 US stocks, paid tier, unlimited requests
- **OANDA**: 28 forex pairs, demo account
- **Binance**: 20 crypto pairs, public API (no auth needed)
- **IG Markets**: 186 instruments (UK/EU/Asia stocks, indices, commodities)
- **Total**: 426 instruments across 8 asset classes

### 1.2 Backtest Engine v2.1
The enhanced backtest engine includes:
- Score-based position sizing (0.5x - 1.5x based on A-F tiers)
- Intraday compounding (equity updates after each trade)
- Trailing stops and breakeven stops
- 6-state regime detection (STRONG_BULL → VOLATILE)
- Regime-aware edge filtering with size adjustments
- ATR-based dynamic stops and targets
- Heat management (portfolio risk tracking)

### 1.3 Risk Management
- Position sizer with score, regime, and momentum multipliers
- Dynamic heat limits (expand with profits, contract with losses)
- Circuit breakers (daily/weekly loss limits)
- Kill switch (emergency shutdown)

### 1.4 Intelligence Layer
- Cost engine (calculates true trading costs including spread, commission, slippage)
- Opportunity scorer (A-F tiers based on edge stacking)
- Regime detector (ADX + ATR percentile classification)
- AI reasoning (Groq Llama 3.1 70B for signal explanation)

### 1.5 Test Coverage
- 828 tests passing
- Unit tests for all components
- Integration tests for backtest engine
- Edge-specific validation tests

---

## Section 2: Critical Gaps

### 2.1 GAP #1: Daily Bars Only (CRITICAL)

**Problem**: The entire backtest runs on daily (1d) bars.

**Impact**:
- Only 1 entry opportunity per instrument per day
- Session-specific edges cannot fire (London Open, NY Open, Power Hour)
- VWAP deviation signals are end-of-day, not intraday extremes
- Gap fill signals detected after the gap has potentially filled
- Opening Range Breakout (ORB) impossible on daily bars

**Evidence**:
```python
# From nexus/scripts/warm_cache.py
TIMEFRAME = "1d"  # <-- Only daily bars cached

# From nexus/backtest/engine.py
bars = await provider.get_bars(symbol, "1d", limit=500)  # <-- Daily only
```

**Required Fix**:
1. Add 5m and 15m bar support to all providers
2. Create intraday cache warming script
3. Modify backtest to iterate over intraday bars
4. Calculate VWAP, ORB, session edges from intraday data

### 2.2 GAP #2: No Session-Aware Scanning (CRITICAL)

**Problem**: All signals generated at end-of-day, not during trading sessions.

**Expected Architecture (from v2.1 doc)**:
```
Session Schedule:
├── 00:00-07:00 UK: Asia session → Forex, Crypto scanners
├── 07:00-09:00 UK: London Open → Forex, UK stock scanners
├── 09:00-14:30 UK: European → EU stock scanners
├── 12:00-14:30 UK: US Pre-Market → Gap scanner
├── 14:30-16:00 UK: US Open → All scanners (highest volume)
├── 16:00-20:00 UK: US Session → US stock, forex scanners
├── 20:00-21:00 UK: Power Hour → US stock scanner
```

**Current Implementation**: None. Backtest iterates day-by-day, not session-by-session.

**Required Fix**:
1. Create SessionScheduler class
2. Run appropriate scanners per session
3. Track session P&L separately
4. Time-weight signals by session (US Open highest, Asia lowest)

### 2.3 GAP #3: No Paper Trading Runner (CRITICAL)

**Problem**: No live execution loop exists. System can only backtest.

**Expected Components**:
- Main loop that runs 24/5
- Real-time data subscription
- Signal generation on new bars
- Paper order submission
- Position tracking
- P&L calculation

**Current Implementation**: None.

**Required Fix**:
1. Create `nexus/execution/runner.py` - main execution loop
2. Create `nexus/execution/paper_broker.py` - simulated order execution
3. Create `nexus/execution/position_tracker.py` - track open positions
4. Create `nexus/scripts/run_paper.py` - CLI to start paper trading

### 2.4 GAP #4: Missing Intraday Edges (HIGH)

**Edges that require intraday data but only have daily implementation**:

| Edge | Requires | Current | Status |
|------|----------|---------|--------|
| VWAP Deviation | 5m intraday VWAP | Daily close vs 20-day avg | ⚠️ Degraded |
| ORB | First 30 min of session | Not implemented | ❌ Missing |
| Power Hour | Last 60 min of session | Not implemented | ❌ Missing |
| Gap Fill | Pre-market gap | Daily open vs close | ⚠️ Degraded |
| London Open | 07:00-09:00 UK | Not implemented | ❌ Missing |
| NY Open | 14:30-16:00 UK | Not implemented | ❌ Missing |
| Asian Range | 00:00-07:00 UK | Not implemented | ❌ Missing |

### 2.5 GAP #5: Signal Cooldown May Be Too Long (MEDIUM)

**Recent Fix**: Reduced insider_cluster cooldown from 12 bars (2.5 weeks) to 3 bars.
**Result**: +160 additional profitable signals, +$11,533 P&L.

**Remaining Concern**: Other edges may have suboptimal cooldowns.

```python
# From nexus/backtest/engine.py
self.SIGNAL_COOLDOWN_BARS = {
    EdgeType.VWAP_DEVIATION: 5,    # 5 days - may be too long for intraday
    EdgeType.RSI_EXTREME: 5,       # 5 days
    EdgeType.BOLLINGER_TOUCH: 12,  # 12 days - very long
    EdgeType.POWER_HOUR: 3,        # 3 days
    EdgeType.ORB: 78,              # One per day (for 5m bars)
    EdgeType.GAP_FILL: 5,          # 5 days
    EdgeType.LONDON_OPEN: 999,     # Once per session
    # ...
}
```

**With intraday bars**: Cooldowns should be in bars (e.g., 78 5m bars = 1 day), not days.

---

## Section 3: Performance Analysis

### 3.1 Current vs Expected Returns

| Metric | Current (Daily) | Expected (Intraday) | Gap |
|--------|-----------------|---------------------|-----|
| Signals/day | 6.6 | 17-29 | -62% to -77% |
| Trades/day | 6.6 | 10-18 | -40% to -63% |
| Daily return | 0.52% | 1.0-1.5% | -50% to -65% |
| Monthly return | ~13% | 15-25% | -15% to -48% |
| Annual return | ~156% | 300-600% | -50% to -74% |

### 3.2 Edge-by-Edge Analysis

**Best Performing (Daily Bars)**:
| Edge | Trades | Win Rate | P&L | Return |
|------|--------|----------|-----|--------|
| Insider Cluster | 393 | 57.5% | +$20,222 | +202% |
| Overnight Premium | 1,632 | 55.1% | +$20,175 | +202% |
| Gap Fill | 618 | 70.2% | +$2,197 | +22% |
| RSI Extreme | 1,112 | 48.6% | +$2,118 | +21% |
| VWAP Deviation | 637 | 50.4% | +$2,268 | +23% |

**Underperforming (Daily Bars)**:
| Edge | Trades | Win Rate | P&L | Issue |
|------|--------|----------|-----|-------|
| Turn of Month | 0 | - | $0 | No signals generated |
| Month End | 0 | - | $0 | No signals generated |
| Power Hour | - | - | - | Not implemented (needs intraday) |
| ORB | - | - | - | Not implemented (needs intraday) |
| London/NY Open | - | - | - | Not implemented (needs sessions) |

### 3.3 Why Turn of Month / Month End Show Zero

**Investigation Needed**: These calendar edges should fire 4-6 days per month but show zero trades in backtest. Possible causes:
1. Scanner not detecting the calendar window correctly
2. Signals being filtered by regime (too strict)
3. Cooldown too long (15-20 bars)
4. Bug in signal generation

---

## Section 4: Code Quality Assessment

### 4.1 Strengths
- Clean separation of concerns (data, scanners, intelligence, risk, execution)
- Comprehensive test coverage (828 tests)
- Pydantic models for data validation
- Async/await for efficient I/O
- Proper logging throughout
- Git tags for version milestones

### 4.2 Areas for Improvement
- Some scanners have hardcoded thresholds (should be configurable)
- Backtest engine is 600+ lines (could be split into smaller modules)
- No type hints in some older modules
- Limited integration tests (mostly unit tests)
- No performance benchmarks

### 4.3 Technical Debt
- `engine.py` and `engine_v2.py` share a lot of code (could use inheritance)
- Cache warming scripts are provider-specific (could be unified)
- Some TODO comments not addressed

---

## Section 5: Recommended Action Plan

### Phase 1: Intraday Data (1-2 weeks)
1. Add `get_bars(timeframe='5m')` support to all providers
2. Create `warm_intraday_cache.py` script
3. Cache 3 months of 5m data for all instruments
4. Validate data quality and gaps

### Phase 2: Intraday Backtest (1-2 weeks)
1. Modify `BacktestEngineV2` to iterate over intraday bars
2. Implement session boundaries (Asia, London, US)
3. Calculate true intraday VWAP
4. Implement ORB, Power Hour, session edges
5. Run full intraday backtest and compare to daily

### Phase 3: Paper Trading (2-3 weeks)
1. Create `runner.py` main execution loop
2. Implement paper broker (simulated fills)
3. Real-time data subscription (websockets where available)
4. Position tracking and P&L
5. Alert generation (Discord, Telegram)
6. Run for 1 week, validate against backtest expectations

### Phase 4: Optimization (1 week)
1. Tune signal cooldowns for intraday
2. Optimize regime multipliers
3. A/B test edge combinations
4. Validate 17-29 signals/day target

---

## Section 6: Questions for Review

1. **Why are Turn of Month and Month End showing zero trades?** These are documented high-probability edges but generate no signals in backtest.

2. **Is the regime filtering too aggressive?** Even with tuned configs, ~10% of signals are filtered by regime. Should we use size adjustment only (no filtering)?

3. **Should cooldowns be bar-based or time-based?** Currently bar-based, but with intraday bars this could mean very short cooldowns.

4. **Is the compounding implementation correct?** The backtest shows intraday compounding but on daily bars - does this make sense?

5. **Why does removing regime filter improve returns?** v2.1 no-regime (+501.8%) outperforms v2.1 tuned (+469.8%). Should regime filtering be removed entirely?

6. **Are the trailing stop parameters optimal?** Current: 1.5 ATR trail, 1.0 ATR breakeven trigger. Should these vary by edge?

7. **Is the score-based sizing working correctly?** A-tier signals should get 1.5x size, but are they actually being detected and sized correctly?

---

## Section 7: Files to Review

### Critical Files
- `nexus/backtest/engine_v2.py` - Main backtest logic
- `nexus/intelligence/regime_detector.py` - Regime configs and filtering
- `nexus/risk/position_sizer.py` - Score-based sizing
- `nexus/scanners/*.py` - Edge detection logic

### Configuration Files
- `nexus/config/settings.py` - All settings
- `.env` - API credentials (not in repo)

### Test Files
- `tests/test_regime_detector.py` - Regime logic tests
- `tests/test_engine_v2.py` - Backtest engine tests
- `tests/test_trailing_stops.py` - Trailing stop tests

### Data Files
- `data/backtest_v2_tuned.json` - Latest backtest results
- `data/cache/` - Cached price data (not in repo)

---

## Appendix A: Command Reference

```bash
# Run tests
python -m pytest tests/ -x --tb=short -q

# Run v1 backtest
python -m nexus.scripts.backtest_all --start 2022-01-01 --end 2024-12-31

# Run v2.1 backtest (tuned)
python -m nexus.scripts.backtest_v2 --start 2022-01-01 --end 2024-12-31

# Run v2.1 without regime filter
python -m nexus.scripts.backtest_v2 --no-regime-filter --start 2022-01-01 --end 2024-12-31

# Compare backtests
python -m nexus.scripts.compare_backtests data/backtest_v1.json data/backtest_v2.json

# Warm cache
python -m nexus.scripts.warm_cache --provider polygon --days 730 --delay 0.5

# Test provider connections
python -c "from nexus.data.polygon import PolygonProvider; import asyncio; asyncio.run(PolygonProvider().connect())"
```

---

## Appendix B: Key Metrics Definitions

- **Win Rate**: Percentage of trades with positive P&L
- **Profit Factor**: Gross profit / Gross loss (>1.0 = profitable)
- **Max Drawdown**: Largest peak-to-trough decline in equity
- **Heat**: Total portfolio risk across all open positions
- **Score Tier**: A (80-100), B (65-79), C (50-64), D (40-49), F (0-39)
- **Regime**: Market state (STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR, VOLATILE)

---

*Last updated: 2026-02-15*
*System version: v2.5-tuned-with-ig*
*Tests: 828 passing*
