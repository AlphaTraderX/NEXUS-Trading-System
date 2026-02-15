# NEXUS TODO - Prioritized Action Items

## 🔴 CRITICAL - Blocking 60%+ of Returns

### TODO-001: ~~Fix Calendar Edges~~ RESOLVED - Not a Bug
**Priority**: ~~P0~~ N/A
**Impact**: Low - Calendar edges ARE generating trades (not 0)
**Location**: `nexus/backtest/engine.py` lines 1176-1289
**Status**: RESOLVED (2026-02-15)

**Findings (debugged 2026-02-15)**:
- TURN_OF_MONTH: **99 trades**, PF 1.28, +$1,544 (15.4%), **VALID** on v2.1 engine
- MONTH_END: **55 trades**, PF 1.05, +$56 (0.6%), INVALID (weak edge, not a bug)
- Calendar edges were never broken — the TODO was written based on incorrect information
- The real 0-trade edges are ORB, Power Hour, London/NY/Asian, Bollinger, Earnings
  (these are DISABLED because they need intraday data that doesn't exist yet)

**No action needed.** Focus on TODO-002 (intraday data) for the real +1000% path.

---

### TODO-002: Implement Intraday Data Pipeline
**Priority**: P0 - Critical
**Impact**: Very High - Unlocks 60-75% of missing signals
**Location**: All provider files, cache system
**Status**: NOT STARTED

**Tasks**:
1. [ ] Add `get_bars(timeframe='5m')` to PolygonProvider
2. [ ] Add `get_bars(timeframe='5m')` to OandaProvider
3. [ ] Add `get_bars(timeframe='5m')` to BinanceProvider
4. [ ] Add `get_bars(timeframe='5m')` to IGProvider
5. [ ] Create `warm_intraday_cache.py` script
6. [ ] Cache 3 months of 5m data for key instruments
7. [ ] Validate data quality and gaps

**Estimated Time**: 1-2 weeks

---

### TODO-003: Create Intraday Backtest Engine
**Priority**: P0 - Critical
**Impact**: Very High - Validates 17-29 signals/day expectation
**Location**: `nexus/backtest/engine_v3.py` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Iterate over 5m bars instead of daily
2. [ ] Calculate true intraday VWAP
3. [ ] Detect session boundaries (Asia, London, US)
4. [ ] Fire session-specific edges at correct times
5. [ ] Compound after each trade (not end of day)
6. [ ] Compare results to daily bar backtest

**Expected Outcome**: 2-3x more signals, proportionally higher returns

---

### TODO-004: Paper Trading Runner
**Priority**: P0 - Critical for validation
**Impact**: High - Validates backtest in real-time
**Location**: `nexus/execution/runner.py` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Create main execution loop
2. [ ] Implement paper broker (simulated fills)
3. [ ] Real-time data subscription (polling or websocket)
4. [ ] Signal generation on new bars
5. [ ] Position tracking and P&L
6. [ ] Alert generation
7. [ ] Run for 1 week, compare to backtest

**Estimated Time**: 2-3 weeks

---

## 🟡 HIGH - Blocking 20-30% of Returns

### TODO-005: Implement ORB Scanner (Opening Range Breakout)
**Priority**: P1
**Impact**: Medium-High - 1-2 signals/day expected
**Location**: `nexus/scanners/orb.py`
**Requires**: TODO-002 (intraday data)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Detect first 30 min range on 5m bars
2. [ ] Signal on breakout with volume confirmation
3. [ ] VWAP filter for quality
4. [ ] Add to scanner orchestrator

---

### TODO-006: Implement Power Hour Scanner
**Priority**: P1
**Impact**: Medium - 2-3 signals/day expected
**Location**: `nexus/scanners/power_hour.py` (new)
**Requires**: TODO-002 (intraday data)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Detect last 60 min of US session
2. [ ] Volume surge detection
3. [ ] Trend continuation signals
4. [ ] Add to scanner orchestrator

---

### TODO-007: Implement Session Scanners (London, NY, Asian)
**Priority**: P1
**Impact**: Medium - 5-10 signals/day expected combined
**Location**: `nexus/scanners/session.py` (new)
**Requires**: TODO-002 (intraday data)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Define session boundaries (UK time)
2. [ ] London Open scanner (07:00-09:00)
3. [ ] NY Open scanner (14:30-16:00)
4. [ ] Asian Range scanner (00:00-07:00)
5. [ ] Add to scanner orchestrator

---

### TODO-008: ~~Optimize Regime Configuration~~ COMPLETE
**Priority**: ~~P1~~ Done
**Impact**: Medium - Improved from +292% to +469.8%
**Location**: `nexus/intelligence/regime_detector.py`
**Status**: COMPLETE (2026-02-15)

**Results**:
- All edges now allowed in ALL regimes (size-adjusted, never blocked)
- Softened multipliers: STRONG_BULL 1.1x, SIDEWAYS 1.0x, BEAR 0.9x, STRONG_BEAR 0.85x, VOLATILE 0.85x
- Insider cluster cooldown reduced from 12→3 bars (+309 signals recovered)
- v2.1 tuned: 4,392 trades, PF 1.46, +$46,980 (+469.8%), MaxDD 26.9%
- Without regime filter: +501.8% (regime sizing still costs ~$3K — acceptable tradeoff for risk reduction)

---

## 🟢 MEDIUM - 10-15% Improvement

### TODO-009: Add News Filter (ForexFactory)
**Priority**: P2
**Impact**: Medium - Avoid trading during high-impact events
**Location**: `nexus/data/economic_calendar.py` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Scrape ForexFactory calendar
2. [ ] Detect high-impact events
3. [ ] 30 min pre/post event buffer
4. [ ] Integrate with signal generator

---

### TODO-010: Add Sentiment Integration (StockTwits)
**Priority**: P2
**Impact**: Medium - Documented edge when combined with volume
**Location**: `nexus/data/sentiment.py` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] StockTwits API integration
2. [ ] Sentiment spike detection
3. [ ] Volume confirmation filter
4. [ ] Add as signal confirmation factor

---

### TODO-011: Add Correlation Monitoring
**Priority**: P2
**Impact**: Medium - Avoid hidden concentration risk
**Location**: `nexus/risk/correlation.py` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] Calculate rolling correlation matrix
2. [ ] Block trades with >0.7 correlation to existing
3. [ ] Track effective portfolio risk

---

### TODO-012: Optimize Signal Cooldowns
**Priority**: P2
**Impact**: Low-Medium - May unlock additional signals
**Location**: `nexus/backtest/engine.py`
**Status**: NOT STARTED

**Tasks**:
1. [ ] Audit all cooldown values
2. [ ] Test 50% reduction across all edges
3. [ ] Find optimal per-edge cooldowns
4. [ ] Verify not overtrading same setup

---

## 🔵 LOW - Nice to Have

### TODO-013: Create Monitoring Dashboard
**Priority**: P3
**Impact**: Low - Operational improvement
**Location**: `nexus/dashboard/` (new)
**Status**: NOT STARTED

**Tasks**:
1. [ ] React frontend
2. [ ] Real-time P&L display
3. [ ] Open positions
4. [ ] Signal history
5. [ ] Regime indicator

---

### TODO-014: Add Email Alerts
**Priority**: P3
**Impact**: Low - Convenience
**Location**: `nexus/delivery/email.py` (new)
**Status**: NOT STARTED

---

### TODO-015: Performance Benchmarking
**Priority**: P3
**Impact**: Low - Optimization
**Status**: NOT STARTED

**Tasks**:
1. [ ] Profile backtest execution time
2. [ ] Identify bottlenecks
3. [ ] Optimize slow paths

---

## 📊 Progress Tracker

| TODO | Status | Started | Completed | Impact |
|------|--------|---------|-----------|--------|
| 001 | RESOLVED | 2026-02-15 | 2026-02-15 | N/A (not a bug) |
| 002 | Not Started | - | - | Very High |
| 003 | Not Started | - | - | Very High |
| 004 | Not Started | - | - | High |
| 005 | Not Started | - | - | Medium-High |
| 006 | Not Started | - | - | Medium |
| 007 | Not Started | - | - | Medium |
| 008 | COMPLETE | 2026-02-15 | 2026-02-15 | +177% improvement |
| 009 | Not Started | - | - | Medium |
| 010 | Not Started | - | - | Medium |
| 011 | Not Started | - | - | Medium |
| 012 | Not Started | - | - | Low-Medium |
| 013 | Not Started | - | - | Low |
| 014 | Not Started | - | - | Low |
| 015 | Not Started | - | - | Low |

---

## Quick Wins (< 1 day each)

1. ~~**Fix Calendar Edges**~~ RESOLVED - Not broken, 99+55 trades generating correctly
2. **Remove Regime Filter** - One-line change, +$8K P&L immediately
3. **Reduce Cooldowns** - Config change, test impact

## Longer Projects (1+ weeks)

1. **Intraday Data Pipeline** - Foundation for everything else
2. **Paper Trading Runner** - Validates system in real-time
3. **Dashboard** - Nice to have but not critical

---

*Last updated: 2026-02-15*
