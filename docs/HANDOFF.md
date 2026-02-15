# NEXUS AI Reviewer Handoff Document

## 🎯 TL;DR - What You Need to Know

**NEXUS is a trading system that should make 15-20% monthly but is stuck at ~13% monthly because it only uses daily bars instead of intraday.**

### The Numbers
| What We Have | What We Should Have |
|--------------|---------------------|
| +469.8% over 3 years | +1,000%+ over 3 years |
| 6.6 trades/day | 17-29 trades/day |
| Daily bars only | 5m, 15m, 1h, 1d bars |
| End-of-day scanning | 24/5 session scanning |
| Backtest only | Paper + Live trading |

### The Core Problem
The system is **architecturally complete** but **operationally limited**. It's like having a Ferrari engine in a car with no wheels.

---

## 🔍 Your Mission

Please review this system and help identify:

1. **Why are we only getting 40% of expected returns?**
   - Is it really just the daily bars limitation?
   - Are there bugs in the signal generation?
   - Are the edge implementations correct?

2. **Why do Turn of Month and Month End edges show ZERO trades?**
   - These are documented high-probability edges
   - They should fire 4-6 days per month
   - Backtest shows zero signals - why?

3. **Is the regime filtering helping or hurting?**
   - v2.1 no-regime: +501.8% return
   - v2.1 tuned regime: +469.8% return
   - Removing regime filter IMPROVES returns - should we remove it?

4. **What's the fastest path to 20%+ monthly returns?**
   - Intraday data pipeline?
   - Fix existing edge bugs?
   - Better regime/scoring tuning?
   - Something else entirely?

---

## 📁 Key Files to Examine

### Start Here (Priority Order)
1. `nexus/backtest/engine_v2.py` - The main backtest logic (~600 lines)
2. `nexus/intelligence/regime_detector.py` - Regime configs and filtering
3. `nexus/scanners/calendar.py` - Turn of Month / Month End (possibly broken)
4. `nexus/backtest/trade_simulator.py` - Trade simulation with trailing stops

### Supporting Files
- `nexus/core/enums.py` - EdgeType definitions
- `nexus/core/registry.py` - 426 instrument registry
- `nexus/risk/position_sizer.py` - Score-based sizing
- `data/backtest_v2_tuned.json` - Latest backtest results

### Architecture Docs (in project knowledge)
- `NEXUS_v2.1_MAXIMUM_OPPORTUNITY.md` - Expected signal frequency
- `NEXUS_v2.0_ARCHITECTURE.md` - Full system design
- `NEXUS_DEEP_RESEARCH_VALIDATION.md` - Academic backing for edges

---

## 🧪 Quick Validation Commands

```bash
# Check tests pass
python -m pytest tests/ -x --tb=short -q
# Expected: 828 passed

# Run backtest
python -m nexus.scripts.backtest_v2 --start 2022-01-01 --end 2024-12-31
# Expected: +469.8% return, 4,392 trades

# Check edge breakdown
cat data/backtest_v2_tuned.json | python -c "import json,sys; d=json.load(sys.stdin); [print(f\"{e['edge']}: {e['trades']} trades, {e['total_pnl_pct']:.1f}%\") for e in d['edges']]"
```

---

## 🚨 Known Issues

### Issue 1: Calendar Edges Not Firing
**Symptom**: Turn of Month and Month End show 0 trades
**Expected**: 4-6 signals per month
**Location**: `nexus/scanners/calendar.py`
**Hypothesis**: Scanner may not be detecting calendar windows correctly

### Issue 2: Intraday Edges Degraded
**Symptom**: VWAP, Gap Fill working but suboptimal
**Expected**: These need 5m bars for true intraday signals
**Current**: Using daily bars (degraded accuracy)
**Location**: `nexus/scanners/vwap.py`, `nexus/scanners/gap.py`

### Issue 3: Regime Filter Reduces Returns
**Symptom**: Removing regime filter adds +$8,000 P&L
**Expected**: Regime filter should IMPROVE risk-adjusted returns
**Location**: `nexus/intelligence/regime_detector.py`
**Hypothesis**: Size multipliers too aggressive (0.5x in VOLATILE)

### Issue 4: Missing Session Edges
**Symptom**: No London Open, NY Open, Power Hour, Asian Range signals
**Expected**: These are ~40% of daily opportunity
**Location**: Not implemented in backtest loop
**Required**: Session-aware scanning loop

---

## 📊 Data Available

### Cached Price Data (2022-2024)
| Provider | Instruments | Bars | Timeframe |
|----------|-------------|------|-----------|
| Polygon | 192 | ~500 each | Daily |
| OANDA | 27 | ~500 each | Daily |
| Binance | 20 | ~500 each | Daily |
| IG Markets | 181 | ~500 each | Daily |

### Backtest Results
- `data/backtest_426_real_baseline.json` - v1 baseline
- `data/backtest_v2_baseline.json` - v2.1 strict regime
- `data/backtest_v2_tuned.json` - v2.1 tuned regime
- `data/backtest_v2_no_regime.json` - v2.1 no regime filter

---

## 🔧 Development Environment

- **Python**: 3.11+
- **Database**: PostgreSQL (not currently used in backtest)
- **Testing**: pytest
- **Data Providers**: Polygon, OANDA, IG Markets, Binance
- **AI**: Groq Llama 3.1 70B (for signal reasoning)
- **Alerts**: Discord webhooks, Telegram bot

---

## 💡 Hypotheses to Test

### Hypothesis 1: Intraday Data is the Bottleneck
**Test**: Add 5m bar support, run intraday backtest
**Expected**: 2-3x more signals, proportionally higher returns

### Hypothesis 2: Calendar Scanner is Broken
**Test**: Add debug logging to `calendar.py`, trace signal generation
**Expected**: Find bug preventing signal generation

### Hypothesis 3: Regime Multipliers Too Aggressive
**Test**: Set all position_size_multipliers to 1.0
**Expected**: Match no-regime performance while keeping edge filtering

### Hypothesis 4: Cooldowns Too Long
**Test**: Reduce all cooldowns by 50%
**Expected**: More signals, need to verify not overtrading

### Hypothesis 5: Score Thresholds Too High
**Test**: Lower min_score from 50 to 40
**Expected**: More signals from lower-tier opportunities

---

## 🎯 Success Criteria

If you can help achieve any of these, it's a win:

1. **Identify why returns are 40% of expected** - Root cause analysis
2. **Fix Turn of Month / Month End edges** - Should add significant P&L
3. **Recommend optimal regime settings** - Filter vs no-filter decision
4. **Design intraday data pipeline** - Architecture for 5m bars
5. **Create paper trading spec** - What's needed for live execution

---

## 📞 Contact

This system is being reviewed to find performance leaks. Please provide:
- Specific code locations of issues found
- Recommended fixes with rationale
- Priority ranking of fixes
- Estimated impact of each fix

Thank you for your review! 🚀
