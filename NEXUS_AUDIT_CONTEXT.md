# NEXUS Trading System - Audit Context
## Complete System Knowledge for Code Review

---

## PROJECT OVERVIEW

NEXUS is a unified, multi-asset automated trading system designed for generational wealth building.

**Goals:**
- Starting capital: £10,000
- Target: £100,000 in 18-24 months
- Monthly returns: 10-15% (opportunity dependent)
- Max drawdown: 10% hard limit

**Tech Stack:**
- Python 3.11+, FastAPI, PostgreSQL, Pydantic
- Data: Polygon.io (US stocks), OANDA (forex)
- Alerts: Discord, Telegram
- AI: Groq Llama 3.1 70B

---

## THE 13 VALIDATED EDGES

Only edges with academic/statistical backing are implemented:

| Edge | Scanner File | Academic Basis | Key Parameters |
|------|--------------|----------------|----------------|
| **Insider Cluster** | insider.py | 2.1% monthly abnormal returns (Alldredge & Blank, 2019) | 3+ insiders buying within 14 days |
| **VWAP Deviation** | vwap.py | Sharpe 2.1 (Zarattini & Aziz study) | >2σ deviation from VWAP |
| **Turn of Month** | calendar.py | 100% of equity premium in 4-day window (McConnell & Xu, 2008) | Last trading day through day 3 |
| **Month End** | calendar.py | $7.5T pension fund rebalancing flows | Last 2 trading days |
| **Gap Fill** | gap.py | 60-92% fill rate for small gaps | Gaps 0.5-3%, fade toward fill |
| **RSI Extreme** | rsi.py | Mean reversion at extremes | Period 2-5 (NOT 14), thresholds 20/80 (NOT 30/70) |
| **Power Hour** | session.py | U-shaped volume pattern confirmed | 20:00-21:00 UK time |
| **Asian Range** | session.py | ICT framework validated | Break of Asian session high/low |
| **ORB** | orb.py | Opening range breakout | Requires volume >120% avg + VWAP alignment |
| **Bollinger Touch** | bollinger.py | 88% mean reversion on band touch | ONLY valid in RANGING regime |
| **London Open** | session.py | High volatility window | 07:00-08:00 UK, needs confirmation |
| **NY Open** | session.py | 24% probability of daily high in first 30 min | 14:30-15:30 UK |
| **Earnings Drift** | earnings.py | Post-earnings momentum | Small/mid-cap only, surprise >5% |

**REMOVED EDGES (arbitraged away):**
- FOMC Days (disappeared after 2015)
- Overnight Premium (transaction costs eliminate edge)
- Pre-Holiday (insignificant in large caps)
- Quarter-End Window Dressing (no evidence)

---

## RISK MANAGEMENT DESIGN

### Position Sizing (risk/position_sizer.py)
- Base risk: 1% per trade
- Score-adjusted: 0.5% (low conviction) to 1.5% (high conviction)
- ATR-based stops (1.5× ATR from entry)
- Regime multiplier: 0.5× in VOLATILE, 1.0× in TRENDING/RANGING

### Portfolio Heat (risk/heat_manager.py)
- Base limit: 25% of equity at risk
- Dynamic: Expands to 35% when profitable, contracts to 15% when losing
- Formula: Sum of (distance to stop × position size) for all positions

### Circuit Breakers (risk/circuit_breaker.py)
- Daily loss -1.5%: WARNING (alert only)
- Daily loss -2.0%: REDUCE (50% position sizes)
- Daily loss -3.0%: STOP (no new trades today)
- Weekly loss -6.0%: STOP (no new trades this week)
- Drawdown -10.0%: FULL STOP (manual review required)

**CRITICAL:** Loss values are NEGATIVE. Comparisons must use <= not >=

### Correlation Monitor (risk/correlation.py)
- Max 3 positions in same sector
- Max 3 same-direction positions per market
- Effective Risk = Nominal Risk × √(N × Average Correlation)
- Alert when correlation > 0.7

### Kill Switch (risk/kill_switch.py)
Triggers:
- Max drawdown hit
- Connection loss > 5 minutes
- Stale data > 30 seconds
- Manual activation

Actions:
- Cancel all pending orders
- Close all positions (optional)
- Disable new trades
- Alert all channels

---

## SCORING SYSTEM (intelligence/scorer.py)

| Score Range | Tier | Position Multiplier |
|-------------|------|---------------------|
| 80-100 | A | 1.5× |
| 65-79 | B | 1.25× |
| 50-64 | C | 1.0× |
| 40-49 | D | 0.5× |
| 0-39 | F | Don't trade |

**Score Components:**
- Primary edge: 15-35 points (based on edge strength)
- Secondary edges: Up to 25 points (confluence)
- Trend alignment: Up to 15 points
- Volume confirmation: Up to 10 points
- Regime alignment: Up to 10 points
- Risk/Reward ratio: Up to 10 points

---

## MARKET REGIME DETECTION (intelligence/regime.py)

Four states:
1. **TRENDING_UP**: ADX > 25, Price > SMA20 > SMA50
2. **TRENDING_DOWN**: ADX > 25, Price < SMA20 < SMA50
3. **RANGING**: Bollinger width < 80% of average
4. **VOLATILE**: ATR > 150% of 30-day average

**Strategy Mapping:**
- TRENDING_UP: TOM, Month End, Insider, ORB, Power Hour
- TRENDING_DOWN: VWAP, RSI, Gap Fill (shorts)
- RANGING: VWAP, RSI, Bollinger, Gap Fill, Asian Range
- VOLATILE: Insider only, 50% position sizes

---

## SIGNAL FLOW

```
Scanner.scan() → List[Opportunity]
    ↓
Orchestrator.run_scan_cycle() → Aggregated opportunities
    ↓
Scorer.score() → ScoredOpportunity with tier
    ↓
SignalGenerator.generate_signal() → NexusSignal
    ↓ (checks circuit breaker, heat, correlation)
OrderManager.create_entry_order() → Order
    ↓
TradeExecutor.execute() → Fill
    ↓
PositionManager.mark_open() → Position tracked
    ↓
AlertManager.send_signal() → Discord + Telegram
    ↓
StorageService.save_signal() → PostgreSQL (if connected)
```

---

## TEST COVERAGE

397 tests passing as of last run.

Key test files:
- `tests/test_execution/` - Order manager, position manager, trade executor
- `nexus/tests/test_risk/` - Position sizer, heat manager, circuit breaker
- `nexus/tests/test_intelligence/` - Scorer, regime, cost engine
- `nexus/tests/test_scanners/` - All 13 edge scanners

---

## KNOWN LIMITATIONS

- Paper trading script (`scripts/paper_trade.py`) is a stub - use `main.py --test` instead
- Database optional - System runs in memory-only mode if `DATABASE_URL` empty
- IBKR not tested - Connection code exists but not validated
- IG Markets not tested - Session auth implemented but untested

---

## LIVE INTEGRATIONS (VERIFIED WORKING)

- Polygon.io - US stock data (SPY, QQQ, IWM prices confirmed)
- OANDA - Forex data + £100K demo account
- Discord - Webhook alerts tested
- Telegram - Bot alerts tested (@NEXUSTS_BOT)
- Groq - AI reasoning configured

---

## WHAT TO AUDIT

- **Risk math correctness** - Will bad math blow up the account?
- **Edge logic accuracy** - Do scanners match academic specifications?
- **Integration integrity** - Does data flow correctly end-to-end?
- **Error handling** - What happens when things fail?
- **Security** - Are credentials safe?

---

## CONTACTS

- **Project Owner:** Stuge
- **Architecture:** Claude (Anthropic)
- **Implementation:** Cursor Agent
