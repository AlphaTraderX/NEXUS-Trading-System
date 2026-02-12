"""
P&L and Compounding Validation Script.

Simulates a sequence of trades to verify:
1. Per-trade P&L calculation (gross, costs, net)
2. Equity compounding (profits increase base for next trade)
3. Position sizing scales with equity growth
4. Tier multipliers work correctly

Run: python -m nexus.scripts.validate_pnl
"""

import logging
from dataclasses import dataclass, field
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Simulation Parameters
# ============================================================================

STARTING_EQUITY = 10000.00
BASE_NOTIONAL_PCT = 16  # 16% of equity per trade
COST_PER_TRADE_PCT = 0.02  # 0.02% round-trip cost (spread + slippage)

# Tier multipliers (from run_paper.py)
TIER_MULTIPLIERS = {
    "A": 1.5,   # Score 80+
    "B": 1.25,  # Score 65-79
    "C": 1.0,   # Score 50-64
    "D": 0.5,   # Score 40-49
}


@dataclass
class SimulatedTrade:
    """A simulated trade for validation."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    tier: str
    score: int

    # Calculated fields
    shares: int = 0
    position_value: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0


@dataclass
class ValidationState:
    """Track equity through simulation."""
    starting_equity: float
    current_equity: float
    trades: List[SimulatedTrade] = field(default_factory=list)

    def total_net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    def total_return_pct(self) -> float:
        return ((self.current_equity - self.starting_equity) / self.starting_equity) * 100


def calculate_position_size(
    current_equity: float,
    entry_price: float,
    tier: str,
    notional_pct: float = BASE_NOTIONAL_PCT,
) -> tuple:
    """
    Calculate position size using current equity (compounding).

    Returns: (shares, position_value)
    """
    tier_mult = TIER_MULTIPLIERS.get(tier, 1.0)
    position_value = current_equity * (notional_pct / 100) * tier_mult
    shares = int(position_value / entry_price) if entry_price > 0 else 0
    actual_value = shares * entry_price

    return shares, actual_value


def calculate_trade_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    shares: int,
    position_value: float,
) -> tuple:
    """
    Calculate trade P&L.

    Returns: (gross_pnl, costs, net_pnl)
    """
    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * shares
    else:  # SHORT
        gross_pnl = (entry_price - exit_price) * shares

    # Costs as % of position value (entry + exit)
    costs = position_value * (COST_PER_TRADE_PCT / 100) * 2
    net_pnl = gross_pnl - costs

    return gross_pnl, costs, net_pnl


def simulate_trades(trades_config: List[dict]) -> ValidationState:
    """
    Simulate a sequence of trades and track compounding.
    """
    state = ValidationState(
        starting_equity=STARTING_EQUITY,
        current_equity=STARTING_EQUITY,
    )

    logger.info("=" * 70)
    logger.info("P&L AND COMPOUNDING VALIDATION")
    logger.info("=" * 70)
    logger.info(f"\nStarting Equity: ${STARTING_EQUITY:,.2f}")
    logger.info(f"Base Notional:   {BASE_NOTIONAL_PCT}% of equity per trade")
    logger.info(f"Cost per Trade:  {COST_PER_TRADE_PCT}% round-trip\n")

    for i, config in enumerate(trades_config, 1):
        trade = SimulatedTrade(
            symbol=config["symbol"],
            direction=config["direction"],
            entry_price=config["entry_price"],
            exit_price=config["exit_price"],
            tier=config["tier"],
            score=config["score"],
        )

        # Record equity BEFORE this trade
        trade.equity_before = state.current_equity

        # Calculate position size using CURRENT equity (key for compounding)
        trade.shares, trade.position_value = calculate_position_size(
            current_equity=state.current_equity,
            entry_price=trade.entry_price,
            tier=trade.tier,
        )

        # Calculate P&L
        trade.gross_pnl, trade.costs, trade.net_pnl = calculate_trade_pnl(
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            shares=trade.shares,
            position_value=trade.position_value,
        )

        # Update equity AFTER trade closes
        state.current_equity += trade.net_pnl
        trade.equity_after = state.current_equity

        state.trades.append(trade)

        # Log trade details
        pnl_sign = "+" if trade.net_pnl >= 0 else ""
        pnl_pct = (trade.net_pnl / trade.equity_before) * 100

        logger.info(f"--- Trade {i}: {trade.symbol} ({trade.tier}-tier, score {trade.score}) ---")
        logger.info(f"  Direction:      {trade.direction}")
        logger.info(f"  Entry:          ${trade.entry_price:.2f}")
        logger.info(f"  Exit:           ${trade.exit_price:.2f}")
        logger.info(f"  Equity Before:  ${trade.equity_before:,.2f}")
        logger.info(f"  Position Size:  {trade.shares} shares @ ${trade.entry_price:.2f} = ${trade.position_value:,.2f}")
        logger.info(f"  Tier Mult:      {TIER_MULTIPLIERS[trade.tier]}x -> {BASE_NOTIONAL_PCT * TIER_MULTIPLIERS[trade.tier]:.1f}% of equity")
        logger.info(f"  Gross P&L:      ${trade.gross_pnl:,.2f}")
        logger.info(f"  Costs:          ${trade.costs:,.2f}")
        logger.info(f"  Net P&L:        {pnl_sign}${trade.net_pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)")
        logger.info(f"  Equity After:   ${trade.equity_after:,.2f}")
        logger.info("")

    return state


def print_summary(state: ValidationState):
    """Print validation summary."""
    logger.info("=" * 70)
    logger.info("COMPOUNDING SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nStarting Equity:  ${state.starting_equity:,.2f}")
    logger.info(f"Final Equity:     ${state.current_equity:,.2f}")
    logger.info(f"Total Net P&L:    ${state.total_net_pnl():,.2f}")
    logger.info(f"Total Return:     {state.total_return_pct():.2f}%")
    logger.info(f"Trades:           {len(state.trades)}")

    winners = [t for t in state.trades if t.net_pnl > 0]
    losers = [t for t in state.trades if t.net_pnl < 0]
    logger.info(f"Winners:          {len(winners)}")
    logger.info(f"Losers:           {len(losers)}")

    if winners:
        logger.info(f"Avg Win:          ${sum(t.net_pnl for t in winners) / len(winners):,.2f}")
    if losers:
        logger.info(f"Avg Loss:         ${sum(t.net_pnl for t in losers) / len(losers):,.2f}")

    # Verify compounding worked
    logger.info("\n" + "=" * 70)
    logger.info("COMPOUNDING VERIFICATION")
    logger.info("=" * 70)

    if len(state.trades) >= 2:
        t1 = state.trades[0]
        t2 = state.trades[1]

        # Check: did trade 2 use updated equity from trade 1?
        expected_equity_for_t2 = t1.equity_after
        actual_equity_used = t2.equity_before

        if abs(expected_equity_for_t2 - actual_equity_used) < 0.01:
            logger.info("PASS: Trade 2 used updated equity from Trade 1")
            logger.info(f"   Trade 1 ended at ${t1.equity_after:,.2f}")
            logger.info(f"   Trade 2 started with ${t2.equity_before:,.2f}")
        else:
            logger.info("FAIL: Compounding not working!")
            logger.info(f"   Expected: ${expected_equity_for_t2:,.2f}")
            logger.info(f"   Actual:   ${actual_equity_used:,.2f}")

    # Show position size growth
    logger.info("\n" + "-" * 70)
    logger.info("POSITION SIZE GROWTH (Compounding Effect)")
    logger.info("-" * 70)

    for i, trade in enumerate(state.trades, 1):
        base_position = STARTING_EQUITY * (BASE_NOTIONAL_PCT / 100) * TIER_MULTIPLIERS[trade.tier]
        actual_position = trade.position_value
        growth_pct = ((actual_position / base_position) - 1) * 100 if base_position > 0 else 0

        logger.info(
            f"Trade {i}: Position ${actual_position:,.2f} "
            f"(vs ${base_position:,.2f} if no compounding = {growth_pct:+.1f}%)"
        )


def main():
    """Run validation with test trades."""

    # Simulate realistic trade sequence
    # Mix of winners, losers, different tiers
    test_trades = [
        # Trade 1: Gap fill winner (C-tier)
        {
            "symbol": "SPY",
            "direction": "LONG",
            "entry_price": 500.00,
            "exit_price": 505.00,  # +1% move
            "tier": "C",
            "score": 55,
        },
        # Trade 2: Overnight winner (C-tier) - should use bigger equity
        {
            "symbol": "QQQ",
            "direction": "LONG",
            "entry_price": 430.00,
            "exit_price": 432.50,  # +0.58% move
            "tier": "C",
            "score": 60,
        },
        # Trade 3: RSI loser (B-tier)
        {
            "symbol": "NVDA",
            "direction": "LONG",
            "entry_price": 850.00,
            "exit_price": 840.00,  # -1.18% move
            "tier": "B",
            "score": 75,
        },
        # Trade 4: VWAP winner (B-tier)
        {
            "symbol": "TSLA",
            "direction": "LONG",
            "entry_price": 180.00,
            "exit_price": 185.00,  # +2.78% move
            "tier": "B",
            "score": 70,
        },
        # Trade 5: A-tier high conviction winner
        {
            "symbol": "AMD",
            "direction": "LONG",
            "entry_price": 160.00,
            "exit_price": 168.00,  # +5% move
            "tier": "A",
            "score": 85,
        },
        # Trade 6: D-tier small position loser
        {
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_price": 185.00,
            "exit_price": 182.00,  # -1.62% move
            "tier": "D",
            "score": 42,
        },
        # Trade 7: Short trade winner (C-tier)
        {
            "symbol": "MARA",
            "direction": "SHORT",
            "entry_price": 25.00,
            "exit_price": 23.50,  # -6% for short = profit
            "tier": "C",
            "score": 60,
        },
        # Trade 8: Short trade loser (B-tier)
        {
            "symbol": "COIN",
            "direction": "SHORT",
            "entry_price": 220.00,
            "exit_price": 226.00,  # +2.7% against short
            "tier": "B",
            "score": 70,
        },
    ]

    state = simulate_trades(test_trades)
    print_summary(state)

    # Final verification
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULT")
    logger.info("=" * 70)

    # Check all conditions
    checks = []

    # 1. Equity changed
    equity_changed = state.current_equity != state.starting_equity
    checks.append(("Equity updates after trades", equity_changed))

    # 2. Compounding works (trade 2 used trade 1's ending equity)
    if len(state.trades) >= 2:
        compounding_works = abs(state.trades[1].equity_before - state.trades[0].equity_after) < 0.01
        checks.append(("Compounding (next trade uses updated equity)", compounding_works))

    # 3. Tier multipliers work
    c_tier = [t for t in state.trades if t.tier == "C"]
    a_tier = [t for t in state.trades if t.tier == "A"]
    if c_tier and a_tier:
        # A-tier should have ~1.5x the position ratio vs C-tier
        c_ratio = c_tier[0].position_value / c_tier[0].equity_before
        a_ratio = a_tier[0].position_value / a_tier[0].equity_before
        expected_ratio = TIER_MULTIPLIERS["A"] / TIER_MULTIPLIERS["C"]  # 1.5
        actual_ratio = a_ratio / c_ratio
        tier_works = 1.4 < actual_ratio < 1.6  # Allow rounding from int(shares)
        checks.append(("Tier multipliers (A=1.5x vs C=1.0x)", tier_works))

    # 4. D-tier has smallest position
    d_tier = [t for t in state.trades if t.tier == "D"]
    if d_tier and c_tier:
        d_ratio = d_tier[0].position_value / d_tier[0].equity_before
        c_ratio_first = c_tier[0].position_value / c_tier[0].equity_before
        d_smaller = d_ratio < c_ratio_first
        checks.append(("D-tier position smaller than C-tier", d_smaller))

    # 5. Costs deducted
    costs_work = all(t.costs > 0 and t.net_pnl < t.gross_pnl for t in state.trades)
    checks.append(("Costs deducted from gross P&L", costs_work))

    # 6. Notional allocation grows with equity after winners
    # (actual position_value depends on stock price + int truncation, so compare
    # the notional TARGET = equity * pct * tier_mult, not the truncated value)
    if len(state.trades) >= 2:
        for i, t in enumerate(state.trades[:-1]):
            if t.net_pnl > 0:
                next_t = state.trades[i + 1]
                target_before = t.equity_before * (BASE_NOTIONAL_PCT / 100) * TIER_MULTIPLIERS[t.tier]
                target_after = next_t.equity_before * (BASE_NOTIONAL_PCT / 100) * TIER_MULTIPLIERS[next_t.tier]
                # Adjust for tier difference to isolate equity growth
                tier_adj = TIER_MULTIPLIERS[next_t.tier] / TIER_MULTIPLIERS[t.tier]
                adjusted_target = target_after / tier_adj if tier_adj else 0
                # After a win, same-tier-adjusted notional must be larger
                positions_grow = adjusted_target > target_before
                checks.append(("Notional allocation grows after wins (compounding)", positions_grow))
                break

    # 7. Short trades calculate P&L correctly
    short_trades = [t for t in state.trades if t.direction == "SHORT"]
    if short_trades:
        short_winner = [t for t in short_trades if t.gross_pnl > 0]
        short_loser = [t for t in short_trades if t.gross_pnl < 0]
        short_pnl_correct = True
        for t in short_winner:
            # Short profit: entry > exit
            if t.entry_price <= t.exit_price:
                short_pnl_correct = False
        for t in short_loser:
            # Short loss: exit > entry
            if t.exit_price <= t.entry_price:
                short_pnl_correct = False
        checks.append(("Short trade P&L direction correct", short_pnl_correct))

    # 8. Equity chain is consistent
    equity_chain_ok = True
    for i in range(1, len(state.trades)):
        if abs(state.trades[i].equity_before - state.trades[i - 1].equity_after) > 0.01:
            equity_chain_ok = False
            break
    checks.append(("Equity chain consistent across all trades", equity_chain_ok))

    # Print results
    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {check_name}")
        if not passed:
            all_pass = False

    logger.info("")
    if all_pass:
        logger.info("ALL CHECKS PASSED - Compounding and P&L tracking working correctly")
    else:
        logger.info("SOME CHECKS FAILED - Review implementation")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
