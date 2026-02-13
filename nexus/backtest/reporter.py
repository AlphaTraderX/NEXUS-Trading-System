"""
Backtest Reporter

Generates formatted reports from multi-asset backtest results.
"""

from typing import Dict, Any

from .models import MultiBacktestResult, EdgePerformance


class BacktestReporter:
    """Generate reports from backtest results."""

    def generate_summary(self, result: MultiBacktestResult) -> str:
        lines = [
            "=" * 70,
            "NEXUS BACKTEST SUMMARY",
            "=" * 70,
            "",
            f"Period: {result.start_date.date()} to {result.end_date.date()}",
            f"Duration: {(result.end_date - result.start_date).days} days",
            "",
            "## ACCOUNT PERFORMANCE",
            f"Starting Balance:  ${result.starting_balance:>12,.2f}",
            f"Ending Balance:    ${result.ending_balance:>12,.2f}",
            f"Net P&L:           ${result.net_pnl:>12,.2f}",
            f"Net Return:        {result.net_return_pct:>12.2f}%",
            f"Max Drawdown:      ${result.max_drawdown:>12,.2f} ({result.max_drawdown_pct:.2f}%)",
            "",
            "## TRADE STATISTICS",
            f"Total Trades:      {result.total_trades:>12}",
            f"Winners:           {result.total_winners:>12}",
            f"Losers:            {result.total_losers:>12}",
            f"Win Rate:          {result.win_rate:>12.1f}%",
            f"Avg Trade P&L:     ${result.avg_trade_pnl:>12,.2f}",
            "",
            "## COSTS",
            f"Gross P&L:         ${result.gross_pnl:>12,.2f}",
            f"Total Costs:       ${result.total_costs:>12,.2f}",
            f"Cost Ratio:        {(result.total_costs / abs(result.gross_pnl) * 100) if result.gross_pnl != 0 else 0:>12.1f}%",
        ]

        return "\n".join(lines)

    def generate_edge_report(self, result: MultiBacktestResult) -> str:
        lines = [
            "",
            "=" * 70,
            "PERFORMANCE BY EDGE TYPE",
            "=" * 70,
        ]

        sorted_edges = sorted(
            result.edge_performance.items(),
            key=lambda x: x[1].net_pnl,
            reverse=True,
        )

        for edge_key, perf in sorted_edges:
            status = "[+]" if perf.is_profitable else "[-]"

            lines.extend([
                "",
                f"{status} {edge_key.upper()}",
                f"   Trades:      {perf.total_trades:>8}  (W:{perf.winners} L:{perf.losers})",
                f"   Win Rate:    {perf.win_rate:>8.1f}%",
                f"   Net P&L:     ${perf.net_pnl:>8,.2f}",
                f"   Expectancy:  {perf.expectancy:>8.2f}R",
                f"   Avg Net %:   {perf.avg_net_pct:>8.3f}%",
                f"   Significance:{perf.statistical_significance:>8}",
            ])

        return "\n".join(lines)

    def generate_timeframe_report(self, result: MultiBacktestResult) -> str:
        lines = [
            "",
            "=" * 70,
            "PERFORMANCE BY TIMEFRAME",
            "=" * 70,
        ]

        for tf_key, perf in result.timeframe_performance.items():
            if perf.total_trades == 0:
                continue

            status = "[+]" if perf.net_pnl > 0 else "[-]"

            lines.extend([
                "",
                f"{status} {tf_key}",
                f"   Trades:      {perf.total_trades:>8}",
                f"   Win Rate:    {perf.win_rate:>8.1f}%",
                f"   Net P&L:     ${perf.net_pnl:>8,.2f}",
            ])

        return "\n".join(lines)

    def generate_full_report(
        self,
        result: MultiBacktestResult,
        ai_analysis: Dict[str, Any] = None,
    ) -> str:
        report = []

        report.append(self.generate_summary(result))
        report.append(self.generate_edge_report(result))
        report.append(self.generate_timeframe_report(result))

        if ai_analysis:
            report.append("")
            report.append("=" * 70)
            report.append("AI ANALYSIS")
            report.append("=" * 70)
            report.append(ai_analysis.get("raw_analysis", ""))

        return "\n".join(report)
