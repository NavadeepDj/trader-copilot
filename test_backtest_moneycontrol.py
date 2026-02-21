"""Backtest dividend strategy: only recommended stocks, metrics + uptrend, stop = entry - 1*ATR, sell at least 1 day before ex."""
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_agents.tools.backtest_dividend import (
    backtest_current_moneycontrol_dividends_filtered,
    backtest_single_event,
)

def _print_result(result, title):
    print(title)
    print("=" * 60)
    if result.get("status") == "error":
        print("Error:", result.get("error_message"))
        return
    print("Source:", result.get("source"))
    print("Metrics used:", result.get("metrics_used", "—"))
    print("Are these recommended?", result.get("are_recommended", "—"))
    if result.get("disclaimer"):
        print("Note:", result["disclaimer"])
    print("Events backtested:", result.get("events_tested", 0))
    print("Win rate %:", result.get("win_rate_pct"))
    print("Avg return %:", result.get("avg_return_pct"))
    print("Total return %:", result.get("total_return_pct"))
    if result.get("results"):
        print("\nPer-event results:")
        for r in result["results"]:
            exit_reason = f" [{r.get('exit_reason', '')}]" if r.get("exit_reason") else ""
            stop_used = f" (stop {r.get('stop_used')})" if r.get("stop_used") is not None else ""
            print(f"  {r.get('company', '?')} ({r.get('symbol')}){stop_used}")
            print(f"    Buy {r.get('buy_date')} @ {r.get('buy_price')} -> Sell {r.get('sell_date')} @ {r.get('sell_price')} => {r.get('return_pct')}%{exit_reason}")
    if result.get("skipped_sample"):
        print("\nSkipped (sample):", result["skipped_sample"][:5])
    print()

# Recommended stocks only: health + 50-DMA + uptrend, stop = entry - 1*ATR, sell at least 1 day before ex
result = backtest_current_moneycontrol_dividends_filtered(sell_days_before_ex=1)
_print_result(result, "Backtest RECOMMENDED dividend stocks (metrics + stop = entry - 1*ATR)")

print("Single event: PI Industries (ann 12-Feb, ex 23-Feb)")
print("=" * 60)
single = backtest_single_event(
    "PIIND.NS",
    announcement_date="2026-02-12",
    ex_date="2026-02-23",
)
if single.get("status") == "success":
    print(f"Buy: {single['buy_date']} @ {single['buy_price']}")
    print(f"Sell: {single['sell_date']} @ {single['sell_price']} => {single['return_pct']}%")
else:
    print("Error:", single.get("error_message"))

print("\nDone.")
