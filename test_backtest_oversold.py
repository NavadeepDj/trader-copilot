"""Run oversold bounce backtest on a single Nifty 50 stock and on the watchlist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_agents.tools.backtest_oversold import backtest_oversold_bounce, backtest_oversold_nifty50

print("=" * 60)
print("1. Single stock: RELIANCE.NS (2 years, RSI<=35, exit RSI>=45)")
print("=" * 60)

r = backtest_oversold_bounce("RELIANCE.NS", years=2, rsi_entry=35, rsi_exit=45, max_hold_days=10, stop_atr_mult=0.8)
if r.get("status") == "error":
    print("Error:", r.get("error_message"))
else:
    print("Symbol:", r.get("symbol"))
    print("Total trades:", r.get("total_trades"))
    print("Win rate %:", r.get("win_rate_pct"))
    print("Avg return %:", r.get("avg_return_pct"))
    if r.get("starting_capital_inr") is not None:
        print("Starting capital (INR):", r["starting_capital_inr"])
        print("Ending capital (INR):", r["ending_capital_inr"])
        print("Total P&L (INR):", r["total_pnl_inr"])
        print("Total P&L %:", r["total_pnl_pct"], "%")
    if r.get("trades"):
        print("\nLast 5 trades:")
        for t in r["trades"][-5:]:
            extra = f"  qty={t.get('qty')}  P&L=INR {t.get('pnl_inr')}" if t.get("qty") is not None else ""
            print(f"  {t['entry_date']} -> {t['exit_date']}  {t['entry_price']} -> {t['exit_price']}  {t['return_pct']}%  [{t['exit_reason']}]{extra}")

print("\n" + "=" * 60)
print("2. Nifty 50 watchlist (first 10 stocks, 2 years)")
print("=" * 60)

n50 = backtest_oversold_nifty50(years=2, max_stocks=10, use_portfolio_sizing=True)
print("Stocks run:", n50.get("stocks_run"))
print("Stocks with trades:", n50.get("stocks_with_trades"))
if n50.get("starting_capital_inr") is not None:
    print("Starting capital (INR):", n50["starting_capital_inr"])
    print("Ending capital (INR):", n50["ending_capital_inr"])
    print("Total P&L (INR):", n50["total_pnl_inr"])
    print("Total P&L %:", n50["total_pnl_pct"], "%")
print("\nTop 5 by win rate:")
for s in n50.get("top_by_win_rate", [])[:5]:
    pnl = f"  P&L INR {s.get('pnl_inr')}" if s.get("pnl_inr") is not None else ""
    print(f"  {s['symbol']}: {s['total_trades']} trades, win_rate={s['win_rate_pct']}%, avg_return={s['avg_return_pct']}%{pnl}")
print("\nTop 5 by avg return:")
for s in n50.get("top_by_avg_return", [])[:5]:
    print(f"  {s['symbol']}: {s['avg_return_pct']}% avg, win_rate={s['win_rate_pct']}%")

print("\nDone.")
