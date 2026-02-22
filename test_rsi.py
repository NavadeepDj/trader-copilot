"""Test RSI oversold backtest"""
from trading_agents.tools.backtest_oversold import backtest_oversold_nifty50
import json

print("Running RSI oversold backtest on Nifty 50 stocks (2 years)...")
print("=" * 60)

r = backtest_oversold_nifty50(years=2, max_stocks=15)

# Print summary
summary = {k: v for k, v in r.items() if k != 'per_stock'}
print("\nSUMMARY:")
print(json.dumps(summary, indent=2))

print("\nPER-STOCK RESULTS:")
print("-" * 60)
for s in r.get('per_stock', []):
    if s.get('total_trades', 0) > 0:
        print(f"  {s['symbol']:15} | Win: {s.get('win_rate_pct', 'N/A'):5}% | Avg: {s.get('avg_return_pct', 'N/A'):6}% | Trades: {s.get('total_trades', 0):2} | PnL: {s.get('pnl_inr', 'N/A')}")
