"""Test the full Moneycontrol -> yfinance dividend pipeline."""
import sys
sys.path.insert(0, ".")

from trading_agents.dividend_agent import scan_dividend_opportunities

print("=" * 60)
print("Full Dividend Pipeline: MC API -> Resolve -> Analyze -> Rank")
print("=" * 60)

result = scan_dividend_opportunities(min_days_to_ex=1)

print(f"\nStatus: {result.get('status')}")
print(f"Source: {result.get('discovery_source')}")
print(f"Discovered: {result.get('dividends_discovered')}")
print(f"Opportunities: {result.get('opportunities_count')}")
print(f"Skipped: {result.get('skipped_count')}")

for opp in result.get("top_opportunities", []):
    print(f"\n  {opp['symbol']} ({opp['company']})")
    print(f"    Health: {opp.get('dividend_health')} (score {opp.get('health_score')})")
    print(f"    Yield: {opp.get('dividend_yield_pct')}% | PE: {opp.get('trailing_pe')}")
    print(f"    Price: {opp.get('current_price')} | 50DMA: {opp.get('dma_50')} | Above: {opp.get('above_50dma')}")
    print(f"    Entry: {opp.get('suggested_entry')} | Stop: {opp.get('suggested_stop')}")
    print(f"    Ex-date: {opp.get('ex_date')} ({opp.get('days_to_ex')} days)")
    print(f"    Exit: {opp.get('suggested_exit')}")
    print(f"    Rank: {opp.get('rank_score')}")

if result.get("skipped_summary"):
    print("\n  SKIPPED:")
    for s in result["skipped_summary"]:
        print(f"    {s.get('symbol', '?')} ({s.get('company', '?')}) -- {s['reason']}")

if result.get("unmapped_companies"):
    print(f"\n  UNMAPPED (no NSE symbol found): {result['unmapped_companies']}")
