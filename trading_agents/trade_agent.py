"""Trade execution sub-agent -- calculates plans and executes paper trades."""

from __future__ import annotations

from typing import Dict

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL
from trading_agents.tools.paper_trading import (
    calculate_trade_plan,
    calculate_trade_plan_from_entry_stop,
    execute_paper_trade,
)


def plan_trade(symbol: str, close: float, atr: float) -> Dict:
    """Calculate a trade plan with entry, stop, target, position size, and risk amount.

    Args:
        symbol: Stock ticker symbol.
        close: Current closing price to use as entry.
        atr: Average True Range for stop-loss calculation.

    Returns:
        dict with the complete trade plan or error.
    """
    return calculate_trade_plan(symbol=symbol, close=close, atr=atr)


def plan_trade_from_dividend(symbol: str, entry: float, stop: float) -> Dict:
    """Calculate a trade plan using entry and stop from a scan (dividend or oversold bounce).

    Use when implementing a dividend pick OR an oversold bounce pick: pass the scan's
    entry (close/suggested_entry) and stop (suggested_stop). Target = 2R (entry + 2 * risk).

    Args:
        symbol: Stock ticker (e.g. ENGINERSIN.NS or RELIANCE.NS).
        entry: Entry price (scan's close or suggested_entry).
        stop: Stop-loss (scan's suggested_stop).

    Returns:
        dict with the complete trade plan or error.
    """
    return calculate_trade_plan_from_entry_stop(symbol=symbol, entry=entry, stop=stop)


def execute_trade(symbol: str, entry: float, stop: float, target: float, qty: int) -> Dict:
    """Execute a paper trade after validating risk rules.

    Args:
        symbol: Stock ticker symbol.
        entry: Entry price.
        stop: Stop-loss price.
        target: Target price.
        qty: Number of shares to buy.

    Returns:
        dict with execution result and portfolio impact.
    """
    return execute_paper_trade(symbol=symbol, entry=entry, stop=stop, target=target, qty=qty)


trade_agent = Agent(
    name="trade_executor",
    model=GEMINI_MODEL,
    description=(
        "Calculates trade plans with proper position sizing and executes paper trades. "
        "Enforces risk rules: 1% risk per trade, max 3 open positions, min 1:2 R:R. "
        "Supports generic plans (plan_trade) and scan-based entry/stop (plan_trade_from_dividend for dividend or oversold)."
    ),
    instruction=(
        "You are the Trade Executor. When asked to trade a stock, "
        "first compute a plan, then present it before using execute_trade.\n\n"
        "WHICH TOOL TO USE:\n"
        "- When implementing a dividend pick: use plan_trade_from_dividend(symbol, suggested_entry, suggested_stop) from the dividend scan.\n"
        "- When implementing an OVERSOLD BOUNCE pick (sideways/bear strategy): use plan_trade_from_dividend(symbol, close, suggested_stop) "
        "with the oversold scan's close and suggested_stop (same tool; entry=close, stop=suggested_stop).\n"
        "- For other trades (breakout, momentum): use plan_trade(symbol, close, atr) with current price and ATR.\n\n"
        "RULES:\n"
        "Never execute without showing the plan first. Always mention: symbol, entry, stop, "
        "target, qty, R:R ratio, and capital required. If risk rules block the trade, explain why."
    ),
    tools=[plan_trade, plan_trade_from_dividend, execute_trade],
)
