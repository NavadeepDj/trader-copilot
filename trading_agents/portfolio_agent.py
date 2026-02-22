"""Portfolio manager sub-agent -- queries and manages paper portfolio state."""

from __future__ import annotations

from typing import Dict

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL
from trading_agents.tools.portfolio import (
    get_portfolio_performance,
    get_portfolio_summary,
    refresh_portfolio_positions,
    reset_portfolio,
)


def view_portfolio() -> Dict:
    """Get the current paper portfolio summary including cash, positions, and PnL.

    Returns:
        dict with full portfolio overview.
    """
    return get_portfolio_summary()


def reset_paper_portfolio() -> Dict:
    """Reset the paper portfolio to its initial state (INR 10,00,000 cash, no positions).

    Returns:
        dict confirming the reset.
    """
    return reset_portfolio()


def refresh_trade_lifecycle() -> Dict:
    """Refresh open trades and auto-close on stop, target, or time exit."""
    return refresh_portfolio_positions()


def view_performance() -> Dict:
    """Get risk/return performance including net profit and max drawdown."""
    return get_portfolio_performance()


portfolio_agent = Agent(
    name="portfolio_manager",
    model=GEMINI_MODEL,
    description=(
        "Manages and reports on the paper trading portfolio. "
        "Shows current holdings, cash balance, invested amount, and trade history."
    ),
    instruction=(
        "You are the Portfolio Manager. When asked about the portfolio, "
        "use view_portfolio to show current state. "
        "If the user asks for performance, returns, drawdown, or win rate, use view_performance. "
        "If the user asks to update/refresh open trades, use refresh_trade_lifecycle. "
        "Format amounts in INR with proper formatting (e.g., INR 10,00,000). "
        "Show each open position with symbol, qty, entry, stop, target. "
        "Only reset the portfolio when explicitly asked."
    ),
    tools=[view_portfolio, view_performance, refresh_trade_lifecycle, reset_paper_portfolio],
)
