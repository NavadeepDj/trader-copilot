"""Paper trade execution and position sizing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
from typing import Dict

from trading_agents.config import (
    ATR_STOP_MULTIPLIER,
    MAX_OPEN_TRADES,
    MIN_REWARD_RISK,
    RISK_PER_TRADE,
)
from trading_agents.models import PortfolioState, Position, TradePlan
from trading_agents.tools.portfolio import load_portfolio, save_portfolio


def calculate_trade_plan(symbol: str, close: float, atr: float) -> Dict:
    """Build a trade plan with entry, stop (1.5xATR), target (2R), and position size.

    Args:
        symbol: Stock ticker.
        close: Current closing price (entry price).
        atr: Average True Range for stop calculation.

    Returns:
        dict with the trade plan details.
    """
    entry = close
    stop = round(max(0.01, entry - (ATR_STOP_MULTIPLIER * atr)), 2)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"status": "error", "error_message": "Risk per share is zero or negative."}

    target = round(entry + (MIN_REWARD_RISK * risk_per_share), 2)
    rr = round((target - entry) / risk_per_share, 2)

    portfolio = load_portfolio()
    risk_amount = round(portfolio.cash * RISK_PER_TRADE, 2)
    qty = int(risk_amount / risk_per_share)
    if qty <= 0:
        return {"status": "error", "error_message": "Position size is zero."}

    return {
        "status": "success",
        "plan": {
            "symbol": symbol.upper(),
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": rr,
            "qty": qty,
            "risk_amount": risk_amount,
            "capital_required": round(qty * entry, 2),
        },
    }


def execute_paper_trade(symbol: str, entry: float, stop: float, target: float, qty: int) -> Dict:
    """Execute a paper trade: validate risk rules, update portfolio, persist state.

    Args:
        symbol: Stock ticker.
        entry: Entry price.
        stop: Stop-loss price.
        target: Target price.
        qty: Number of shares.

    Returns:
        dict with execution result and updated portfolio summary.
    """
    portfolio = load_portfolio()

    if len(portfolio.open_positions) >= MAX_OPEN_TRADES:
        return {"status": "SKIPPED", "reason": f"Already at max {MAX_OPEN_TRADES} open trades."}

    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"status": "SKIPPED", "reason": "Invalid stop (>= entry)."}

    rr = (target - entry) / risk_per_share
    if rr < MIN_REWARD_RISK:
        return {"status": "SKIPPED", "reason": f"R:R {rr:.1f} below minimum {MIN_REWARD_RISK}."}

    capital_needed = qty * entry
    if capital_needed > portfolio.cash:
        qty = int(portfolio.cash / entry)
        if qty <= 0:
            return {"status": "SKIPPED", "reason": "Insufficient cash."}
        capital_needed = qty * entry

    portfolio.cash -= capital_needed
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    portfolio.open_positions.append(
        Position(
            symbol=symbol.upper(),
            qty=qty,
            entry=entry,
            stop=stop,
            target=target,
            opened_at=now_str,
        )
    )
    action = f"OPEN {symbol.upper()} qty={qty} entry={entry:.2f} stop={stop:.2f} target={target:.2f}"
    portfolio.actions_log.append(f"[{now_str}] {action}")

    save_portfolio(portfolio)

    return {
        "status": "OPENED",
        "symbol": symbol.upper(),
        "qty": qty,
        "entry": entry,
        "stop": stop,
        "target": target,
        "cash_remaining": round(portfolio.cash, 2),
        "open_positions_count": len(portfolio.open_positions),
    }
