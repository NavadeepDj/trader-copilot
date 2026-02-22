"""Portfolio state persistence and queries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from trading_agents.config import INITIAL_CAPITAL, MAX_HOLD_DAYS
from trading_agents.models import PortfolioState, Position
from trading_agents.tools.market_data import fetch_stock_data


# Use absolute path based on project root to avoid relative path issues
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORY_DIR = _PROJECT_ROOT / "memory"
PORTFOLIO_FILE = MEMORY_DIR / "portfolio.json"


def load_portfolio() -> PortfolioState:
    """Load portfolio state from disk, or return a fresh one."""
    if not PORTFOLIO_FILE.exists():
        return PortfolioState(cash=INITIAL_CAPITAL)

    raw = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    return PortfolioState(
        cash=raw.get("cash", INITIAL_CAPITAL),
        open_positions=[Position(**p) for p in raw.get("open_positions", [])],
        closed_trades=raw.get("closed_trades", []),
        realized_pnl=raw.get("realized_pnl", 0.0),
        actions_log=raw.get("actions_log", []),
        equity_curve=raw.get("equity_curve", []),
    )


def save_portfolio(state: PortfolioState) -> None:
    """Persist portfolio state to disk."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump()
    payload["actions_log"] = payload["actions_log"][-50:]
    payload["equity_curve"] = payload.get("equity_curve", [])[-1000:]
    payload["closed_trades"] = payload.get("closed_trades", [])[-2000:]
    PORTFOLIO_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parse_opened_at(opened_at: str) -> datetime | None:
    if not opened_at:
        return None
    try:
        return datetime.strptime(opened_at.replace(" IST", ""), "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _latest_bar(symbol: str) -> Dict:
    try:
        data = fetch_stock_data(symbol=symbol, days=30)
    except Exception as exc:
        return {"status": "error", "error_message": f"Quote fetch exception for {symbol}: {exc}"}
    if data.get("status") != "success":
        return data
    return {
        "status": "success",
        "close": float(data["closes"][-1]),
        "high": float(data["highs"][-1]),
        "low": float(data["lows"][-1]),
        "trade_date": str(data["last_trade_date"]),
        "fetched_at_ist": data.get("fetched_at_ist"),
    }


def _record_equity_snapshot(
    state: PortfolioState,
    *,
    portfolio_value: float,
    invested_mtm: float,
    unrealized_pnl: float,
    source: str,
) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prev_peak = max((pt.get("peak", 0.0) for pt in state.equity_curve), default=portfolio_value)
    peak = max(prev_peak, portfolio_value)
    drawdown_pct = ((portfolio_value - peak) / peak * 100.0) if peak > 0 else 0.0
    state.equity_curve.append(
        {
            "timestamp": now_str,
            "source": source,
            "cash": round(state.cash, 2),
            "invested_mtm": round(invested_mtm, 2),
            "portfolio_value": round(portfolio_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(state.realized_pnl, 2),
            "peak": round(peak, 2),
            "drawdown_pct": round(drawdown_pct, 4),
        }
    )


def refresh_portfolio_positions(max_hold_days: int = MAX_HOLD_DAYS) -> Dict:
    """Advance open trades through lifecycle using latest daily bar.

    Exit priority is conservative when both target and stop are hit in one bar:
    STOP_HIT is applied before TARGET_HIT.
    """
    state = load_portfolio()
    if not state.open_positions:
        return {
            "status": "success",
            "message": "No open positions to refresh.",
            "closed_now": 0,
            "open_positions_count": 0,
        }

    updated_positions: list[Position] = []
    closed_now: list[Dict] = []
    errors: list[str] = []

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    for pos in state.open_positions:
        bar = _latest_bar(pos.symbol)
        if bar.get("status") != "success":
            errors.append(f"{pos.symbol}: {bar.get('error_message', 'price fetch failed')}")
            updated_positions.append(pos)
            continue

        low = bar["low"]
        high = bar["high"]
        close = bar["close"]
        opened_dt = _parse_opened_at(pos.opened_at)
        holding_days = (datetime.now() - opened_dt).days if opened_dt else 0

        exit_reason = None
        exit_price = None

        if low <= pos.stop:
            exit_reason = "STOP_HIT"
            exit_price = pos.stop
        elif high >= pos.target:
            exit_reason = "TARGET_HIT"
            exit_price = pos.target
        elif holding_days >= max_hold_days:
            exit_reason = "TIME_EXIT"
            exit_price = close

        if exit_reason is None or exit_price is None:
            updated_positions.append(pos)
            continue

        proceeds = pos.qty * exit_price
        pnl = (exit_price - pos.entry) * pos.qty
        state.cash += proceeds
        state.realized_pnl += pnl

        closed = {
            "symbol": pos.symbol,
            "qty": pos.qty,
            "entry": round(pos.entry, 2),
            "stop": round(pos.stop, 2),
            "target": round(pos.target, 2),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "pnl_inr": round(pnl, 2),
            "pnl_pct": round(((exit_price - pos.entry) / pos.entry) * 100, 2) if pos.entry > 0 else None,
            "opened_at": pos.opened_at,
            "closed_at": now_str,
            "last_trade_date": bar.get("trade_date"),
            "holding_days": holding_days,
        }
        state.closed_trades.append(closed)
        state.actions_log.append(
            f"[{now_str}] CLOSE {pos.symbol} qty={pos.qty} exit={exit_price:.2f} reason={exit_reason} pnl={pnl:.2f}"
        )
        closed_now.append(closed)

    state.open_positions = updated_positions

    # Snapshot after lifecycle updates.
    invested_mtm = 0.0
    unrealized_pnl = 0.0
    for pos in state.open_positions:
        bar = _latest_bar(pos.symbol)
        if bar.get("status") != "success":
            invested_mtm += pos.qty * pos.entry
            continue
        m2m = pos.qty * bar["close"]
        invested_mtm += m2m
        unrealized_pnl += (bar["close"] - pos.entry) * pos.qty

    portfolio_value = state.cash + invested_mtm
    _record_equity_snapshot(
        state,
        portfolio_value=portfolio_value,
        invested_mtm=invested_mtm,
        unrealized_pnl=unrealized_pnl,
        source="lifecycle_refresh",
    )
    save_portfolio(state)

    return {
        "status": "success",
        "closed_now": len(closed_now),
        "closed_positions": closed_now,
        "open_positions_count": len(state.open_positions),
        "errors": errors if errors else None,
    }


def get_portfolio_summary() -> Dict:
    """Return portfolio summary with mark-to-market valuation and drawdown stats."""
    refresh_portfolio_positions()
    p = load_portfolio()

    open_positions: list[Dict] = []
    total_invested_entry = 0.0
    total_invested_mtm = 0.0
    unrealized_pnl = 0.0
    quote_errors: list[str] = []

    for pos in p.open_positions:
        total_invested_entry += pos.qty * pos.entry
        bar = _latest_bar(pos.symbol)
        if bar.get("status") != "success":
            quote_errors.append(f"{pos.symbol}: {bar.get('error_message', 'quote unavailable')}")
            current_price = pos.entry
            last_trade_date = None
        else:
            current_price = bar["close"]
            last_trade_date = bar.get("trade_date")

        market_value = pos.qty * current_price
        position_unrealized = (current_price - pos.entry) * pos.qty
        total_invested_mtm += market_value
        unrealized_pnl += position_unrealized

        open_positions.append(
            {
                "symbol": pos.symbol,
                "qty": pos.qty,
                "entry": round(pos.entry, 2),
                "stop": round(pos.stop, 2),
                "target": round(pos.target, 2),
                "current_price": round(current_price, 2),
                "invested": round(pos.qty * pos.entry, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(position_unrealized, 2),
                "opened_at": pos.opened_at,
                "last_trade_date": last_trade_date,
            }
        )

    portfolio_value = p.cash + total_invested_mtm
    total_pnl = p.realized_pnl + unrealized_pnl
    _record_equity_snapshot(
        p,
        portfolio_value=portfolio_value,
        invested_mtm=total_invested_mtm,
        unrealized_pnl=unrealized_pnl,
        source="portfolio_summary",
    )
    save_portfolio(p)

    max_drawdown_pct = min((pt.get("drawdown_pct", 0.0) for pt in p.equity_curve), default=0.0)

    return {
        "status": "success",
        "cash": round(p.cash, 2),
        "total_invested": round(total_invested_entry, 2),
        "total_invested_mtm": round(total_invested_mtm, 2),
        "portfolio_value": round(portfolio_value, 2),
        "net_profit_inr": round(portfolio_value - INITIAL_CAPITAL, 2),
        "net_profit_pct": round(((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2),
        "open_positions_count": len(open_positions),
        "open_positions": open_positions,
        "realized_pnl": round(p.realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "equity_points": len(p.equity_curve),
        "recent_equity_curve": p.equity_curve[-30:],
        "recent_closed_trades": p.closed_trades[-10:],
        "recent_actions": p.actions_log[-10:],
        "quote_errors": quote_errors if quote_errors else None,
    }


def get_portfolio_performance() -> Dict:
    """Return risk/return performance metrics from lifecycle + equity curve."""
    summary = get_portfolio_summary()
    p = load_portfolio()

    closed = p.closed_trades
    wins = [t for t in closed if (t.get("pnl_inr") or 0.0) > 0]
    losses = [t for t in closed if (t.get("pnl_inr") or 0.0) < 0]
    gross_profit = sum(float(t.get("pnl_inr", 0.0)) for t in wins)
    gross_loss = abs(sum(float(t.get("pnl_inr", 0.0)) for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    return {
        "status": "success",
        "starting_capital": INITIAL_CAPITAL,
        "portfolio_value": summary["portfolio_value"],
        "net_profit_inr": summary["net_profit_inr"],
        "net_profit_pct": summary["net_profit_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "total_closed_trades": len(closed),
        "win_rate_pct": round((len(wins) / len(closed)) * 100, 2) if closed else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "gross_profit_inr": round(gross_profit, 2),
        "gross_loss_inr": round(gross_loss, 2),
        "realized_pnl": summary["realized_pnl"],
        "unrealized_pnl": summary["unrealized_pnl"],
        "open_positions_count": summary["open_positions_count"],
        "equity_points": summary["equity_points"],
        "last_equity_points": summary["recent_equity_curve"][-10:],
    }


def reset_portfolio() -> Dict:
    """Reset portfolio to initial clean state."""
    fresh = PortfolioState(cash=INITIAL_CAPITAL)
    fresh.equity_curve = [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "reset",
            "cash": round(INITIAL_CAPITAL, 2),
            "invested_mtm": 0.0,
            "portfolio_value": round(INITIAL_CAPITAL, 2),
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "peak": round(INITIAL_CAPITAL, 2),
            "drawdown_pct": 0.0,
        }
    ]
    save_portfolio(fresh)
    return {"status": "success", "message": f"Portfolio reset to INR {INITIAL_CAPITAL:,.0f}"}
