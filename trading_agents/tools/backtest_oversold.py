"""Backtest oversold bounce (RSI <= threshold) strategy for Nifty 50 stocks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import yfinance as yf

from trading_agents.config import INITIAL_CAPITAL, NSE_WATCHLIST, RISK_PER_TRADE
from trading_agents.tools.technical import compute_atr, compute_rsi_series

IST = timezone(timedelta(hours=5, minutes=30))


def _ensure_nse(symbol: str) -> str:
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        return symbol.upper() + ".NS"
    return symbol


def backtest_oversold_bounce(
    symbol: str,
    years: int = 2,
    rsi_entry: float = 35.0,
    rsi_exit: float = 45.0,
    max_hold_days: int = 10,
    stop_atr_mult: float = 0.8,
    require_below_50dma: bool = True,
    use_portfolio_sizing: bool = True,
    initial_capital: float | None = None,
) -> Dict:
    """Backtest oversold bounce: buy when RSI <= rsi_entry (and below 50-DMA), exit at stop, RSI>=rsi_exit, or max_hold_days.

    When use_portfolio_sizing is True, uses paper portfolio rules (1%% risk per trade) to compute
    qty per trade and running capital, so you get total_pnl_inr and amount gained/lost.

    Args:
        symbol: NSE ticker (e.g. RELIANCE.NS).
        years: Years of history.
        rsi_entry, rsi_exit, max_hold_days, stop_atr_mult: Strategy params.
        require_below_50dma: Only enter when price below 50-DMA.
        use_portfolio_sizing: If True, compute qty from 1%% risk and show P&L in INR.
        initial_capital: Starting capital in INR (default from config INITIAL_CAPITAL).

    Returns:
        dict with status, trades, total_trades, win_rate_pct, avg_return_pct; if use_portfolio_sizing
        then also starting_capital, ending_capital, total_pnl_inr, total_pnl_pct, and per-trade qty/pnl_inr.
    """
    symbol = _ensure_nse(symbol)
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{years}y", interval="1d")
    except Exception as e:
        return {"status": "error", "error_message": str(e), "symbol": symbol}

    if hist is None or hist.empty or len(hist) < 60:
        return {
            "status": "error",
            "error_message": f"Insufficient history for {symbol} (need >= 60 days).",
            "symbol": symbol,
        }

    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    closes = hist["Close"].astype(float).tolist()
    highs = hist["High"].astype(float).tolist()
    lows = hist["Low"].astype(float).tolist()
    dates = hist.index.normalize().tolist()

    rsi_series = compute_rsi_series(closes, period=14)
    n = len(closes)
    trades: List[Dict] = []
    i = 50
    while i < n - 1:
        rsi_i = rsi_series[i] if i < len(rsi_series) else None
        if rsi_i is None or rsi_i > rsi_entry:
            i += 1
            continue
        dma_50_i = sum(closes[i - 49 : i + 1]) / 50
        if require_below_50dma and closes[i] >= dma_50_i:
            i += 1
            continue
        atr_i = compute_atr(highs[: i + 1], lows[: i + 1], closes[: i + 1])
        if atr_i <= 0:
            i += 1
            continue
        entry_price = closes[i]
        stop_price = max(0.01, entry_price - stop_atr_mult * atr_i)
        entry_date = dates[i]
        exit_date = None
        exit_price = None
        exit_reason = None
        for j in range(i + 1, min(i + max_hold_days + 1, n)):
            if lows[j] <= stop_price:
                exit_date = dates[j]
                exit_price = stop_price
                exit_reason = "stop_hit"
                break
            rsi_j = rsi_series[j] if j < len(rsi_series) else None
            if rsi_j is not None and rsi_j >= rsi_exit:
                exit_date = dates[j]
                exit_price = closes[j]
                exit_reason = "rsi_exit"
                break
            if (j - i) >= max_hold_days:
                exit_date = dates[j]
                exit_price = closes[j]
                exit_reason = "max_days"
                break
        if exit_date is None or exit_price is None:
            i += 1
            continue
        ret_pct = round((exit_price - entry_price) / entry_price * 100, 2)
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "stop_price": round(stop_price, 2),
            "return_pct": ret_pct,
            "exit_reason": exit_reason,
            "rsi_at_entry": round(rsi_i, 2),
        })
        i = j + 1

    if not trades:
        return {
            "status": "success",
            "symbol": symbol,
            "strategy": "OVERSOLD_BOUNCE",
            "total_trades": 0,
            "trades": [],
            "win_rate_pct": None,
            "avg_return_pct": None,
            "params": {
                "rsi_entry": rsi_entry,
                "rsi_exit": rsi_exit,
                "max_hold_days": max_hold_days,
                "stop_atr_mult": stop_atr_mult,
            },
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }

    returns = [t["return_pct"] for t in trades]
    wins = sum(1 for r in returns if r > 0)
    total = len(trades)

    start_cap = float(initial_capital if initial_capital is not None else INITIAL_CAPITAL)
    if use_portfolio_sizing and start_cap > 0:
        capital = start_cap
        for t in trades:
            risk_per_share = t["entry_price"] - t["stop_price"]
            if risk_per_share <= 0:
                t["qty"] = 0
                t["pnl_inr"] = 0.0
                t["capital_after"] = round(capital, 2)
                continue
            risk_amount = capital * RISK_PER_TRADE
            qty = int(risk_amount / risk_per_share)
            qty = min(qty, int(capital / t["entry_price"])) if t["entry_price"] > 0 else 0
            qty = max(0, qty)
            pnl_inr = round(qty * (t["exit_price"] - t["entry_price"]), 2)
            capital += pnl_inr
            t["qty"] = qty
            t["pnl_inr"] = pnl_inr
            t["capital_after"] = round(capital, 2)
        total_pnl_inr = round(capital - start_cap, 2)
        total_pnl_pct = round((capital - start_cap) / start_cap * 100, 2) if start_cap else 0
        return {
            "status": "success",
            "symbol": symbol,
            "strategy": "OVERSOLD_BOUNCE",
            "total_trades": total,
            "win_rate_pct": round(wins / total * 100, 1),
            "avg_return_pct": round(sum(returns) / total, 2),
            "total_return_pct": round(sum(returns), 2),
            "starting_capital_inr": round(start_cap, 2),
            "ending_capital_inr": round(capital, 2),
            "total_pnl_inr": total_pnl_inr,
            "total_pnl_pct": total_pnl_pct,
            "trades": trades[-20:],
            "params": {
                "rsi_entry": rsi_entry,
                "rsi_exit": rsi_exit,
                "max_hold_days": max_hold_days,
                "stop_atr_mult": stop_atr_mult,
            },
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }
    return {
        "status": "success",
        "symbol": symbol,
        "strategy": "OVERSOLD_BOUNCE",
        "total_trades": total,
        "win_rate_pct": round(wins / total * 100, 1),
        "avg_return_pct": round(sum(returns) / total, 2),
        "total_return_pct": round(sum(returns), 2),
        "trades": trades[-20:],
        "params": {
            "rsi_entry": rsi_entry,
            "rsi_exit": rsi_exit,
            "max_hold_days": max_hold_days,
            "stop_atr_mult": stop_atr_mult,
        },
        "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def backtest_oversold_nifty50(
    years: int = 2,
    rsi_entry: float = 35.0,
    rsi_exit: float = 45.0,
    max_hold_days: int = 10,
    stop_atr_mult: float = 0.8,
    max_stocks: int = 10,
    use_portfolio_sizing: bool = True,
    initial_capital: float | None = None,
) -> Dict:
    """Run oversold bounce backtest on Nifty 50 watchlist; aggregate and rank. With use_portfolio_sizing, splits capital across stocks and reports total P&L in INR.

    Args:
        years: Years of history per stock.
        rsi_entry, rsi_exit, max_hold_days, stop_atr_mult: Strategy params.
        max_stocks: Max number of stocks to backtest (default 10).
        use_portfolio_sizing: If True, run each stock with (initial_capital / max_stocks) and sum P&L.
        initial_capital: Total capital in INR (default INITIAL_CAPITAL).

    Returns:
        dict with per-stock results, top performers; if use_portfolio_sizing then starting_capital_inr, ending_capital_inr, total_pnl_inr, total_pnl_pct.
    """
    cap = float(initial_capital if initial_capital is not None else INITIAL_CAPITAL)
    cap_per_stock = cap / max_stocks if use_portfolio_sizing else None
    results: List[Dict] = []
    total_ending = 0.0
    for sym in NSE_WATCHLIST[:max_stocks]:
        out = backtest_oversold_bounce(
            symbol=sym,
            years=years,
            rsi_entry=rsi_entry,
            rsi_exit=rsi_exit,
            max_hold_days=max_hold_days,
            stop_atr_mult=stop_atr_mult,
            use_portfolio_sizing=use_portfolio_sizing,
            initial_capital=cap_per_stock,
        )
        if out.get("status") == "success" and out.get("total_trades", 0) > 0:
            row = {
                "symbol": out["symbol"],
                "total_trades": out["total_trades"],
                "win_rate_pct": out["win_rate_pct"],
                "avg_return_pct": out["avg_return_pct"],
                "total_return_pct": out["total_return_pct"],
            }
            if use_portfolio_sizing and "ending_capital_inr" in out:
                row["starting_capital_inr"] = out["starting_capital_inr"]
                row["ending_capital_inr"] = out["ending_capital_inr"]
                row["pnl_inr"] = out["total_pnl_inr"]
                total_ending += out["ending_capital_inr"]
            results.append(row)
        elif out.get("status") == "success" and out.get("total_trades", 0) == 0 and use_portfolio_sizing and cap_per_stock:
            total_ending += cap_per_stock
            results.append({"symbol": sym, "total_trades": 0, "win_rate_pct": None, "avg_return_pct": None})
        elif out.get("status") == "error":
            results.append({
                "symbol": out.get("symbol", sym),
                "error": out.get("error_message", "unknown"),
            })
            if use_portfolio_sizing and cap_per_stock:
                total_ending += cap_per_stock

    by_win = sorted([r for r in results if "win_rate_pct" in r], key=lambda x: x.get("win_rate_pct", 0), reverse=True)
    by_avg_ret = sorted([r for r in results if "avg_return_pct" in r], key=lambda x: x.get("avg_return_pct", -999), reverse=True)

    out = {
        "status": "success",
        "strategy": "OVERSOLD_BOUNCE",
        "universe": "Nifty 50 (watchlist)",
        "stocks_run": len(NSE_WATCHLIST[:max_stocks]),
        "stocks_with_trades": len(by_win),
        "per_stock": results,
        "top_by_win_rate": by_win[:5],
        "top_by_avg_return": by_avg_ret[:5],
        "params": {
            "years": years,
            "rsi_entry": rsi_entry,
            "rsi_exit": rsi_exit,
            "max_hold_days": max_hold_days,
            "stop_atr_mult": stop_atr_mult,
        },
        "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }
    if use_portfolio_sizing and cap_per_stock:
        out["starting_capital_inr"] = round(cap, 2)
        out["ending_capital_inr"] = round(total_ending, 2)
        out["total_pnl_inr"] = round(total_ending - cap, 2)
        out["total_pnl_pct"] = round((total_ending - cap) / cap * 100, 2) if cap else 0
    return out


def get_top_oversold_nifty50(
    years: int = 2,
    top_n: int = 5,
    rank_by: str = "win_rate",
) -> Dict:
    """Run oversold backtest on full Nifty 50 and return only the top N stocks (data-driven, not LLM guess).

    Use when the user wants 'top 5 RSI/oversold stocks': we COMPUTE them from backtest
    (win rate or avg return), not from Gemini. Then user can focus on these for scan/paper trade.

    Args:
        years: Years of history.
        top_n: Number of top stocks to return (default 5).
        rank_by: "win_rate" or "avg_return".

    Returns:
        dict with top_symbols, top_stocks, and backtest params.
    """
    out = backtest_oversold_nifty50(years=years, max_stocks=50)
    if out.get("status") != "success":
        return out
    if rank_by == "avg_return":
        top_list = out.get("top_by_avg_return", [])[:top_n]
    else:
        top_list = out.get("top_by_win_rate", [])[:top_n]
    top_symbols = [s["symbol"] for s in top_list]
    return {
        "status": "success",
        "message": "Top stocks are from BACKTEST results, not LLM guess. Use these for scan/paper trade.",
        "top_n": top_n,
        "rank_by": rank_by,
        "top_symbols": top_symbols,
        "top_stocks": top_list,
        "params": out.get("params", {}),
        "backtest_end_ist": out.get("backtest_end_ist"),
    }


def get_best_oversold_nifty50(
    years: int = 2,
    max_stocks: int = 50,
    min_win_rate_pct: float = 50.0,
    min_avg_return_pct: float = 0.0,
    min_trades: int = 3,
) -> Dict:
    """Select BEST Nifty 50 stocks for oversold bounce: backtest, then keep only those that meet quality bars.

    Criteria (all must pass): win_rate >= min_win_rate_pct, avg_return >= min_avg_return_pct,
    total_trades >= min_trades. Rank by win_rate then avg_return. Use this to get a shortlist
    of stocks that have actually worked in the backtest, not just "top 5 by one metric".

    Args:
        years: Years of history.
        max_stocks: How many Nifty 50 stocks to run (default 50 = all).
        min_win_rate_pct: Minimum win rate (default 50).
        min_avg_return_pct: Minimum average return per trade in % (default 0).
        min_trades: Minimum number of trades in backtest (default 3) so result is not noise.

    Returns:
        dict with best_symbols, best_stocks (full rows), criteria_used, and backtest summary.
    """
    out = backtest_oversold_nifty50(years=years, max_stocks=max_stocks, use_portfolio_sizing=True)
    if out.get("status") != "success":
        return out
    per_stock = out.get("per_stock", [])
    best = [
        r for r in per_stock
        if r.get("win_rate_pct") is not None
        and r.get("avg_return_pct") is not None
        and r.get("total_trades", 0) >= min_trades
        and r.get("win_rate_pct", 0) >= min_win_rate_pct
        and r.get("avg_return_pct", -999) >= min_avg_return_pct
    ]
    best.sort(key=lambda x: (x.get("win_rate_pct", 0), x.get("avg_return_pct", -999)), reverse=True)
    return {
        "status": "success",
        "message": "Best = backtested stocks that meet min win rate, min avg return, and min trades.",
        "criteria": {
            "min_win_rate_pct": min_win_rate_pct,
            "min_avg_return_pct": min_avg_return_pct,
            "min_trades": min_trades,
        },
        "best_symbols": [s["symbol"] for s in best],
        "best_stocks": best,
        "total_passed": len(best),
        "starting_capital_inr": out.get("starting_capital_inr"),
        "ending_capital_inr": out.get("ending_capital_inr"),
        "total_pnl_inr": out.get("total_pnl_inr"),
        "backtest_end_ist": out.get("backtest_end_ist"),
    }
