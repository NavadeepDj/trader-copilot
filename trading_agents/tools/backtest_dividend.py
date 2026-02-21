"""Backtest dividend momentum strategy: buy after announcement or N days before ex-date, sell before ex-date."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))

# Entry rules: "fixed_days_before_ex" = buy exactly N trading days before ex; "after_announcement" = buy on first trading day on/after simulated announcement date
ENTRY_RULE_FIXED = "fixed_days_before_ex"
ENTRY_RULE_ANNOUNCEMENT = "after_announcement"


def _ensure_nse_symbol(symbol: str) -> str:
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        return symbol.upper() + ".NS"
    return symbol


def backtest_dividend_momentum(
    symbol: str,
    years: int = 2,
    days_before_ex_to_buy: int = 5,
    sell_days_before_ex: int = 1,
    entry_rule: str = ENTRY_RULE_FIXED,
    announcement_days_before_ex: int = 21,
) -> Dict:
    """Backtest the dividend momentum strategy on historical data.

    Note: The *live* dividend scan uses Moneycontrol API, which provides real
    announcement_date and ex_date. Backtest uses *historical* data from yfinance,
    which has ex-dates and amounts but no announcement dates. So here we support:

    1. fixed_days_before_ex (default): buy exactly N trading days before ex-date.
    2. after_announcement: simulate "buy after announcement" by assuming
       announcement is announcement_days_before_ex calendar days before ex-date;
       we buy on the first trading day on or after that date (and before ex-date).

    Exit: always sell sell_days_before_ex trading day(s) before ex-date to avoid
    the ex-date price drop.

    Args:
        symbol: NSE ticker (e.g. ENGINERSIN.NS or ENGINERSIN).
        years: Number of years of history to use (default 2).
        days_before_ex_to_buy: For fixed_days_before_ex: buy this many trading days before ex (default 5).
        sell_days_before_ex: Sell this many trading days before ex-date (default 1).
        entry_rule: "fixed_days_before_ex" or "after_announcement".
        announcement_days_before_ex: For after_announcement: assume announcement is this many calendar days before ex (default 21).

    Returns:
        dict with status, trades list, total_trades, win_rate, avg_return_pct,
        total_return_pct, params (including entry_rule), and optional error_message.
    """
    symbol = _ensure_nse_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to fetch dividends for {symbol}: {e}",
            "symbol": symbol,
        }

    if dividends is None or (hasattr(dividends, "empty") and dividends.empty):
        div_series = pd.Series(dtype=float)
    else:
        div_series = dividends

    if div_series.empty or len(div_series) == 0:
        return {
            "status": "success",
            "symbol": symbol,
            "strategy": "dividend_momentum",
            "message": "No historical dividend data found for this symbol.",
            "total_trades": 0,
            "trades": [],
            "win_rate_pct": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }

    try:
        hist = ticker.history(period=f"{years}y", interval="1d")
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to fetch price history for {symbol}: {e}",
            "symbol": symbol,
        }

    min_required = (days_before_ex_to_buy + sell_days_before_ex + 2) if entry_rule == ENTRY_RULE_FIXED else (announcement_days_before_ex + 5)
    if hist is None or hist.empty or len(hist) < min_required:
        return {
            "status": "error",
            "error_message": f"Insufficient price history for {symbol} (need at least {min_required} trading days).",
            "symbol": symbol,
        }

    # Trading days as sorted date index (use tz-naive for comparison)
    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    trading_days = hist.index.normalize().unique().sort_values()
    trading_days_list = trading_days.tolist()

    trades: List[Dict] = []
    for ex_ts, div_amount in div_series.items():
        ex_date = pd.Timestamp(ex_ts).normalize()
        if ex_date.tzinfo is not None:
            ex_date = ex_date.replace(tzinfo=None)

        # Sell date = last trading day strictly before ex_date
        before_ex = trading_days[trading_days < ex_date]
        if len(before_ex) == 0:
            continue
        sell_date = before_ex[-1]
        try:
            sell_idx = trading_days_list.index(sell_date)
        except ValueError:
            continue

        if entry_rule == ENTRY_RULE_ANNOUNCEMENT:
            # Buy on first trading day on or after simulated announcement date (and before sell_date so we hold at least 1 day)
            announcement_cutoff = ex_date - pd.Timedelta(days=announcement_days_before_ex)
            # Trading days that are >= announcement_cutoff and <= sell_date (and we need buy_date < sell_date for a hold)
            in_window = (trading_days >= announcement_cutoff) & (trading_days < sell_date)
            candidates = trading_days[in_window]
            if len(candidates) == 0:
                continue
            buy_date = candidates[0]
        else:
            # Buy date = N trading days before sell_date
            buy_idx = sell_idx - days_before_ex_to_buy
            if buy_idx < 0:
                continue
            buy_date = trading_days_list[buy_idx]

        try:
            buy_price = float(hist.loc[buy_date, "Close"])
            sell_price = float(hist.loc[sell_date, "Close"])
        except (KeyError, TypeError):
            continue

        if buy_price <= 0:
            continue
        ret_pct = round((sell_price - buy_price) / buy_price * 100, 2)
        trades.append({
            "ex_date": str(ex_date.date()),
            "buy_date": str(buy_date.date()),
            "sell_date": str(sell_date.date()),
            "buy_price": round(buy_price, 2),
            "sell_price": round(sell_price, 2),
            "return_pct": ret_pct,
            "dividend_amount": round(float(div_amount), 2),
        })

    if not trades:
        return {
            "status": "success",
            "symbol": symbol,
            "strategy": "dividend_momentum",
            "message": "No valid trades in the backtest window (strategy conditions not met for any ex-date).",
            "total_trades": 0,
            "trades": [],
            "win_rate_pct": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "dividends_found": len(div_series),
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }

    wins = sum(1 for t in trades if t["return_pct"] > 0)
    total = len(trades)
    returns = [t["return_pct"] for t in trades]
    avg_ret = round(sum(returns) / total, 2)
    win_rate = round(wins / total * 100, 1)

    return {
        "status": "success",
        "symbol": symbol,
        "strategy": "dividend_momentum",
        "total_trades": total,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "total_return_pct": round(sum(returns), 2),
        "trades": trades[-15:],
        "dividends_found": len(div_series),
        "params": {
            "years": years,
            "entry_rule": entry_rule,
            "days_before_ex_to_buy": days_before_ex_to_buy,
            "sell_days_before_ex": sell_days_before_ex,
            "announcement_days_before_ex": announcement_days_before_ex if entry_rule == ENTRY_RULE_ANNOUNCEMENT else None,
        },
        "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def _parse_date(d: Any) -> date | None:
    """Parse announcement_date or ex_date from str (YYYY-MM-DD) or date."""
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.strptime(d[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
    return None


def backtest_single_event(
    symbol: str,
    announcement_date: str | date,
    ex_date: str | date,
    sell_days_before_ex: int = 1,
    stop_price: float | None = None,
) -> Dict:
    """Backtest one dividend event using real announcement and ex dates (e.g. from Moneycontrol).

    Buy: first trading day on or after announcement_date.
    Sell: last trading day strictly before ex_date (i.e. 1 trading day before ex).
    If stop_price is set (e.g. scan's suggested_stop): we check daily Low; if Low <= stop
    on any day between buy and sell, we exit at stop that day (limit loss).

    Args:
        symbol: NSE ticker (e.g. PIINDUSTRIES.NS).
        announcement_date: Announcement date as YYYY-MM-DD or date.
        ex_date: Ex-dividend date as YYYY-MM-DD or date.
        sell_days_before_ex: Sell this many trading days before ex (default 1).
        stop_price: Optional. If price hits this (daily Low <= stop), exit at stop to limit loss.

    Returns:
        dict with status, buy_date, sell_date, buy_price, sell_price, return_pct,
        exit_reason ("target_date" | "stop_hit"), or error_message.
    """
    symbol = _ensure_nse_symbol(symbol)
    ann = _parse_date(announcement_date)
    ex = _parse_date(ex_date)
    if ann is None or ex is None:
        return {
            "status": "error",
            "error_message": "Invalid announcement_date or ex_date (use YYYY-MM-DD).",
            "symbol": symbol,
        }
    if ex <= ann:
        return {
            "status": "error",
            "error_message": "ex_date must be after announcement_date.",
            "symbol": symbol,
        }

    try:
        ticker = yf.Ticker(symbol)
        start = (ann - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (ex + timedelta(days=5)).strftime("%Y-%m-%d")
        hist = ticker.history(start=start, end=end, interval="1d")
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to fetch prices for {symbol}: {e}",
            "symbol": symbol,
        }

    if hist is None or hist.empty or len(hist) < 2:
        return {
            "status": "error",
            "error_message": f"Insufficient price data for {symbol} in window {start} to {end}.",
            "symbol": symbol,
        }

    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    trading_days = hist.index.normalize().unique().sort_values()
    ann_pd = pd.Timestamp(ann)
    ex_pd = pd.Timestamp(ex)

    # Sell = last trading day strictly before ex_date
    before_ex = trading_days[trading_days < ex_pd]
    if len(before_ex) == 0:
        return {
            "status": "error",
            "error_message": f"No trading day before ex_date {ex} in data for {symbol}.",
            "symbol": symbol,
        }
    sell_date = before_ex[-1]

    # Buy = first trading day on or after announcement_date, and strictly before sell_date
    buy_candidates = trading_days[(trading_days >= ann_pd) & (trading_days < sell_date)]
    if len(buy_candidates) == 0:
        return {
            "status": "error",
            "error_message": f"No buy date in [announcement, sell_date) for {symbol}.",
            "symbol": symbol,
        }
    buy_date = buy_candidates[0]

    try:
        buy_price = float(hist.loc[buy_date, "Close"])
        sell_price = float(hist.loc[sell_date, "Close"])
    except (KeyError, TypeError):
        return {
            "status": "error",
            "error_message": f"Missing close price for {symbol} on buy/sell date.",
            "symbol": symbol,
        }
    if buy_price <= 0:
        return {"status": "error", "error_message": "Invalid buy price.", "symbol": symbol}

    # Optional stop-loss: if daily Low hits stop between buy and sell, exit at stop
    exit_reason = "target_date"
    actual_sell_date = sell_date
    actual_sell_price = sell_price
    if stop_price is not None and stop_price < buy_price and "Low" in hist.columns:
        for d in trading_days:
            if d <= buy_date or d >= sell_date:
                continue
            try:
                low = float(hist.loc[d, "Low"])
                if low <= stop_price:
                    actual_sell_date = d
                    actual_sell_price = stop_price
                    exit_reason = "stop_hit"
                    break
            except (KeyError, TypeError):
                continue

    ret_pct = round((actual_sell_price - buy_price) / buy_price * 100, 2)
    return {
        "status": "success",
        "symbol": symbol,
        "announcement_date": ann.isoformat(),
        "ex_date": ex.isoformat(),
        "buy_date": str(buy_date.date()),
        "sell_date": str(actual_sell_date.date()),
        "buy_price": round(buy_price, 2),
        "sell_price": round(actual_sell_price, 2),
        "return_pct": ret_pct,
        "exit_reason": exit_reason,
        "strategy": "buy_after_announcement_sell_before_ex",
    }


def backtest_moneycontrol_events(
    candidates: List[Dict],
    sell_days_before_ex: int = 1,
) -> Dict:
    """Backtest dividend events from Moneycontrol list using real announcement_date and ex_date.

    TIMING ONLY (no metric filters): We simulate buy on the first trading day on or
    after announcement_date and sell 1 trading day before ex-date. We do NOT check
    dividend health, 50-DMA, or other conditions at buy time. So every event with
    announcement in the past is included. Use backtest_current_moneycontrol_dividends_filtered
    to only backtest stocks that pass the scan's health + technical filters.

    Why same sell date (e.g. 20 Feb)? Ex-date is the same for many (e.g. 23 Feb);
    "1 trading day before ex" is then the same calendar day for all (e.g. 20 Feb).

    Args:
        candidates: List of dicts from fetch_moneycontrol_dividends (or similar),
                   each with keys: symbol, announcement_date (YYYY-MM-DD), ex_date (YYYY-MM-DD),
                   and optionally company.
        sell_days_before_ex: Sell this many trading days before ex (default 1).

    Returns:
        dict with status, events_tested, results (per-event), win_rate_pct, avg_return_pct, etc.
    """
    today = date.today()
    results: List[Dict] = []
    skipped: List[Dict] = []

    for c in candidates:
        symbol = c.get("symbol")
        ann_str = c.get("announcement_date")
        ex_str = c.get("ex_date")
        company = c.get("company", c.get("stockName", "?"))

        if not symbol or not ex_str:
            skipped.append({"company": company or "?", "reason": "missing symbol or ex_date"})
            continue

        ann = _parse_date(ann_str)
        ex = _parse_date(ex_str)
        if ann is None or ex is None:
            skipped.append({"company": company or "?", "reason": "invalid announcement_date or ex_date"})
            continue
        if ex <= today:
            skipped.append({"company": company or "?", "reason": "ex_date already passed (no need to backtest)"})
            continue
        if ann > today:
            skipped.append({"company": company or "?", "reason": "announcement_date in the future"})
            continue

        stop_price = c.get("suggested_stop")
        if stop_price is not None:
            try:
                stop_price = float(stop_price)
            except (TypeError, ValueError):
                stop_price = None
        out = backtest_single_event(
            symbol=symbol,
            announcement_date=ann,
            ex_date=ex,
            sell_days_before_ex=sell_days_before_ex,
            stop_price=stop_price,
        )
        if out.get("status") == "success":
            out["company"] = company
            if stop_price is not None:
                out["stop_used"] = round(stop_price, 2)
            results.append(out)
        else:
            skipped.append({
                "company": company or "?",
                "symbol": symbol,
                "reason": out.get("error_message", "backtest failed"),
            })

    if not results:
        return {
            "status": "success",
            "source": "Moneycontrol events (real announcement + ex dates)",
            "message": "No events could be backtested (announcement in past, ex in future).",
            "events_tested": 0,
            "results": [],
            "skipped": skipped[:15],
            "win_rate_pct": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }

    returns = [r["return_pct"] for r in results]
    wins = sum(1 for r in returns if r > 0)
    total = len(results)
    return {
        "status": "success",
        "source": "Moneycontrol events (real announcement + ex dates)",
        "metrics_used": "None (timing only: buy on/after announcement, sell 1 day before ex). No health, 50-DMA, or ATR/stop applied unless you used the filtered backtest with suggested_stop.",
        "are_recommended": "No. This list is ALL Moneycontrol events with announcement in past. Use backtest_current_moneycontrol_dividends_filtered for only scan-recommended stocks.",
        "disclaimer": "Timing only: buy = first trading day on/after announcement, sell = 1 day before ex. No health/50-DMA filters. No stop-loss (raw list has no suggested_stop). Same sell date when ex-date is same (e.g. 20 Feb = last trading day before 23 Feb).",
        "events_tested": total,
        "results": results,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:10],
        "win_rate_pct": round(wins / total * 100, 1),
        "avg_return_pct": round(sum(returns) / total, 2),
        "total_return_pct": round(sum(returns), 2),
        "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def backtest_current_moneycontrol_dividends(sell_days_before_ex: int = 1) -> Dict:
    """Backtest the current Moneycontrol dividend list using real announcement + ex dates.

    TIMING ONLY: No health/50-DMA filters. Buy = first trading day on/after announcement,
    sell = 1 day before ex. Same sell date for all when ex-date is the same (e.g. 20 Feb).

    Fetches the latest dividend list from Moneycontrol, then for each event where
    announcement_date is already in the past (and ex_date is still in the future),
    simulates that timing. Use backtest_current_moneycontrol_dividends_filtered to
    only backtest stocks that pass the scan's conditions (health + 50-DMA etc.).

    Args:
        sell_days_before_ex: Sell this many trading days before ex (default 1).

    Returns:
        Same shape as backtest_moneycontrol_events: events_tested, results, win_rate_pct, disclaimer, etc.
    """
    from trading_agents.tools.dividend_data import fetch_moneycontrol_dividends

    mc = fetch_moneycontrol_dividends()
    if mc.get("status") != "success":
        return {
            "status": "error",
            "error_message": mc.get("error_message", "Failed to fetch Moneycontrol dividend list."),
        }
    candidates = mc.get("candidates", [])
    return backtest_moneycontrol_events(candidates, sell_days_before_ex=sell_days_before_ex)


def backtest_current_moneycontrol_dividends_filtered(sell_days_before_ex: int = 1) -> Dict:
    """Backtest only stocks that PASS the scan's conditions (health + 50-DMA + trend).

    Runs the full dividend scan first (health, above 50-DMA, uptrend, etc.), then
    backtests only those opportunities using real announcement + ex dates. So you see
    results for "stocks we would have recommended", not every raw Moneycontrol event.

    Buy/sell timing is still: buy first trading day on/after announcement, sell 1 day before ex.
    We do not re-check 50-DMA at the historical buy date (that would need point-in-time data).

    Args:
        sell_days_before_ex: Sell this many trading days before ex (default 1).

    Returns:
        Same shape as backtest_moneycontrol_events, with disclaimer that only scan-passed stocks are included.
    """
    from trading_agents.dividend_agent import scan_dividend_opportunities

    scan = scan_dividend_opportunities(min_days_to_ex=1)
    if scan.get("status") != "success":
        return {
            "status": "error",
            "error_message": scan.get("error_message", "Scan failed."),
        }
    opportunities = scan.get("top_opportunities", [])
    # Build candidate-like list with suggested_stop so backtest can simulate stop-loss
    candidates = [
        {
            "symbol": o.get("symbol"),
            "announcement_date": o.get("announcement_date"),
            "ex_date": o.get("ex_date"),
            "company": o.get("company"),
            "suggested_stop": o.get("suggested_stop"),
        }
        for o in opportunities
        if o.get("symbol") and o.get("ex_date")
    ]
    if not candidates:
        return {
            "status": "success",
            "source": "Moneycontrol events (filtered by scan: health + 50-DMA + trend)",
            "message": "No scan opportunities to backtest (or none with announcement in past).",
            "events_tested": 0,
            "results": [],
            "win_rate_pct": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "backtest_end_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }
    out = backtest_moneycontrol_events(candidates, sell_days_before_ex=sell_days_before_ex)
    from trading_agents.config import DIVIDEND_STOP_ATR_MULTIPLIER
    out["metrics_used"] = (
        "Scan: dividend health, above 50-DMA, uptrend. "
        f"Stop-loss: entry - {DIVIDEND_STOP_ATR_MULTIPLIER}*ATR (tighter); exit at stop if daily Low hits before sell date."
    )
    out["are_recommended"] = "Yes. Only stocks that passed the scan (health + 50-DMA + uptrend). Stop = entry - 1*ATR (configurable)."
    out["disclaimer"] = (
        "Only stocks that passed the scan (dividend health, above 50-DMA, good uptrend). "
        f"Stop = entry - {DIVIDEND_STOP_ATR_MULTIPLIER}*ATR. Sell = at least 1 trading day before ex-date (or at stop if hit). "
        "Same sell date when ex-date is same (e.g. 20 Feb before 23 Feb ex). "
        "Strategy tends to work better in BULL/NEUTRAL markets; in bear or weak periods you may see more losses."
    )
    out["source"] = "Moneycontrol events (filtered: health + 50-DMA + uptrend; stop = entry - 1*ATR)"
    return out
