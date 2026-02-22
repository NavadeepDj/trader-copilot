"""Stock scanner sub-agent -- scans NSE watchlist for breakout candidates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL, NSE_WATCHLIST
from trading_agents.regime_agent import analyze_regime
from trading_agents.tools.backtest_oversold import (
    backtest_oversold_bounce,
    backtest_oversold_nifty50,
    get_best_oversold_nifty50,
    get_top_oversold_nifty50,
)
from trading_agents.tools.market_data import fetch_stock_data
from trading_agents.tools.news_data import fetch_stock_news
from trading_agents.tools.technical import compute_atr, compute_index_metrics, compute_rsi, detect_breakout

IST = timezone(timedelta(hours=5, minutes=30))


def scan_watchlist_breakouts(watchlist: str = "") -> Dict:
    """Scan NSE watchlist stocks for 20-day breakout candidates with live data.

    Args:
        watchlist: Comma-separated stock symbols to scan. Leave empty to use default NSE watchlist.

    Returns:
        dict with breakout candidates and scan metadata.
    """
    if watchlist.strip():
        symbols = [s.strip() for s in watchlist.split(",")]
    else:
        symbols = NSE_WATCHLIST

    candidates: List[Dict] = []
    scanned: List[str] = []
    errors: List[str] = []

    for sym in symbols:
        data = fetch_stock_data(symbol=sym)
        if data.get("status") != "success":
            errors.append(f"{sym}: {data.get('error_message', 'fetch failed')}")
            continue

        scanned.append(sym)
        result = detect_breakout(
            symbol=data["symbol"],
            closes=data["closes"],
            volumes=data["volumes"],
            highs=data["highs"],
            lows=data["lows"],
        )
        if result.get("status") == "success" and result.get("is_breakout"):
            candidates.append(result)

    candidates.sort(key=lambda x: x.get("volume_ratio", 0), reverse=True)

    return {
        "status": "success",
        "stocks_scanned": len(scanned),
        "breakout_count": len(candidates),
        "candidates": candidates,
        "scan_errors": errors if errors else None,
    }


def get_stock_analysis(symbol: str) -> Dict:
    """Get detailed breakout analysis for a single stock.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with breakout analysis, ATR, and technical metrics.
    """
    data = fetch_stock_data(symbol=symbol)
    if data.get("status") != "success":
        return data

    result = detect_breakout(
        symbol=data["symbol"],
        closes=data["closes"],
        volumes=data["volumes"],
        highs=data["highs"],
        lows=data["lows"],
    )
    result["last_trade_date"] = data["last_trade_date"]
    result["source"] = data["source"]
    return result


def scan_announcement_momentum(watchlist: str = "") -> Dict:
    """Scan NSE stocks for announcement-driven momentum candidates.

    Identifies stocks with recent news (last 3 days) combined with
    significant price movement (>2% in 5 days) and above-average volume.
    The agent interprets whether news is a significant corporate announcement.

    Args:
        watchlist: Comma-separated stock symbols. Leave empty for default NSE watchlist.

    Returns:
        dict with momentum candidates, their news headlines, and price metrics.
    """
    if watchlist.strip():
        symbols = [s.strip() for s in watchlist.split(",")]
    else:
        symbols = NSE_WATCHLIST

    candidates: List[Dict] = []
    scanned: List[str] = []
    errors: List[str] = []

    for sym in symbols:
        data = fetch_stock_data(symbol=sym)
        if data.get("status") != "success":
            errors.append(f"{sym}: {data.get('error_message', 'fetch failed')}")
            continue

        scanned.append(sym)
        closes = data["closes"]
        volumes = data["volumes"]

        if len(closes) < 6 or len(volumes) < 21:
            continue

        price_change_5d = (closes[-1] / closes[-6]) - 1.0
        avg_20d_volume = sum(volumes[-21:-1]) / 20
        volume_ratio = round(volumes[-1] / max(avg_20d_volume, 1), 2)
        dma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]

        has_momentum = abs(price_change_5d) > 0.02 and volume_ratio > 1.0

        if not has_momentum:
            continue

        news = fetch_stock_news(symbol=sym)
        articles = news.get("articles", [])
        recent_news = [a for a in articles if (a.get("days_ago") or 999) <= 3]

        if not recent_news:
            continue

        candidates.append({
            "symbol": data["symbol"],
            "close": round(closes[-1], 2),
            "price_change_5d": round(price_change_5d, 4),
            "price_change_5d_pct": f"{price_change_5d:+.2%}",
            "volume_ratio": volume_ratio,
            "above_50dma": closes[-1] > dma_50,
            "dma_50": round(dma_50, 2),
            "direction": "BULLISH" if price_change_5d > 0 else "BEARISH",
            "recent_news_count": len(recent_news),
            "news_headlines": [
                {"title": a["title"], "publisher": a["publisher"], "days_ago": a["days_ago"]}
                for a in recent_news[:5]
            ],
        })

    candidates.sort(key=lambda x: abs(x.get("price_change_5d", 0)), reverse=True)

    return {
        "status": "success",
        "strategy": "ANNOUNCEMENT_MOMENTUM",
        "stocks_scanned": len(scanned),
        "momentum_candidates": len(candidates),
        "candidates": candidates,
        "scan_errors": errors if errors else None,
    }


def scan_oversold_bounce(
    watchlist: str = "",
    rsi_max: float = 35.0,
    require_below_50dma: bool = True,
) -> Dict:
    """Scan for oversold stocks (RSI <= threshold) for mean-reversion / bounce in sideways or bear markets.

    Use when regime is SIDEWAYS or BEAR: buy oversold dips with tight stops, target mean reversion.

    Args:
        watchlist: Comma-separated symbols. Empty = default NSE watchlist.
        rsi_max: Max RSI to consider oversold (default 35; use 30 for deeper oversold).
        require_below_50dma: If True, only include stocks trading below 50-DMA (typical oversold).

    Returns:
        dict with oversold candidates, RSI, distance from 50-DMA, suggested stop (e.g. entry - 0.8*ATR).
    """
    if watchlist.strip():
        symbols = [s.strip() for s in watchlist.split(",")]
    else:
        symbols = NSE_WATCHLIST

    candidates: List[Dict] = []
    scanned: List[str] = []
    errors: List[str] = []

    for sym in symbols:
        data = fetch_stock_data(symbol=sym)
        if data.get("status") != "success":
            errors.append(f"{sym}: {data.get('error_message', 'fetch failed')}")
            continue

        scanned.append(sym)
        closes = data["closes"]
        highs = data["highs"]
        lows = data["lows"]

        if len(closes) < 60:
            continue

        rsi = compute_rsi(closes, period=14)
        if rsi is None or rsi > rsi_max:
            continue

        metrics = compute_index_metrics(closes)
        if metrics.get("status") != "success":
            continue
        dma_50 = metrics["dma_50"]
        close = closes[-1]
        below_50 = close < dma_50
        if require_below_50dma and not below_50:
            continue

        atr = compute_atr(highs, lows, closes)
        # Tighter stop for mean reversion
        stop = round(max(0.01, close - 0.8 * atr), 2)

        pct_below_50dma = round((1 - close / dma_50) * 100, 2) if dma_50 else 0

        candidates.append({
            "symbol": data["symbol"],
            "close": round(close, 2),
            "rsi": rsi,
            "dma_50": round(dma_50, 2),
            "below_50dma": below_50,
            "pct_below_50dma": pct_below_50dma,
            "atr": round(atr, 2),
            "suggested_stop": stop,
            "strategy_note": "Oversold bounce / mean reversion; use tight stop; works in sideways/bear.",
        })

    candidates.sort(key=lambda x: x.get("rsi", 100))

    return {
        "status": "success",
        "strategy": "OVERSOLD_BOUNCE",
        "regime_suitability": "SIDEWAYS / BEAR",
        "stocks_scanned": len(scanned),
        "oversold_count": len(candidates),
        "candidates": candidates,
        "params": {"rsi_max": rsi_max, "require_below_50dma": require_below_50dma},
        "scan_errors": errors if errors else None,
    }


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.01, float(value)), 2)


def _signal_row_for_symbol(symbol: str, regime: str) -> Dict:
    try:
        data = fetch_stock_data(symbol=symbol)
    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "error_message": f"Quote fetch exception for {symbol}: {exc}",
        }
    if data.get("status") != "success":
        return {
            "status": "error",
            "symbol": symbol,
            "error_message": data.get("error_message", "price fetch failed"),
        }

    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]

    close = float(closes[-1])
    atr = float(compute_atr(highs, lows, closes))
    rsi = compute_rsi(closes, period=14)
    dma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else close
    breakout_result = detect_breakout(
        symbol=data["symbol"],
        closes=closes,
        volumes=volumes,
        highs=highs,
        lows=lows,
    )
    is_breakout = bool(breakout_result.get("is_breakout")) if breakout_result.get("status") == "success" else False
    volume_ratio = float(breakout_result.get("volume_ratio", 0.0)) if breakout_result.get("status") == "success" else None

    signal = "HOLD"
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rationale = "No clear setup right now."

    if regime == "BULL":
        if is_breakout and atr > 0:
            signal = "BUY"
            entry = close
            stop = close - 1.5 * atr
            risk = entry - stop
            target = entry + 2.0 * risk
            rationale = (
                f"Breakout confirmed above 20D high with volume ratio {volume_ratio:.2f} and price above 50-DMA."
                if volume_ratio is not None
                else "Breakout confirmed above 20D high and price above 50-DMA."
            )
        elif close < dma_50 and (rsi is not None and rsi < 45):
            signal = "SELL"
            rationale = "Price below 50-DMA with weak RSI in bull strategy context; reduce risk or exit longs."
        else:
            rationale = "Trend is bull but no confirmed breakout trigger yet."
    else:
        oversold_buy = (rsi is not None and rsi <= 35 and close < dma_50 and atr > 0)
        overbought_sell = (rsi is not None and rsi >= 65 and close > dma_50)
        if oversold_buy:
            signal = "BUY"
            entry = close
            stop = close - 0.8 * atr
            risk = entry - stop
            target = entry + 2.0 * risk
            rationale = "Oversold bounce setup (RSI <= 35) below 50-DMA with tight stop."
        elif overbought_sell:
            signal = "SELL"
            rationale = "Overbought in non-bull regime (RSI >= 65); book profits / avoid fresh long exposure."
        else:
            rationale = "No oversold bounce trigger and no clear risk-off trigger."

    return {
        "status": "success",
        "symbol": data["symbol"],
        "display_symbol": data["symbol"].replace(".NS", ""),
        "signal": signal,
        "current_price": _round_price(close),
        "entry": _round_price(entry),
        "stop": _round_price(stop),
        "target": _round_price(target),
        "rationale": rationale,
        "last_trade_date": data.get("last_trade_date"),
        "metrics": {
            "rsi": round(rsi, 2) if rsi is not None else None,
            "atr": _round_price(atr),
            "dma_50": _round_price(dma_50),
            "above_50dma": close > dma_50,
            "is_breakout": is_breakout,
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        },
    }


def _attach_signal_news(row: Dict, max_news: int = 2, news_days: int = 1) -> Dict:
    """Attach same-day/recent news snippets to one signal row."""
    symbol = row.get("symbol")
    if not symbol:
        row["news_today_count"] = 0
        row["news"] = []
        return row

    try:
        news = fetch_stock_news(symbol=symbol)
    except Exception as exc:
        row["news_today_count"] = 0
        row["news"] = []
        row["news_error"] = f"News fetch exception: {exc}"
        return row

    if news.get("status") != "success":
        row["news_today_count"] = 0
        row["news"] = []
        row["news_error"] = news.get("error_message", "news unavailable")
        return row

    articles = news.get("articles", [])
    recent = [a for a in articles if isinstance(a.get("days_ago"), int) and a.get("days_ago") <= news_days]
    today = [a for a in articles if a.get("days_ago") == 0]

    row["news_today_count"] = len(today)
    row["news_recent_count"] = len(recent)
    row["news"] = [
        {
            "title": a.get("title", ""),
            "publisher": a.get("publisher", "Unknown"),
            "days_ago": a.get("days_ago"),
            "published": a.get("published"),
        }
        for a in recent[:max_news]
    ]
    return row


def get_nifty50_signal_board(limit: int = 50, include_news: bool = True, max_news: int = 2, news_days: int = 1) -> Dict:
    """Build regime-aware BUY/SELL/HOLD signals for Nifty 50 watchlist symbols."""
    try:
        regime = analyze_regime()
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Failed to analyze market regime: {exc}",
        }
    if regime.get("status") != "success":
        return regime

    chosen_regime = regime["regime"]
    symbols = NSE_WATCHLIST[: max(1, int(limit))]

    rows: List[Dict] = []
    errors: List[str] = []
    for sym in symbols:
        row = _signal_row_for_symbol(sym, chosen_regime)
        if row.get("status") != "success":
            errors.append(f"{sym}: {row.get('error_message', 'signal build failed')}")
            continue
        if include_news:
            row = _attach_signal_news(row, max_news=max_news, news_days=news_days)
        rows.append(row)

    order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    rows.sort(key=lambda r: (order.get(r.get("signal", "HOLD"), 3), r.get("display_symbol", "")))

    counts = {
        "BUY": sum(1 for r in rows if r.get("signal") == "BUY"),
        "HOLD": sum(1 for r in rows if r.get("signal") == "HOLD"),
        "SELL": sum(1 for r in rows if r.get("signal") == "SELL"),
    }

    return {
        "status": "success",
        "generated_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "regime": chosen_regime,
        "strategy": regime.get("strategy"),
        "strategy_suggestions": regime.get("strategy_suggestions"),
        "stocks_requested": len(symbols),
        "stocks_scanned": len(rows),
        "signal_counts": counts,
        "signals": rows,
        "news_params": {
            "include_news": include_news,
            "max_news": max_news,
            "news_days": news_days,
        },
        "scan_errors": errors if errors else None,
        "source": "Yahoo Finance (yfinance)",
    }


scanner_agent = Agent(
    name="stock_scanner",
    model=GEMINI_MODEL,
    description=(
        "Scans NSE stocks for trade candidates using live market data. "
        "Supports breakout (bull), announcement momentum, and oversold bounce (sideways/bear)."
    ),
    instruction=(
        "You are the Stock Scanner. You have three scanning strategies:\n\n"
        "1. BREAKOUT SCAN: Use scan_watchlist_breakouts to find stocks breaking "
        "their 20-day high with volume confirmation. Best in BULL regime. Rank by volume ratio.\n\n"
        "2. ANNOUNCEMENT MOMENTUM: Use scan_announcement_momentum to find stocks with "
        "recent news-driven price moves. Interpret headlines; recommend only material news.\n\n"
        "3. OVERSOLD BOUNCE (for SIDEWAYS / BEAR): Use scan_oversold_bounce to find "
        "stocks with RSI <= 35 (oversold), often below 50-DMA. Strategy: buy the dip, "
        "tight stop (e.g. entry - 0.8*ATR), target mean reversion. When presenting "
        "candidates, say the user can 'implement' or 'paper trade [symbol]' to execute via trade_executor.\n\n"
        "4. BACKTEST OVERSOLD: Use backtest_oversold_bounce(symbol) for one stock, or "
        "backtest_oversold_nifty50() for the watchlist.\n"
        "5. TOP 5 RSI/OVERSOLD: When user asks for 'top 5 benefiting RSI stocks', call get_top_oversold_nifty50() "
        "and return its result (do NOT guess from memory).\n"
        "6. BEST STOCKS FROM NIFTY 50: When user asks how to SELECT the best stocks from Nifty 50, "
        "call get_best_oversold_nifty50(). It runs the backtest and keeps only stocks that meet "
        "min win rate (e.g. 50%%) and min avg return (e.g. 0%%) and min trades (e.g. 3). Return "
        "best_symbols and best_stocks so the user can focus scanning/paper trading on these only.\n\n"
        "For individual stock analysis, use get_stock_analysis.\n"
        "When asked for strategies in bear or sideways markets, run scan_oversold_bounce and explain."
    ),
    tools=[
        scan_watchlist_breakouts,
        get_stock_analysis,
        scan_announcement_momentum,
        scan_oversold_bounce,
        backtest_oversold_bounce,
        backtest_oversold_nifty50,
        get_top_oversold_nifty50,
        get_best_oversold_nifty50,
    ],
)
