"""Stock scanner sub-agent -- scans NSE watchlist for breakout candidates."""

from __future__ import annotations

from typing import Dict, List

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL, NSE_WATCHLIST
from trading_agents.tools.backtest_oversold import (
    backtest_oversold_bounce,
    backtest_oversold_nifty50,
    get_top_oversold_nifty50,
)
from trading_agents.tools.market_data import fetch_stock_data
from trading_agents.tools.news_data import fetch_stock_news
from trading_agents.tools.technical import compute_atr, compute_index_metrics, compute_rsi, detect_breakout


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
        "5. TOP 5 RSI/OVERSOLD STOCKS (CRITICAL): When the user asks for 'top 5 benefiting RSI stocks' "
        "or 'top 5 oversold stocks for Nifty 50', you MUST call get_top_oversold_nifty50() and "
        "return ITS result. Do NOT guess or list stocks from memory — Gemini cannot know which "
        "stocks benefit most; only the backtest does. Present the top_symbols and top_stocks "
        "from the tool; then the user can run scan_oversold_bounce on those symbols only for profit.\n\n"
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
    ],
)
