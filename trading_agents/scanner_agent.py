"""Stock scanner sub-agent -- scans NSE watchlist for breakout candidates."""

from __future__ import annotations

from typing import Dict, List

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL, NSE_WATCHLIST
from trading_agents.tools.market_data import fetch_stock_data
from trading_agents.tools.news_data import fetch_stock_news
from trading_agents.tools.technical import detect_breakout


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


scanner_agent = Agent(
    name="stock_scanner",
    model=GEMINI_MODEL,
    description=(
        "Scans NSE stocks for trade candidates using live market data. "
        "Supports two strategies: 20-day breakout scanning and announcement momentum detection."
    ),
    instruction=(
        "You are the Stock Scanner. You have two scanning strategies:\n\n"
        "1. BREAKOUT SCAN: Use scan_watchlist_breakouts to find stocks breaking "
        "their 20-day high with volume confirmation. Report candidates ranked by volume ratio.\n\n"
        "2. ANNOUNCEMENT MOMENTUM: Use scan_announcement_momentum to find stocks with "
        "recent news-driven price moves (>2% in 5 days + above-average volume + fresh news). "
        "When presenting these candidates, interpret the news headlines -- highlight "
        "significant corporate announcements (earnings, results, buybacks, M&A, contracts) "
        "vs. generic market noise. Only recommend stocks where the news is genuinely material.\n\n"
        "For individual stock analysis, use get_stock_analysis.\n"
        "When asked to scan broadly, run BOTH strategies and present results together."
    ),
    tools=[scan_watchlist_breakouts, get_stock_analysis, scan_announcement_momentum],
)
