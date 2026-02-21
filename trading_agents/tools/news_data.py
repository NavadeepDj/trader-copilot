"""Live stock news fetching via yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))


def fetch_stock_news(symbol: str) -> Dict:
    """Fetch recent news articles for a stock via yfinance.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with status, list of recent articles (title, summary, date, publisher).
    """
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        symbol = symbol.upper() + ".NS"

    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Failed to fetch news for '{symbol}': {exc}",
        }

    if not raw_news:
        return {
            "status": "success",
            "symbol": symbol,
            "article_count": 0,
            "articles": [],
            "fetched_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    articles: List[Dict] = []
    for item in raw_news:
        content = item.get("content", {})
        pub_date = content.get("pubDate", "")
        provider = content.get("provider", {})

        articles.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "published": pub_date,
            "publisher": provider.get("displayName", "Unknown"),
        })

    now_ist = datetime.now(IST)

    recent_articles: List[Dict] = []
    for art in articles:
        pub = art["published"]
        if pub:
            try:
                pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                days_ago = (now_ist - pub_dt.astimezone(IST)).days
                art["days_ago"] = days_ago
            except (ValueError, TypeError):
                art["days_ago"] = None
        else:
            art["days_ago"] = None
        recent_articles.append(art)

    recent_articles.sort(key=lambda a: a.get("days_ago") or 999)

    return {
        "status": "success",
        "symbol": symbol,
        "article_count": len(recent_articles),
        "articles": recent_articles,
        "fetched_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
    }
