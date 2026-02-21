"""Dividend announcement strategy agent -- finds and evaluates dividend plays."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL
from trading_agents.tools.dividend_data import search_upcoming_dividends
from trading_agents.tools.fundamental_data import (
    assess_dividend_health,
    get_stock_fundamentals,
)
from trading_agents.tools.market_data import fetch_stock_data
from trading_agents.tools.technical import compute_atr, compute_index_metrics

IST = timezone(timedelta(hours=5, minutes=30))


def scan_dividend_opportunities(min_days_to_ex: int = 3) -> Dict:
    """Discover dividend opportunities across the entire Indian stock market.

    Flow:
      1. Gemini Google Search discovers ALL recently announced dividends
         (searches Moneycontrol, BSE, NSE, Tickertape, etc.)
      2. For each discovered stock, fetches fundamentals + historical data
         from yfinance to assess dividend health and technical trend.
      3. Ranks and returns actionable opportunities.

    Args:
        min_days_to_ex: Minimum days until ex-date to consider (default 3).

    Returns:
        dict with ranked dividend opportunities.
    """
    search_result = search_upcoming_dividends()

    if search_result.get("status") != "success":
        return {
            "status": "error",
            "error_message": f"Dividend discovery failed: {search_result.get('error_message', 'unknown')}",
            "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    candidates = search_result.get("candidates", [])
    if not candidates:
        return {
            "status": "success",
            "strategy": "DIVIDEND_ANNOUNCEMENT",
            "message": "No upcoming dividend announcements found via web search.",
            "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    filtered = [c for c in candidates if c["days_to_ex_date"] >= min_days_to_ex]

    opportunities: List[Dict] = []
    skipped: List[Dict] = []

    for candidate in filtered:
        sym = candidate.get("symbol")
        company = candidate.get("company", "?")
        ex_date = candidate.get("ex_date", "?")
        days_left = candidate.get("days_to_ex_date", 0)

        base = {
            "company": company,
            "symbol": sym,
            "ex_date": ex_date,
            "days_to_ex": days_left,
            "dividend_amount_rs": candidate.get("dividend_amount_rs"),
            "announcement_date": candidate.get("announcement_date"),
        }

        if sym is None:
            skipped.append({**base, "skip_reason": "NSE symbol not identified by search"})
            continue

        print(f"[dividend] Analyzing {sym} ({company})...")

        health = assess_dividend_health(sym)
        if health.get("status") != "success":
            skipped.append({**base, "skip_reason": "yfinance data not available"})
            continue

        verdict = health.get("dividend_health", "CAUTION")
        health_score = health.get("health_score", 0)

        if verdict == "DESPERATE":
            skipped.append({
                **base,
                "skip_reason": f"DESPERATE dividend (score {health_score})",
            })
            continue

        stock_data = fetch_stock_data(symbol=sym)
        if stock_data.get("status") != "success":
            skipped.append({**base, "skip_reason": "price history fetch failed"})
            continue

        closes = stock_data["closes"]
        highs = stock_data["highs"]
        lows = stock_data["lows"]

        metrics = compute_index_metrics(closes)
        if metrics.get("status") != "success":
            skipped.append({**base, "skip_reason": "insufficient price history"})
            continue

        above_50dma = closes[-1] > metrics["dma_50"]
        uptrend = metrics["dma_50_slope"] > 0
        atr = compute_atr(highs, lows, closes)

        fundamentals = get_stock_fundamentals(sym)
        f_data = fundamentals.get("fundamentals", {})
        div_yield = f_data.get("dividendYield")
        trailing_pe = f_data.get("trailingPE")

        rank_score = health_score
        if div_yield and div_yield > 0:
            rank_score += div_yield * 10
        if uptrend:
            rank_score += 15
        if above_50dma:
            rank_score += 10
        if days_left >= 5:
            rank_score += 10

        opportunities.append({
            **base,
            "analysis_status": "fully_analyzed",
            "dividend_health": verdict,
            "health_score": health_score,
            "health_reasons": health.get("reasons", []),
            "dividend_yield_pct": round(div_yield, 2) if div_yield else None,
            "trailing_pe": round(trailing_pe, 2) if trailing_pe else None,
            "current_price": round(closes[-1], 2),
            "dma_50": round(metrics["dma_50"], 2),
            "above_50dma": above_50dma,
            "uptrend": uptrend,
            "return_20d_pct": round(metrics["return_20d"] * 100, 2),
            "atr": round(atr, 2),
            "suggested_entry": round(closes[-1], 2),
            "suggested_stop": round(closes[-1] - 1.5 * atr, 2),
            "suggested_exit": f"1-2 days before ex-date ({ex_date})",
            "rank_score": round(rank_score, 1),
        })

    opportunities.sort(key=lambda x: x.get("rank_score", 0), reverse=True)

    skipped_summary = [
        {"symbol": s.get("symbol", "?"), "company": s.get("company", "?"),
         "reason": s.get("skip_reason", "unknown")}
        for s in skipped
    ]

    return {
        "status": "success",
        "strategy": "DIVIDEND_ANNOUNCEMENT",
        "discovery_source": "Gemini Google Search (entire Indian market)",
        "dividends_discovered": len(candidates),
        "analyzed": len(filtered),
        "opportunities_count": len(opportunities),
        "top_opportunities": opportunities[:8],
        "skipped_count": len(skipped),
        "skipped_summary": skipped_summary[:10],
        "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }


def analyze_dividend_stock(symbol: str) -> Dict:
    """Deep-dive analysis of a single stock for dividend opportunity.

    Combines fundamentals, dividend health, technical trend, and news
    into a comprehensive assessment.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with full dividend analysis: health, technicals, entry/exit levels.
    """
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        symbol = symbol.upper() + ".NS"

    fundamentals = get_stock_fundamentals(symbol)
    if fundamentals.get("status") != "success":
        return fundamentals

    health = assess_dividend_health(symbol)
    stock_data = fetch_stock_data(symbol=symbol)

    analysis: Dict = {
        "status": "success",
        "symbol": symbol,
        "fundamentals": fundamentals.get("fundamentals", {}),
        "dividend_health": health.get("dividend_health", "UNKNOWN"),
        "health_score": health.get("health_score", 0),
        "health_reasons": health.get("reasons", []),
    }

    if stock_data.get("status") == "success":
        closes = stock_data["closes"]
        highs = stock_data["highs"]
        lows = stock_data["lows"]

        metrics = compute_index_metrics(closes)
        atr = compute_atr(highs, lows, closes)

        if metrics.get("status") == "success":
            analysis["technicals"] = {
                "current_price": round(closes[-1], 2),
                "dma_50": round(metrics["dma_50"], 2),
                "above_50dma": closes[-1] > metrics["dma_50"],
                "dma_50_slope": metrics["dma_50_slope"],
                "uptrend": metrics["dma_50_slope"] > 0,
                "return_20d_pct": round(metrics["return_20d"] * 100, 2),
                "volatility": metrics["volatility"],
                "atr": round(atr, 2),
            }

            analysis["trade_levels"] = {
                "suggested_entry": round(closes[-1], 2),
                "suggested_stop": round(closes[-1] - 1.5 * atr, 2),
                "risk_per_share": round(1.5 * atr, 2),
            }

    f = fundamentals.get("fundamentals", {})
    ex_info = f.get("exDividendDate")
    if ex_info:
        analysis["ex_dividend_date"] = ex_info

    analysis["fetched_at_ist"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return analysis


dividend_agent = Agent(
    name="dividend_scanner",
    model=GEMINI_MODEL,
    description=(
        "Scans for dividend announcement trade opportunities in NSE stocks. "
        "Finds upcoming dividends, assesses dividend health (HEALTHY vs DESPERATE), "
        "applies technical filters, and recommends entry/exit levels."
    ),
    instruction=(
        "You are the Dividend Strategy Scanner. Your specialty is finding "
        "profitable trades around dividend announcements.\n\n"
        "STRATEGY THESIS:\n"
        "Stocks tend to rise between dividend announcement and ex-date. "
        "We buy after announcement, sell 1-2 days BEFORE ex-date to avoid "
        "the ex-date price drop.\n\n"
        "HOW IT WORKS:\n"
        "scan_dividend_opportunities searches the ENTIRE Indian stock market "
        "via Google Search (Moneycontrol, BSE, NSE, Tickertape). For each "
        "company found, it automatically fetches fundamentals and historical "
        "data from yfinance to assess health and technicals. No limited watchlist.\n\n"
        "TOOLS:\n"
        "1. scan_dividend_opportunities: Discovers dividend announcements "
        "market-wide, then analyzes each with yfinance data. Returns ranked "
        "opportunities with health, technicals, and entry/exit.\n"
        "2. analyze_dividend_stock: Deep-dive on a single stock.\n\n"
        "CRITICAL RULES:\n"
        "- NEVER recommend stocks with DESPERATE dividend health.\n"
        "- Always mention the ex-date and how many trading days remain.\n"
        "- Always recommend selling BEFORE ex-date, not on or after.\n"
        "- Flag CAUTION stocks explicitly.\n"
        "- Show concrete entry price, stop-loss, and exit timing.\n"
        "- If no good opportunities exist, say so honestly.\n\n"
        "FORMAT:\n"
        "- Keep responses SHORT. No lengthy preambles or methodology recaps.\n"
        "- Present in a table: Symbol, Yield, Health, Ex-Date, Days Left, "
        "Entry, Stop, Verdict.\n"
        "- 1-2 sentences per opportunity after the table.\n"
        "- Skipped stocks: one line each (symbol + reason).\n"
        "- Be data-driven, cite specific numbers."
    ),
    tools=[scan_dividend_opportunities, analyze_dividend_stock],
)
