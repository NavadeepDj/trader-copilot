"""Dividend announcement discovery via yfinance watchlist scan + Gemini Google Search."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import yfinance as yf

from trading_agents.config import NSE_WATCHLIST, call_gemini_with_fallback, create_genai_client

IST = timezone(timedelta(hours=5, minutes=30))

import logging as _logging

_yf_logger = _logging.getLogger("yfinance")


def _validate_symbol(symbol: str) -> bool:
    """Quick check whether a symbol exists on yfinance (suppresses 404 noise)."""
    prev_level = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return price is not None
    except Exception:
        return False
    finally:
        _yf_logger.setLevel(prev_level)


def _try_fix_symbol(raw_symbol: str) -> str | None:
    """Try common NSE symbol variations if the original doesn't validate.

    Returns the first valid .NS symbol found, or None.
    """
    base = raw_symbol.upper().replace(".NS", "")

    candidates = [
        base,
        base.replace(" ", ""),
        base.replace("-", ""),
        base[:10],
    ]
    if base.endswith("LTD"):
        candidates.append(base[:-3])
    if base.endswith("LIMITED"):
        candidates.append(base[:-7])
    if not base.endswith("IND") and len(base) <= 15:
        candidates.append(base + "IND")

    seen: set = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        sym = c + ".NS"
        if _validate_symbol(sym):
            return sym

    return None


def _parse_ex_date(ex_ts) -> date | None:
    """Convert various exDividendDate formats to a date object."""
    if ex_ts is None:
        return None
    if isinstance(ex_ts, (int, float)):
        return date.fromtimestamp(ex_ts)
    if isinstance(ex_ts, str):
        try:
            return datetime.fromisoformat(ex_ts).date()
        except ValueError:
            return None
    if isinstance(ex_ts, datetime):
        return ex_ts.date()
    if isinstance(ex_ts, date):
        return ex_ts
    return None


def scan_watchlist_dividends() -> Dict:
    """Scan the NSE watchlist (~80 stocks) for upcoming ex-dividend dates.

    Suppresses yfinance error logs during scanning to keep output clean.

    Returns:
        dict with upcoming dividend candidates and metadata.
    """
    today = datetime.now(IST).date()
    candidates: List[Dict] = []
    scanned: List[str] = []
    errors: List[str] = []

    prev_level = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)

    try:
        for sym in NSE_WATCHLIST:
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info or {}
            except Exception as exc:
                errors.append(f"{sym}: {exc}")
                continue

            scanned.append(sym)

            ex_date = _parse_ex_date(info.get("exDividendDate"))
            if ex_date is None or ex_date <= today:
                continue

            days_to_ex = (ex_date - today).days
            div_rate = info.get("dividendRate")
            trailing_pe = info.get("trailingPE")
            payout_ratio = info.get("payoutRatio")
            earnings_growth = info.get("earningsGrowth")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

            if div_rate and current_price and float(current_price) > 0:
                yield_pct = round((float(div_rate) / float(current_price)) * 100, 2)
            else:
                raw_yield = info.get("dividendYield")
                if raw_yield is not None:
                    y = float(raw_yield)
                    yield_pct = round(y * 100 if y < 1.0 else y, 2)
                else:
                    yield_pct = None

            candidates.append({
                "symbol": sym,
                "company": info.get("shortName", sym),
                "current_price": round(float(current_price), 2) if current_price else None,
                "ex_dividend_date": ex_date.isoformat(),
                "days_to_ex_date": days_to_ex,
                "dividend_rate_rs": round(float(div_rate), 2) if div_rate else None,
                "dividend_yield_pct": yield_pct,
                "trailing_pe": round(float(trailing_pe), 2) if trailing_pe else None,
                "payout_ratio_pct": round(float(payout_ratio) * 100, 1) if payout_ratio else None,
                "earnings_growth_pct": round(float(earnings_growth) * 100, 1) if earnings_growth else None,
            })
    finally:
        _yf_logger.setLevel(prev_level)

    candidates.sort(key=lambda c: c["days_to_ex_date"])
    print(f"[dividend] Watchlist scan: {len(scanned)} stocks scanned, "
          f"{len(candidates)} with upcoming dividends")

    return {
        "status": "success",
        "source": f"yfinance watchlist scan ({len(scanned)} stocks)",
        "stocks_scanned": len(scanned),
        "upcoming_dividends": len(candidates),
        "candidates": candidates,
        "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "scan_errors": errors[:5] if errors else None,
    }


def search_upcoming_dividends() -> Dict:
    """Use Gemini with Google Search grounding to find upcoming NSE dividend announcements.

    Gemini searches the web (Moneycontrol, BSE, NSE, Tickertape, etc.) for
    recently announced dividends with future ex-dates, then returns structured data.

    Returns:
        dict with dividend candidates discovered via web search.
    """
    try:
        from google.genai import types
    except ImportError:
        return {
            "status": "error",
            "error_message": "google-genai not installed.",
        }

    try:
        client = create_genai_client()
    except ValueError as exc:
        return {
            "status": "error",
            "error_message": f"No credentials configured: {exc}",
        }

    today = datetime.now(IST).date()
    prompt = (
        f"Today is {today.isoformat()}. "
        "Search the Indian stock market for ALL companies that have recently "
        "announced dividends with ex-dates in the next 30-45 days.\n\n"
        "Search these sources thoroughly:\n"
        "- Moneycontrol upcoming dividends page\n"
        "- BSE India corporate actions\n"
        "- NSE India corporate actions\n"
        "- Tickertape dividends calendar\n"
        "- Economic Times dividends section\n\n"
        "IMPORTANT for nse_symbol: Use the EXACT NSE trading symbol as it "
        "appears on NSE India website. Examples: PIIND, ENGINERSIN, NBCC, "
        "CASTROLIND, COALINDIA, NHPC, SANOFI, RELIANCE, TCS.\n"
        "Do NOT abbreviate or modify the symbol.\n\n"
        "Return ONLY a valid JSON array (no markdown, no explanation) with objects "
        "having these exact fields:\n"
        '- "company_name": string (full company name)\n'
        '- "nse_symbol": string (EXACT NSE ticker without .NS suffix)\n'
        '- "dividend_amount_rs": number (dividend per share in Rs, or null)\n'
        '- "dividend_type": string ("Interim" or "Final" or null)\n'
        '- "announcement_date": string (YYYY-MM-DD format, or null)\n'
        '- "ex_date": string (YYYY-MM-DD format)\n\n'
        "Include ALL companies you can find -- aim for 10-30 results. "
        "Cover large-cap, mid-cap, and small-cap companies. "
        "Only include stocks with ex-dates AFTER today."
    )

    try:
        response = call_gemini_with_fallback(
            client=client,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Gemini search failed (all models exhausted): {exc}",
        }

    raw_text = response.text.strip() if response.text else ""
    if not raw_text:
        return {
            "status": "error",
            "error_message": "Gemini returned empty response.",
        }

    candidates = _parse_gemini_dividend_response(raw_text, today)

    return {
        "status": "success",
        "source": "Gemini Google Search grounding",
        "dividends_found": len(candidates),
        "candidates": candidates,
        "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }


def _parse_gemini_dividend_response(text: str, today: date) -> List[Dict]:
    """Extract structured dividend data from Gemini's response text."""
    json_match = re.search(r"\[[\s\S]*\]", text)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    candidates: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        company = item.get("company_name", "")
        symbol = item.get("nse_symbol", "")
        ex_date_str = item.get("ex_date", "")

        if not company or not ex_date_str:
            continue

        try:
            ex_date = datetime.strptime(ex_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        if ex_date <= today:
            continue

        if symbol and not symbol.upper().endswith(".NS"):
            symbol = symbol.upper() + ".NS"

        if symbol and not _validate_symbol(symbol):
            fixed = _try_fix_symbol(symbol)
            if fixed:
                print(f"[dividend] Fixed symbol: {symbol} -> {fixed}")
                symbol = fixed
            else:
                print(f"[dividend] Skipping {symbol} ({company}) -- not found on yfinance")
                continue

        candidates.append({
            "company": company,
            "symbol": symbol if symbol else None,
            "dividend_amount_rs": item.get("dividend_amount_rs"),
            "dividend_type": item.get("dividend_type"),
            "announcement_date": item.get("announcement_date"),
            "ex_date": ex_date.isoformat(),
            "days_to_ex_date": (ex_date - today).days,
            "source": "gemini_search",
        })

    candidates.sort(key=lambda c: c["days_to_ex_date"])
    return candidates
