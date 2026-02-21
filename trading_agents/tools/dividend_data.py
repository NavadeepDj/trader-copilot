"""Dividend discovery via Moneycontrol API + yfinance symbol validation."""

from __future__ import annotations

import logging as _logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import requests
import yfinance as yf

from trading_agents.config import call_gemini_with_fallback, create_genai_client

IST = timezone(timedelta(hours=5, minutes=30))

_yf_logger = _logging.getLogger("yfinance")

_mc_cache: Dict | None = None
_mc_cache_time: datetime | None = None
_MC_CACHE_TTL_SECONDS = 300  # 5-minute cache

_MC_API_URL = "https://api.moneycontrol.com/mcapi/v1/ecalendar/corporate-action"
_MC_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.moneycontrol.com/",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Symbol validation helpers
# ---------------------------------------------------------------------------

def _validate_symbol(symbol: str) -> bool:
    """Quick check whether a symbol exists on yfinance (suppresses 404 noise)."""
    prev_level = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)
    try:
        info = yf.Ticker(symbol).info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return price is not None
    except Exception:
        return False
    finally:
        _yf_logger.setLevel(prev_level)


def _derive_nse_candidates(stock_name: str, mc_url: str | None) -> List[str]:
    """Generate candidate NSE tickers from a Moneycontrol company name + URL slug."""
    candidates: List[str] = []

    slug_match = re.search(r"/stockpricequote/[^/]+/([^/]+)/", mc_url or "")
    slug = slug_match.group(1).upper() if slug_match else None

    if slug:
        base = slug
        for suffix in ("LIMITED", "LTD", "INDIA", "IND"):
            if base.endswith(suffix) and len(base) > len(suffix):
                candidates.append(base[: -len(suffix)])
        candidates.append(base)
        for n in (12, 10, 8, 6, 4):
            if len(base) > n:
                candidates.append(base[:n])

    clean = re.sub(r"[^A-Za-z0-9]", "", stock_name).upper()
    candidates.append(clean)
    candidates.append(clean[:10])

    if "(" in stock_name:
        before_paren = stock_name.split("(")[0].strip().upper().replace(" ", "").replace(".", "")
        candidates.append(before_paren)

    words = re.findall(r"[A-Za-z]+", stock_name.upper())
    if len(words) >= 2:
        candidates.append(words[0] + words[1][:3])
        candidates.append(words[0][:3] + words[1][:3])
    if words:
        candidates.append(words[0])

    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _resolve_nse_symbol(stock_name: str, mc_url: str | None) -> str | None:
    """Try to find a valid yfinance .NS symbol for a Moneycontrol stock.

    1. Algorithmic: try slug/name-derived candidates on yfinance.
    2. Gemini fallback: ask Gemini for the exact NSE symbol.
    """
    prev_level = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)
    try:
        for candidate in _derive_nse_candidates(stock_name, mc_url):
            sym = candidate + ".NS"
            try:
                info = yf.Ticker(sym).info or {}
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if price:
                    print(f"[dividend] Mapped '{stock_name}' -> {sym}")
                    return sym
            except Exception:
                continue
    finally:
        _yf_logger.setLevel(prev_level)

    return _resolve_symbol_via_gemini(stock_name)


def _resolve_symbol_via_gemini(stock_name: str) -> str | None:
    """Ask Gemini for the exact NSE ticker when algorithmic mapping fails."""
    try:
        client = create_genai_client()
    except (ValueError, Exception):
        return None

    prompt = (
        f'What is the exact NSE (National Stock Exchange of India) ticker symbol '
        f'for "{stock_name}"? Reply with ONLY the ticker symbol in uppercase, '
        f'nothing else. Example: ENGINERSIN'
    )

    try:
        response = call_gemini_with_fallback(client=client, contents=prompt)
        text = (response.text or "").strip().upper()
        symbol = re.sub(r"[^A-Z0-9&-]", "", text)
        if not symbol or len(symbol) > 20:
            return None

        sym = symbol + ".NS"
        if _validate_symbol(sym):
            print(f"[dividend] Gemini resolved '{stock_name}' -> {sym}")
            return sym
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Moneycontrol API discovery
# ---------------------------------------------------------------------------

def fetch_moneycontrol_dividends() -> Dict:
    """Fetch upcoming dividend announcements from the Moneycontrol API.

    Calls the corporate-action calendar API with event=D (dividends) and
    duration=UP (upcoming). Paginates through all pages. Results are cached
    for 5 minutes to avoid duplicate API calls and symbol resolution.

    Returns:
        dict with dividend candidates including company name, ex-date,
        dividend amount, LTP, market cap, and resolved NSE symbol.
    """
    global _mc_cache, _mc_cache_time
    now = datetime.now(IST)
    if (_mc_cache is not None
            and _mc_cache_time is not None
            and (now - _mc_cache_time).total_seconds() < _MC_CACHE_TTL_SECONDS):
        print("[dividend] Using cached Moneycontrol results "
              f"({int((now - _mc_cache_time).total_seconds())}s old)")
        return _mc_cache

    today = now.date()
    all_items: List[Dict] = []

    for page in range(1, 6):
        params = {
            "indexId": "All",
            "page": page,
            "event": "D",
            "apiVersion": 161,
            "orderBy": "asc",
            "deviceType": "W",
            "duration": "UP",
        }
        try:
            resp = requests.get(
                _MC_API_URL, params=params, headers=_MC_HEADERS, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", {}).get("list", [])
            if not items:
                break
            all_items.extend(items)
        except Exception as exc:
            print(f"[dividend] Moneycontrol API page {page} error: {exc}")
            if page == 1:
                return {
                    "status": "error",
                    "error_message": f"Moneycontrol API request failed: {exc}",
                }
            break

    candidates: List[Dict] = []
    symbol_failures: List[str] = []

    for item in all_items:
        stock_name = item.get("stockName", "")
        ex_date_str = item.get("exDate", "")
        if not stock_name or not ex_date_str:
            continue

        try:
            ex_date = datetime.strptime(ex_date_str, "%d/%m/%Y").date()
        except (ValueError, TypeError):
            continue

        if ex_date <= today:
            continue

        mc_url = item.get("url")
        nse_symbol = _resolve_nse_symbol(stock_name, mc_url)

        if not nse_symbol:
            symbol_failures.append(stock_name)
            print(f"[dividend] Could not resolve NSE symbol for '{stock_name}' -- skipping")
            continue

        ltp_str = (item.get("lastValue") or "").replace(",", "")
        try:
            ltp = round(float(ltp_str), 2) if ltp_str else None
        except ValueError:
            ltp = None

        div_str = item.get("dividend", "")
        try:
            div_amount = round(float(div_str), 2) if div_str and div_str != "-" else None
        except ValueError:
            div_amount = None

        ann_date_str = item.get("announcementDate", "")
        try:
            ann_date = datetime.strptime(ann_date_str, "%d/%m/%Y").date().isoformat()
        except (ValueError, TypeError):
            ann_date = None

        event_type = item.get("eventType", "")
        div_type = None
        if "Interim" in event_type:
            div_type = "Interim"
        elif "Final" in event_type:
            div_type = "Final"

        candidates.append({
            "company": stock_name,
            "symbol": nse_symbol,
            "ex_date": ex_date.isoformat(),
            "days_to_ex_date": (ex_date - today).days,
            "dividend_amount_rs": div_amount,
            "dividend_type": div_type,
            "announcement_date": ann_date,
            "ltp": ltp,
            "market_cap": item.get("marketCap"),
            "source": "moneycontrol_api",
        })

    candidates.sort(key=lambda c: c["days_to_ex_date"])
    print(f"[dividend] Moneycontrol: {len(all_items)} records, "
          f"{len(candidates)} resolved, {len(symbol_failures)} unmapped")

    result = {
        "status": "success",
        "source": "Moneycontrol API (corporate actions calendar)",
        "total_records": len(all_items),
        "dividends_found": len(candidates),
        "candidates": candidates,
        "unmapped_companies": symbol_failures if symbol_failures else None,
        "scan_date_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }

    _mc_cache = result
    _mc_cache_time = datetime.now(IST)
    return result
