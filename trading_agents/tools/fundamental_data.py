"""Stock fundamental data and dividend health assessment via yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict

import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))

_FUNDAMENTAL_KEYS = [
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "dividendRate",
    "payoutRatio",
    "earningsGrowth",
    "revenueGrowth",
    "debtToEquity",
    "returnOnEquity",
    "marketCap",
    "bookValue",
    "priceToBook",
    "currentPrice",
    "regularMarketPrice",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "shortName",
    "sector",
    "industry",
]


def get_stock_fundamentals(symbol: str) -> Dict:
    """Fetch key fundamental metrics for an NSE stock.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with fundamental metrics extracted from yfinance .info.
    """
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        symbol = symbol.upper() + ".NS"

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Failed to fetch fundamentals for '{symbol}': {exc}",
        }

    fundamentals: Dict = {"symbol": symbol}
    _PCT_KEYS = {"payoutRatio", "earningsGrowth", "revenueGrowth", "returnOnEquity"}
    for key in _FUNDAMENTAL_KEYS:
        val = info.get(key)
        if val is not None:
            if isinstance(val, float):
                if key in _PCT_KEYS:
                    fundamentals[key] = round(val * 100, 2)
                    fundamentals[f"{key}_display"] = f"{val * 100:.2f}%"
                elif key == "dividendYield":
                    pass  # handled below with manual computation
                else:
                    fundamentals[key] = round(val, 2)
            else:
                fundamentals[key] = val

    price = fundamentals.get("currentPrice") or fundamentals.get("regularMarketPrice")
    if price:
        fundamentals["current_price"] = round(float(price), 2)

    div_rate = info.get("dividendRate")
    current_price = float(price) if price else None
    if div_rate and current_price and current_price > 0:
        computed_yield = (float(div_rate) / current_price) * 100
        fundamentals["dividendYield"] = round(computed_yield, 2)
        fundamentals["dividendYield_display"] = f"{computed_yield:.2f}%"
        fundamentals["dividendRate"] = round(float(div_rate), 2)
    elif info.get("dividendYield") is not None:
        raw = float(info["dividendYield"])
        pct = raw * 100 if raw < 1.0 else raw
        fundamentals["dividendYield"] = round(pct, 2)
        fundamentals["dividendYield_display"] = f"{pct:.2f}%"

    return {
        "status": "success",
        "symbol": symbol,
        "fundamentals": fundamentals,
        "fetched_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }


def assess_dividend_health(symbol: str) -> Dict:
    """Classify a stock's dividend as HEALTHY, CAUTION, or DESPERATE.

    Combines earnings growth, payout ratio, PE ratio, and return on equity
    to determine whether the dividend is sustainable.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with health verdict, score, reasoning, and underlying fundamentals.
    """
    result = get_stock_fundamentals(symbol)
    if result.get("status") != "success":
        return result

    f = result["fundamentals"]

    earnings_growth = f.get("earningsGrowth")  # already in % form
    payout_ratio = f.get("payoutRatio")        # already in % form
    trailing_pe = f.get("trailingPE")
    roe = f.get("returnOnEquity")              # already in % form
    div_yield = f.get("dividendYield")         # already in % form
    debt_to_equity = f.get("debtToEquity")

    reasons: list[str] = []
    score = 50  # neutral start

    if earnings_growth is not None:
        if earnings_growth > 10:
            score += 20
            reasons.append(f"Strong earnings growth ({earnings_growth:.1f}%)")
        elif earnings_growth > 0:
            score += 10
            reasons.append(f"Positive earnings growth ({earnings_growth:.1f}%)")
        elif earnings_growth > -10:
            score -= 10
            reasons.append(f"Weak earnings growth ({earnings_growth:.1f}%)")
        else:
            score -= 25
            reasons.append(f"Declining earnings ({earnings_growth:.1f}%)")

    if payout_ratio is not None:
        if payout_ratio < 40:
            score += 15
            reasons.append(f"Conservative payout ratio ({payout_ratio:.0f}%)")
        elif payout_ratio < 70:
            score += 5
            reasons.append(f"Moderate payout ratio ({payout_ratio:.0f}%)")
        elif payout_ratio < 100:
            score -= 10
            reasons.append(f"High payout ratio ({payout_ratio:.0f}%)")
        else:
            score -= 25
            reasons.append(f"Unsustainable payout ratio ({payout_ratio:.0f}%) - paying more than earned")

    if trailing_pe is not None:
        if 0 < trailing_pe < 20:
            score += 10
            reasons.append(f"Reasonable PE ({trailing_pe:.1f})")
        elif trailing_pe < 40:
            score += 0
            reasons.append(f"Moderate PE ({trailing_pe:.1f})")
        elif trailing_pe > 0:
            score -= 10
            reasons.append(f"Expensive PE ({trailing_pe:.1f})")
        else:
            score -= 20
            reasons.append(f"Negative PE ({trailing_pe:.1f}) - company is loss-making")

    if roe is not None:
        if roe > 15:
            score += 10
            reasons.append(f"Good ROE ({roe:.1f}%)")
        elif roe > 8:
            score += 5
            reasons.append(f"Decent ROE ({roe:.1f}%)")
        else:
            score -= 5
            reasons.append(f"Weak ROE ({roe:.1f}%)")

    if debt_to_equity is not None:
        if debt_to_equity > 200:
            score -= 10
            reasons.append(f"High debt-to-equity ({debt_to_equity:.0f})")

    score = max(0, min(100, score))

    if score >= 65:
        verdict = "HEALTHY"
    elif score >= 40:
        verdict = "CAUTION"
    else:
        verdict = "DESPERATE"

    return {
        "status": "success",
        "symbol": f.get("symbol", symbol),
        "company": f.get("shortName", symbol),
        "dividend_health": verdict,
        "health_score": score,
        "reasons": reasons,
        "key_metrics": {
            "earnings_growth_pct": earnings_growth,
            "payout_ratio_pct": payout_ratio,
            "trailing_pe": trailing_pe,
            "roe_pct": roe,
            "dividend_yield_pct": div_yield,
            "debt_to_equity": debt_to_equity,
        },
        "fetched_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }
