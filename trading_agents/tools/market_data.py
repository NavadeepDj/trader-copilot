"""Live NSE market data fetching via yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
from typing import Dict

import yfinance as yf

from trading_agents.config import DATA_LOOKBACK_DAYS


def fetch_index_data(symbol: str = "^NSEI", days: int = DATA_LOOKBACK_DAYS) -> Dict:
    """Fetch daily OHLCV data for a market index (default: Nifty 50).

    Args:
        symbol: Yahoo Finance ticker symbol for the index.
        days: Number of calendar days of history to fetch.

    Returns:
        dict with status, closes, highs, lows, volumes, metadata.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{days}d", interval="1d")

    if hist is None or hist.empty:
        return {
            "status": "error",
            "error_message": f"No data returned for '{symbol}'. Market may be closed or symbol invalid.",
        }

    closes = [round(float(v), 2) for v in hist["Close"].dropna().tolist()]
    highs = [round(float(v), 2) for v in hist["High"].dropna().tolist()]
    lows = [round(float(v), 2) for v in hist["Low"].dropna().tolist()]
    volumes = [int(v) for v in hist["Volume"].dropna().tolist()]

    if len(closes) < 60:
        return {
            "status": "error",
            "error_message": f"Only {len(closes)} trading days available for '{symbol}'. Need at least 60.",
        }

    last_ts = str(hist.index[-1])

    return {
        "status": "success",
        "symbol": symbol,
        "source": "Yahoo Finance (yfinance)",
        "fetched_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "last_trade_date": last_ts,
        "trading_days": len(closes),
        "latest_close": closes[-1],
        "last_5_closes": closes[-5:],
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
    }


def fetch_stock_data(symbol: str, days: int = DATA_LOOKBACK_DAYS) -> Dict:
    """Fetch daily OHLCV data for an individual NSE stock.

    Args:
        symbol: Yahoo Finance ticker (e.g. 'RELIANCE.NS').
        days: Number of calendar days of history to fetch.

    Returns:
        dict with status, closes, highs, lows, volumes, metadata.
    """
    if not symbol.upper().endswith(".NS") and not symbol.startswith("^"):
        symbol = symbol.upper() + ".NS"

    return fetch_index_data(symbol=symbol, days=days)
