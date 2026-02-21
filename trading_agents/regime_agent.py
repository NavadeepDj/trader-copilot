"""Regime classification sub-agent -- determines market regime from live Nifty data."""

from __future__ import annotations

from typing import Dict

from google.adk.agents import Agent

from trading_agents.config import (
    BEAR_RETURN_20D_MAX,
    BULL_RETURN_20D_MIN,
    DEFAULT_INDEX,
    GEMINI_MODEL,
)
from trading_agents.tools.market_data import fetch_index_data
from trading_agents.tools.technical import compute_index_metrics


def analyze_regime(index_symbol: str = DEFAULT_INDEX) -> Dict:
    """Fetch live index data, compute metrics, and classify market regime.

    Args:
        index_symbol: Yahoo Finance index symbol. Default is Nifty 50 (^NSEI).

    Returns:
        dict with regime classification, metrics, and supporting evidence.
    """
    data = fetch_index_data(symbol=index_symbol)
    if data.get("status") != "success":
        return data

    metrics = compute_index_metrics(data["closes"])
    if metrics.get("status") != "success":
        return metrics

    close = metrics["close"]
    dma_50 = metrics["dma_50"]
    slope = metrics["dma_50_slope"]
    ret_20d = metrics["return_20d"]

    is_bull = close > dma_50 and slope > 0 and ret_20d >= BULL_RETURN_20D_MIN
    is_bear = close < dma_50 and slope < 0 and ret_20d <= BEAR_RETURN_20D_MAX

    if is_bull:
        regime = "BULL"
        strategy = "TREND_BREAKOUT"
        reasoning = (
            f"Nifty is trading at {close} ABOVE its 50-DMA ({dma_50}), "
            f"trend slope is positive ({slope:+.2f}), "
            f"and 20-day return is {ret_20d:+.2%}. "
            "Momentum supports breakout-style entries."
        )
    elif is_bear:
        regime = "BEAR"
        strategy = "NO_TRADE"
        reasoning = (
            f"Nifty is trading at {close} BELOW its 50-DMA ({dma_50}), "
            f"trend slope is negative ({slope:+.2f}), "
            f"and 20-day return is {ret_20d:+.2%}. "
            "Risk rules block new trades in bear regimes."
        )
    else:
        regime = "SIDEWAYS"
        strategy = "NO_TRADE"
        reasoning = (
            f"Nifty at {close} vs 50-DMA {dma_50}, "
            f"slope {slope:+.2f}, 20d return {ret_20d:+.2%}. "
            "Signals are mixed -- no strong directional conviction."
        )

    return {
        "status": "success",
        "regime": regime,
        "strategy": strategy,
        "reasoning": reasoning,
        "index_symbol": data["symbol"],
        "source": data["source"],
        "fetched_at_ist": data["fetched_at_ist"],
        "last_trade_date": data["last_trade_date"],
        "last_5_closes": data["last_5_closes"],
        "metrics": metrics,
    }


regime_agent = Agent(
    name="regime_analyst",
    model=GEMINI_MODEL,
    description=(
        "Analyzes the current Indian stock market regime using live Nifty 50 data. "
        "Classifies the market as BULL, SIDEWAYS, or BEAR and recommends a trading strategy."
    ),
    instruction=(
        "You are the Regime Analyst. When asked about market conditions, "
        "use the analyze_regime tool to fetch live Nifty 50 data and classify the regime. "
        "Always include: regime, strategy recommendation, supporting metrics, "
        "data source, fetch timestamp, and the last 5 closing prices as proof of freshness. "
        "Be concise and data-driven."
    ),
    tools=[analyze_regime],
)
