"""Bull vs Bear debate agent system -- adversarial evaluation before trading."""

from __future__ import annotations

from typing import Dict

from google.adk.agents import Agent

from trading_agents.config import GEMINI_MODEL
from trading_agents.scanner_agent import get_stock_analysis
from trading_agents.tools.news_data import fetch_stock_news


def analyze_stock_for_debate(symbol: str) -> Dict:
    """Fetch comprehensive stock data (technicals + news) for debate evaluation.

    Args:
        symbol: NSE stock ticker (e.g. 'RELIANCE' or 'RELIANCE.NS').

    Returns:
        dict with technical analysis, news headlines, and key metrics.
    """
    technicals = get_stock_analysis(symbol=symbol)
    news = fetch_stock_news(symbol=symbol)

    return {
        "status": "success",
        "symbol": technicals.get("symbol", symbol),
        "technicals": technicals,
        "news": news,
    }


bull_agent = Agent(
    name="bull_advocate",
    model=GEMINI_MODEL,
    description=(
        "Argues the bullish case FOR buying a stock. "
        "Presents data-backed reasons why the trade should be taken."
    ),
    instruction=(
        "You are the Bull Advocate in a trade debate. Your job is to construct "
        "the STRONGEST possible case FOR buying the stock under evaluation.\n\n"
        "USE the analyze_stock_for_debate tool to get the latest data, then argue:\n"
        "- Price action: Is it above 50-DMA? Breaking out? Momentum?\n"
        "- Volume: Is volume confirming the move? Above average?\n"
        "- News catalysts: Any positive announcements, earnings beats, contract wins?\n"
        "- Trend: Is the broader market (regime) supportive?\n"
        "- Risk/reward: Is the setup attractive at current levels?\n\n"
        "Be persuasive but ALWAYS cite specific numbers from the data. "
        "Structure your argument with clear bullet points. "
        "End with a one-line conviction statement."
    ),
    tools=[analyze_stock_for_debate],
)


bear_agent = Agent(
    name="bear_advocate",
    model=GEMINI_MODEL,
    description=(
        "Argues the bearish case AGAINST buying a stock. "
        "Presents data-backed risks and reasons to avoid the trade."
    ),
    instruction=(
        "You are the Bear Advocate in a trade debate. Your job is to construct "
        "the STRONGEST possible case AGAINST buying the stock under evaluation.\n\n"
        "USE the analyze_stock_for_debate tool to get the latest data, then argue:\n"
        "- Overextension: How far is price from 50-DMA? Is it stretched?\n"
        "- Volume quality: Is volume declining? Is the breakout on weak volume?\n"
        "- News risks: Any negative headlines? Sector headwinds? Regulatory concerns?\n"
        "- Resistance: Is the stock near historical resistance levels?\n"
        "- Valuation: Any signs of overvaluation or unsustainable moves?\n"
        "- Macro risk: Is the broader market weak or uncertain?\n\n"
        "Be critical and skeptical but ALWAYS cite specific numbers from the data. "
        "Structure your argument with clear bullet points. "
        "End with a one-line risk warning."
    ),
    tools=[analyze_stock_for_debate],
)


debate_agent = Agent(
    name="trade_debate_judge",
    model=GEMINI_MODEL,
    description=(
        "Coordinates a Bull vs Bear debate on a stock and delivers a final verdict. "
        "First hears the bullish case, then the bearish case, then judges."
    ),
    instruction=(
        "You are the Trade Debate Judge. When asked to evaluate whether a stock "
        "should be traded, follow this EXACT protocol:\n\n"
        "STEP 1: Delegate to bull_advocate and ask it to present the case FOR "
        "buying the stock.\n\n"
        "STEP 2: Delegate to bear_advocate and ask it to present the case AGAINST "
        "buying the stock.\n\n"
        "STEP 3: After hearing BOTH sides, deliver your VERDICT with this structure:\n"
        "- VERDICT: BUY or SKIP\n"
        "- CONFIDENCE: 0-100%\n"
        "- BULL SUMMARY: strongest 1-2 points from the bull case\n"
        "- BEAR SUMMARY: strongest 1-2 points from the bear case\n"
        "- JUDGE REASONING: 2-3 sentences explaining your decision, weighing "
        "both sides objectively\n\n"
        "You may also use analyze_stock_for_debate yourself to independently "
        "verify any claims made by either side.\n\n"
        "IMPORTANT: You MUST hear from BOTH advocates before delivering your verdict. "
        "Never skip the bear case just because the bull case is strong."
    ),
    sub_agents=[bull_agent, bear_agent],
    tools=[analyze_stock_for_debate],
)
