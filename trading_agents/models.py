"""Pydantic models for the trading assistant."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    BEAR = "BEAR"


class Strategy(str, Enum):
    TREND_BREAKOUT = "TREND_BREAKOUT"
    PULLBACK = "PULLBACK"
    NO_TRADE = "NO_TRADE"


class IndexMetrics(BaseModel):
    close: float
    dma_50: float
    dma_50_slope: float
    return_20d: float
    volatility: float


class StockData(BaseModel):
    symbol: str
    closes: List[float]
    highs: List[float]
    lows: List[float]
    volumes: List[float]
    last_timestamp: str
    source: str = "Yahoo Finance"


class BreakoutResult(BaseModel):
    symbol: str
    close: float
    prev_20d_high: float
    volume_ratio: float
    above_50dma: bool
    is_breakout: bool


class TradePlan(BaseModel):
    symbol: str
    entry: float
    stop: float
    target: float
    rr: float
    qty: int = 0
    risk_amount: float = 0.0


class Position(BaseModel):
    symbol: str
    qty: int
    entry: float
    stop: float
    target: float
    opened_at: str = ""


class PortfolioState(BaseModel):
    cash: float = 1_000_000.0
    open_positions: List[Position] = Field(default_factory=list)
    closed_trades: List[dict] = Field(default_factory=list)
    realized_pnl: float = 0.0
    actions_log: List[str] = Field(default_factory=list)
    equity_curve: List[dict] = Field(default_factory=list)


class AgentDecision(BaseModel):
    regime: MarketRegime
    strategy: Strategy
    reasoning: str
    allocation: Optional[dict] = None
