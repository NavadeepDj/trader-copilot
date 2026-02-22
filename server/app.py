"""FastAPI application serving the trading dashboard and ADK agent API."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment from trading_agents/.env
_env_path = Path(__file__).resolve().parent.parent / "trading_agents" / ".env"
load_dotenv(_env_path)

# Ensure project root is on sys.path so trading_agents is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from trading_agents.agent import root_agent
from trading_agents.scanner_agent import get_nifty50_signal_board
from trading_agents.tools.portfolio import (
    get_portfolio_performance,
    get_portfolio_summary,
    refresh_portfolio_positions,
    reset_portfolio,
)
from trading_agents.regime_agent import analyze_regime

app = FastAPI(title="Agentic Trading Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ADK runner and session
_runner = InMemoryRunner(agent=root_agent, app_name="trading_assistant")
_USER_ID = "dashboard_user"
_session_id: str | None = None


async def _get_session_id() -> str:
    global _session_id
    if _session_id is None:
        session = await _runner.session_service.create_session(
            app_name="trading_assistant", user_id=_USER_ID
        )
        _session_id = session.id
    return _session_id


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = await _get_session_id()
    user_content = types.Content(
        role="user", parts=[types.Part.from_text(text=req.message)]
    )

    reply_parts: list[str] = []
    async for event in _runner.run_async(
        user_id=_USER_ID, session_id=session_id, new_message=user_content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and part.text.strip():
                    reply_parts.append(part.text.strip())

    reply = "\n\n".join(reply_parts) if reply_parts else "No response from agent."
    return ChatResponse(reply=reply)


@app.get("/api/regime")
async def regime():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyze_regime)
    return result


@app.get("/api/portfolio")
async def portfolio():
    return get_portfolio_summary()


@app.get("/api/portfolio/performance")
async def portfolio_performance():
    return get_portfolio_performance()


@app.post("/api/portfolio/refresh")
async def portfolio_refresh():
    return refresh_portfolio_positions()


@app.post("/api/portfolio/reset")
async def portfolio_reset():
    return reset_portfolio()


@app.get("/api/signals/nifty50")
async def nifty50_signals(
    limit: int = 50,
    include_news: bool = True,
    max_news: int = 2,
    news_days: int = 1,
):
    loop = asyncio.get_event_loop()
    fn = partial(
        get_nifty50_signal_board,
        limit,
        include_news,
        max_news,
        news_days,
    )
    return await loop.run_in_executor(None, fn)
