# Architecture -- Agentic Indian Stock Market Trading Assistant

## System Design

This application is a **multi-agent system** built on **Google ADK** (Agent Development Kit).
A root coordinator agent delegates tasks to five specialist sub-agents, including an adversarial
debate system (Bull vs Bear) for trade evaluation. All data comes from live NSE market feeds
via Yahoo Finance.

## Agent Hierarchy

```
trading_assistant (root coordinator)
  |
  +-- regime_analyst          -> analyzes Nifty 50 to classify BULL / SIDEWAYS / BEAR
  +-- stock_scanner           -> scans 20 NSE stocks for breakouts + announcement momentum
  +-- trade_debate_judge      -> coordinates Bull vs Bear debate, delivers verdict
  |     +-- bull_advocate     -> argues FOR the trade with data
  |     +-- bear_advocate     -> argues AGAINST the trade with data
  +-- trade_executor          -> calculates trade plans and executes paper trades
  +-- portfolio_manager       -> reports portfolio state, holdings, PnL
```

Total: 7 agents (1 root + 4 sub-agents + 2 debate sub-agents)

## Data Flow

```
User (browser) ---> FastAPI server ---> ADK root_agent
                                           |
                +-------+--------+---------+--------+--------+
                |       |        |                  |        |
          regime    scanner   debate_judge      executor  portfolio
                |       |        |        |         |        |
          fetch_idx  fetch_stk  bull     bear    paper_trade  load/save
          (yfinance) (yfinance) (data+news) (data+news) (memory/) (memory/)
                     fetch_news
                     (yfinance)
```

## Tool Functions

### market_data.py
- `fetch_index_data(symbol, days)` -- fetches index OHLCV from Yahoo Finance
- `fetch_stock_data(symbol, days)` -- fetches individual stock OHLCV

### news_data.py
- `fetch_stock_news(symbol)` -- fetches recent news articles via yfinance

### technical.py
- `compute_index_metrics(closes)` -- 50-DMA, slope, 20d return, volatility
- `compute_atr(highs, lows, closes, period)` -- Average True Range
- `detect_breakout(symbol, closes, volumes, highs, lows)` -- 20-day breakout detection

### debate_agent.py
- `analyze_stock_for_debate(symbol)` -- combined technicals + news for debate evaluation

### scanner_agent.py
- `scan_watchlist_breakouts(watchlist)` -- 20-day breakout scan across watchlist
- `get_stock_analysis(symbol)` -- single-stock breakout analysis
- `scan_announcement_momentum(watchlist)` -- news-driven momentum scan

### paper_trading.py
- `calculate_trade_plan(symbol, close, atr)` -- entry, stop, target, position size
- `execute_paper_trade(symbol, entry, stop, target, qty)` -- execute with risk rules

### portfolio.py
- `load_portfolio()` / `save_portfolio()` -- JSON persistence
- `get_portfolio_summary()` -- formatted portfolio overview
- `reset_portfolio()` -- reset to initial capital

## Scanning Strategies

| Strategy | Criteria | Tool |
|---|---|---|
| Breakout | close > 20d high + volume > 1.2x avg + above 50-DMA | `scan_watchlist_breakouts` |
| Announcement Momentum | recent news (3d) + price move > 2% (5d) + above-avg volume | `scan_announcement_momentum` |

## Debate Protocol

```
1. Judge delegates to Bull Advocate (argues FOR with data)
2. Judge delegates to Bear Advocate (argues AGAINST with data)
3. Judge delivers: VERDICT (BUY/SKIP) + CONFIDENCE (0-100%) + REASONING
```

## Risk Rules (hardcoded, not LLM-decided)

| Rule               | Value         |
|--------------------|---------------|
| Risk per trade     | 1% of cash    |
| Max open trades    | 3             |
| Min reward:risk    | 2:1           |
| Stop-loss          | 1.5 x ATR     |
| BEAR regime        | No new trades |

## Tech Stack

- **Agent framework:** Google ADK (Python)
- **LLM:** Gemini (auto-fallback across 4 models)
- **Data:** Yahoo Finance via yfinance (OHLCV + news)
- **Backend:** FastAPI + uvicorn
- **Frontend:** HTML/CSS/JS (single-page dashboard)
- **Memory:** JSON file (memory/portfolio.json)
- **Market:** NSE (Nifty 50 + top 20 liquid stocks)
- **Timezone:** IST (UTC+5:30)
