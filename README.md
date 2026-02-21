# Regime-Aware Agentic Equity Trading Assistant (Paper Trading Mode)

A **multi-agent AI system** for the Indian stock market (NSE), built entirely on **Google ADK** (Agent Development Kit). The system uses live market data to classify market regimes, scan for breakout stocks, execute paper trades with risk management, and manage a virtual portfolio -- all through natural language conversation.

> **Disclaimer:** This is a paper trading assistant for educational and hackathon demonstration purposes only. No real money is involved. This is not financial advice.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Agentic Architecture Overview](#agentic-architecture-overview)
3. [Agent Hierarchy and Communication](#agent-hierarchy-and-communication)
4. [Detailed Agent Documentation](#detailed-agent-documentation)
5. [Detailed Tool Documentation](#detailed-tool-documentation)
6. [How Calculations Work](#how-calculations-work)
7. [Decision-Making Flow](#decision-making-flow)
8. [Risk Rules and Protocols](#risk-rules-and-protocols)
9. [Data Sources and Watchlist](#data-sources-and-watchlist)
10. [Memory and Persistence](#memory-and-persistence)
11. [Web Dashboard](#web-dashboard)
12. [Model Fallback System](#model-fallback-system)
13. [Project Completion Status](#project-completion-status)
14. [Setup and Running](#setup-and-running)
15. [Project Structure](#project-structure)
16. [Example Prompts](#example-prompts)
17. [Tech Stack](#tech-stack)

---

## What This Project Does

The system performs seven core functions through an agentic AI architecture:

1. **Regime Detection** -- Fetches live Nifty 50 data and classifies the market as BULL, SIDEWAYS, or BEAR using technical indicators (50-DMA, trend slope, 20-day momentum).

2. **Breakout Scanning** -- Scans 20 of the most liquid Nifty 50 stocks for 20-day price breakouts with volume confirmation and trend alignment.

3. **Announcement Momentum Scanning** -- Fetches live news headlines for watchlist stocks and identifies announcement-driven momentum candidates (stocks with recent news + significant price move + above-average volume).

4. **Bull vs Bear Debate** -- Before any trade, a Bull Advocate argues FOR the trade and a Bear Advocate argues AGAINST, using real data. A Judge agent weighs both sides and delivers a BUY/SKIP verdict with a confidence percentage.

5. **Trade Planning** -- Calculates precise entry price, stop-loss (ATR-based), target price (2:1 reward-risk), position size (1% risk rule), and capital required.

6. **Paper Trade Execution** -- Simulates trade execution with full risk rule validation, updates the virtual portfolio, and persists state to disk.

7. **Portfolio Management** -- Tracks cash balance, open positions, invested capital, realized P&L, and trade history.

**What makes this "agentic":** The LLM (Gemini) acts as the reasoning layer that decides *which* tools to call and *how* to interpret results, while all calculations, data fetching, and risk enforcement are handled by deterministic Python tool functions. The LLM never computes indicators or overrides risk rules -- it orchestrates and explains. The adversarial debate pattern adds a layer where agents *collaborate through disagreement* to reach better trading decisions.

---

## Agentic Architecture Overview

This system follows the **Agent + Tools + Memory + Orchestration** pattern prescribed by Google ADK:

```
+------------------------------------------------------------------+
|                        USER (Natural Language)                     |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                     root_agent (trading_assistant)                 |
|                    Google ADK LLM Coordinator                     |
|                                                                    |
|  Reads user intent --> Selects appropriate sub-agent --> Returns   |
|  consolidated response with data, reasoning, and explanation      |
+------------------------------------------------------------------+
     |            |                  |              |             |
     v            v                  v              v             v
+---------+ +-----------+ +------------------+ +----------+ +-----------+
| regime  | |  stock    | | trade_debate     | |  trade   | | portfolio |
| analyst | |  scanner  | | judge            | | executor | | manager   |
+---------+ +-----------+ +------------------+ +----------+ +-----------+
     |         |    |         |           |         |             |
     v         v    v         v           v         v             v
 [analyze]  [scan] [scan]  [bull]     [bear]    [plan]      [view]
 [regime]   [brkout][annc]  [advocate]  [advocate] [execute]    [reset]
     |         |    |         |           |         |             |
     v         v    v         v           v         v             v
  yfinance  yfinance      yfinance +  yfinance + portfolio   portfolio
  (index)   (stocks)      news data   news data  .json        .json
```

### Key Design Principles

| Principle | Implementation |
|---|---|
| **LLM as orchestrator, not calculator** | The Gemini model decides *what to do*, but all math (DMA, ATR, position sizing) is done in deterministic Python functions |
| **Tools are pure functions** | Each tool takes explicit inputs, returns structured dicts, has no side effects (except portfolio persistence) |
| **Risk rules are hardcoded** | The LLM cannot override stop-loss, position size, or max trades. These are enforced in code |
| **Live data only** | All market data comes from Yahoo Finance in real-time. No synthetic or cached data |
| **Stateful memory** | Portfolio state persists to `memory/portfolio.json` across sessions |

---

## Agent Hierarchy and Communication

### How ADK Agent Communication Works

Google ADK uses a **delegation model** for multi-agent orchestration:

1. **User message** arrives at the `root_agent` (trading_assistant)
2. The root agent's LLM reads the user intent and decides which sub-agent to delegate to
3. ADK **transfers control** to the chosen sub-agent
4. The sub-agent's LLM decides which of its tools to call
5. Tool functions execute deterministically and return structured results
6. The sub-agent's LLM interprets the results and formulates a response
7. Control returns to the root agent, which may delegate to another sub-agent or respond to the user

### Agent Delegation Protocol

The root agent uses its `instruction` prompt to determine delegation:

| User Intent | Delegated To | Example |
|---|---|---|
| Market conditions, regime | `regime_analyst` | "What's the market doing?" |
| Stock scanning, finding opportunities | `stock_scanner` | "Scan for breakout stocks" |
| Announcement-driven momentum | `stock_scanner` | "Scan for news-driven movers" |
| Evaluate / debate a stock | `trade_debate_judge` | "Should I buy RELIANCE?" |
| Trade planning, execution | `trade_executor` | "Plan a trade for RELIANCE" |
| Portfolio queries, reset | `portfolio_manager` | "Show my portfolio" |
| Full workflow (scan-to-trade) | Sequential: regime -> scanner -> debate -> trade | "Run a full market scan and trade" |

### ADK Protocols Used

- **Agent-to-SubAgent Delegation:** Root agent uses ADK's `sub_agents` parameter. ADK handles context passing and response routing automatically.
- **Tool Registration:** Each sub-agent registers its tools via the `tools` parameter. ADK auto-generates tool schemas from Python function signatures and docstrings.
- **Session Management:** ADK maintains conversation history per session. The `InMemoryRunner` (for FastAPI) or ADK's built-in session service (for `adk web`) handles this.
- **Model Binding:** Each agent has `model=GEMINI_MODEL`, resolved once at startup via the fallback system.

---

## Detailed Agent Documentation

### 1. Root Agent -- `trading_assistant`

**File:** `trading_agents/agent.py`

| Property | Value |
|---|---|
| Name | `trading_assistant` |
| Role | Coordinator / Orchestrator |
| Model | Resolved via fallback (default: `gemini-2.5-flash`) |
| Tools | None (delegates everything) |
| Sub-agents | `regime_analyst`, `stock_scanner`, `trade_debate_judge`, `trade_executor`, `portfolio_manager` |

**What it does:**
- Receives all user messages
- Interprets intent using the LLM
- Delegates to the appropriate specialist sub-agent
- Can chain multiple sub-agents for complex workflows (e.g., "scan and trade" triggers regime check -> stock scan -> trade execution)
- Enforces response formatting rules (INR currency format, data source citation, paper trading disclaimer)

**What it does NOT do:**
- Never calls tools directly
- Never computes indicators or makes calculations
- Never overrides sub-agent decisions

---

### 2. Regime Analyst -- `regime_analyst`

**File:** `trading_agents/regime_agent.py`

| Property | Value |
|---|---|
| Name | `regime_analyst` |
| Role | Market regime classification |
| Tools | `analyze_regime` |
| Data Source | Live Nifty 50 (^NSEI) via Yahoo Finance |

**What it does:**
- Fetches live Nifty 50 daily price data (140 calendar days)
- Computes 5 technical metrics (close, 50-DMA, DMA slope, 20-day return, volatility)
- Classifies the market into one of three regimes: **BULL**, **SIDEWAYS**, or **BEAR**
- Recommends a trading strategy based on the regime
- Provides the last 5 closing prices as proof of data freshness

**Classification Logic (deterministic, not LLM-decided):**

```
BULL:     close > 50-DMA  AND  DMA slope > 0  AND  20d return >= 0%
BEAR:     close < 50-DMA  AND  DMA slope < 0  AND  20d return <= -3%
SIDEWAYS: everything else
```

**Strategy Mapping:**

| Regime | Strategy | Reasoning |
|---|---|---|
| BULL | `TREND_BREAKOUT` | Momentum is positive, breakout entries are viable |
| BEAR | `NO_TRADE` | Risk rules block all new trades in bear markets |
| SIDEWAYS | `NO_TRADE` | Mixed signals, no directional conviction |

---

### 3. Stock Scanner -- `stock_scanner`

**File:** `trading_agents/scanner_agent.py`

| Property | Value |
|---|---|
| Name | `stock_scanner` |
| Role | Breakout and announcement momentum scanning |
| Tools | `scan_watchlist_breakouts`, `get_stock_analysis`, `scan_announcement_momentum` |
| Data Source | Live stock data + news for 20 NSE stocks via Yahoo Finance |

**Two scanning strategies:**

**Strategy 1 -- Breakout Scan:**
- Iterates through the 20-stock NSE watchlist
- Fetches 140 days of OHLCV data for each stock
- Applies breakout detection criteria
- Ranks candidates by volume ratio (highest first)

**Breakout Criteria (all three must be true):**

```
1. Today's close > highest close of the previous 20 trading days
2. Today's volume > 1.2x the average volume of the previous 20 days
3. Today's close > 50-day moving average
```

**Strategy 2 -- Announcement Momentum Scan:**
- Iterates through the NSE watchlist
- For each stock: fetches OHLCV data + live news headlines
- Identifies stocks with all three conditions:
  - Recent news (published within last 3 days)
  - Significant price move (> 2% in 5 days)
  - Above-average volume (volume ratio > 1.0)
- Returns candidates with news headlines for the agent to interpret
- The LLM classifies whether news is material (earnings, buyback, M&A, contracts) vs. noise

When asked to scan broadly, the agent runs BOTH strategies and presents combined results.

---

### 4. Trade Debate Judge -- `trade_debate_judge` (Bull vs Bear)

**File:** `trading_agents/debate_agent.py`

| Property | Value |
|---|---|
| Name | `trade_debate_judge` |
| Role | Adversarial evaluation of trade candidates |
| Tools | `analyze_stock_for_debate` |
| Sub-agents | `bull_advocate`, `bear_advocate` |

This agent implements the **adversarial debate pattern** -- two opposing agents argue for and against a trade, then a judge delivers the verdict.

**Debate Protocol:**

```
Step 1: Judge delegates to bull_advocate
        Bull fetches stock data + news, argues FOR the trade
        (cites: breakout confirmation, volume, DMA alignment, positive catalysts)

Step 2: Judge delegates to bear_advocate
        Bear fetches stock data + news, argues AGAINST the trade
        (cites: overextension, weak volume, negative news, resistance levels)

Step 3: Judge reads both arguments and delivers:
        - VERDICT: BUY or SKIP
        - CONFIDENCE: 0-100%
        - BULL SUMMARY: strongest bull points
        - BEAR SUMMARY: strongest bear points
        - REASONING: 2-3 sentence synthesis
```

**Sub-agent: `bull_advocate`**
- Constructs the strongest possible case FOR buying
- Has access to `analyze_stock_for_debate` tool (technicals + news)
- Always argues bullish, citing specific data points

**Sub-agent: `bear_advocate`**
- Constructs the strongest possible case AGAINST buying
- Has access to `analyze_stock_for_debate` tool (technicals + news)
- Always argues bearish, citing risks and concerns

Both advocates independently fetch and analyze data, ensuring genuine adversarial perspectives rather than a single model arguing with itself.

---

### 5. Trade Executor -- `trade_executor`

**File:** `trading_agents/trade_agent.py`

| Property | Value |
|---|---|
| Name | `trade_executor` |
| Role | Trade planning and paper execution |
| Tools | `plan_trade`, `execute_trade` |
| Data Source | Portfolio state from `memory/portfolio.json` |

**What it does:**
- Calculates a complete trade plan: entry, stop-loss, target, quantity, capital required
- Presents the plan to the user for confirmation before executing
- Validates all risk rules before opening a position
- Updates the portfolio state and persists to disk
- Adjusts position size if insufficient cash (scales down quantity)

**Two-Step Protocol:**
1. `plan_trade` -- calculates and shows the plan (no side effects)
2. `execute_trade` -- validates rules and opens the position (modifies portfolio)

The agent is instructed to **never execute without showing the plan first**.

---

### 6. Portfolio Manager -- `portfolio_manager`

**File:** `trading_agents/portfolio_agent.py`

| Property | Value |
|---|---|
| Name | `portfolio_manager` |
| Role | Portfolio reporting and management |
| Tools | `view_portfolio`, `reset_paper_portfolio` |
| Data Source | `memory/portfolio.json` |

**What it does:**
- Loads portfolio state from disk
- Reports: cash balance, total invested, portfolio value, open positions, realized P&L, recent trade history
- Formats all amounts in INR with Indian number formatting
- Can reset the portfolio to its initial state (INR 10,00,000 cash, no positions)

---

## Detailed Tool Documentation

### Data Layer Tools (`tools/market_data.py`)

#### `fetch_index_data(symbol, days)`

| Parameter | Default | Description |
|---|---|---|
| `symbol` | `"^NSEI"` | Yahoo Finance ticker for the index |
| `days` | `140` | Calendar days of history to fetch |

**What it returns:**
- `closes`, `highs`, `lows`, `volumes` -- arrays of daily OHLCV data
- `latest_close` -- most recent closing price
- `last_5_closes` -- proof of data freshness
- `last_trade_date` -- timestamp of the most recent trading day
- `fetched_at_ist` -- when the API call was made (IST timezone)
- `source` -- always `"Yahoo Finance (yfinance)"`

**Error handling:** Returns `{"status": "error", "error_message": "..."}` if the market is closed, symbol is invalid, or fewer than 60 trading days are available.

#### `fetch_stock_data(symbol, days)`

Same as `fetch_index_data` but automatically appends `.NS` suffix for NSE stocks (e.g., `"RELIANCE"` becomes `"RELIANCE.NS"`).

---

### News Data Tools (`tools/news_data.py`)

#### `fetch_stock_news(symbol)`

Fetches recent news articles for a stock using yfinance's `.news` property.

| Parameter | Default | Description |
|---|---|---|
| `symbol` | (required) | NSE stock ticker (e.g. `"RELIANCE"` or `"RELIANCE.NS"`) |

**What it returns:**
- `articles` -- list of recent news articles, each with:
  - `title` -- headline text
  - `summary` -- article summary
  - `published` -- ISO timestamp of publication
  - `publisher` -- source name (e.g. "Reuters", "Economic Times")
  - `days_ago` -- computed age of the article in days
- `article_count` -- number of articles found
- `fetched_at_ist` -- IST timestamp of the API call
- Articles are sorted by recency (newest first)

---

### Debate Tools (`debate_agent.py`)

#### `analyze_stock_for_debate(symbol)`

Combines technical analysis and news data into a single comprehensive view for debate evaluation.

**What it returns:**
- `technicals` -- full output from `get_stock_analysis()` (close, 20d high, volume ratio, DMA, ATR, breakout status)
- `news` -- full output from `fetch_stock_news()` (headlines, summaries, dates, publishers)

Used by `bull_advocate`, `bear_advocate`, and `trade_debate_judge` to independently verify claims.

---

### Technical Analysis Tools (`tools/technical.py`)

#### `compute_index_metrics(closes)`

Requires at least 60 daily closing prices.

**Calculations:**

| Metric | Formula | Description |
|---|---|---|
| `close` | `closes[-1]` | Latest closing price |
| `dma_50` | `mean(closes[-50:])` | Simple 50-day moving average |
| `dma_50_slope` | `dma_50_today - dma_50_5days_ago` | Trend direction of the 50-DMA. Computed as: `mean(closes[-50:]) - mean(closes[-55:-5])` |
| `return_20d` | `(closes[-1] / closes[-21]) - 1` | 20-trading-day price return (percentage) |
| `volatility` | `pstdev(daily_returns[-20:]) * sqrt(252)` | Annualized volatility from 20-day daily returns |

#### `compute_atr(highs, lows, closes, period=14)`

Average True Range -- measures average daily price volatility.

**Formula:**

```
For each day i:
    True Range = max(
        high[i] - low[i],             # intraday range
        |high[i] - close[i-1]|,       # gap up
        |low[i] - close[i-1]|         # gap down
    )

ATR = mean(true_ranges[-14:])          # average over last 14 days
```

ATR is used to set the stop-loss distance. Higher ATR = wider stop = fewer shares (to maintain fixed risk).

#### `detect_breakout(symbol, closes, volumes, highs, lows)`

**Breakout detection algorithm:**

```
Step 1: Get today's close
Step 2: Calculate the highest close in the previous 20 days (excluding today)
         prev_20d_high = max(closes[-21:-1])
Step 3: Calculate volume ratio
         avg_20d_volume = mean(volumes[-21:-1])
         volume_ratio = today's volume / avg_20d_volume
Step 4: Calculate 50-DMA
         dma_50 = mean(closes[-50:])
Step 5: Check all three conditions:
         is_breakout = (close > prev_20d_high)
                   AND (volume_ratio > 1.2)
                   AND (close > dma_50)
Step 6: Compute ATR for trade planning
```

**Returns:** close, prev_20d_high, volume_ratio, above_50dma, dma_50, atr, is_breakout

---

### Paper Trading Tools (`tools/paper_trading.py`)

#### `calculate_trade_plan(symbol, close, atr)`

**Position sizing algorithm:**

```
1. entry = close (buy at current price)
2. stop = entry - (1.5 x ATR)                     # stop-loss distance
3. risk_per_share = entry - stop                    # INR risked per share
4. target = entry + (2.0 x risk_per_share)          # 2:1 reward-risk
5. risk_amount = portfolio_cash x 0.01              # 1% of available cash
6. qty = floor(risk_amount / risk_per_share)        # number of shares
7. capital_required = qty x entry                   # total capital needed
```

**Example:**

```
Stock: TCS at INR 4,200, ATR = 85
stop = 4200 - (1.5 x 85) = INR 4,072.50
risk_per_share = 4200 - 4072.50 = INR 127.50
target = 4200 + (2 x 127.50) = INR 4,455.00
risk_amount = 10,00,000 x 0.01 = INR 10,000
qty = floor(10000 / 127.50) = 78 shares
capital_required = 78 x 4200 = INR 3,27,600
```

#### `execute_paper_trade(symbol, entry, stop, target, qty)`

**Validation checks (in order):**

1. Open positions < 3 (max trades limit)
2. Stop-loss is below entry price (risk_per_share > 0)
3. Reward-risk ratio >= 2.0
4. Sufficient cash available (auto-scales qty down if needed)

If all checks pass: deducts cash, creates a `Position` object, appends to `actions_log`, saves to disk.

**Possible outcomes:** `OPENED` (success), `SKIPPED` (rule violation with reason).

---

### Portfolio Tools (`tools/portfolio.py`)

#### `load_portfolio()` / `save_portfolio(state)`

- **Storage:** `memory/portfolio.json`
- **Format:** JSON with fields: `cash`, `open_positions`, `closed_trades`, `realized_pnl`, `actions_log`
- **Actions log:** Capped at 50 most recent entries to prevent unbounded growth

#### `get_portfolio_summary()`

Returns a structured dict with:
- Cash balance, total invested amount, total portfolio value
- List of open positions (symbol, qty, entry, stop, target, invested amount, open date)
- Realized P&L from closed trades
- Last 10 actions from the trade log

#### `reset_portfolio()`

Resets to: INR 10,00,000 cash, no positions, no history.

---

## How Calculations Work

### Regime Detection -- Step by Step

```
1. fetch_index_data("^NSEI") pulls 140 days of Nifty 50 OHLCV from Yahoo Finance
2. compute_index_metrics() calculates:
   - 50-DMA (simple moving average of last 50 closes)
   - DMA slope (difference between current 50-DMA and 50-DMA from 5 days ago)
   - 20-day return (percentage change over last 20 trading days)
   - Annualized volatility (standard deviation of daily returns * sqrt(252))
3. Regime classification:
   - If close > 50-DMA AND slope > 0 AND 20d_return >= 0% --> BULL
   - If close < 50-DMA AND slope < 0 AND 20d_return <= -3% --> BEAR
   - Otherwise --> SIDEWAYS
4. LLM formats the result with reasoning, but does NOT change the classification
```

### Breakout Scanning -- Step by Step

```
1. For each of the 20 NSE watchlist stocks:
   a. fetch_stock_data() pulls 140 days of OHLCV
   b. detect_breakout() checks:
      - Is today's close above the highest close of the previous 20 days?
      - Is today's volume at least 1.2x the 20-day average volume?
      - Is the stock trading above its 50-DMA?
   c. If all three are true --> it's a breakout candidate
2. Candidates are ranked by volume_ratio (highest first)
3. ATR is computed for each candidate (needed for trade planning)
4. LLM presents the results but does NOT filter or re-rank
```

### Announcement Momentum -- Step by Step

```
1. For each of the 20 NSE watchlist stocks:
   a. fetch_stock_data() pulls 140 days of OHLCV
   b. Calculate 5-day price change: (close / close_5_days_ago) - 1
   c. Calculate volume ratio: today's volume / 20-day average volume
   d. Check momentum criteria:
      - |5-day price change| > 2% (significant move in either direction)
      - volume ratio > 1.0 (above-average activity)
   e. If momentum criteria met, fetch news via fetch_stock_news()
   f. Filter for recent news (published within last 3 days)
   g. If recent news exists --> it's an announcement momentum candidate
2. Candidates are ranked by absolute price change (largest move first)
3. Direction labeled BULLISH (positive move) or BEARISH (negative move)
4. LLM interprets news headlines to determine if announcement is material
```

### Bull vs Bear Debate -- Step by Step

```
1. trade_debate_judge receives "evaluate RELIANCE"
2. Delegates to bull_advocate:
   - Bull calls analyze_stock_for_debate("RELIANCE")
   - Gets technicals (breakout status, DMA, volume, ATR) + news
   - Constructs data-backed argument FOR buying
   - Returns bullish case with specific numbers
3. Delegates to bear_advocate:
   - Bear calls analyze_stock_for_debate("RELIANCE")
   - Gets the same data independently
   - Constructs data-backed argument AGAINST buying
   - Returns bearish case with specific risks
4. Judge reads both arguments, optionally verifies claims
5. Delivers verdict: BUY/SKIP + confidence % + synthesis
```

### Trade Execution -- Step by Step

```
1. plan_trade(symbol, close, atr):
   - Calculates stop = entry - 1.5*ATR
   - Calculates target = entry + 2*(entry - stop)
   - Loads portfolio to check available cash
   - Sizes position: qty = floor(1% of cash / risk_per_share)
   - Returns the plan (no side effects)

2. LLM presents plan to user, waits for confirmation

3. execute_trade(symbol, entry, stop, target, qty):
   - Loads portfolio from disk
   - Validates: max trades not exceeded
   - Validates: stop < entry
   - Validates: reward:risk >= 2.0
   - Validates: sufficient cash (auto-adjusts qty if needed)
   - Deducts cash from portfolio
   - Creates Position record
   - Logs the action
   - Saves portfolio to disk
   - Returns execution confirmation
```

---

## Decision-Making Flow

### Full Scan-to-Trade Workflow

```
User: "Run a full market scan and trade the best opportunity"

  Step 1: Root agent delegates to regime_analyst
          |
          v
  regime_analyst calls analyze_regime()
          |
          +-- If BEAR or SIDEWAYS: Returns "NO_TRADE" recommendation
          |   Root agent informs user, workflow stops
          |
          +-- If BULL: Returns "TREND_BREAKOUT" recommendation
              Root agent proceeds to step 2
          |
          v
  Step 2: Root agent delegates to stock_scanner
          |
          v
  stock_scanner runs breakout scan AND/OR announcement momentum scan
          |
          +-- If 0 candidates found: Returns empty list
          |   Root agent informs user, workflow stops
          |
          +-- If candidates found: Returns ranked list
              Root agent picks the top candidate, proceeds to step 3
          |
          v
  Step 3: Root agent delegates to trade_debate_judge
          |
          v
  trade_debate_judge runs Bull vs Bear debate:
          +-> bull_advocate: argues FOR with data
          +-> bear_advocate: argues AGAINST with data
          +-> Judge verdict: BUY/SKIP + confidence %
          |
          +-- If SKIP: Root agent informs user, workflow stops
          +-- If BUY: Root agent proceeds to step 4
          |
          v
  Step 4: Root agent delegates to trade_executor
          |
          v
  trade_executor calls plan_trade(symbol, close, atr)
          +-- Shows the trade plan to user
          v
  trade_executor calls execute_trade(symbol, entry, stop, target, qty)
          +-- If risk rules pass: OPENED, portfolio updated
          +-- If rules fail: SKIPPED with reason
          |
          v
  Step 5: Root agent consolidates and presents final summary
```

---

## Risk Rules and Protocols

All risk rules are **hardcoded in Python** -- the LLM cannot override them.

| Rule | Value | Where Enforced |
|---|---|---|
| Risk per trade | 1% of available cash | `paper_trading.py` -- `calculate_trade_plan()` |
| Max open positions | 3 | `paper_trading.py` -- `execute_paper_trade()` |
| Minimum reward:risk | 2:1 | `paper_trading.py` -- `execute_paper_trade()` |
| Stop-loss distance | 1.5 x ATR | `paper_trading.py` -- `calculate_trade_plan()` |
| No trading in BEAR regime | Strategy = NO_TRADE | `regime_agent.py` -- `analyze_regime()` (advisory, enforced by LLM instruction) |
| Volume confirmation | > 1.2x 20-day average | `technical.py` -- `detect_breakout()` |
| Trend alignment | Close must be above 50-DMA | `technical.py` -- `detect_breakout()` |
| Minimum data | 60 trading days required | `market_data.py` and `technical.py` |
| Auto-scale on low cash | Reduces qty to fit available cash | `paper_trading.py` -- `execute_paper_trade()` |
| Action log cap | Last 50 entries only | `portfolio.py` -- `save_portfolio()` |

---

## Data Sources and Watchlist

### Data Provider

**Yahoo Finance** via the `yfinance` Python library. Data is fetched live on every request (no caching).

- **Index:** Nifty 50 (`^NSEI`), Bank Nifty (`^NSEBANK`)
- **Interval:** Daily (1d)
- **Lookback:** 140 calendar days (~90 trading days, enough for 50-DMA calculation)

### NSE Watchlist (20 Stocks)

All stocks are from the **Nifty 50** -- the most liquid large-cap stocks on the National Stock Exchange of India:

| # | Symbol | Company |
|---|---|---|
| 1 | RELIANCE.NS | Reliance Industries |
| 2 | TCS.NS | Tata Consultancy Services |
| 3 | HDFCBANK.NS | HDFC Bank |
| 4 | INFY.NS | Infosys |
| 5 | ICICIBANK.NS | ICICI Bank |
| 6 | HINDUNILVR.NS | Hindustan Unilever |
| 7 | ITC.NS | ITC Limited |
| 8 | SBIN.NS | State Bank of India |
| 9 | BHARTIARTL.NS | Bharti Airtel |
| 10 | KOTAKBANK.NS | Kotak Mahindra Bank |
| 11 | LT.NS | Larsen & Toubro |
| 12 | AXISBANK.NS | Axis Bank |
| 13 | ASIANPAINT.NS | Asian Paints |
| 14 | MARUTI.NS | Maruti Suzuki |
| 15 | TITAN.NS | Titan Company |
| 16 | SUNPHARMA.NS | Sun Pharma |
| 17 | BAJFINANCE.NS | Bajaj Finance |
| 18 | WIPRO.NS | Wipro |
| 19 | HCLTECH.NS | HCL Technologies |
| 20 | TATAMOTORS.NS | Tata Motors |

---

## Memory and Persistence

The system uses **file-based memory** via JSON:

- **File:** `memory/portfolio.json`
- **Created automatically** on first trade or portfolio query
- **Schema:**

```json
{
  "cash": 1000000.0,
  "open_positions": [
    {
      "symbol": "RELIANCE.NS",
      "qty": 78,
      "entry": 2800.0,
      "stop": 2672.5,
      "target": 3055.0,
      "opened_at": "2026-02-21 16:00 IST"
    }
  ],
  "closed_trades": [],
  "realized_pnl": 0.0,
  "actions_log": [
    "[2026-02-21 16:00 IST] OPEN RELIANCE.NS qty=78 entry=2800.00 stop=2672.50 target=3055.00"
  ]
}
```

Portfolio state survives server restarts. The actions log is capped at 50 entries.

---

## Web Dashboard

The project includes a **FastAPI-powered web dashboard** with a dark-themed UI.

**File:** `server/app.py` (backend), `server/static/` (frontend)

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the HTML dashboard |
| `/api/chat` | POST | Sends a message to the root agent, returns AI response |
| `/api/regime` | GET | Calls `analyze_regime()` directly, returns regime data |
| `/api/portfolio` | GET | Returns current portfolio summary |
| `/api/portfolio/reset` | POST | Resets portfolio to initial state |

### Dashboard Layout

```
+-------------------------------------------+-------------------+
|                                           |  Market Regime    |
|           Chat Interface                  |  [BULL/BEAR/...]  |
|                                           |  Close: 22,400    |
|  User: "Scan for breakouts"              |  50-DMA: 22,100   |
|  Agent: "Found 3 breakout candidates..." |  20d Ret: +2.1%   |
|                                           +-------------------+
|                                           |  Portfolio        |
|                                           |  Cash: 10,00,000  |
|  [Type a message...] [Send]              |  Invested: 0      |
|                                           |  P&L: 0           |
+-------------------------------------------+-------------------+
```

### How the Dashboard Communicates with the Agent

1. User types a message in the chat input
2. Frontend sends `POST /api/chat` with `{"message": "..."}` 
3. Backend creates an ADK `InMemoryRunner` session
4. Message is sent to the `root_agent` via `runner.run_async()`
5. Agent processes the request (may delegate to sub-agents, call tools)
6. All response text parts are collected and joined
7. Response is returned as `{"reply": "..."}` to the frontend
8. Regime and portfolio cards are refreshed independently via their own API endpoints

---

## Model Fallback System

To handle Gemini API rate limits and outages (common during hackathons), the system automatically probes models at startup:

**File:** `trading_agents/config.py`

**Fallback order:**

| Priority | Model | Characteristics |
|---|---|---|
| 1 | `gemini-3-flash-preview` | Latest, fastest |
| 2 | `gemini-2.5-flash` | Stable, reliable |
| 3 | `gemini-2.0-flash` | Older, very available |
| 4 | `gemini-2.5-pro` | More powerful, usually low traffic |

At startup, `_pick_available_model()` sends a minimal `"ping"` request to each model in order. The first model that responds successfully is used for all agents. Startup logs show which model was selected:

```
[config] gemini-3-flash-preview unavailable (503), trying next...
[config] Using model: gemini-2.5-flash
```

---

## Project Completion Status

### Version Roadmap

This project was designed with a three-phase roadmap:

| Version | Focus | Status |
|---|---|---|
| **V1 -- Hackathon MVP** | Regime + single strategy + paper trading | **COMPLETED** |
| V2 -- Enhanced Intelligence | Multi-strategy, adaptive allocation, performance tracking | Not started |
| V3 -- Production System | Full portfolio engine, broker integration, live execution | Not started |

### V1 Completion Checklist

| Component | Status | Details |
|---|---|---|
| Multi-agent architecture (ADK) | Done | Root + 5 sub-agents + 2 debate sub-agents (7 total) |
| Live market data (yfinance) | Done | NSE index + 20 stock watchlist |
| Live news data (yfinance) | Done | Stock news headlines for announcement detection |
| Regime classification (BULL/SIDEWAYS/BEAR) | Done | Deterministic rules on 50-DMA, slope, momentum |
| Breakout scanner (20-day high + volume) | Done | Scans all 20 watchlist stocks |
| Announcement momentum scanner | Done | News + price move + volume confirmation |
| Bull vs Bear debate system | Done | Adversarial evaluation with BUY/SKIP verdict |
| Trade planning (entry/stop/target/qty) | Done | ATR-based stops, 1% risk sizing |
| Paper trade execution | Done | Full risk validation, portfolio updates |
| Portfolio persistence (JSON) | Done | Cash, positions, P&L, trade history |
| Risk rules (hardcoded) | Done | 1% risk, max 3 trades, 2:1 R:R, no bear trading |
| Web dashboard (FastAPI) | Done | Chat + regime card + portfolio card |
| Model fallback system | Done | Auto-probes 4 Gemini models at startup |
| LLM reasoning and explanation | Done | Agent explains every decision with data |
| Data source transparency | Done | Every response includes source + IST timestamp |

### Is This an Agentic AI System?

**Yes.** This system satisfies the core requirements of an agentic AI architecture:

| Agentic Property | How It's Implemented |
|---|---|
| **Autonomous reasoning** | The LLM decides which sub-agent to delegate to and how to interpret tool results |
| **Tool use** | 11 deterministic tool functions registered with ADK and called by agents |
| **Multi-agent orchestration** | Root agent coordinates 5 sub-agents (7 total agents including debate sub-agents) |
| **Adversarial reasoning** | Bull and Bear agents debate trade decisions from opposing perspectives before a Judge delivers a verdict |
| **Memory / State** | Portfolio state persists to disk across sessions |
| **Human-in-the-loop** | Trade plans are shown before execution; user confirms via chat |
| **Explainability** | Every decision includes reasoning, metrics, data source, and timestamp |
| **Safety guardrails** | Risk rules are hardcoded in Python, not LLM-decided |

### What V2 Would Add

- Multiple strategies (pullback, dividend momentum) alongside trend breakout
- Adaptive capital allocation (agent decides weight per strategy)
- Performance-aware trading (win rate tracking, drawdown monitoring)
- Scheduled/continuous scanning instead of on-demand only
- Trade closing (currently only opens, no close/exit functionality)
- Live P&L tracking using current market prices

### What V3 Would Add

- Real broker integration (Zerodha/Angel One API)
- Portfolio-level risk engine (correlation, diversification scoring)
- Volatility targeting and scenario stress testing
- Kill-switch and loss limits
- User risk profiling
- Audit log and compliance layer

---

## Setup and Running

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Set up API key

```bash
copy trading_agents\.env.example trading_agents\.env
```

Edit `trading_agents/.env` and add your Gemini API key:

```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Run with ADK (terminal mode)

```bash
adk run trading_agents
```

### 4. Run with ADK (web UI)

```bash
adk web
```

Open http://localhost:8000 and select **trading_agents** in the dropdown.
(Other folders like `data`, `docs` may appear in the list -- ignore them.)

### 5. Run the custom dashboard

```bash
python -m uvicorn server.app:app --reload --port 8080
```

Open http://localhost:8080

---

## Project Structure

```
trading_agents/                  # ADK agent package
  __init__.py                    #   package init (imports agent module)
  agent.py                       #   root_agent -- coordinator, delegates to sub-agents
  regime_agent.py                #   regime_analyst -- classifies BULL/SIDEWAYS/BEAR
  scanner_agent.py               #   stock_scanner -- breakout + announcement momentum
  debate_agent.py                #   trade_debate_judge + bull/bear advocates
  trade_agent.py                 #   trade_executor -- plans and executes paper trades
  portfolio_agent.py             #   portfolio_manager -- reports and resets portfolio
  config.py                      #   model fallback, risk rules, thresholds, watchlist
  models.py                      #   Pydantic data models for type safety
  .env / .env.example            #   Gemini API key configuration
  tools/
    market_data.py               #     live NSE data fetching (yfinance)
    news_data.py                 #     live stock news fetching (yfinance)
    technical.py                 #     indicator calculations (DMA, ATR, breakout)
    paper_trading.py             #     trade planning and execution logic
    portfolio.py                 #     portfolio persistence (JSON read/write)
server/
  app.py                         #   FastAPI backend (chat API + dashboard serving)
  static/
    index.html                   #     dashboard HTML layout
    style.css                    #     dark theme styling
    app.js                       #     frontend JavaScript (chat, regime, portfolio)
memory/
  portfolio.json                 #   persistent paper portfolio state
docs/
  ARCHITECTURE.md                #   system architecture documentation
requirements.txt                 #   Python dependencies
.gitignore                       #   excludes .venv, .env, __pycache__, .adk, memory/
```

---

## Example Prompts

| Prompt | What Happens |
|---|---|
| "What is the current market regime?" | Fetches live Nifty 50 data, computes metrics, classifies regime |
| "Scan for breakout stocks" | Scans all 20 watchlist stocks for breakout candidates |
| "Scan for news-driven movers" | Scans watchlist for announcement momentum candidates |
| "Analyze RELIANCE" | Single-stock breakout analysis with ATR |
| "Should I buy RELIANCE?" | Runs Bull vs Bear debate, delivers BUY/SKIP verdict with confidence |
| "Debate TCS" | Full adversarial evaluation with bull case, bear case, and judge verdict |
| "Plan a trade for TCS at 4200 with ATR 85" | Calculates entry, stop, target, position size |
| "Execute the trade" | Validates risk rules and opens the paper position |
| "Show my portfolio" | Displays cash, positions, P&L, recent actions |
| "Reset portfolio" | Resets to INR 10,00,000 with no positions |
| "Run a full market scan and trade" | Regime -> scan -> debate -> trade planning -> execution |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | Google ADK (Agent Development Kit) |
| LLM | Google Gemini (2.0/2.5/3.0 Flash/Pro with auto-fallback) |
| Market Data | Yahoo Finance via `yfinance` |
| Data Models | Pydantic v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (dark theme, single-page) |
| Memory | JSON file persistence |
| Target Market | NSE (National Stock Exchange of India) |
| Language | Python 3.10+ |
