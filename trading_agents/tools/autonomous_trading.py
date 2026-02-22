"""Autonomous trading orchestrator - full flow from analysis to execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

from trading_agents.regime_agent import analyze_regime
from trading_agents.tools.portfolio import get_portfolio_summary, get_portfolio_performance
from trading_agents.tools.demo_tools import show_dividend_strategy_demo, show_rsi_strategy_demo
from trading_agents.tools.backtest_oversold import backtest_oversold_bounce, get_best_oversold_nifty50
from trading_agents.tools.backtest_dividend import backtest_current_moneycontrol_dividends_filtered
from trading_agents.scanner_agent import scan_oversold_bounce, scan_watchlist_breakouts
from trading_agents.dividend_agent import scan_dividend_opportunities

IST = timezone(timedelta(hours=5, minutes=30))


def analyze_and_recommend_strategy() -> Dict:
    """Analyze market, read portfolio, and recommend the best strategy for current conditions.
    
    This is STEP 1 of the autonomous trading flow. It:
    1. Analyzes current market regime (BULL/SIDEWAYS/BEAR)
    2. Reads portfolio state (cash, positions, available capital)
    3. Recommends the appropriate strategy based on regime
    4. Shows backtest proof of the recommended strategy
    
    Use when: User says "analyze market and invest", "maximize gains", "help me trade",
    or wants to understand what strategy to use and why.
    
    Returns:
        dict with regime, portfolio_state, recommended_strategy, backtest_proof, and next_steps.
    """
    # Step 1: Analyze market regime
    regime_data = analyze_regime()
    regime = regime_data.get("regime", "UNKNOWN") if regime_data.get("status") == "success" else "UNKNOWN"
    
    # Step 2: Get portfolio state
    portfolio = get_portfolio_summary()
    cash = portfolio.get("cash_inr", 0)
    open_positions = portfolio.get("open_positions", [])
    num_positions = len(open_positions)
    max_positions = 3  # From config
    can_open_new = num_positions < max_positions
    
    # Step 3: Determine strategy based on regime
    if regime == "BULL":
        strategy = "DIVIDEND_MOMENTUM"
        strategy_desc = "Buy dividend stocks in uptrend after announcement, sell before ex-date"
        backtest_func = "dividend"
    else:  # SIDEWAYS or BEAR
        strategy = "RSI_OVERSOLD_BOUNCE"
        strategy_desc = "Buy oversold stocks (RSI <= 35) below 50-DMA, exit on RSI recovery or stop"
        backtest_func = "rsi"
    
    # Step 4: Get backtest proof
    if backtest_func == "dividend":
        demo = show_dividend_strategy_demo()
        backtest_proof = {
            "strategy": "DIVIDEND_MOMENTUM",
            "example_win": demo.get("successful_trade", {}),
            "example_loss": demo.get("failed_trade", {}),
            "key_insight": demo.get("key_insight"),
        }
    else:
        demo = show_rsi_strategy_demo()
        backtest_proof = {
            "strategy": "RSI_OVERSOLD_BOUNCE",
            "best_performers": demo.get("best_performers", [])[:3],
            "recent_trades": demo.get("recent_trades_in_sideways_market", {}).get("trades", []),
            "why_works_now": demo.get("why_rsi_works_now", []),
        }
    
    return {
        "status": "success",
        "step": "1_ANALYSIS_COMPLETE",
        "market_analysis": {
            "regime": regime,
            "nifty_close": regime_data.get("metrics", {}).get("close"),
            "nifty_50dma": regime_data.get("metrics", {}).get("dma_50"),
            "trend_slope": regime_data.get("metrics", {}).get("dma_50_slope"),
            "position_vs_dma": "BELOW" if regime_data.get("metrics", {}).get("close", 0) < regime_data.get("metrics", {}).get("dma_50", 0) else "ABOVE",
        },
        "portfolio_state": {
            "available_cash_inr": cash,
            "open_positions": num_positions,
            "max_positions": max_positions,
            "can_open_new_position": can_open_new,
            "positions": [{"symbol": p.get("symbol"), "qty": p.get("qty"), "entry": p.get("entry")} for p in open_positions],
        },
        "recommended_strategy": {
            "name": strategy,
            "description": strategy_desc,
            "reason": f"Market is {regime}. {strategy_desc}.",
        },
        "backtest_proof": backtest_proof,
        "next_steps": [
            "STEP 2: I will now scan for opportunities using this strategy",
            "STEP 3: Show you the top candidates with entry/stop/target",
            "STEP 4: Wait for your confirmation before executing",
            "STEP 5: Execute trade and update portfolio",
        ],
        "awaiting_confirmation": "Say 'continue' or 'proceed' to scan for opportunities, or 'stop' to cancel.",
        "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def scan_opportunities_for_regime() -> Dict:
    """Scan for trading opportunities based on current market regime.
    
    This is STEP 2 of the autonomous trading flow. It:
    1. Re-checks market regime
    2. Runs the appropriate scanner (dividend or RSI oversold)
    3. Returns top candidates with entry/stop/target levels
    
    Use after analyze_and_recommend_strategy when user confirms to proceed.
    
    Returns:
        dict with regime, scan_results, top_candidates, and trade_plans.
    """
    # Get current regime
    regime_data = analyze_regime()
    regime = regime_data.get("regime", "UNKNOWN") if regime_data.get("status") == "success" else "UNKNOWN"
    
    # Get portfolio to check capacity
    portfolio = get_portfolio_summary()
    cash = portfolio.get("cash_inr", 0)
    open_positions = len(portfolio.get("open_positions", []))
    can_trade = open_positions < 3 and cash > 10000
    
    if not can_trade:
        return {
            "status": "blocked",
            "reason": "Cannot open new positions. Max 3 positions reached or insufficient cash.",
            "open_positions": open_positions,
            "cash_inr": cash,
            "suggestion": "Wait for existing positions to close, or add more capital.",
        }
    
    # Run appropriate scan
    if regime == "BULL":
        # Dividend scan
        scan = scan_dividend_opportunities(min_days_to_ex=3)
        if scan.get("status") != "success" or not scan.get("top_opportunities"):
            return {
                "status": "no_opportunities",
                "regime": regime,
                "strategy": "DIVIDEND_MOMENTUM",
                "message": "No dividend opportunities found that pass all filters. Try RSI oversold scan instead.",
            }
        
        candidates = []
        for opp in scan.get("top_opportunities", [])[:5]:
            candidates.append({
                "symbol": opp.get("symbol"),
                "company": opp.get("company"),
                "entry": opp.get("latest_close"),
                "stop": opp.get("suggested_stop"),
                "target": round(opp.get("latest_close", 0) + 2 * (opp.get("latest_close", 0) - opp.get("suggested_stop", 0)), 2),
                "ex_date": opp.get("ex_date"),
                "days_to_ex": opp.get("days_to_ex"),
                "trend_strength": opp.get("trend_strength"),
            })
        
        return {
            "status": "success",
            "step": "2_SCAN_COMPLETE",
            "regime": regime,
            "strategy": "DIVIDEND_MOMENTUM",
            "total_found": len(scan.get("top_opportunities", [])),
            "top_candidates": candidates,
            "selection_criteria": scan.get("scan_criteria"),
            "next_step": "Review candidates above. Say 'trade [SYMBOL]' to execute, or 'trade top' for the best one.",
            "awaiting_confirmation": True,
            "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }
    else:
        # RSI oversold scan
        scan = scan_oversold_bounce(min_rsi=None, max_rsi=35)
        if scan.get("status") != "success" or not scan.get("oversold_stocks"):
            # Try getting best historical performers
            best = get_best_oversold_nifty50(years=2, min_win_rate_pct=50, min_trades=3)
            if best.get("status") == "success" and best.get("best_stocks"):
                return {
                    "status": "success",
                    "step": "2_SCAN_COMPLETE",
                    "regime": regime,
                    "strategy": "RSI_OVERSOLD_BOUNCE",
                    "message": "No currently oversold stocks. Showing historically best performers for this strategy.",
                    "best_historical": best.get("best_stocks", [])[:5],
                    "suggestion": "Wait for these stocks to become oversold (RSI < 35) before entering.",
                    "awaiting_confirmation": False,
                    "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
                }
            return {
                "status": "no_opportunities",
                "regime": regime,
                "strategy": "RSI_OVERSOLD_BOUNCE",
                "message": "No oversold stocks found. Market may not be oversold enough yet.",
            }
        
        candidates = []
        for stock in scan.get("oversold_stocks", [])[:5]:
            entry = stock.get("close")
            stop = stock.get("suggested_stop")
            risk = entry - stop if entry and stop else 0
            target = round(entry + 2 * risk, 2) if risk > 0 else entry
            
            candidates.append({
                "symbol": stock.get("symbol"),
                "entry": entry,
                "stop": stop,
                "target": target,
                "rsi": stock.get("rsi"),
                "pct_below_50dma": stock.get("pct_below_50dma"),
                "risk_reward": "1:2",
            })
        
        return {
            "status": "success",
            "step": "2_SCAN_COMPLETE",
            "regime": regime,
            "strategy": "RSI_OVERSOLD_BOUNCE",
            "total_found": len(scan.get("oversold_stocks", [])),
            "top_candidates": candidates,
            "selection_criteria": "RSI <= 35, below 50-DMA, with 0.8*ATR stop-loss",
            "next_step": "Review candidates above. Say 'trade [SYMBOL]' to execute, or 'trade top' for the best one.",
            "awaiting_confirmation": True,
            "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        }


def prepare_trade_for_execution(symbol: str) -> Dict:
    """Prepare a trade plan for a specific symbol from the scan results.
    
    This is STEP 3 of the autonomous trading flow. It:
    1. Gets current price and technical data for the symbol
    2. Calculates position size based on portfolio risk rules
    3. Returns complete trade plan awaiting user confirmation
    
    Args:
        symbol: Stock symbol to trade (e.g., KOTAKBANK or KOTAKBANK.NS)
    
    Returns:
        dict with complete trade plan including entry, stop, target, qty, capital required.
    """
    from trading_agents.tools.paper_trading import calculate_trade_plan_from_entry_stop
    from trading_agents.tools.market_data import fetch_stock_data
    from trading_agents.tools.technical import compute_stock_metrics
    
    # Normalize symbol
    if not symbol.upper().endswith('.NS') and not symbol.startswith('^'):
        symbol = symbol.upper() + '.NS'
    
    # Get current price data
    stock_data = fetch_stock_data(symbol)
    if stock_data.get("status") != "success":
        return {"status": "error", "message": f"Could not fetch data for {symbol}"}
    
    closes = stock_data.get("closes", [])
    highs = stock_data.get("highs", [])
    lows = stock_data.get("lows", [])
    
    if not closes:
        return {"status": "error", "message": f"No price data for {symbol}"}
    
    close = closes[-1]
    
    # Get technical metrics
    metrics = compute_stock_metrics(closes, highs, lows)
    atr = metrics.get("atr", close * 0.02)  # Default 2% if no ATR
    
    # Calculate stop (0.8 * ATR for RSI strategy)
    stop = round(close - 0.8 * atr, 2)
    
    # Get trade plan
    plan = calculate_trade_plan_from_entry_stop(symbol=symbol, entry=close, stop=stop)
    
    if plan.get("status") != "success":
        return plan
    
    return {
        "status": "success",
        "step": "3_TRADE_PLAN_READY",
        "trade_plan": {
            "symbol": symbol,
            "action": "BUY",
            "entry_price": plan.get("entry"),
            "stop_loss": plan.get("stop"),
            "target_price": plan.get("target"),
            "quantity": plan.get("qty"),
            "capital_required_inr": plan.get("capital_required_inr"),
            "risk_amount_inr": plan.get("risk_inr"),
            "risk_reward_ratio": plan.get("rr_ratio"),
            "risk_pct_of_capital": plan.get("risk_pct"),
        },
        "confirmation_required": True,
        "instruction": f"Trade plan ready for {symbol}. Say 'execute' or 'confirm' to place the paper trade, or 'cancel' to abort.",
        "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def execute_confirmed_trade(symbol: str, entry: float, stop: float, target: float, qty: int) -> Dict:
    """Execute a confirmed paper trade and update portfolio.
    
    This is STEP 4 of the autonomous trading flow. Call this ONLY after user confirms.
    
    Args:
        symbol: Stock symbol
        entry: Entry price
        stop: Stop-loss price
        target: Target price
        qty: Quantity to buy
    
    Returns:
        dict with execution result and updated portfolio state.
    """
    from trading_agents.tools.paper_trading import execute_paper_trade
    
    # Execute the trade
    result = execute_paper_trade(symbol=symbol, entry=entry, stop=stop, target=target, qty=qty)
    
    if result.get("status") != "success":
        return result
    
    # Get updated portfolio
    portfolio = get_portfolio_summary()
    
    return {
        "status": "success",
        "step": "4_TRADE_EXECUTED",
        "execution": {
            "symbol": result.get("symbol"),
            "action": "BOUGHT",
            "entry": result.get("entry"),
            "stop": result.get("stop"),
            "target": result.get("target"),
            "qty": result.get("qty"),
            "invested_inr": result.get("invested_inr"),
        },
        "portfolio_after": {
            "cash_remaining_inr": portfolio.get("cash_inr"),
            "open_positions": len(portfolio.get("open_positions", [])),
            "total_invested_inr": portfolio.get("total_invested_inr"),
        },
        "next_steps": [
            "Position is now OPEN and will be tracked",
            "Stop-loss and target will auto-trigger on price hit",
            "Use 'refresh portfolio' to check status",
            "Say 'continue trading' to look for more opportunities",
        ],
        "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }


def check_trading_loop_status() -> Dict:
    """Check if autonomous trading can continue (positions available, cash available).
    
    Use this to determine if the trading loop should continue or stop.
    
    Returns:
        dict with can_continue flag and current portfolio state.
    """
    portfolio = get_portfolio_summary()
    cash = portfolio.get("cash_inr", 0)
    open_positions = portfolio.get("open_positions", [])
    num_positions = len(open_positions)
    
    # Check conditions
    has_cash = cash > 10000  # Minimum cash to trade
    has_capacity = num_positions < 3  # Max 3 positions
    can_continue = has_cash and has_capacity
    
    stop_reason = None
    if not has_cash:
        stop_reason = "Insufficient cash (< INR 10,000)"
    elif not has_capacity:
        stop_reason = "Maximum positions reached (3 open)"
    
    return {
        "status": "success",
        "can_continue_trading": can_continue,
        "stop_reason": stop_reason,
        "portfolio_state": {
            "cash_inr": cash,
            "open_positions": num_positions,
            "max_positions": 3,
            "positions": [
                {
                    "symbol": p.get("symbol"),
                    "qty": p.get("qty"),
                    "entry": p.get("entry"),
                    "current_pnl": p.get("unrealized_pnl_inr"),
                }
                for p in open_positions
            ],
        },
        "suggestion": "Say 'continue trading' to find more opportunities" if can_continue else f"Trading paused: {stop_reason}. Wait for positions to close or add capital.",
        "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    }
