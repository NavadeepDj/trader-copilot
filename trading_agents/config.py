"""Risk rules, thresholds, and NSE-specific defaults."""

import os
import google.genai as genai

GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]


def _pick_available_model() -> str:
    """Try each model with a minimal API call; return the first that responds."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[config] WARNING: GOOGLE_API_KEY not set, defaulting to first model")
        return GEMINI_FALLBACK_MODELS[0]

    client = genai.Client(api_key=api_key)
    for model in GEMINI_FALLBACK_MODELS:
        try:
            client.models.generate_content(model=model, contents="ping")
            print(f"[config] Using model: {model}")
            return model
        except Exception as exc:
            status = getattr(exc, "status_code", "") or type(exc).__name__
            print(f"[config] {model} unavailable ({status}), trying next...")
    print(f"[config] WARNING: all models failed, defaulting to {GEMINI_FALLBACK_MODELS[0]}")
    return GEMINI_FALLBACK_MODELS[0]


GEMINI_MODEL = _pick_available_model()

# --------------- Risk management ---------------
RISK_PER_TRADE = 0.01          # 1 % of portfolio per trade
MAX_OPEN_TRADES = 3
MIN_REWARD_RISK = 2.0          # minimum R:R ratio
ATR_STOP_MULTIPLIER = 1.5      # stop = entry - 1.5 * ATR
INITIAL_CAPITAL = 1_000_000.0  # INR paper portfolio

# --------------- Regime thresholds ---------------
BULL_RETURN_20D_MIN = 0.0
BEAR_RETURN_20D_MAX = -0.03

# --------------- Default index ---------------
DEFAULT_INDEX = "^NSEI"        # Nifty 50
BANK_INDEX = "^NSEBANK"        # Bank Nifty

# --------------- NSE watchlist (top liquid Nifty 50 stocks) ---------------
NSE_WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TATAMOTORS.NS",
]

DATA_LOOKBACK_DAYS = 140       # enough for 50-DMA + buffer
