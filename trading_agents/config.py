"""Risk rules, thresholds, and NSE-specific defaults."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
import google.genai as genai

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]

_503_RETRY_ATTEMPTS = 3
_503_RETRY_BASE_DELAY = 2  # seconds


def create_genai_client() -> genai.Client:
    """Create a google.genai Client using Vertex AI or API key based on env vars.

    Reads GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
    and GOOGLE_API_KEY from environment (loaded from trading_agents/.env).
    """
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() in ("TRUE", "1", "YES")

    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise ValueError(
                "GOOGLE_GENAI_USE_VERTEXAI is TRUE but GOOGLE_CLOUD_PROJECT is not set."
            )
        print(f"[config] Using Vertex AI (project={project}, location={location})")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("[config] Using Google AI Studio (API key)")
        return genai.Client(api_key=api_key)

    raise ValueError(
        "No credentials found. Set either GOOGLE_GENAI_USE_VERTEXAI=TRUE with "
        "GOOGLE_CLOUD_PROJECT, or set GOOGLE_API_KEY."
    )


def _is_503(exc: Exception) -> bool:
    """Check if an exception is a 503 UNAVAILABLE error."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 503:
        return True
    return "503" in str(exc) or "UNAVAILABLE" in str(exc).upper()


def call_gemini_with_fallback(client: genai.Client, contents, config=None):
    """Call Gemini with automatic model fallback and retry on 503.

    Tries each model in GEMINI_FALLBACK_MODELS. For 503 errors, retries the
    same model with exponential backoff before moving to the next.
    For 404/other errors, skips to the next model immediately.

    Returns:
        The generate_content response from the first model that succeeds.

    Raises:
        Last exception if all models and retries are exhausted.
    """
    last_exc = None

    for model in GEMINI_FALLBACK_MODELS:
        for attempt in range(_503_RETRY_ATTEMPTS):
            try:
                kwargs = {"model": model, "contents": contents}
                if config is not None:
                    kwargs["config"] = config
                response = client.models.generate_content(**kwargs)
                if attempt > 0 or model != GEMINI_FALLBACK_MODELS[0]:
                    print(f"[gemini] Success with {model} (attempt {attempt + 1})")
                return response
            except Exception as exc:
                last_exc = exc
                if _is_503(exc):
                    delay = _503_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[gemini] {model} 503 (attempt {attempt + 1}/{_503_RETRY_ATTEMPTS}), "
                          f"retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    status = getattr(exc, "status_code", "") or type(exc).__name__
                    print(f"[gemini] {model} failed ({status}), trying next model...")
                    break  # non-503 error, skip to next model

    raise last_exc  # type: ignore[misc]


def _pick_available_model() -> str:
    """Probe each model; return the first that responds. Retries 503 errors."""
    try:
        client = create_genai_client()
    except ValueError as exc:
        print(f"[config] WARNING: {exc} -- defaulting to first model")
        return GEMINI_FALLBACK_MODELS[0]

    for model in GEMINI_FALLBACK_MODELS:
        for attempt in range(_503_RETRY_ATTEMPTS):
            try:
                client.models.generate_content(model=model, contents="ping")
                print(f"[config] Using model: {model}")
                return model
            except Exception as exc:
                if _is_503(exc):
                    delay = _503_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[config] {model} 503 (attempt {attempt + 1}/{_503_RETRY_ATTEMPTS}), "
                          f"retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    status = getattr(exc, "status_code", "") or type(exc).__name__
                    print(f"[config] {model} unavailable ({status}), trying next...")
                    break  # non-503, skip to next model

    print(f"[config] WARNING: all models failed, defaulting to {GEMINI_FALLBACK_MODELS[0]}")
    return GEMINI_FALLBACK_MODELS[0]


GEMINI_MODEL = _pick_available_model()

# --------------- Risk management ---------------
RISK_PER_TRADE = 0.01          # 1 % of portfolio per trade
MAX_OPEN_TRADES = 3
MIN_REWARD_RISK = 2.0          # minimum R:R ratio
ATR_STOP_MULTIPLIER = 1.5      # stop = entry - 1.5 * ATR (breakout/momentum)
DIVIDEND_STOP_ATR_MULTIPLIER = 0.8   # dividend: stop = entry - 0.8*ATR (smaller = closer stop, less loss when hit; use 0.5–0.6 for even tighter)
INITIAL_CAPITAL = 1_000_000.0  # INR paper portfolio
MAX_HOLD_DAYS = 10             # lifecycle time exit for open paper trades

# --------------- Regime thresholds ---------------
BULL_RETURN_20D_MIN = 0.0
BEAR_RETURN_20D_MAX = -0.03

# --------------- Default index ---------------
DEFAULT_INDEX = "^NSEI"        # Nifty 50
BANK_INDEX = "^NSEBANK"        # Bank Nifty

# --------------- NSE watchlist (Nifty 50 -- used by breakout/momentum scanner) --
NSE_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS", "INDUSINDBK.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "NESTLEIND.NS", "DRREDDY.NS",
    "CIPLA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DIVISLAB.NS",
    "BPCL.NS", "TATACONSUM.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS", "TRENT.NS", "BEL.NS",
]

DATA_LOOKBACK_DAYS = 140       # enough for 50-DMA + buffer
