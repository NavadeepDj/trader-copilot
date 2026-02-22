"""Debug yfinance fetch for IOC.NS"""
import yfinance as yf

symbol = "IOC.NS"
start = "2025-04-20"
end = "2025-08-13"

print(f"Testing yfinance for {symbol} from {start} to {end}")
print("-" * 50)

try:
    ticker = yf.Ticker(symbol)
    print(f"1. Ticker created: {ticker}")
    
    # Check ticker info
    try:
        info = ticker.info
        print(f"2. Ticker info keys: {list(info.keys())[:10] if info else 'None'}")
    except Exception as e:
        print(f"2. Ticker info error: {e}")
    
    # Try to fetch history
    print(f"3. Fetching history...")
    hist = ticker.history(start=start, end=end, interval="1d")
    
    print(f"4. History type: {type(hist)}")
    print(f"5. History shape: {hist.shape if hasattr(hist, 'shape') else 'N/A'}")
    print(f"6. History empty: {hist.empty if hasattr(hist, 'empty') else 'N/A'}")
    
    if hist is not None and not hist.empty:
        print(f"7. First few rows:\n{hist.head()}")
    else:
        print("7. No data returned")
        
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
