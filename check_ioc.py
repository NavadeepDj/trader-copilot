import yfinance as yf

# Check IOC's trend around buy date (Apr 30, 2025)
t = yf.Ticker('IOC.NS')
h = t.history(start='2025-01-01', end='2025-05-15')
print(f"IOC data points: {len(h)}")

if len(h) >= 50:
    h['DMA50'] = h['Close'].rolling(50).mean()
    print('=== IOC around announcement (Apr 30) ===')
    for i in range(-15, 0):
        try:
            d = h.index[i]
            c = h['Close'].iloc[i]
            dma = h['DMA50'].iloc[i]
            status = 'ABOVE' if c > dma else 'BELOW'
            print(f'{d.date()}: Close={c:.2f}, 50-DMA={dma:.2f} -> {status}')
        except Exception as e:
            print(f"Error: {e}")
else:
    print("Not enough data for 50-DMA")
    print(h[['Close']].tail(15))
