import yfinance as yf

# Try different variations for Sonata Software
symbols = ['SONATSOFTW.NS', 'SONATA.NS', 'SONATASOFTWARE.NS', 'SONSOFT.NS', 'SONATSOFT.NS']
for s in symbols:
    try:
        t = yf.Ticker(s)
        h = t.history(period='5d')
        if not h.empty:
            close = h['Close'].iloc[-1]
            print(f'{s} - FOUND! Last close: {close:.2f}')
        else:
            print(f'{s} - empty')
    except Exception as e:
        print(f'{s} - error: {e}')
