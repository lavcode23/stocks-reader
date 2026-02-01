
import yfinance as yf
import pandas as pd

SECTORS = {
    "IT": ["TCS.NS","INFY.NS","WIPRO.NS"],
    "BANK": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS"],
    "PHARMA": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"]
}

def weekly_return(ticker):
    df = yf.download(ticker, period="10d", interval="1d", progress=False)
    if len(df) < 5:
        return 0.0
    return float((df["Close"].iloc[-1] / df["Close"].iloc[-5]) - 1)

def sector_scores():
    scores = {}
    for s, stocks in SECTORS.items():
        vals = []
        for t in stocks:
            try:
                vals.append(weekly_return(t))
            except:
                pass
        scores[s] = sum(vals)/len(vals) if vals else 0
    return scores

def top_stocks(sector, n=2):
    stocks = SECTORS[sector]
    ranked = sorted(stocks, key=lambda x: weekly_return(x), reverse=True)
    return ranked[:n]
