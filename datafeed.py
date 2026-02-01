import pandas as pd
import yfinance as yf

def fetch_ohlc(ticker: str, start: str, end: str | None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None, errors="ignore")
    return df.dropna(how="all")

def fetch_many(tickers: list[str], start: str, end: str | None, interval: str = "1d") -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        out[t] = fetch_ohlc(t, start, end, interval=interval)
    return out
