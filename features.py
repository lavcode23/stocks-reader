import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def momentum(close: pd.Series, days: int) -> pd.Series:
    return close.pct_change(days)

def realized_vol(close: pd.Series, days: int = 20) -> pd.Series:
    r = close.pct_change()
    return r.rolling(days).std() * np.sqrt(252)
