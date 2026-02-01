import numpy as np
import pandas as pd

def beta_to_benchmark(asset_close: pd.Series, bench_close: pd.Series, lookback: int = 90) -> float:
    a = asset_close.pct_change().dropna()
    b = bench_close.pct_change().dropna()
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty:
        return 0.0
    df = df.tail(max(20, min(lookback, len(df))))
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    if x.std() == 0:
        return 0.0
    return float(np.cov(y, x)[0, 1] / np.var(x))

def hedge_ratio(port_beta: float, cap: float = 1.0) -> float:
    return max(0.0, min(cap, abs(port_beta)))
