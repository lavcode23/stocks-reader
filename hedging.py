import numpy as np
import pandas as pd

def beta_to_benchmark(asset_close: pd.Series, bench_close: pd.Series, lookback: int = 90) -> float:
    a = asset_close.pct_change().dropna()
    b = bench_close.pct_change().dropna()
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < lookback:
        lookback = max(20, len(df))
    df = df.tail(lookback)
    x = df.iloc[:,1].values
    y = df.iloc[:,0].values
    if x.std() == 0:
        return 0.0
    beta = float(np.cov(y, x)[0,1] / np.var(x))
    return beta

def hedge_ratio(port_beta: float, cap: float = 1.0) -> float:
    # hedge ratio in [0, cap]
    return max(0.0, min(cap, abs(port_beta)))
