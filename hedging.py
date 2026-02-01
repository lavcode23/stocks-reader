import numpy as np
import pandas as pd

def beta(asset: pd.Series, bench: pd.Series, lookback=60):
    a = asset.pct_change().dropna()
    b = bench.pct_change().dropna()
    df = pd.concat([a,b],axis=1).dropna().tail(lookback)
    if len(df)<10: return 0.0
    return float(np.cov(df.iloc[:,0],df.iloc[:,1])[0,1]/df.iloc[:,1].var())

def hedge_ratio(port_beta, cap=1.0):
    return max(0.0, min(cap, abs(port_beta)))
