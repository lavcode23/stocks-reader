import numpy as np
import pandas as pd
import yfinance as yf
from strategy import simulate_trade


def weekly_dates(df):
    d = pd.DataFrame(index=df.index)
    d["w"] = d.index.to_period("W-FRI")
    return d.groupby("w").tail(1).index


def run_backtest(sectors, start, sl, tp):

    stocks = sorted({x for v in sectors.values() for x in v})
    prices = yf.download(stocks, start=start, group_by="ticker", progress=False)
    index = yf.download("^NSEI", start=start, progress=False)

    signals = weekly_dates(index)

    rows = []

    for sd in signals:
        scores = {}

        for s, tks in sectors.items():
            vals = []
            for t in tks:
                try:
                    df = prices[t].loc[:sd]
                    r = df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1
                    vals.append(r)
                except:
                    pass
            scores[s] = np.mean(vals) if vals else -999

        best = max(scores, key=scores.get)
        picks = sectors[best][:2]

        for t in picks:
            try:
                df = prices[t]
                tr = simulate_trade(df, sd, sl, tp)
                if tr:
                    rows.append({
                        "ticker": t,
                        "entry_date": tr.entry_date,
                        "exit_date": tr.exit_date,
                        "return": tr.ret
                    })
            except:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0

    win = (df["return"] > 0).mean()

    return df, float(win)
