import numpy as np
import pandas as pd
import yfinance as yf
import importlib.util
from pathlib import Path

BASE = Path(__file__).parent

# Load strategy by absolute path
spec = importlib.util.spec_from_file_location("strategy", BASE / "strategy.py")
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

make_trade_plan = strategy.make_trade_plan
simulate_trade = strategy.simulate_trade


def weekly_dates(df):
    d = pd.DataFrame(index=df.index)
    d["w"] = d.index.to_period("W-FRI")
    return d.groupby("w").tail(1).index


def backtest(sectors, start, stop_loss, take_profit, holding_days):

    stocks = sorted({x for v in sectors.values() for x in v})
    prices = yf.download(stocks, start=start, progress=False, group_by="ticker")
    index = yf.download("^NSEI", start=start, progress=False)

    signals = weekly_dates(index)

    trades = []

    for sd in signals:
        sector_scores = {}

        for s, tks in sectors.items():
            vals = []
            for t in tks:
                try:
                    df = prices[t].loc[:sd]
                    r = df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1
                    vals.append(r)
                except:
                    pass
            sector_scores[s] = np.mean(vals) if vals else -999

        best = max(sector_scores, key=sector_scores.get)
        picks = sectors[best][:2]

        for t in picks:
            try:
                df = prices[t]
                plan = make_trade_plan(t, df, sd, holding_days, stop_loss, take_profit)
                if plan:
                    r = simulate_trade(df, plan)
                    trades.append(r.__dict__)
            except:
                continue

    df = pd.DataFrame(trades)
    if df.empty:
        return {"trades": df, "win_rate": 0, "avg_return": 0}

    df["win"] = df["gross_return"] > 0

    return {
        "trades": df,
        "win_rate": float(df["win"].mean()),
        "avg_return": float(df["gross_return"].mean()),
    }
