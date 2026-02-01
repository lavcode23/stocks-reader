import numpy as np
import pandas as pd
import yfinance as yf

from strategy import make_trade_plan, simulate_trade


def weekly_dates(index_df):
    d = pd.DataFrame(index=index_df.index)
    d["week"] = d.index.to_period("W-FRI")
    return d.groupby("week").tail(1).index


def backtest(
    sectors,
    start,
    stop_loss,
    take_profit,
    holding_days=5,
):
    all_stocks = sorted({x for v in sectors.values() for x in v})
    prices = yf.download(all_stocks, start=start, progress=False, group_by="ticker")

    benchmark = yf.download("^NSEI", start=start, progress=False)

    signals = weekly_dates(benchmark)

    trades = []

    for sd in signals:
        sector_scores = {}

        for s, tickers in sectors.items():
            vals = []
            for t in tickers:
                try:
                    df = prices[t]
                    df = df.loc[df.index <= sd]
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
                    res = simulate_trade(df, plan)
                    trades.append(res.__dict__)
            except:
                continue

    df = pd.DataFrame(trades)
    df["win"] = df["gross_return"] > 0

    return {
        "trades": df,
        "win_rate": float(df["win"].mean()),
        "avg_return": float(df["gross_return"].mean()),
    }
