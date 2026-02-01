from dataclasses import dataclass
import pandas as pd


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    reason: str
    ret: float


def simulate_trade(df, signal_date, sl, tp, hold=5):

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    future = df[df.index > signal_date]
    if future.empty:
        return None

    entry_date = future.index[0]
    entry = float(future.iloc[0]["Open"])

    stop = entry * (1 - sl)
    take = entry * (1 + tp)

    window = df[df.index >= entry_date].iloc[:hold]

    exit_price = window.iloc[-1]["Close"]
    exit_date = window.index[-1]
    reason = "time"

    for d, r in window.iterrows():
        if r["Low"] <= stop:
            exit_price = stop
            exit_date = d
            reason = "stop"
            break
        if r["High"] >= take:
            exit_price = take
            exit_date = d
            reason = "take"
            break

    ret = exit_price / entry - 1

    return Trade("X", entry_date, exit_date, entry, float(exit_price), reason, float(ret))
