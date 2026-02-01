import pandas as pd
from features import atr

def compute_trade_plan(
    df: pd.DataFrame,
    equity: float,
    risk_per_trade_pct: float,
    entry_mode: str,
    breakout_buffer_pct: float,
    stop_mode: str,
    atr_period: int,
    atr_mult: float,
    take_profit_R: float,
) -> dict:
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if len(df) < 25:
        raise ValueError("Not enough history to compute plan")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ENTRY
    if entry_mode == "open":
        entry = float(last["Open"])
        trigger = entry
    else:
        trigger = float(prev["High"]) * (1.0 + breakout_buffer_pct)
        entry = trigger  # assume filled at trigger if breakout happens

    # STOP
    if stop_mode == "atr":
        a = atr(df, atr_period).iloc[-1]
        if pd.isna(a):
            a = (df["High"] - df["Low"]).rolling(atr_period).mean().iloc[-1]
        stop = entry - float(atr_mult * a)
    else:
        stop = float(prev["Low"])

    stop_dist = max(1e-9, entry - stop)

    # POSITION SIZE (risk-based)
    risk_rupees = equity * risk_per_trade_pct
    qty = int(risk_rupees / stop_dist)
    qty = max(qty, 1)

    stop_loss_amount = qty * stop_dist

    # TAKE PROFIT
    take = entry + take_profit_R * stop_dist

    return {
        "entry": entry,
        "trigger": trigger,
        "stop": stop,
        "take": take,
        "qty": qty,
        "stop_loss_amount": stop_loss_amount,
        "risk_rupees": risk_rupees,
        "R": stop_dist,
    }
