import pandas as pd
from features import atr

def compute_trade_plan(df: pd.DataFrame, equity: float, risk_per_trade_pct: float,
                       entry_mode: str, breakout_buffer_pct: float,
                       stop_mode: str, atr_period: int, atr_mult: float,
                       take_profit_R: float):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # --- Entry price ---
    if entry_mode == "open":
        entry = float(last["Open"])
        trigger = entry
    else:
        trigger = float(prev["High"]) * (1 + breakout_buffer_pct)
        entry = trigger  # in backtest we assume filled at trigger

    # --- Stop loss ---
    if stop_mode == "atr":
        a = atr(df, atr_period).iloc[-1]
        if pd.isna(a):
            a = (df["High"] - df["Low"]).rolling(atr_period).mean().iloc[-1]
        stop = entry - float(atr_mult * a)
    else:
        stop = float(prev["Low"])  # simple swing stop

    stop_dist = max(1e-9, entry - stop)

    # --- Position sizing (risk-based) ---
    risk_rupees = equity * risk_per_trade_pct
    qty = int(risk_rupees / stop_dist)
    qty = max(qty, 1)

    stop_loss_amount = qty * stop_dist

    # --- Take profit ---
    take = entry + take_profit_R * stop_dist

    return {
        "entry": entry,
        "trigger": trigger,
        "stop": stop,
        "take": take,
        "qty": qty,
        "stop_loss_amount": stop_loss_amount,
        "risk_rupees": risk_rupees,
        "R": stop_dist
    }
