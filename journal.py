import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import numpy as np

JOURNAL_PATH = Path("paper_journal.csv")

COLUMNS = [
    "timestamp","symbol","sector","entry","stop","target","qty",
    "ml_prob","R_multiple","max_loss",
    "status","entry_filled_price","exit_price","pnl"
]

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_journal():
    if JOURNAL_PATH.exists():
        return pd.read_csv(JOURNAL_PATH)
    return pd.DataFrame(columns=COLUMNS)

def _save(df):
    df.to_csv(JOURNAL_PATH, index=False)

def append_trades(df_signals):
    j = load_journal()
    rows = []
    ts = _now()
    for _, r in df_signals.iterrows():
        rows.append({
            "timestamp": ts,
            "symbol": r["symbol"],
            "sector": r["sector"],
            "entry": r["entry_buy_above"],
            "stop": r["stop_loss"],
            "target": r["target"],
            "qty": int(r["qty"]),
            "ml_prob": r["ml_prob"],
            "R_multiple": r["R_multiple"],
            "max_loss": r["max_loss_₹"],
            "status": "OPEN",
            "entry_filled_price": "",
            "exit_price": "",
            "pnl": ""
        })
    out = pd.concat([j, pd.DataFrame(rows)], ignore_index=True)
    _save(out)
    return out

# ---------------- AUTO PAPER FILL ----------------

def _flatten_cols(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def auto_fill_open_trades():
    """
    Uses latest 1m data to:
    - Fill entry if price >= entry
    - Close at stop/target if hit
    """
    j = load_journal()
    if j.empty:
        return j

    open_trades = j[j["status"] == "OPEN"].copy()
    if open_trades.empty:
        return j

    symbols = open_trades["symbol"].unique().tolist()
    data = yf.download(symbols, period="1d", interval="1m", group_by="ticker", progress=False)

    for idx, r in open_trades.iterrows():
        sym = r["symbol"]
        try:
            df = data[sym] if sym in data else None
            if df is None or len(df) == 0:
                continue
            df = _flatten_cols(df).dropna()

            last = float(df["Close"].iloc[-1])
            low = float(df["Low"].iloc[-1])
            high = float(df["High"].iloc[-1])

            entry = float(r["entry"])
            stop = float(r["stop"])
            target = float(r["target"])
            qty = int(r["qty"])

            # Entry fill
            if r["entry_filled_price"] == "" and last >= entry:
                j.loc[idx, "entry_filled_price"] = round(entry, 2)

            # Exit only after entry filled
            if j.loc[idx, "entry_filled_price"] != "":
                fill = float(j.loc[idx, "entry_filled_price"])

                exit_price = None
                if low <= stop:
                    exit_price = stop
                elif high >= target:
                    exit_price = target

                if exit_price is not None:
                    pnl = (exit_price - fill) * qty
                    j.loc[idx, "exit_price"] = round(exit_price, 2)
                    j.loc[idx, "pnl"] = round(pnl, 2)
                    j.loc[idx, "status"] = "CLOSED"

        except Exception:
            continue

    _save(j)
    return j

# ---------------- PERFORMANCE STATS ----------------

def performance_stats():
    j = load_journal()
    if j.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}

    closed = j[j["status"] == "CLOSED"].copy()
    if closed.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}

    pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)

    # build equity curve from pnl
    equity = pnl.cumsum()
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_drawdown": 0.0}

    returns = equity.diff().fillna(0.0)
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd = float(drawdown.min())

    return {
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(abs(max_dd), 2)
    }
