import pandas as pd
from pathlib import Path
from datetime import datetime

JOURNAL_PATH = Path("paper_journal.csv")

COLUMNS = [
    "timestamp","symbol","sector","entry","stop","target","qty",
    "ml_prob","R_multiple","max_loss",
    "status","entry_filled_price","exit_price","pnl"
]

def load_journal():
    if JOURNAL_PATH.exists():
        return pd.read_csv(JOURNAL_PATH)
    return pd.DataFrame(columns=COLUMNS)

def append_trades(df_signals):
    j = load_journal()
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    out.to_csv(JOURNAL_PATH, index=False)
    return out

def update_trade(symbol, entry_filled_price=None, exit_price=None):
    j = load_journal()
    mask = (j["symbol"] == symbol) & (j["status"] == "OPEN")
    if entry_filled_price is not None:
        j.loc[mask, "entry_filled_price"] = entry_filled_price
    if exit_price is not None:
        j.loc[mask, "exit_price"] = exit_price
        # compute pnl if both prices present
        rows = j.loc[mask]
        for idx, r in rows.iterrows():
            try:
                pnl = (float(exit_price) - float(r["entry"])) * int(r["qty"])
                j.loc[idx, "pnl"] = round(pnl, 2)
                j.loc[idx, "status"] = "CLOSED"
            except:
                pass
    j.to_csv(JOURNAL_PATH, index=False)
    return j
