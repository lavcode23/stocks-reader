import yfinance as yf
import pandas as pd
import pickle
import yaml
import os
import numpy as np

from ml_ranker import make_dataset, train_model, predict_prob

MODEL_PATH = "ml_model.pkl"

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def load_or_train_model(tickers):
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False)
    pool = []
    for t in tickers:
        if t in data:
            df = _flatten_cols(data[t]).dropna()
            if len(df) > 120:
                ds = make_dataset(df)
                if not ds.empty:
                    pool.append(ds)

    if not pool:
        raise RuntimeError("Not enough data to train ML model.")

    model = train_model(pd.concat(pool))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def run_realtime(account_equity: float = 100000.0, max_positions: int = 3):

    cfg = load_cfg()
    sectors = cfg["universe"]["sectors"]
    tickers = sorted({t for s in sectors.values() for t in s})

    # ---- Market regime filter (beginner safety) ----
    nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False).dropna()
    nifty = _flatten_cols(nifty)
    ma50 = nifty["Close"].rolling(50).mean().iloc[-1]
    market_ok = bool(nifty["Close"].iloc[-1] > ma50)

    if not market_ok:
        return {
            "market_ok": False,
            "signals": pd.DataFrame(),
            "note": "NO TRADE: NIFTY below 50D MA (bear/sideways regime)."
        }

    model = load_or_train_model(tickers)

    data = yf.download(tickers, period="60d", interval="1d", group_by="ticker", progress=False)

    signals = []
    risk_pct = float(cfg["trade"]["risk_per_trade_pct"])
    atr_mult = float(cfg["trade"]["atr_mult"])
    rr = float(cfg["trade"]["take_profit_R"])

    for sector, symbols in sectors.items():
        for sym in symbols:
            if sym not in data:
                continue

            df = _flatten_cols(data[sym]).dropna()
            if len(df) < 40:
                continue

            ds = make_dataset(df)
            if ds.empty:
                continue

            prob = predict_prob(model, ds)

            # confidence filter for beginners
            if prob < 0.55:
                continue

            atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])
            close = float(df["Close"].iloc[-1])
            if np.isnan(atr) or atr <= 0:
                continue

            # Entry trigger: buy above close by 0.2% (simple breakout trigger)
            entry = close * 1.002
            stop = entry - atr * atr_mult
            target = entry + rr * (entry - stop)

            risk_amt = account_equity * risk_pct
            per_share_risk = max(entry - stop, 1.0)
            qty = int(risk_amt / per_share_risk)
            if qty <= 0:
                continue

            max_loss = qty * per_share_risk
            r_multiple = (target - entry) / per_share_risk

            signals.append({
                "sector": sector,
                "symbol": sym,
                "ml_prob": round(prob, 3),
                "entry_buy_above": round(entry, 2),
                "stop_loss": round(stop, 2),
                "target": round(target, 2),
                "qty": int(qty),
                "max_loss_₹": round(max_loss, 0),
                "R_multiple": round(r_multiple, 2),
                "validity": "Today only"
            })

    df = pd.DataFrame(signals)
    if df.empty:
        return {
            "market_ok": True,
            "signals": df,
            "note": "No high-confidence trades today (prob < 0.55 filtered out)."
        }

    df = df.sort_values(["ml_prob", "R_multiple"], ascending=[False, False]).head(max_positions)

    return {
        "market_ok": True,
        "signals": df,
        "note": "Trade only if entry is triggered. If not triggered, skip."
    }
