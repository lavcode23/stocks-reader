import yfinance as yf
import pandas as pd
import pickle
import yaml
import os

from ml_ranker import make_dataset, train_model, predict_prob

ACCOUNT = 100000
RISK = 0.003
ATR_MULT = 1.5
RR = 2.0

MODEL_PATH = "ml_model.pkl"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

sectors = cfg["universe"]["sectors"]
tickers = sorted({t for s in sectors.values() for t in s})


def load_or_train_model():

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,"rb") as f:
            return pickle.load(f)

    # auto train if missing
    data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False)

    pool = []
    for t in tickers:
        if t in data:
            df = data[t].dropna()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            if len(df) > 80:
                ds = make_dataset(df)
                if not ds.empty:
                    pool.append(ds)

    model = train_model(pd.concat(pool))

    with open(MODEL_PATH,"wb") as f:
        pickle.dump(model,f)

    return model


model = load_or_train_model()


def run_realtime():

    data = yf.download(tickers, period="30d", interval="1d", group_by="ticker", progress=False)

    signals = []

    for sector, symbols in sectors.items():
        for sym in symbols:
            if sym not in data:
                continue

            df = data[sym].dropna()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            if len(df) < 25:
                continue

            ds = make_dataset(df)
            if ds.empty:
                continue

            prob = predict_prob(model, ds)

            atr = (df["High"]-df["Low"]).rolling(14).mean().iloc[-1]
            close = df["Close"].iloc[-1]

            entry = close * 1.002
            stop = entry - atr * ATR_MULT
            target = entry + RR*(entry-stop)

            qty = int((ACCOUNT*RISK)/max(entry-stop,1))

            signals.append({
                "sector":sector,
                "symbol":sym,
                "prob":round(prob,3),
                "entry":round(entry,2),
                "stop":round(stop,2),
                "target":round(target,2),
                "qty":qty
            })

    df = pd.DataFrame(signals)
    return df.sort_values("prob",ascending=False).head(5)
