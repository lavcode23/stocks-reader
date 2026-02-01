import yfinance as yf
import pandas as pd
import numpy as np

from ml_ranker import make_dataset, train_model, predict_prob


ACCOUNT_EQUITY = 100000
RISK_PER_TRADE = 0.003
ATR_MULT = 1.5
RR = 2.0


def run_realtime(cfg):

    sectors = cfg["universe"]["sectors"]
    tickers = sorted({t for s in sectors.values() for t in s})

    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", progress=False)

    cleaned = {}
    for t in tickers:
        if t in data:
            df = data[t].dropna()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            if len(df) > 60:
                cleaned[t] = df

    # -------- train ML once --------
    pool = []
    for df in cleaned.values():
        ds = make_dataset(df)
        if not ds.empty:
            pool.append(ds)

    model = train_model(pd.concat(pool))

    signals = []

    for sector, symbols in sectors.items():
        for sym in symbols:
            if sym not in cleaned:
                continue

            df = cleaned[sym]
            ds = make_dataset(df)
            if ds.empty:
                continue

            prob = predict_prob(model, ds)

            atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
            close = df["Close"].iloc[-1]

            entry = close * 1.002
            stop = entry - atr * ATR_MULT
            target = entry + RR * (entry - stop)

            risk_amt = ACCOUNT_EQUITY * RISK_PER_TRADE
            qty = int(risk_amt / max(entry - stop, 1))

            signals.append({
                "sector": sector,
                "symbol": sym,
                "ml_prob": round(prob,3),
                "entry": round(entry,2),
                "stop": round(stop,2),
                "target": round(target,2),
                "qty": qty
            })

    df = pd.DataFrame(signals)
    df = df.sort_values("ml_prob", ascending=False).head(5)

    return df


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    trades = run_realtime(cfg)
    print(trades)
