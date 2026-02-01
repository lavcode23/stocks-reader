import yfinance as yf
import pandas as pd
import numpy as np

from ml_ranker import make_dataset, train_model, predict_prob
from hedging import beta, hedge_ratio
from options_proxy import protective_put_proxy


HOLD_DAYS = 5
STEP = 5   # weekly


def run_backtest(cfg: dict):

    start = cfg["backtest"]["start"]
    sectors = cfg["universe"]["sectors"]

    top_n = int(cfg["signals"]["top_stocks"])
    equity = float(cfg["trade"]["equity_start"])
    risk = float(cfg["trade"]["risk_per_trade_pct"])
    atr_mult = float(cfg["trade"]["atr_mult"])
    rr = float(cfg["trade"]["take_profit_R"])

    trades = []

    # ---------------- Batch download ----------------
    tickers = sorted({t for s in sectors.values() for t in s})
    data = yf.download(tickers, start=start, group_by="ticker", progress=False)

    # flatten columns
    cleaned = {}
    for t in tickers:
        if t in data:
            df = data[t].dropna()
            if len(df) > 120:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                cleaned[t] = df

    bench = yf.download("^NSEI", start=start, progress=False).dropna()

    dates = bench.index

    # ---------------- Train ML once ----------------
    pool = []
    for df in cleaned.values():
        ds = make_dataset(df)
        if not ds.empty:
            pool.append(ds)

    model = None
    if pool:
        big = pd.concat(pool)
        if len(big) > 300:
            model = train_model(big)

    # ---------------- Weekly loop ----------------
    for i in range(120, len(dates) - HOLD_DAYS - 1, STEP):

        today = dates[i]

        scores = []

        for sector, symbols in sectors.items():
            for sym in symbols:
                if sym not in cleaned:
                    continue

                df = cleaned[sym].loc[:today].copy()
                if len(df) < 60:
                    continue

                df["ret"] = df["Close"].pct_change()

                mom = float(df["Close"].pct_change(20).iloc[-1])
                vol = float(df["ret"].rolling(20).std().iloc[-1])

                if np.isnan(mom) or np.isnan(vol) or vol == 0:
                    continue

                base = mom / (vol + 1e-6)

                ds = make_dataset(df)
                if model is not None and not ds.empty:
                    p = predict_prob(model, ds)
                else:
                    p = 0.5

                scores.append((sector, sym, base + 0.25*(p-0.5), df))

        if not scores:
            continue

        ranked = sorted(scores, key=lambda x: x[2], reverse=True)[:top_n]

        # ---------------- Trades ----------------
        for sector, sym, _, df in ranked:

            hist = df.iloc[:-1]
            if len(hist) < 60:
                continue

            entry = float(hist["Close"].iloc[-1])
            atr = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
            if atr <= 0:
                continue

            stop = entry - atr * atr_mult
            target = entry + atr * atr_mult * rr

            risk_amt = equity * risk
            qty = max(int(risk_amt / max(entry - stop, 0.01)), 1)

            future = cleaned[sym].loc[today:].iloc[1:HOLD_DAYS+1]
            if len(future) < 2:
                continue

            exit_price = float(future["Close"].iloc[-1])
            exit_reason = "TIME"

            for _, r in future.iterrows():
                low = float(r["Low"])
                high = float(r["High"])

                if low <= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    break
                if high >= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    break

            gross_pnl = (exit_price - entry) * qty

            bench_slice = bench.loc[today:].iloc[1:HOLD_DAYS+1]
            bench_ret = float(bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[0] - 1)

            try:
                b = beta(hist["Close"], bench["Close"])
            except:
                b = 0.0

            h = hedge_ratio(b)
            hedged_pnl = gross_pnl - (h * bench_ret * equity)

            weekly_ret = protective_put_proxy(hedged_pnl / equity)
            equity *= (1 + weekly_ret)

            trades.append({
                "date": today,
                "sector": sector,
                "symbol": sym,
                "entry": round(entry,2),
                "stop": round(stop,2),
                "target": round(target,2),
                "qty": qty,
                "exit_price": round(exit_price,2),
                "exit_reason": exit_reason,
                "gross_pnl": round(gross_pnl,2),
                "hedge_ratio": round(h,2),
                "final_week_return": round(weekly_ret*100,2),
                "equity": round(equity,2),
            })

    df_trades = pd.DataFrame(trades)

    if not df_trades.empty:
        winrate = float((df_trades["final_week_return"] > 0).mean()) * 100
    else:
        winrate = 0.0

    return {
        "trades": df_trades,
        "final_equity": round(equity,2),
        "winrate": round(winrate,2),
    }
