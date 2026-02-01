import yfinance as yf
import pandas as pd
import numpy as np

from ml_ranker import make_dataset, train_model, predict_prob
from hedging import beta, hedge_ratio
from options_proxy import protective_put_proxy


def run_backtest(cfg: dict):

    start = cfg["backtest"]["start"]
    sectors = cfg["universe"]["sectors"]

    top_n = int(cfg["signals"]["top_stocks"])
    equity = float(cfg["trade"]["equity_start"])
    risk = float(cfg["trade"]["risk_per_trade_pct"])
    atr_mult = float(cfg["trade"]["atr_mult"])
    rr = float(cfg["trade"]["take_profit_R"])

    trades = []

    # NIFTY benchmark
    bench = yf.download("^NSEI", start=start, progress=False).dropna()

    for sector, symbols in sectors.items():

        scores = []

        # -------------------------------
        # Momentum + volatility ranking
        # -------------------------------
        for sym in symbols:
            try:
                df = yf.download(sym, start=start, progress=False)

                if df is None or len(df) < 120:
                    continue

                df = df.dropna()
                df["ret"] = df["Close"].pct_change()

                mom = float(df["Close"].pct_change(20).iloc[-1])
                vol = float(df["ret"].rolling(20).std().iloc[-1])

                if np.isnan(mom) or np.isnan(vol) or vol == 0:
                    continue

                base_score = float(mom / (vol + 1e-6))
                scores.append((sym, base_score, df))

            except Exception:
                continue

        if not scores:
            continue

        # -------------------------------
        # ML ranking
        # -------------------------------
        pool = []
        for _, _, d in scores:
            ds = make_dataset(d)
            if not ds.empty:
                pool.append(ds)

        model = None
        if pool:
            big = pd.concat(pool)
            if len(big) > 300:
                model = train_model(big)

        ranked = []
        for sym, base_score, df in scores:
            ds = make_dataset(df)
            if model is not None and not ds.empty:
                p = predict_prob(model, ds)
            else:
                p = 0.5
            final_score = float(base_score + 0.25 * (p - 0.5))
            ranked.append((sym, final_score, df))

        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]

        # -------------------------------
        # Trade simulation (daily)
        # -------------------------------
        for sym, _, df in ranked:

            entry = float(df["Close"].iloc[-1])
            atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

            if np.isnan(atr) or atr == 0:
                continue

            stop = entry - atr * atr_mult
            target = entry + atr * atr_mult * rr

            risk_amt = equity * risk
            qty = max(int(risk_amt / max(entry - stop, 0.01)), 1)

            # ---- realistic exit over next 5 days ----
            future = df.tail(6).copy()

            exit_price = float(future["Close"].iloc[-1])
            exit_reason = "TIME"

            for _, r in future.iterrows():
                if float(r["Low"]) <= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    break
                if float(r["High"]) >= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    break

            gross_pnl = (exit_price - entry) * qty

            # -------------------------------
            # Hedge vs NIFTY
            # -------------------------------
            try:
                b = beta(df["Close"], bench["Close"])
            except Exception:
                b = 0.0

            h = hedge_ratio(b)
            bench_ret = float(bench["Close"].pct_change().iloc[-1]) if len(bench) > 2 else 0.0
            hedged_pnl = gross_pnl - (h * bench_ret * equity)

            # -------------------------------
            # Options proxy
            # -------------------------------
            weekly_ret = hedged_pnl / equity
            weekly_ret = protective_put_proxy(weekly_ret)

            equity *= (1 + weekly_ret)

            trades.append({
                "sector": sector,
                "symbol": sym,
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "target": round(target, 2),
                "qty": qty,
                "exit_price": round(exit_price, 2),
                "exit_reason": exit_reason,
                "gross_pnl": round(gross_pnl, 2),
                "hedge_ratio": round(h, 2),
                "final_week_return": round(weekly_ret * 100, 2),
                "equity": round(equity, 2),
            })

    df_trades = pd.DataFrame(trades)

    if not df_trades.empty:
        winrate = float((df_trades["final_week_return"] > 0).mean()) * 100.0
    else:
        winrate = 0.0

    return {
        "trades": df_trades,
        "final_equity": round(equity, 2),
        "winrate": round(winrate, 2),
    }
