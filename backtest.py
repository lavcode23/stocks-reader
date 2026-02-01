import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest(cfg: dict):

    start = cfg["backtest"]["start"]
    sectors = cfg["universe"]["sectors"]

    top_n = cfg["signals"]["top_stocks"]
    equity = float(cfg["trade"]["equity_start"])
    risk = float(cfg["trade"]["risk_per_trade_pct"])
    atr_mult = float(cfg["trade"]["atr_mult"])
    rr = float(cfg["trade"]["take_profit_R"])

    trades = []

    for sector, symbols in sectors.items():

        scores = []

        for sym in symbols:
            try:
                df = yf.download(sym, start=start, progress=False)

                if len(df) < 60:
                    continue

                df["ret"] = df["Close"].pct_change()
                mom = df["Close"].pct_change(20).iloc[-1]
                vol = df["ret"].rolling(20).std().iloc[-1]

                score = mom / (vol + 1e-6)
                scores.append((sym, score, df))

            except:
                continue

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        for sym, _, df in scores:

            entry = float(df["Close"].iloc[-1])
            atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

            stop = entry - atr * atr_mult
            target = entry + atr * atr_mult * rr

            risk_amt = equity * risk
            qty = max(int(risk_amt / max(entry - stop, 0.01)), 1)

            exit_price = target
            pnl = (exit_price - entry) * qty

            equity += pnl

            trades.append({
                "sector": sector,
                "symbol": sym,
                "entry": round(entry,2),
                "stop": round(stop,2),
                "target": round(target,2),
                "qty": qty,
                "pnl": round(pnl,2),
                "equity": round(equity,2)
            })

    df_trades = pd.DataFrame(trades)

    if not df_trades.empty:
        winrate = float((df_trades.pnl > 0).mean())
    else:
        winrate = 0.0

    return {
        "trades": df_trades,
        "final_equity": round(equity,2),
        "winrate": round(winrate*100,2)
    }
