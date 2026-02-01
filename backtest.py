import numpy as np
import pandas as pd

from datafeed import fetch_many
from features import momentum
from sector_model import sector_scores
from ml_ranker import make_dataset, train_model, predict_proba
from strategy_rules import compute_trade_plan
from hedging import beta_to_benchmark, hedge_ratio
from options_proxy import apply_protective_put_proxy

def _weekly_signal_dates(benchmark_df: pd.DataFrame) -> list[pd.Timestamp]:
    idx = pd.to_datetime(benchmark_df.index)
    d = pd.DataFrame(index=idx)
    d["w"] = d.index.to_period("W-FRI")
    return list(d.groupby("w").tail(1).index)

def _slice_upto(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.loc[df.index <= t]

def _next_window(df: pd.DataFrame, start_t: pd.Timestamp, days: int) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    fut = df.loc[df.index > start_t]
    return fut.iloc[:days] if not fut.empty else pd.DataFrame()

def run_backtest(cfg: dict) -> dict:
    sectors = cfg["universe"]["sectors"]

    interval = cfg["data"]["timeframe"]
    start = cfg["backtest"]["start"]
    end = cfg["backtest"].get("end", None)

    momentum_days = int(cfg["data"]["momentum_days"])
    top_sectors_n = int(cfg["signals"]["top_sectors"])
    top_stocks_n = int(cfg["signals"]["top_stocks"])

    ml_enabled = bool(cfg["ml"]["enabled"])
    ml_model_name = str(cfg["ml"]["model"])
    ml_min_rows = int(cfg["ml"]["min_train_rows"])

    entry_mode = str(cfg["trade"]["entry_mode"])
    breakout_buffer_pct = float(cfg["trade"]["breakout_buffer_pct"])
    equity = float(cfg["trade"]["equity_start"])
    risk_per_trade_pct = float(cfg["trade"]["risk_per_trade_pct"])
    stop_mode = str(cfg["trade"]["stop_mode"])
    atr_period = int(cfg["trade"]["atr_period"])
    atr_mult = float(cfg["trade"]["atr_mult"])
    take_profit_R = float(cfg["trade"]["take_profit_R"])
    holding_days = int(cfg["trade"]["holding_days"])

    hedge_enabled = bool(cfg["hedge"]["enabled"])
    hedge_inst = str(cfg["hedge"]["instrument"])
    beta_lb = int(cfg["hedge"]["beta_lookback_days"])
    hedge_cap = float(cfg["hedge"]["hedge_ratio_cap"])

    opt_enabled = bool(cfg["options_proxy"]["enabled"])
    opt_prem_bps = float(cfg["options_proxy"]["premium_bps_per_week"])
    opt_crash = float(cfg["options_proxy"]["crash_protection_pct"])

    slippage_bps = float(cfg["costs"]["slippage_bps"])
    fee_bps = float(cfg["costs"]["fee_bps"])
    roundtrip_cost = 2.0 * (slippage_bps + fee_bps) / 10000.0

    # ---- Fetch all data ----
    all_stocks = sorted({t for s in sectors.values() for t in s})
    tickers = all_stocks + [hedge_inst]
    data = fetch_many(tickers, start=start, end=end, interval=interval)

    bench = data.get(hedge_inst, pd.DataFrame())
    if bench.empty:
        raise RuntimeError(f"Benchmark/hedge instrument has no data: {hedge_inst}")

    signal_dates = _weekly_signal_dates(bench)

    trade_rows = []
    plan_rows = []
    equity_rows = []

    for sd in signal_dates:
        # 1) compute each stock's 5d momentum as of signal date (no lookahead)
        returns_5d = {}
        latest_features_rows = {}  # for ML prediction per ticker

        for t in all_stocks:
            hist = _slice_upto(data.get(t, pd.DataFrame()), sd)
            if hist.empty or len(hist) <= max(25, momentum_days + 2):
                continue
            hist = hist.dropna()
            if len(hist) <= momentum_days:
                continue

            ret5 = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-momentum_days] - 1)
            returns_5d[t] = ret5

            # build dataset row for ML from recent history
            ds = make_dataset(hist)
            if not ds.empty:
                latest_features_rows[t] = ds.tail(1)

        # 2) sector demand scoring (market-internal)
        s_scores = sector_scores(sectors, returns_5d)
        ranked_sectors = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_sectors = [s for s, _ in ranked_sectors[:top_sectors_n]]

        # 3) candidate stocks = constituents of chosen sectors
        candidates = []
        for s in chosen_sectors:
            for t in sectors[s]:
                if t in returns_5d:
                    candidates.append((s, t, returns_5d[t]))

        if not candidates:
            continue

        # 4) ML ranking (optional): train one pooled model up to sd
        ml_model = None
        if ml_enabled:
            pooled = []
            for t in all_stocks:
                hist = _slice_upto(data.get(t, pd.DataFrame()), sd)
                if hist.empty:
                    continue
                ds = make_dataset(hist)
                if not ds.empty:
                    pooled.append(ds)
            if pooled:
                pool_df = pd.concat(pooled, axis=0, ignore_index=True)
                if len(pool_df) >= ml_min_rows:
                    ml_model = train_model(pool_df, model_name=ml_model_name)

        scored = []
        for s, t, m5 in candidates:
            p_up = None
            if ml_model is not None and t in latest_features_rows:
                try:
                    p_up = predict_proba(ml_model, latest_features_rows[t])
                except Exception:
                    p_up = None
            # final score: momentum + ML boost
            score = float(m5 + (0.25 * (p_up - 0.5) if p_up is not None else 0.0))
            scored.append((s, t, score, m5, p_up))

        scored.sort(key=lambda x: x[2], reverse=True)
        picks = scored[:top_stocks_n]
        if not picks:
            continue

        # 5) build trade plans (entry/stop/take/qty/₹risk) using data up to sd
        # allocate equity across picks (equal risk budget)
        per_trade_equity = equity / len(picks)

        bench_hist = _slice_upto(bench, sd)
        if bench_hist.empty:
            continue

        # estimate portfolio beta (avg of stock betas) for hedge ratio
        betas = []
        for _, t, *_ in picks:
            hist = _slice_upto(data.get(t, pd.DataFrame()), sd)
            if hist.empty:
                continue
            try:
                b = beta_to_benchmark(hist["Close"], bench_hist["Close"], lookback=beta_lb)
                betas.append(b)
            except Exception:
                pass

        port_beta = float(np.mean(betas)) if betas else 0.0
        h_ratio = hedge_ratio(port_beta, cap=hedge_cap) if hedge_enabled else 0.0

        # next-week benchmark return for hedge P&L approximation
        bench_fut = _next_window(bench, sd, holding_days + 1)
        bench_week_ret = 0.0
        if not bench_fut.empty and len(bench_fut) >= 2:
            bench_week_ret = float(bench_fut["Close"].iloc[-1] / bench_fut["Open"].iloc[0] - 1)

        # simulate each stock trade over next window
        pos_rets = []
        week_trades = []

        for s, t, final_score, m5, p_up in picks:
            hist = _slice_upto(data.get(t, pd.DataFrame()), sd)
            fut = _next_window(data.get(t, pd.DataFrame()), sd, holding_days + 1)

            if hist.empty or fut.empty or len(fut) < 2:
                continue

            # build plan using last bar of hist
            plan = compute_trade_plan(
                df=hist.tail(200),
                equity=per_trade_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                entry_mode=entry_mode,
                breakout_buffer_pct=breakout_buffer_pct,
                stop_mode=stop_mode,
                atr_period=atr_period,
                atr_mult=atr_mult,
                take_profit_R=take_profit_R,
            )

            entry = float(plan["entry"])
            stop = float(plan["stop"])
            take = float(plan["take"])
            qty = int(plan["qty"])
            sl_amount = float(plan["stop_loss_amount"])

            # simulate fill + intraday rules using future daily candles
            # entry day: first future row open/high/low
            entry_day = fut.iloc[0]

            filled = False
            entry_px = None
            entry_date = fut.index[0]
            # breakout fill logic: must trade through trigger
            if entry_mode == "open":
                filled = True
                entry_px = float(entry_day["Open"])
            else:
                trigger = float(plan["trigger"])
                if float(entry_day["High"]) >= trigger:
                    filled = True
                    entry_px = trigger

            if not filled:
                # no trade
                plan_rows.append({
                    "signal_date": sd, "sector": s, "ticker": t,
                    "final_score": final_score, "momentum5d": m5, "ml_p_up": p_up,
                    "status": "NOT_FILLED",
                    "entry": entry, "stop": stop, "take": take, "qty": qty,
                    "stop_loss_amount": sl_amount
                })
                continue

            # now walk forward day by day from entry day to holding window end
            exit_reason = "TIME_EXIT"
            exit_date = fut.index[min(len(fut)-1, holding_days)]
            exit_px = float(fut.iloc[min(len(fut)-1, holding_days)]["Close"])

            # conservative: stop checked before take
            # note: if stop/take hit on entry day after fill, we still detect using Low/High
            for i in range(0, min(len(fut), holding_days + 1)):
                row = fut.iloc[i]
                d = fut.index[i]
                low = float(row["Low"])
                high = float(row["High"])
                if low <= stop:
                    exit_reason = "STOP"
                    exit_date = d
                    exit_px = stop
                    break
                if high >= take:
                    exit_reason = "TAKE"
                    exit_date = d
                    exit_px = take
                    break

            gross_ret = float(exit_px / entry_px - 1.0)
            net_ret = gross_ret - roundtrip_cost

            week_trades.append({
                "signal_date": sd,
                "sector": s,
                "ticker": t,
                "entry_date": pd.to_datetime(entry_date),
                "exit_date": pd.to_datetime(exit_date),
                "entry_price": float(entry_px),
                "stop_price": stop,
                "take_price": take,
                "qty": qty,
                "stop_loss_amount": sl_amount,
                "exit_price": float(exit_px),
                "exit_reason": exit_reason,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "momentum5d": m5,
                "ml_p_up": p_up,
                "final_score": final_score,
            })

            plan_rows.append({
                "signal_date": sd,
                "sector": s,
                "ticker": t,
                "final_score": final_score,
                "momentum5d": m5,
                "ml_p_up": p_up,
                "status": "FILLED",
                "entry": float(entry_px),
                "stop": stop,
                "take": take,
                "qty": qty,
                "stop_loss_amount": sl_amount,
                "risk_rupees": float(plan["risk_rupees"]),
            })

            pos_rets.append(net_ret)

        if not pos_rets:
            continue

        # equal-weight portfolio return
        port_ret = float(np.mean(pos_rets))

        # hedge (short benchmark proxy) reduces market exposure
        hedged_ret = port_ret - (h_ratio * bench_week_ret)

        # options proxy (insurance premium + crash cap)
        final_week_ret = hedged_ret
        if opt_enabled:
            final_week_ret = apply_protective_put_proxy(
                weekly_return=hedged_ret,
                premium_bps=opt_prem_bps,
                crash_protect_pct=opt_crash,
            )

        # update equity
        equity *= (1.0 + final_week_ret)

        for tr in week_trades:
            tr["portfolio_beta_est"] = port_beta
            tr["hedge_ratio"] = h_ratio
            tr["bench_week_ret"] = bench_week_ret
            tr["portfolio_week_ret_before_hedge"] = port_ret
            tr["portfolio_week_ret_after_hedge"] = hedged_ret
            tr["portfolio_week_ret_final"] = final_week_ret
            trade_rows.append(tr)

        equity_rows.append({
            "signal_date": sd,
            "equity": equity,
            "port_beta_est": port_beta,
            "hedge_ratio": h_ratio,
            "bench_week_ret": bench_week_ret,
            "portfolio_ret_before_hedge": port_ret,
            "portfolio_ret_after_hedge": hedged_ret,
            "portfolio_ret_final": final_week_ret,
        })

    trades_df = pd.DataFrame(trade_rows)
    plans_df = pd.DataFrame(plan_rows)
    equity_df = pd.DataFrame(equity_rows)

    if trades_df.empty:
        summary = {
            "trades": 0,
            "win_rate": 0.0,
            "final_equity": equity,
            "total_return": 0.0,
        }
        return {"trade_plans": plans_df, "trades": trades_df, "equity": equity_df, "summary": summary}

    trades_df["win"] = trades_df["net_return"] > 0
    win_rate = float(trades_df["win"].mean())

    total_return = float(equity / float(cfg["trade"]["equity_start"]) - 1.0) if cfg["trade"]["equity_start"] else 0.0

    summary = {
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "final_equity": float(equity),
        "total_return": total_return,
        "hedge_enabled": hedge_enabled,
        "ml_enabled": ml_enabled,
        "options_proxy_enabled": opt_enabled,
    }

    return {"trade_plans": plans_df, "trades": trades_df, "equity": equity_df, "summary": summary}
