import streamlit as st
import yaml
import pandas as pd
import yfinance as yf
from datetime import datetime

from backtest import run_backtest
from realtime_engine import run_realtime

st.set_page_config(layout="wide", page_title="AI Trading Terminal")

ACCOUNT = 100000

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

st.title("📊 AI Trading Terminal")

tab_live, tab_positions, tab_research, tab_settings = st.tabs(
    ["🚦 Live", "📂 Positions", "🧪 Research", "⚙️ Settings"]
)

# ======================================================
# LIVE TERMINAL
# ======================================================

with tab_live:

    st.subheader("Portfolio Snapshot")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Account", f"₹{ACCOUNT}")
    c2.metric("Risk / Trade", f"{cfg['trade']['risk_per_trade_pct']*100:.2f}%")
    c3.metric("RR", cfg["trade"]["take_profit_R"])
    c4.metric("Top Stocks", cfg["signals"]["top_stocks"])

    if st.button("🔄 Generate Signals"):

        with st.spinner("Running realtime engine..."):
            df = run_realtime()

        if df.empty:
            st.warning("No signals.")
            st.stop()

        # Derived metrics
        df["risk_₹"] = (ACCOUNT * cfg["trade"]["risk_per_trade_pct"]).round(0)
        df["R"] = ((df["target"]-df["entry"])/(df["entry"]-df["stop"])).round(2)
        df["position_value"] = (df["entry"] * df["qty"]).round(0)

        st.subheader("Today Trade Plan")

        st.dataframe(df, use_container_width=True)

        e1,e2,e3 = st.columns(3)
        e1.metric("Positions", len(df))
        e2.metric("Total Exposure ₹", int(df["position_value"].sum()))
        e3.metric("Total Risk ₹", int(df["risk_₹"].sum()))

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Trades", csv, "today_trades.csv")

        st.session_state["live_positions"] = df

# ======================================================
# POSITIONS (Paper)
# ======================================================

with tab_positions:

    st.subheader("Paper Positions")

    if "live_positions" not in st.session_state:
        st.info("Generate signals first.")
        st.stop()

    pos = st.session_state["live_positions"].copy()

    prices = yf.download(pos["symbol"].tolist(), period="1d", interval="1m", group_by="ticker", progress=False)

    pnl = []

    for _, r in pos.iterrows():
        try:
            p = prices[r["symbol"]]["Close"].iloc[-1]
            pnl.append((p - r["entry"]) * r["qty"])
        except:
            pnl.append(0)

    pos["current_pnl"] = pnl

    st.dataframe(pos, use_container_width=True)

    st.metric("Total Unrealized PnL ₹", int(sum(pnl)))

# ======================================================
# RESEARCH
# ======================================================

with tab_research:

    if st.button("Run Backtest"):

        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]

        c1,c2,c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        st.subheader("Trades")
        st.dataframe(trades, use_container_width=True)

        st.subheader("Equity Curve")
        eq = trades[["equity"]].copy()
        eq["i"] = range(len(eq))
        st.line_chart(eq.set_index("i"))

        st.subheader("Exit Reasons")
        st.bar_chart(trades["exit_reason"].value_counts())

# ======================================================
# SETTINGS
# ======================================================

with tab_settings:

    st.subheader("Strategy Controls")

    cfg["signals"]["top_stocks"] = st.slider("Top Stocks",1,5,cfg["signals"]["top_stocks"])
    cfg["trade"]["risk_per_trade_pct"] = st.slider("Risk %",0.001,0.01,cfg["trade"]["risk_per_trade_pct"],0.001)
    cfg["trade"]["atr_mult"] = st.slider("ATR Mult",1.0,4.0,cfg["trade"]["atr_mult"],0.1)
    cfg["trade"]["take_profit_R"] = st.slider("Reward Ratio",1.0,3.0,cfg["trade"]["take_profit_R"],0.1)

    st.info("Settings apply next run.")
