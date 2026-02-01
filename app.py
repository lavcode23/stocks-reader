import streamlit as st
import yaml
import pandas as pd

from backtest import run_backtest
from realtime_engine import run_realtime

st.set_page_config(layout="wide", page_title="AI Trading System")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ACCOUNT = 100000

st.title("🇮🇳 AI Sector Rotation Trading System")

tab_live, tab_backtest, tab_settings = st.tabs(
    ["🚦 Live Trading", "🧪 Research Backtest", "⚙️ Settings"]
)

# =====================================================
# LIVE TAB
# =====================================================

with tab_live:

    st.subheader("📌 Portfolio Snapshot")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Account Equity", f"₹{ACCOUNT}")
    c2.metric("Risk / Trade", f"{cfg['trade']['risk_per_trade_pct']*100:.2f}%")
    c3.metric("Max Positions", "5")
    c4.metric("RR", cfg["trade"]["take_profit_R"])

    if st.button("🔄 Refresh Live Signals"):

        with st.spinner("Generating signals..."):
            df = run_realtime()

        if df.empty:
            st.warning("No signals today.")
        else:

            # Derived trader info
            df["risk_per_trade_₹"] = (
                ACCOUNT * cfg["trade"]["risk_per_trade_pct"]
            ).round(0)

            df["R_multiple"] = (
                (df["target"] - df["entry"]) /
                (df["entry"] - df["stop"])
            ).round(2)

            df["position_value"] = (df["entry"] * df["qty"]).round(0)

            st.subheader("🚦 Today’s Trade Plan")

            st.dataframe(df, use_container_width=True)

            st.subheader("📊 Exposure Summary")

            e1,e2,e3 = st.columns(3)
            e1.metric("Total Positions", len(df))
            e2.metric("Total Exposure ₹", int(df["position_value"].sum()))
            e3.metric("Total Risk ₹", int(df["risk_per_trade_₹"].sum()))

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download Signals CSV", csv, "signals.csv")

# =====================================================
# BACKTEST TAB
# =====================================================

with tab_backtest:

    st.subheader("🧪 Strategy Research")

    if st.button("Run Backtest"):

        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]

        c1,c2,c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        if not trades.empty:

            st.subheader("📄 Trades")
            st.dataframe(trades, use_container_width=True)

            st.subheader("📈 Equity Curve")
            eq = trades[["equity"]].copy()
            eq["i"] = range(len(eq))
            st.line_chart(eq.set_index("i"))

            st.subheader("📉 Exit Breakdown")
            st.bar_chart(trades["exit_reason"].value_counts())

# =====================================================
# SETTINGS TAB
# =====================================================

with tab_settings:

    st.subheader("⚙️ Strategy Parameters")

    cfg["backtest"]["start"] = st.text_input(
        "Backtest Start",
        value=cfg["backtest"]["start"]
    )

    cfg["signals"]["top_stocks"] = st.slider(
        "Top Stocks",
        1,5,
        cfg["signals"]["top_stocks"]
    )

    cfg["trade"]["risk_per_trade_pct"] = st.slider(
        "Risk per Trade %",
        0.001,0.01,
        cfg["trade"]["risk_per_trade_pct"],
        0.001
    )

    cfg["trade"]["atr_mult"] = st.slider(
        "ATR Multiplier",
        1.0,4.0,
        cfg["trade"]["atr_mult"],
        0.1
    )

    cfg["trade"]["take_profit_R"] = st.slider(
        "Reward Ratio",
        1.0,3.0,
        cfg["trade"]["take_profit_R"],
        0.1
    )

    st.info("Settings apply on next refresh / backtest.")
