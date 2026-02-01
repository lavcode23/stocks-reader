import streamlit as st
import yaml
import pandas as pd

from backtest import run_backtest
from realtime_engine import run_realtime

st.set_page_config(layout="wide", page_title="AI Sector Rotation")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

st.title("🇮🇳 AI Sector Rotation System")

tab1, tab2 = st.tabs(["🚦 Live Signals", "🧪 Research Backtest"])

# =====================================================
# LIVE TAB
# =====================================================

with tab1:

    st.subheader("Today's Trade Signals")

    col1,col2,col3 = st.columns(3)
    col1.metric("Account Equity", "₹100,000")
    col2.metric("Risk / Trade", "0.3%")
    col3.metric("Positions", "Top 5")

    if st.button("🔄 Refresh Signals"):

        with st.spinner("Fetching live market data..."):
            df = run_realtime()

        if df.empty:
            st.warning("No signals today.")
        else:
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv, "signals.csv")

# =====================================================
# BACKTEST TAB
# =====================================================

with tab2:

    st.subheader("Strategy Backtest")

    cfg["backtest"]["start"] = st.text_input(
        "Backtest Start",
        value=cfg["backtest"]["start"]
    )

    run = st.button("Run Backtest")

    if run:

        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]

        c1,c2,c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        st.dataframe(trades, use_container_width=True)

        if not trades.empty:
            eq = trades[["equity"]].copy()
            eq["i"] = range(len(eq))
            eq = eq.set_index("i")
            st.line_chart(eq)

