import streamlit as st
import yaml
import pandas as pd
from backtest import run_backtest

st.set_page_config(layout="wide")
st.title("🇮🇳 Sector Rotation Weekly Strategy")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

sl = st.sidebar.slider("Stop Loss %", 0.005, 0.1, cfg["trade"]["stop_loss_pct"], 0.005)
tp = st.sidebar.slider("Take Profit %", 0.005, 0.1, cfg["trade"]["take_profit_pct"], 0.005)

if st.sidebar.button("Run Backtest"):

    with st.spinner("Running..."):
        trades, win = run_backtest(
            cfg["universe"]["sectors"],
            cfg["backtest"]["start"],
            sl,
            tp
        )

    st.metric("Win Rate", f"{win*100:.2f}%")

    if not trades.empty:
        trades["equity"] = (1 + trades["return"]).cumprod()
        st.line_chart(trades.set_index("entry_date")["equity"])
        st.dataframe(trades)
    else:
        st.warning("No trades")
