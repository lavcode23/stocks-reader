import streamlit as st
import pandas as pd
from pathlib import Path
import importlib.util

# -------------------------------------------------
# Absolute module loader (no Python imports)
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src" / "bharat_sector_demand_hedged_weekly"

def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

config = load_module("config", BASE / "config.py")
backtest_mod = load_module("backtest", BASE / "backtest.py")

load_config = config.load_config
backtest = backtest_mod.backtest

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(page_title="Bharat Sector Demand AI", layout="wide")
st.title("🇮🇳 Bharat Sector Demand – Weekly Strategy")

ROOT_CONFIG = ROOT / "config" / "default.yaml"
cfg = load_config(ROOT_CONFIG)

st.sidebar.header("Backtest Controls")

start = st.sidebar.text_input("Start date", cfg["backtest"]["start"])
stop_loss = st.sidebar.slider("Stop Loss %", 0.005, 0.1, float(cfg["trade"]["stop_loss_pct"]), 0.005)
take_profit = st.sidebar.slider("Take Profit %", 0.005, 0.1, float(cfg["trade"]["take_profit_pct"]), 0.005)
holding_days = st.sidebar.slider("Holding Days", 3, 10, 5)

if st.sidebar.button("Run Backtest"):

    with st.spinner("Running..."):
        res = backtest(
            sectors=cfg["universe"]["sectors"],
            start=start,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_days=holding_days,
        )

    trades = res["trades"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", len(trades))
    c2.metric("Win Rate", f"{res['win_rate']*100:.2f}%")
    c3.metric("Avg Return", f"{res['avg_return']*100:.2f}%")

    st.subheader("Trades")
    st.dataframe(trades)

    if not trades.empty:
        trades["equity"] = (1 + trades["gross_return"]).cumprod()
        st.line_chart(trades.set_index("entry_date")["equity"])

else:
    st.info("Press Run Backtest")
