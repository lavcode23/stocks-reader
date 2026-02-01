import streamlit as st
import yaml
import pandas as pd

from backtest import run_backtest
from export_excel import export_results_to_bytes

st.set_page_config(layout="wide", page_title="Sector Rotation AI")

st.title("🇮🇳 Daily-First Sector Rotation Strategy (Research Prototype)")

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

st.sidebar.header("Run Controls")

start = st.sidebar.text_input("Backtest Start (YYYY-MM-DD)", value=cfg["backtest"]["start"])
cfg["backtest"]["start"] = start

cfg["ml"]["enabled"] = st.sidebar.checkbox("Enable ML Ranking", value=cfg["ml"]["enabled"])
cfg["hedge"]["enabled"] = st.sidebar.checkbox("Enable Beta Hedge", value=cfg["hedge"]["enabled"])
cfg["options_proxy"]["enabled"] = st.sidebar.checkbox("Enable Options Proxy", value=cfg["options_proxy"]["enabled"])

cfg["trade"]["entry_mode"] = st.sidebar.selectbox("Entry Mode", ["breakout", "open"], index=0 if cfg["trade"]["entry_mode"]=="breakout" else 1)
cfg["trade"]["stop_mode"] = st.sidebar.selectbox("Stop Mode", ["atr", "swing"], index=0 if cfg["trade"]["stop_mode"]=="atr" else 1)

cfg["trade"]["risk_per_trade_pct"] = st.sidebar.slider("Risk per trade (% of equity)", 0.001, 0.02, float(cfg["trade"]["risk_per_trade_pct"]), 0.001)
cfg["trade"]["atr_mult"] = st.sidebar.slider("ATR Multiplier (stop)", 1.0, 4.0, float(cfg["trade"]["atr_mult"]), 0.1)
cfg["trade"]["take_profit_R"] = st.sidebar.slider("Take Profit (R multiple)", 0.5, 3.0, float(cfg["trade"]["take_profit_R"]), 0.1)
cfg["signals"]["top_stocks"] = st.sidebar.slider("Top Stocks", 1, 6, int(cfg["signals"]["top_stocks"]), 1)

run = st.sidebar.button("🚀 Run Backtest")

st.markdown("""
This system outputs **trade plans** with:

- **Entry price** (open or breakout trigger)
- **Stop-loss price**
- **Stop-loss ₹ amount** (position sizing based on risk)
- **Take-profit price**
- **Quantity**
- **Hedge ratio** (beta-based)
- **Options-proxy** adjusted portfolio return

⚠️ No strategy can guarantee win rate. This app measures it on history.
""")

if not run:
    st.info("Adjust sidebar settings and click **Run Backtest**.")
    st.stop()

with st.spinner("Running full pipeline (daily-first)…"):
    try:
    result = run_backtest(cfg)
except Exception as e:
    st.exception(e)
    st.stop()


plans = result["trade_plans"]
trades = result["trades"]
equity = result["equity"]
summary = result["summary"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", summary.get("trades", 0))
c2.metric("Win rate", f"{summary.get('win_rate', 0.0)*100:.2f}%")
c3.metric("Final equity", f"₹{summary.get('final_equity', 0.0):,.0f}")
c4.metric("Total return", f"{summary.get('total_return', 0.0)*100:.2f}%")

st.subheader("📌 Trade Plans (Entry / SL / SL ₹ / TP / Qty)")
if not plans.empty:
    st.dataframe(plans.sort_values(["signal_date","final_score"], ascending=[False, False]), use_container_width=True)
else:
    st.warning("No plans generated.")

st.subheader("🧾 Executed Trades (Backtest)")
if not trades.empty:
    st.dataframe(trades.sort_values(["signal_date","ticker"], ascending=[False, True]), use_container_width=True)
else:
    st.warning("No trades executed.")

st.subheader("📈 Equity Curve")
if not equity.empty:
    chart = equity.copy()
    chart["signal_date"] = pd.to_datetime(chart["signal_date"])
    chart = chart.sort_values("signal_date")
    st.line_chart(chart.set_index("signal_date")["equity"])
else:
    st.warning("No equity curve.")

# Excel export
xlsx_bytes = export_results_to_bytes(plans, trades, equity, summary)
st.download_button(
    label="⬇️ Download Excel Report",
    data=xlsx_bytes,
    file_name="sector_rotation_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
