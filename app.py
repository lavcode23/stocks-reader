import streamlit as st
import yaml
import pandas as pd

from backtest import run_backtest

st.set_page_config(layout="wide", page_title="Sector Rotation AI")

st.title("🇮🇳 Daily-First Sector Rotation Strategy (Research Prototype)")

# -------------------------------
# Load config
# -------------------------------

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# -------------------------------
# Sidebar controls
# -------------------------------

st.sidebar.header("Run Controls")

cfg["backtest"]["start"] = st.sidebar.text_input(
    "Backtest Start (YYYY-MM-DD)",
    value=cfg["backtest"]["start"]
)

cfg["signals"]["top_stocks"] = st.sidebar.slider(
    "Top stocks per sector",
    1,
    5,
    int(cfg["signals"]["top_stocks"]),
    1
)

cfg["trade"]["risk_per_trade_pct"] = st.sidebar.slider(
    "Risk per trade (% of equity)",
    0.001,
    0.02,
    float(cfg["trade"]["risk_per_trade_pct"]),
    0.001
)

cfg["trade"]["atr_mult"] = st.sidebar.slider(
    "ATR Multiplier (Stop)",
    1.0,
    4.0,
    float(cfg["trade"]["atr_mult"]),
    0.1
)

cfg["trade"]["take_profit_R"] = st.sidebar.slider(
    "Take Profit (R multiple)",
    0.5,
    3.0,
    float(cfg["trade"]["take_profit_R"]),
    0.1
)

run = st.sidebar.button("🚀 Run Backtest")

st.markdown("""
This prototype:

• Ranks stocks by momentum / volatility per sector  
• Builds ATR-based stop loss  
• Risk-sizes each position  
• Simulates exits at target  
• Reports trades + win rate  

⚠️ Research only. Not financial advice.
""")

# -------------------------------
# Run backtest
# -------------------------------

if not run:
    st.info("Adjust parameters in sidebar and click **Run Backtest**.")
    st.stop()

try:
    result = run_backtest(cfg)
except Exception as e:
    st.exception(e)
    st.stop()

# -------------------------------
# Results
# -------------------------------

trades = result["trades"]
final_equity = result["final_equity"]
winrate = result["winrate"]

c1, c2, c3 = st.columns(3)
c1.metric("Total Trades", 0 if trades is None else len(trades))
c2.metric("Win Rate (%)", winrate)
c3.metric("Final Equity (₹)", final_equity)

st.subheader("📄 Trades")

if trades is None or trades.empty:
    st.warning("No trades generated.")
else:
    st.dataframe(trades, use_container_width=True)

# -------------------------------
# Equity curve
# -------------------------------

if trades is not None and not trades.empty:
    st.subheader("📈 Equity Curve")

    eq = trades[["equity"]].copy()
    eq["step"] = range(len(eq))
    eq = eq.set_index("step")

    st.line_chart(eq)

st.success("Backtest completed.")
