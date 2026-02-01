import streamlit as st
import pandas as pd
from pathlib import Path

from bharat_sector_demand_hedged_weekly.config import load_config
from bharat_sector_demand_hedged_weekly.backtest import backtest

# ---------------- Page setup ----------------

st.set_page_config(page_title="Bharat Sector Demand AI", layout="wide")

st.title("🇮🇳 Bharat Sector Demand – Weekly Trading Strategy")

st.markdown("""
This app:

• Ranks sectors weekly by momentum  
• Picks top stocks  
• Enters next day open  
• Uses stop-loss + take-profit  
• Exits after ~1 week if neither hit  
• Backtests the logic  

⚠️ Research prototype only — NOT financial advice.
""")

# ---------------- Load config ----------------

CONFIG_PATH = Path("config/default.yaml")

try:
    cfg = load_config(CONFIG_PATH)
except Exception as e:
    st.error(f"Failed to load config: {e}")
    st.stop()

# ---------------- Sidebar controls ----------------

st.sidebar.header("Backtest Controls")

start_date = st.sidebar.text_input(
    "Backtest start (YYYY-MM-DD)",
    value=cfg.get("backtest", {}).get("start", "2018-01-01"),
)

stop_loss = st.sidebar.slider(
    "Stop Loss %",
    0.005,
    0.10,
    float(cfg.get("trade", {}).get("stop_loss_pct", 0.02)),
    0.005,
)

take_profit = st.sidebar.slider(
    "Take Profit %",
    0.005,
    0.10,
    float(cfg.get("trade", {}).get("take_profit_pct", 0.02)),
    0.005,
)

holding_days = st.sidebar.slider(
    "Holding days (approx 1 week)",
    3,
    10,
    5,
)

run_btn = st.sidebar.button("🚀 Run Backtest")

# ---------------- Main logic ----------------

if not run_btn:
    st.info("Adjust parameters in sidebar and click **Run Backtest**.")
    st.stop()

sectors = cfg.get("universe", {}).get("sectors", {})

if not sectors:
    st.error("No sectors found in config/default.yaml")
    st.stop()

with st.spinner("Running backtest (this may take 1–2 minutes)..."):
    try:
        result = backtest(
            sectors=sectors,
            start=start_date,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_days=holding_days,
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

trades = result["trades"]

# ---------------- Results ----------------

st.subheader("📊 Results")

c1, c2, c3 = st.columns(3)

c1.metric("Total Trades", len(trades))
c2.metric("Win Rate", f"{result['win_rate']*100:.2f}%")
c3.metric("Avg Trade Return", f"{result['avg_return']*100:.2f}%")

# ---------------- Trades table ----------------

st.subheader("📄 Trades")

if trades.empty:
    st.warning("No trades generated.")
else:
    trades = trades.sort_values("entry_date", ascending=False)
    st.dataframe(trades, use_container_width=True)

# ---------------- Simple equity curve ----------------

if not trades.empty:
    st.subheader("📈 Equity Curve (Equal Weight)")

    trades["cum"] = (1 + trades["gross_return"]).cumprod()

    st.line_chart(trades.set_index("entry_date")["cum"])

# ---------------- Footer ----------------

st.markdown("""
---

### Notes

• Entry = next trading day OPEN  
• Exit = stop-loss / take-profit / time exit  
• Win rate shown is REAL historical result  
• No guarantees in markets  

Next improvements:
- volatility filter  
- trend regime filter  
- hedging  
- position sizing  
""")
