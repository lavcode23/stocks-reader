import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# -------------------------------------------------
# Make local modules importable (NO packages)
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "src" / "bharat_sector_demand_hedged_weekly"
sys.path.insert(0, str(MODULE_DIR))

from config import load_config
from backtest import backtest

# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------

st.set_page_config(page_title="Bharat Sector Demand AI", layout="wide")

st.title("🇮🇳 Bharat Sector Demand – Weekly Trading Strategy")

st.markdown("""
This prototype:

• Ranks sectors weekly by momentum  
• Picks top stocks  
• Enters next day OPEN  
• Uses stop-loss + take-profit  
• Exits after ~1 week  
• Runs historical backtest  

⚠️ Research only. Not financial advice.
""")

# -------------------------------------------------
# Load config
# -------------------------------------------------

CONFIG_PATH = ROOT / "config" / "default.yaml"

try:
    cfg = load_config(CONFIG_PATH)
except Exception as e:
    st.error(f"Failed to load config: {e}")
    st.stop()

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------

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
    "Holding Days",
    3,
    10,
    5,
)

run_btn = st.sidebar.button("🚀 Run Backtest")

# -------------------------------------------------
# Run backtest
# -------------------------------------------------

if not run_btn:
    st.info("Adjust parameters in sidebar and click **Run Backtest**.")
    st.stop()

sectors = cfg.get("universe", {}).get("sectors", {})

if not sectors:
    st.error("No sectors defined in config/default.yaml")
    st.stop()

with st.spinner("Running backtest (may take ~1 minute)..."):
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

# -------------------------------------------------
# Results
# -------------------------------------------------

st.subheader("📊 Results")

c1, c2, c3 = st.columns(3)

c1.metric("Total Trades", len(trades))
c2.metric("Win Rate", f"{result['win_rate']*100:.2f}%")
c3.metric("Avg Trade Return", f"{result['avg_return']*100:.2f}%")

# -------------------------------------------------
# Trades table
# -------------------------------------------------

st.subheader("📄 Trades")

if trades.empty:
    st.warning("No trades generated.")
else:
    trades = trades.sort_values("entry_date", ascending=False)
    st.dataframe(trades, use_container_width=True)

# -------------------------------------------------
# Equity curve
# -------------------------------------------------

if not trades.empty:
    st.subheader("📈 Equity Curve (Equal Weight)")

    trades["cum_equity"] = (1 + trades["gross_return"]).cumprod()
    st.line_chart(trades.set_index("entry_date")["cum_equity"])

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("""
---

### Strategy Rules

• Entry: next trading day OPEN  
• Exit: stop-loss / take-profit / time exit  
• Portfolio: equal weight  
• Rebalance: weekly  

Next upgrades:
- volatility filter  
- trend regime  
- hedging  
- position sizing  
""")
