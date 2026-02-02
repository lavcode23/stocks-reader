import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from pathlib import Path

from realtime_engine import run_realtime
from backtest import run_backtest
from journal import (
    load_journal,
    append_trades,
    auto_fill_open_trades,
    performance_stats,
)

# ======================================================
# AUTO BOOTSTRAP (demo journal so UI is never empty)
# ======================================================
DEMO_PATH = Path("paper_journal.csv")

if not DEMO_PATH.exists():
    demo = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=15).astype(str),
        "symbol": ["DEMO"]*15,
        "sector": ["Demo"]*15,
        "entry": np.random.uniform(90,110,15),
        "stop": np.random.uniform(85,95,15),
        "target": np.random.uniform(115,130,15),
        "qty": [10]*15,
        "ml_prob": np.random.uniform(0.55,0.7,15),
        "R_multiple": np.random.uniform(1.5,2.5,15),
        "max_loss_₹": np.random.uniform(200,500,15),
        "status": ["CLOSED"]*15,
        "entry_filled_price": np.random.uniform(90,110,15),
        "exit_price": np.random.uniform(95,125,15),
        "pnl": np.random.uniform(-200,500,15)
    })
    demo.to_csv(DEMO_PATH, index=False)

# ======================================================
# PAGE + CONFIG
# ======================================================
st.set_page_config(layout="wide", page_title="Pro Trading Cockpit")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ACCOUNT_DEFAULT = 100000.0

st.title("📊 Pro Trading Cockpit")

tab_live, tab_journal, tab_backtest, tab_settings, tab_explain = st.tabs(
    ["🚦 Live", "📒 Journal", "🧪 Backtest", "⚙️ Settings", "📘 Explain"]
)

# ======================================================
# LIVE
# ======================================================
with tab_live:
    st.subheader("Live Trading")

    c1, c2 = st.columns(2)
    account = c1.number_input("Account Equity (₹)", min_value=10000.0, value=ACCOUNT_DEFAULT, step=1000.0)
    max_positions = c2.selectbox("Max Trades Today", [1,2,3,4,5], index=2)

    if st.button("🔄 Generate Signals"):
        with st.spinner("Generating live signals..."):
            out = run_realtime(account_equity=float(account), max_positions=int(max_positions))
        st.session_state["live"] = out

    if "live" not in st.session_state:
        st.info("Click **Generate Signals** to start.")
    else:
        df = st.session_state["live"]["signals"]
        st.info(st.session_state["live"]["note"])

        if df.empty:
            st.warning("No trades today.")
        else:
            st.dataframe(df, use_container_width=True)

            if st.button("📒 Add to Journal"):
                append_trades(df)
                st.success("Signals added to journal.")

# ======================================================
# JOURNAL
# ======================================================
with tab_journal:
    st.subheader("Paper Trade Journal")

    j1, j2, j3, j4 = st.columns(4)

    if j1.button("⚡ Auto Fill (Live Prices)"):
        with st.spinner("Auto-filling entries/exits..."):
            auto_fill_open_trades()
        st.success("Auto fill complete.")

    if j2.button("🔄 Reload Journal"):
        st.experimental_rerun()

    if j3.button("🧹 Clear Journal (Demo Only)"):
        if DEMO_PATH.exists():
            DEMO_PATH.unlink()
        st.warning("Journal cleared. Reload app.")
        st.stop()

    journal = load_journal()

    j4.download_button(
        "⬇ Export CSV",
        journal.to_csv(index=False).encode(),
        "paper_journal.csv"
    )

    if journal.empty:
        st.info("Journal empty.")
    else:
        st.dataframe(journal, use_container_width=True)

        stats = performance_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", stats["sharpe"])
        c2.metric("Max Drawdown ₹", stats["max_drawdown"])
        c3.metric("Trades", len(journal))

        closed = journal[journal["status"]=="CLOSED"]
        if not closed.empty:
            pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
            st.line_chart(pnl.cumsum())
            st.bar_chart(pnl)

# ======================================================
# BACKTEST
# ======================================================
with tab_backtest:
    st.subheader("Strategy Backtest")

    b1, b2 = st.columns(2)

    cfg["backtest"]["start"] = b1.text_input("Backtest Start Date", cfg["backtest"]["start"])
    show_params = b2.checkbox("Show Current Parameters", value=True)

    if show_params:
        st.json(cfg)

    if st.button("▶ Run Backtest"):
        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        if trades is not None and not trades.empty:
            st.dataframe(trades, use_container_width=True)
            st.line_chart(trades["equity"])

# ======================================================
# SETTINGS
# ======================================================
with tab_settings:
    st.subheader("Engine Settings")

    s1, s2, s3, s4 = st.columns(4)

    cfg["signals"]["top_stocks"] = s1.slider("Top Stocks",1,5,int(cfg["signals"]["top_stocks"]))
    cfg["trade"]["risk_per_trade_pct"] = s2.slider("Risk %",0.001,0.01,float(cfg["trade"]["risk_per_trade_pct"]),0.001)
    cfg["trade"]["atr_mult"] = s3.slider("ATR Mult",1.0,4.0,float(cfg["trade"]["atr_mult"]),0.1)
    cfg["trade"]["take_profit_R"] = s4.slider("Reward R",1.0,3.0,float(cfg["trade"]["take_profit_R"]),0.1)

    cA, cB = st.columns(2)

    if cA.button("✅ Apply (Session Only)"):
        st.success("Applied for this session (edit config.yaml to persist).")

    if cB.button("🔄 Reset UI"):
        st.experimental_rerun()

    st.caption("To permanently save settings, edit config.yaml in your repo.")

# ======================================================
# EXPLAIN
# ======================================================
with tab_explain:
    st.subheader("How to Use This System")

    st.markdown("""
### 🚦 Live
Generate signals → follow Entry / Stop / Target → add to Journal.

### 📒 Journal
Auto-fill simulates live execution.
Sharpe + Drawdown tell you if system is stable.

### 🧪 Backtest
Historical validation only. Do not mix emotions with backtest.

### ⚙️ Settings
Controls risk and aggressiveness.

### Golden Rules
- Max 3 trades/day
- Always place stop
- Skip if entry not hit
- Never exceed suggested quantity
""")

    if st.button("Show Example Workflow"):
        st.info("""
1) Go to Live → Generate Signals  
2) Review trades → Add to Journal  
3) Journal → Auto Fill  
4) Watch equity curve grow  
5) Adjust settings → repeat
""")
