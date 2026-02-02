import streamlit as st
import pandas as pd
import yaml

from realtime_engine import run_realtime
from backtest import run_backtest

st.set_page_config(layout="wide", page_title="Beginner Trading Cockpit")

ACCOUNT_DEFAULT = 100000.0

st.title("🧭 Beginner Trading Cockpit (Research + Live Signals)")

tab_live, tab_backtest, tab_help = st.tabs(["🚦 Live Trade Plan", "🧪 Backtest", "📘 How to Trade"])

# ---------------------------------------------------------
# LIVE
# ---------------------------------------------------------
with tab_live:
    st.subheader("Today’s Decision")

    c1, c2, c3 = st.columns(3)
    account = c1.number_input("Account Equity (₹)", min_value=10000.0, value=ACCOUNT_DEFAULT, step=1000.0)
    max_positions = c2.selectbox("Max Trades Today", [1, 2, 3, 4, 5], index=2)
    mode = c3.selectbox("Mode", ["Beginner (recommended)", "Advanced"])

    if st.button("🔄 Generate Today’s Trade Plan"):
        with st.spinner("Building your trade plan..."):
            out = run_realtime(account_equity=float(account), max_positions=int(max_positions))

        market_ok = out["market_ok"]
        note = out["note"]
        df = out["signals"]

        if not market_ok:
            st.error("⛔ NO TRADE TODAY")
            st.info(note)
            st.stop()

        st.success("✅ TRADE ALLOWED TODAY")
        st.info(note)

        if df.empty:
            st.warning("No trades to take today.")
            st.stop()

        st.subheader("Trade Cards (Do exactly this)")

        # render beginner cards
        for i, row in df.reset_index(drop=True).iterrows():
            with st.container(border=True):
                st.markdown(f"### {i+1}) {row['symbol']}  —  Sector: **{row['sector']}**  |  Confidence: **{row['ml_prob']}**")

                a, b, c, d = st.columns(4)
                a.metric("✅ Buy above", f"₹{row['entry_buy_above']}")
                b.metric("🛑 Stop loss", f"₹{row['stop_loss']}")
                c.metric("🎯 Target", f"₹{row['target']}")
                d.metric("📦 Quantity", f"{int(row['qty'])}")

                e, f, g = st.columns(3)
                e.metric("💸 Max Loss", f"₹{int(row['max_loss_₹'])}")
                f.metric("📈 R Multiple", f"{row['R_multiple']}R")
                g.metric("⏱ Validity", row["validity"])

                st.markdown(
                    """
**How to place orders (simple):**
1) Place a **BUY Stop-Limit** / **BUY Stop** at **Buy above** price.  
2) Immediately place a **STOP LOSS** at the Stop price.  
3) Place a **Target SELL Limit** at the Target price.  
4) If buy is NOT triggered today → **DO NOTHING** (skip trade).
                    """
                )

        st.subheader("Download")
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Today’s Trades (CSV)", csv, "today_trade_plan.csv")

        if mode.startswith("Advanced"):
            st.subheader("Full Table")
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------
with tab_backtest:
    st.subheader("Research Backtest (learning only)")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        st.dataframe(trades, use_container_width=True)

        if not trades.empty:
            eq = trades[["equity"]].copy()
            eq["i"] = range(len(eq))
            st.line_chart(eq.set_index("i"))


# ---------------------------------------------------------
# HELP
# ---------------------------------------------------------
with tab_help:
    st.subheader("How a beginner should use this output")

    st.markdown(
        """
### Rules (non-negotiable)
- Trade only when the app says **TRADE ALLOWED**.
- Take maximum **3 trades** per day.
- Always place **Stop Loss** immediately after entry.
- If entry price is not triggered → **skip**.
- Never increase quantity beyond what app shows.

### What “Max Loss” means
That is the maximum damage if stop is hit.

### Accuracy is not the goal
Goal is: **small losses + bigger wins + discipline**.

### Paper trade first
Use this for 2 weeks on paper before real money.
        """
    )
