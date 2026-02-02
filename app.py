import streamlit as st
import pandas as pd
import yaml

from realtime_engine import run_realtime
from backtest import run_backtest
from journal import load_journal, append_trades, auto_fill_open_trades, performance_stats

st.set_page_config(layout="wide", page_title="Beginner Trading Cockpit")

ACCOUNT_DEFAULT = 100000.0

st.title("🧭 Beginner Trading Cockpit (Live + Journal + Research)")

tab_live, tab_journal, tab_backtest, tab_help = st.tabs(
    ["🚦 Live Trade Plan", "📒 Paper Journal", "🧪 Backtest", "📘 How to Trade"]
)

# ---------------------------------------------------------
# LIVE
# ---------------------------------------------------------
with tab_live:
    st.subheader("Today’s Decision")

    c1, c2, c3 = st.columns(3)
    account = c1.number_input("Account Equity (₹)", min_value=10000.0, value=ACCOUNT_DEFAULT, step=1000.0)
    max_positions = c2.selectbox("Max Trades Today", [1, 2, 3, 4, 5], index=2)
    broker = c3.selectbox("Broker Format", ["Zerodha", "Upstox"])

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

        st.subheader("Trade Cards (Follow exactly)")
        for i, row in df.reset_index(drop=True).iterrows():
            with st.container(border=True):
                st.markdown(f"### {i+1}) {row['symbol']} — Sector: **{row['sector']}** | Confidence: **{row['ml_prob']}**")

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
**Order steps:**
1) BUY Stop at *Buy above*
2) Place STOP LOSS
3) Place TARGET
4) If not triggered today → SKIP
                    """
                )

        st.subheader("🧾 One-Click Copy Orders")
        if broker == "Zerodha":
            lines = [f"{r['symbol']},BUY,{int(r['qty'])},{r['entry_buy_above']},SL:{r['stop_loss']},TGT:{r['target']}"
                     for _, r in df.iterrows()]
            st.text_area("Zerodha (reference)", "\n".join(lines), height=150)
        else:
            lines = [f"{r['symbol']} | BUY | QTY {int(r['qty'])} | ENTRY {r['entry_buy_above']} | SL {r['stop_loss']} | TGT {r['target']}"
                     for _, r in df.iterrows()]
            st.text_area("Upstox (reference)", "\n".join(lines), height=150)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Today’s Trades (CSV)", csv, "today_trade_plan.csv")

        if st.button("📒 Add These Trades to Paper Journal"):
            j = append_trades(df)
            st.success(f"Added {len(df)} trades to journal.")
            st.dataframe(j.tail(20), use_container_width=True)

# ---------------------------------------------------------
# PAPER JOURNAL
# ---------------------------------------------------------
with tab_journal:
    st.subheader("📒 Paper Trade Journal")

    if st.button("⚡ Auto-Fill Using Live Prices"):
        with st.spinner("Checking live prices..."):
            j2 = auto_fill_open_trades()
        st.success("Auto-fill complete.")

    journal = load_journal()
    if journal.empty:
        st.info("Journal is empty. Add trades from Live tab.")
        st.stop()

    st.dataframe(journal, use_container_width=True)

    stats = performance_stats()
    s1, s2 = st.columns(2)
    s1.metric("Sharpe (paper)", stats["sharpe"])
    s2.metric("Max Drawdown ₹", stats["max_drawdown"])

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
    st.markdown(
        """
### Beginner Rules
- Trade only when app says **TRADE ALLOWED**.
- Max 3 trades/day.
- Always place Stop Loss immediately.
- If Buy not triggered → SKIP.
- Never increase quantity.

### Auto Paper Fill
Use **Auto-Fill Using Live Prices** to simulate fills & exits automatically.

### Performance
Sharpe and Max Drawdown are calculated from paper trades.

Paper trade for 2–3 weeks before real money.
        """
    )
