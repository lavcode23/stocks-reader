import streamlit as st
import pandas as pd
import yaml

from realtime_engine import run_realtime
from backtest import run_backtest
from journal import load_journal, append_trades, update_trade

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

        # Render Trade Cards
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
**Order steps (simple):**
1) Place a **BUY Stop** at *Buy above*.  
2) Immediately place **STOP LOSS** at *Stop loss*.  
3) Place **Target SELL Limit** at *Target*.  
4) If Buy is NOT triggered today → **SKIP**.
                    """
                )

        # ---------------- One-click Copy Orders ----------------
        st.subheader("🧾 One-Click Copy Orders")

        if broker == "Zerodha":
            # Simple Kite CSV-like lines (copy-paste)
            lines = []
            for _, r in df.iterrows():
                lines.append(f"{r['symbol']},BUY,{int(r['qty'])},{r['entry_buy_above']},SL:{r['stop_loss']},TGT:{r['target']}")
            kite_text = "\n".join(lines)
            st.text_area("Copy into Zerodha (reference format)", kite_text, height=150)

        if broker == "Upstox":
            lines = []
            for _, r in df.iterrows():
                lines.append(f"{r['symbol']} | BUY | QTY {int(r['qty'])} | ENTRY {r['entry_buy_above']} | SL {r['stop_loss']} | TGT {r['target']}")
            upstox_text = "\n".join(lines)
            st.text_area("Copy into Upstox (reference format)", upstox_text, height=150)

        # CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Today’s Trades (CSV)", csv, "today_trade_plan.csv")

        # ---------------- Add to Paper Journal ----------------
        if st.button("📒 Add These Trades to Paper Journal"):
            j = append_trades(df)
            st.success(f"Added {len(df)} trades to journal.")
            st.dataframe(j.tail(20), use_container_width=True)


# ---------------------------------------------------------
# PAPER JOURNAL
# ---------------------------------------------------------
with tab_journal:
    st.subheader("📒 Paper Trade Journal")

    journal = load_journal()
    if journal.empty:
        st.info("Journal is empty. Add trades from Live tab.")
        st.stop()

    st.dataframe(journal, use_container_width=True)

    st.subheader("Update Trade (when filled / exited)")

    col1, col2, col3 = st.columns(3)
    sym = col1.selectbox("Symbol", journal[journal["status"]=="OPEN"]["symbol"].unique())
    entry_fill = col2.text_input("Entry Filled Price (optional)")
    exit_price = col3.text_input("Exit Price (optional)")

    if st.button("Update Selected Trade"):
        ef = float(entry_fill) if entry_fill else None
        xp = float(exit_price) if exit_price else None
        j2 = update_trade(sym, ef, xp)
        st.success("Trade updated.")
        st.dataframe(j2.tail(20), use_container_width=True)

    # Journal summary
    st.subheader("Journal Summary")
    closed = journal[journal["status"]=="CLOSED"].copy()
    if not closed.empty:
        pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0)
        c1,c2,c3 = st.columns(3)
        c1.metric("Closed Trades", len(closed))
        c2.metric("Total PnL ₹", int(pnl.sum()))
        c3.metric("Win Rate", f"{(pnl>0).mean()*100:.1f}%")

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
- If Buy price not triggered → SKIP.
- Never increase quantity.

### Paper Journal
Use **Paper Journal** tab:
- Add trades from Live.
- When entry fills, update Entry Filled Price.
- When exited, update Exit Price.
- App calculates PnL.

### Brokers
The copy formats are **reference templates**.
Paste into Zerodha/Upstox and place orders manually.

Paper trade for 2–3 weeks before real money.
        """
    )
