import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import yaml

from realtime_engine import run_realtime
from backtest import run_backtest
from journal import (
    load_journal,
    append_trades,
    auto_fill_open_trades,
    performance_stats,
)

st.set_page_config(layout="wide", page_title="Trading Cockpit")

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data(ttl=60 * 30)
def fetch_nifty(period="1y", interval="1d"):
    df = yf.download("^NSEI", period=period, interval=interval, progress=False)
    df = df.dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=60 * 15)
def fetch_symbol(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df = df.dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def plot_price_with_levels(df, entry=None, stop=None, target=None, title=""):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close")

    if entry is not None:
        ax.axhline(float(entry), linestyle="--", label=f"Entry ₹{entry}")
    if stop is not None:
        ax.axhline(float(stop), linestyle="--", label=f"Stop ₹{stop}")
    if target is not None:
        ax.axhline(float(target), linestyle="--", label=f"Target ₹{target}")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_hist(values, title, bins=12):
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def equity_and_drawdown_from_journal(journal_df, start_equity=0.0):
    """
    Builds equity curve based on CLOSED pnl values (paper journal).
    Returns (equity_series, drawdown_series).
    """
    closed = journal_df[journal_df["status"] == "CLOSED"].copy()
    if closed.empty:
        return None, None

    pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    equity = start_equity + pnl.cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    return equity, dd

# -----------------------------
# Load config
# -----------------------------

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ACCOUNT_DEFAULT = 100000.0

st.title("📊 Pro Trading Cockpit (Live + Journal + Research)")

tab_live, tab_journal, tab_research, tab_settings, tab_help = st.tabs(
    ["🚦 Live Terminal", "📒 Paper Journal", "🧪 Research", "⚙️ Settings", "📘 Explain"]
)

# =========================================================
# LIVE TERMINAL
# =========================================================

with tab_live:
    st.subheader("1) Market Regime Dashboard (Decide TRADE / NO TRADE)")

    left, mid, right = st.columns([1.2, 1.2, 1.2])
    account = left.number_input("Account Equity (₹)", min_value=10000.0, value=ACCOUNT_DEFAULT, step=1000.0)
    max_positions = mid.selectbox("Max Trades Today", [1, 2, 3, 4, 5], index=2)
    view_mode = right.selectbox("View Mode", ["Beginner", "Pro"])

    nifty = fetch_nifty(period="1y", interval="1d")
    nifty["MA50"] = nifty["Close"].rolling(50).mean()
    nifty["MA200"] = nifty["Close"].rolling(200).mean()

    # Regime logic (simple, readable)
    close = float(nifty["Close"].iloc[-1])
    ma50 = float(nifty["MA50"].iloc[-1]) if not np.isnan(nifty["MA50"].iloc[-1]) else close
    ma200 = float(nifty["MA200"].iloc[-1]) if not np.isnan(nifty["MA200"].iloc[-1]) else close

    if close > ma50 and ma50 > ma200:
        regime = "BULLISH"
        st.success("✅ TRADE MODE: Market is BULLISH (Close > MA50 > MA200)")
    elif close < ma50 and ma50 < ma200:
        regime = "BEARISH"
        st.error("⛔ NO TRADE MODE: Market is BEARISH (Close < MA50 < MA200)")
    else:
        regime = "SIDEWAYS"
        st.warning("⚠️ CAUTION MODE: Market is SIDEWAYS / MIXED (trade smaller or fewer positions)")

    # NIFTY chart
    st.caption("NIFTY trend with moving averages (helps decide if we should trade aggressively or not).")
    plot_price_with_levels(
        nifty.tail(240),
        entry=None,
        stop=None,
        target=None,
        title="NIFTY (^NSEI) — Close with MA50 / MA200"
    )
    # overlay MAs via simple chart
    fig, ax = plt.subplots()
    ax.plot(nifty.tail(240).index, nifty.tail(240)["Close"], label="Close")
    ax.plot(nifty.tail(240).index, nifty.tail(240)["MA50"], label="MA50")
    ax.plot(nifty.tail(240).index, nifty.tail(240)["MA200"], label="MA200")
    ax.set_title("NIFTY: Close vs MA50/MA200")
    ax.legend()
    st.pyplot(fig)

    st.divider()
    st.subheader("2) Live Signals (What to trade today)")

    colA, colB, colC = st.columns([1, 1, 1])
    if colA.button("🔄 Generate Today’s Trade Plan"):
        with st.spinner("Generating signals..."):
            out = run_realtime(account_equity=float(account), max_positions=int(max_positions))
        st.session_state["live_out"] = out

    if "live_out" not in st.session_state:
        st.info("Click **Generate Today’s Trade Plan** to see actionable trades.")
        st.stop()

    out = st.session_state["live_out"]
    df = out.get("signals", pd.DataFrame())
    note = out.get("note", "")

    st.info(note)

    if df is None or df.empty:
        st.warning("No trades today based on current filters (this is normal).")
        st.stop()

    # Extra trader metrics (exposure / risk etc.)
    df = df.copy()
    per_trade_risk = float(account) * float(cfg["trade"]["risk_per_trade_pct"])
    df["risk_per_trade_₹"] = round(per_trade_risk, 0)
    df["position_value_₹"] = (df["entry_buy_above"] * df["qty"]).round(0)
    df["reward_₹"] = ((df["target"] - df["entry_buy_above"]) * df["qty"]).round(0)

    # Summary at top
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Signals", len(df))
    s2.metric("Total Exposure ₹", int(df["position_value_₹"].sum()))
    s3.metric("Total Risk ₹", int(df["risk_per_trade_₹"].sum()))
    s4.metric("Avg ML Confidence", round(float(df["ml_prob"].mean()), 3))

    # --- Sector strength visual
    st.subheader("3) Sector Strength View (Quick understanding)")
    sec = df.groupby("sector").agg(
        avg_prob=("ml_prob", "mean"),
        count=("symbol", "count")
    ).reset_index()

    fig, ax = plt.subplots()
    ax.bar(sec["sector"], sec["avg_prob"])
    ax.set_title("Average ML Confidence by Sector (Higher = stronger today)")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Avg ML Probability")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.bar(sec["sector"], sec["count"])
    ax2.set_title("Signals count by Sector")
    ax2.set_xlabel("Sector")
    ax2.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    # --- Probability distribution
    st.subheader("4) Confidence Distribution (Are we seeing strong signals?)")
    plot_hist(df["ml_prob"].values, "ML Probability Histogram", bins=10)

    # --- Main table
    st.subheader("5) Trade Table (Actionable numbers)")
    show_cols = [
        "sector","symbol","ml_prob",
        "entry_buy_above","stop_loss","target",
        "qty","max_loss_₹","R_multiple",
        "risk_per_trade_₹","position_value_₹","reward_₹",
        "validity"
    ]
    st.dataframe(df[show_cols], use_container_width=True)

    # --- Trade Cards with chart
    st.subheader("6) Trade Cards + Chart (Beginner-friendly execution)")
    pick = st.selectbox("Select a symbol to view chart with Entry/Stop/Target", df["symbol"].tolist())

    row = df[df["symbol"] == pick].iloc[0]
    sym_df = fetch_symbol(pick, period="6mo", interval="1d")
    st.caption("Chart shows last ~6 months close. Lines show today’s trade levels.")
    plot_price_with_levels(
        sym_df.tail(120),
        entry=row["entry_buy_above"],
        stop=row["stop_loss"],
        target=row["target"],
        title=f"{pick} — Price with Entry/Stop/Target"
    )

    st.markdown(
        """
### How to execute (simple rules)
- **Buy only if price crosses “Buy above”** (do not buy early).
- Immediately place the **Stop loss**.
- Place the **Target**.
- If entry is not triggered today → **skip** (no trade is also a decision).
"""
    )

    # Copy Orders formats (useful button)
    st.subheader("7) Copy Orders (Reference Formats)")
    broker = st.radio("Choose format", ["Zerodha", "Upstox"], horizontal=True)

    if broker == "Zerodha":
        lines = [
            f"{r['symbol']},BUY,{int(r['qty'])},{r['entry_buy_above']},SL:{r['stop_loss']},TGT:{r['target']}"
            for _, r in df.iterrows()
        ]
        st.text_area("Copy-Paste (Reference)", "\n".join(lines), height=140)
    else:
        lines = [
            f"{r['symbol']} | BUY | QTY {int(r['qty'])} | ENTRY {r['entry_buy_above']} | SL {r['stop_loss']} | TGT {r['target']}"
            for _, r in df.iterrows()
        ]
        st.text_area("Copy-Paste (Reference)", "\n".join(lines), height=140)

    # Export
    st.subheader("8) Export")
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download Today’s Signals (CSV)", csv, "today_signals.csv")

    if st.button("📒 Add these signals to Paper Journal"):
        j = append_trades(df)
        st.success(f"Added {len(df)} trades to journal.")
        st.dataframe(j.tail(20), use_container_width=True)


# =========================================================
# PAPER JOURNAL
# =========================================================

with tab_journal:
    st.subheader("Paper Journal (Auto-fill + Performance)")

    col1, col2, col3 = st.columns([1, 1, 2])
    if col1.button("⚡ Auto-Fill Using Live Prices"):
        with st.spinner("Checking live prices and updating entries/exits..."):
            auto_fill_open_trades()
        st.success("Auto-fill done. Refreshing journal below.")

    journal = load_journal()
    if journal.empty:
        st.info("Journal is empty. Add trades from Live tab.")
        st.stop()

    # Show OPEN vs CLOSED
    open_df = journal[journal["status"] == "OPEN"].copy()
    closed_df = journal[journal["status"] == "CLOSED"].copy()

    a, b = st.columns(2)
    a.metric("Open Trades", len(open_df))
    b.metric("Closed Trades", len(closed_df))

    # Performance
    stats = performance_stats()
    p1, p2 = st.columns(2)
    p1.metric("Sharpe (paper)", stats["sharpe"])
    p2.metric("Max Drawdown ₹", stats["max_drawdown"])

    st.subheader("1) Open Trades (what’s running)")
    if open_df.empty:
        st.info("No open trades right now.")
    else:
        st.dataframe(open_df, use_container_width=True)

    st.subheader("2) Closed Trades (results)")
    if closed_df.empty:
        st.info("No closed trades yet.")
    else:
        st.dataframe(closed_df, use_container_width=True)

        pnl = pd.to_numeric(closed_df["pnl"], errors="coerce").fillna(0.0)

        st.subheader("3) PnL Distribution (are losses controlled?)")
        plot_hist(pnl.values, "Closed-trade PnL Histogram", bins=14)

        st.subheader("4) Equity Curve + Drawdown (most important)")
        equity, dd = equity_and_drawdown_from_journal(journal, start_equity=0.0)
        if equity is not None:
            eq_df = pd.DataFrame({"equity": equity.values})
            dd_df = pd.DataFrame({"drawdown": dd.values})

            st.line_chart(eq_df)
            st.line_chart(dd_df)

            st.caption("Drawdown shows how much you are down from peak. Lower and shorter drawdowns are better.")

    st.subheader("5) Export Journal")
    st.download_button(
        "⬇ Download Paper Journal CSV",
        journal.to_csv(index=False).encode(),
        "paper_journal.csv"
    )


# =========================================================
# RESEARCH
# =========================================================

with tab_research:
    st.subheader("Research Backtest (Learning / Diagnostics)")

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            res = run_backtest(cfg)

        trades = res["trades"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(trades))
        c2.metric("Win Rate", f"{res['winrate']}%")
        c3.metric("Final Equity", f"₹{res['final_equity']}")

        if trades is None or trades.empty:
            st.warning("No trades were produced by backtest.")
        else:
            st.subheader("Trade Log")
            st.dataframe(trades, use_container_width=True)

            st.subheader("Equity Curve")
            eq = trades[["equity"]].copy()
            eq["i"] = range(len(eq))
            st.line_chart(eq.set_index("i"))

            if "exit_reason" in trades.columns:
                st.subheader("Exit Reasons")
                st.bar_chart(trades["exit_reason"].value_counts())


# =========================================================
# SETTINGS
# =========================================================

with tab_settings:
    st.subheader("Settings (These control your engine)")

    st.caption("These are read from config.yaml. Adjust here and re-run signals/backtests.")

    cfg["signals"]["top_stocks"] = st.slider("Top Stocks", 1, 5, int(cfg["signals"]["top_stocks"]))
    cfg["trade"]["risk_per_trade_pct"] = st.slider("Risk per Trade (%)", 0.001, 0.01, float(cfg["trade"]["risk_per_trade_pct"]), 0.001)
    cfg["trade"]["atr_mult"] = st.slider("ATR Multiplier", 1.0, 4.0, float(cfg["trade"]["atr_mult"]), 0.1)
    cfg["trade"]["take_profit_R"] = st.slider("Reward Ratio (R)", 1.0, 3.0, float(cfg["trade"]["take_profit_R"]), 0.1)

    st.warning("These sliders update UI only. If you want them saved permanently, edit config.yaml in your repo.")


# =========================================================
# HELP / EXPLAIN
# =========================================================

with tab_help:
    st.subheader("What each section means (so nothing feels like a waste)")

    st.markdown(
        """
### Market Regime Dashboard
This tells you whether the overall market is supportive.
- **BULLISH:** Take normal trades.
- **SIDEWAYS:** Fewer trades / smaller risk.
- **BEARISH:** Avoid trading (most strategies fail here).

### Sector Strength View
Shows where strength is concentrated today.  
If signals cluster in one sector, it means money is flowing there.

### Confidence Distribution
If probabilities are mostly near 0.50–0.55, signals are weak.  
If probabilities are higher (0.60+), the day is stronger.

### Trade Cards + Chart
The chart with Entry/Stop/Target prevents beginner mistakes:
- no early entry
- stop must exist
- target is clear
- risk is known

### Paper Journal
Auto-fill makes your learning fast:
- it simulates fills and exits based on live prices
- Sharpe / Drawdown show whether the system is stable
        """
    )
