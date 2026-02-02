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
# ---------------- AUTO BOOTSTRAP (for first-time users) ----------------
from pathlib import Path

DEMO_PATH = Path("paper_journal.csv")

if not DEMO_PATH.exists():
    demo = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=20).astype(str),
        "symbol": ["DEMO"]*20,
        "sector": ["Demo"]*20,
        "entry": np.random.uniform(90,110,20),
        "stop": np.random.uniform(85,95,20),
        "target": np.random.uniform(115,130,20),
        "qty": [10]*20,
        "ml_prob": np.random.uniform(0.55,0.7,20),
        "R_multiple": np.random.uniform(1.5,2.5,20),
        "max_loss": np.random.uniform(200,500,20),
        "status": ["CLOSED"]*20,
        "entry_filled_price": np.random.uniform(90,110,20),
        "exit_price": np.random.uniform(95,125,20),
        "pnl": np.random.uniform(-200,500,20)
    })
    demo.to_csv(DEMO_PATH,index=False)

# -----------------------------
# Page
# -----------------------------
st.set_page_config(layout="wide", page_title="Pro Trading Cockpit")

# -----------------------------
# Config + constants
# -----------------------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ACCOUNT_DEFAULT = 100000.0
RISK_GATE_PCT = 0.01  # 1% of account

# -----------------------------
# Plotly (preferred) with fallback
# -----------------------------
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    PLOTLY_OK = False

# -----------------------------
# Data helpers (cached)
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_nifty(period="1y", interval="1d"):
    df = yf.download("^NSEI", period=period, interval=interval, progress=False).dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=60 * 15)
def fetch_symbol(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False).dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def compute_atr(df, n=14):
    tr = (df["High"] - df["Low"]).rolling(n).mean()
    return float(tr.iloc[-1]) if len(tr) >= n and not np.isnan(tr.iloc[-1]) else np.nan

def compute_mom(df, n=20):
    if len(df) < n + 1:
        return np.nan
    return float(df["Close"].pct_change(n).iloc[-1])

def compute_vol(df, n=20):
    if len(df) < n + 1:
        return np.nan
    r = df["Close"].pct_change()
    v = r.rolling(n).std().iloc[-1]
    return float(v) if not np.isnan(v) else np.nan

# -----------------------------
# Charts
# -----------------------------
def show_candlestick(df, title="", entry=None, stop=None, target=None):
    df = df.copy()
    df = df.tail(120)

    if PLOTLY_OK:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                )
            ]
        )
        if entry is not None:
            fig.add_hline(y=float(entry), line_dash="dash", annotation_text=f"Entry {entry}")
        if stop is not None:
            fig.add_hline(y=float(stop), line_dash="dash", annotation_text=f"Stop {stop}")
        if target is not None:
            fig.add_hline(y=float(target), line_dash="dash", annotation_text=f"Target {target}")

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: close line
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close")
        if entry is not None:
            ax.axhline(float(entry), linestyle="--", label=f"Entry {entry}")
        if stop is not None:
            ax.axhline(float(stop), linestyle="--", label=f"Stop {stop}")
        if target is not None:
            ax.axhline(float(target), linestyle="--", label=f"Target {target}")
        ax.set_title(title + " (fallback close chart)")
        ax.legend()
        st.pyplot(fig)

def show_heatmap_sector(df_signals):
    """
    TradingView-style grid: rows=sector, cols=["count","avg_prob","avg_R"]
    """
    if df_signals.empty:
        st.info("No signals to show in heatmap.")
        return

    sec = df_signals.groupby("sector").agg(
        count=("symbol", "count"),
        avg_prob=("ml_prob", "mean"),
        avg_R=("R_multiple", "mean"),
        avg_max_loss=("max_loss_₹", "mean"),
    ).reset_index()

    # Normalize to 0..1 for heatmap-like look
    hm = sec.set_index("sector")[["count", "avg_prob", "avg_R", "avg_max_loss"]].copy()

    # make it readable: min-max scale each col
    for c in hm.columns:
        mn, mx = hm[c].min(), hm[c].max()
        if mx != mn:
            hm[c] = (hm[c] - mn) / (mx - mn)
        else:
            hm[c] = 0.5

    if PLOTLY_OK:
        fig = px.imshow(
            hm.T,
            aspect="auto",
            title="Sector Heatmap Grid (relative strength by metric)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(hm.T, use_container_width=True)

def hist(values, title):
    values = pd.Series(values).dropna().values
    if len(values) == 0:
        st.info("Not enough values for chart.")
        return
    fig, ax = plt.subplots()
    ax.hist(values, bins=12)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# Risk gate
# -----------------------------
def apply_risk_gate(df, account):
    """
    Blocks/auto-reduces trades if total risk > 1% of account.
    Uses max_loss_₹ (already computed by engine) as per-trade worst case.
    """
    if df.empty:
        return df, 0.0, 0.0, "NO_TRADES"

    gate = float(account) * RISK_GATE_PCT
    df2 = df.copy()
    total_risk = float(pd.to_numeric(df2["max_loss_₹"], errors="coerce").fillna(0.0).sum())

    if total_risk <= gate:
        return df2, total_risk, gate, "PASS"

    # Auto-reduce: keep best signals until within gate
    df2 = df2.sort_values(["ml_prob", "R_multiple"], ascending=[False, False]).copy()
    kept = []
    running = 0.0
    for _, r in df2.iterrows():
        rsk = float(r["max_loss_₹"])
        if running + rsk <= gate:
            kept.append(r)
            running += rsk

    if len(kept) == 0:
        return df2.iloc[0:0], total_risk, gate, "BLOCK"  # block all
    return pd.DataFrame(kept), total_risk, gate, "REDUCED"

# -----------------------------
# UI
# -----------------------------
st.title("📊 Pro Trading Cockpit — Visual, Explainable, Actionable")

tab_live, tab_journal, tab_research, tab_settings, tab_help = st.tabs(
    ["🚦 Live Terminal", "📒 Paper Journal", "🧪 Research", "⚙️ Settings", "📘 Explain"]
)

# =========================================================
# LIVE TERMINAL
# =========================================================
with tab_live:
    st.subheader("Today Checklist (Beginner-proof)")

    # checklist panel
    with st.container(border=True):
        ck1 = st.checkbox("1) I will trade ONLY if market regime says TRADE", value=True)
        ck2 = st.checkbox("2) I will place STOP LOSS immediately after entry", value=True)
        ck3 = st.checkbox("3) I will not increase quantity beyond suggested qty", value=True)
        ck4 = st.checkbox("4) If entry not triggered today, I will SKIP", value=True)
        ck5 = st.checkbox("5) Total risk must be <= 1% of account (risk gate)", value=True)

        if not all([ck1, ck2, ck3, ck4, ck5]):
            st.warning("Checklist not fully accepted. This app will still show signals, but **do not trade live** until checklist is followed.")

    st.divider()
    st.subheader("1) Market Regime Dashboard")

    c1, c2, c3 = st.columns(3)
    account = c1.number_input("Account Equity (₹)", min_value=10000.0, value=ACCOUNT_DEFAULT, step=1000.0)
    max_positions = c2.selectbox("Max Trades Today", [1, 2, 3, 4, 5], index=2)
    mode = c3.selectbox("Mode", ["Beginner", "Pro"])

    nifty = fetch_nifty(period="1y", interval="1d")
    nifty["MA50"] = nifty["Close"].rolling(50).mean()
    nifty["MA200"] = nifty["Close"].rolling(200).mean()

    close = float(nifty["Close"].iloc[-1])
    ma50 = float(nifty["MA50"].iloc[-1]) if not np.isnan(nifty["MA50"].iloc[-1]) else close
    ma200 = float(nifty["MA200"].iloc[-1]) if not np.isnan(nifty["MA200"].iloc[-1]) else close

    if close > ma50 and ma50 > ma200:
        regime = "BULLISH"
        st.success("✅ TRADE MODE: Close > MA50 > MA200 (trend supportive)")
    elif close < ma50 and ma50 < ma200:
        regime = "BEARISH"
        st.error("⛔ NO TRADE MODE: Close < MA50 < MA200 (avoid trading)")
    else:
        regime = "SIDEWAYS"
        st.warning("⚠️ CAUTION MODE: Mixed trend (reduce trades / reduce risk)")

    # Candlestick for NIFTY (yes, candlestick)
    st.caption("Candlestick view helps you see momentum, volatility, and trend visually.")
    show_candlestick(nifty.tail(220), title="NIFTY (^NSEI) Candlestick")

    st.divider()
    st.subheader("2) Generate Live Signals")

    if st.button("🔄 Generate Today’s Trade Plan"):
        with st.spinner("Running realtime engine..."):
            out = run_realtime(account_equity=float(account), max_positions=int(max_positions))
        st.session_state["live_out"] = out

    if "live_out" not in st.session_state:
        st.info("Click **Generate Today’s Trade Plan** to see signals.")
        st.stop()

    out = st.session_state["live_out"]
    note = out.get("note", "")
    df = out.get("signals", pd.DataFrame())

    st.info(note)

    if df is None or df.empty:
        st.warning("No trades today (filters may be strict — that is OK).")
        st.stop()

    # Risk gate
    df_gate, total_risk_before, gate, gate_status = apply_risk_gate(df, account)

    if gate_status == "PASS":
        st.success(f"✅ Risk Gate PASS: total risk ₹{int(total_risk_before)} ≤ ₹{int(gate)} (1% of account)")
    elif gate_status == "REDUCED":
        st.warning(
            f"⚠️ Risk Gate REDUCED trades: requested risk ₹{int(total_risk_before)} > ₹{int(gate)}. "
            f"Showing only best trades within risk limit."
        )
    else:
        st.error(
            f"⛔ Risk Gate BLOCK: requested risk ₹{int(total_risk_before)} > ₹{int(gate)} and none fit. "
            f"Reduce risk per trade or reduce positions."
        )
        df_gate = df_gate.iloc[0:0]

    # Use gated df for the rest of UI
    df_show = df_gate.copy()
    if df_show.empty:
        st.stop()

    # Derived metrics
    df_show["position_value_₹"] = (df_show["entry_buy_above"] * df_show["qty"]).round(0)
    df_show["reward_₹"] = ((df_show["target"] - df_show["entry_buy_above"]) * df_show["qty"]).round(0)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Signals (after risk gate)", len(df_show))
    s2.metric("Total Exposure ₹", int(df_show["position_value_₹"].sum()))
    s3.metric("Total Risk ₹", int(pd.to_numeric(df_show["max_loss_₹"], errors="coerce").fillna(0.0).sum()))
    s4.metric("Avg Confidence", round(float(df_show["ml_prob"].mean()), 3))

    st.divider()
    st.subheader("3) Sector Heatmap Grid (TradingView-style)")

    show_heatmap_sector(df_show)

    st.divider()
    st.subheader("4) Confidence Distribution")
    hist(df_show["ml_prob"].values, "ML Confidence Histogram (higher cluster = stronger day)")

    st.divider()
    st.subheader("5) Signals Table (Actionable)")

    cols = [
        "sector","symbol","ml_prob",
        "entry_buy_above","stop_loss","target",
        "qty","max_loss_₹","R_multiple",
        "position_value_₹","reward_₹",
        "validity"
    ]
    st.dataframe(df_show[cols], use_container_width=True)

    st.divider()
    st.subheader("6) Trade Chart + Why This Trade?")

    pick = st.selectbox("Pick a trade to inspect deeply", df_show["symbol"].tolist())
    r = df_show[df_show["symbol"] == pick].iloc[0]

    sym_df = fetch_symbol(pick, period="6mo", interval="1d")
    show_candlestick(
        sym_df,
        title=f"{pick} Candlestick with Entry/Stop/Target",
        entry=r["entry_buy_above"],
        stop=r["stop_loss"],
        target=r["target"],
    )

    # Why this trade panel
    st.markdown("### ✅ Why this trade?")
    mom20 = compute_mom(sym_df, 20)
    vol20 = compute_vol(sym_df, 20)
    atr14 = compute_atr(sym_df, 14)
    last_close = float(sym_df["Close"].iloc[-1])

    # simple interpretability
    reasons = []
    if float(r["ml_prob"]) >= 0.60:
        reasons.append("High ML confidence (≥ 0.60) — stronger edge.")
    else:
        reasons.append("Moderate ML confidence — keep size small and follow stop strictly.")

    if float(r["R_multiple"]) >= 1.8:
        reasons.append("Good Reward:Risk (R multiple high) — winners can pay for losses.")
    else:
        reasons.append("Lower Reward:Risk — consider skipping unless market is strong.")

    if not np.isnan(mom20) and mom20 > 0:
        reasons.append(f"Positive 20D momentum ({mom20*100:.2f}%).")
    elif not np.isnan(mom20):
        reasons.append(f"Weak/negative 20D momentum ({mom20*100:.2f}%) — higher risk of failure.")

    if regime == "BULLISH":
        reasons.append("Market regime supports longs (BULLISH).")
    elif regime == "SIDEWAYS":
        reasons.append("Market is SIDEWAYS — trades are harder; reduce count/risk.")
    else:
        reasons.append("Market is BEARISH — avoid trading (but signal may exist due to data lag).")

    with st.container(border=True):
        cA, cB, cC, cD = st.columns(4)
        cA.metric("ML Confidence", r["ml_prob"])
        cB.metric("R Multiple", r["R_multiple"])
        cC.metric("ATR(14)", f"{atr14:.2f}" if not np.isnan(atr14) else "NA")
        cD.metric("20D Momentum", f"{mom20*100:.2f}%" if not np.isnan(mom20) else "NA")

        st.write("**Interpretation:**")
        for x in reasons:
            st.write(f"- {x}")

        st.caption(
            "This panel is intentionally simple: it explains the trade in human language without pretending to predict the future."
        )

    st.divider()
    st.subheader("7) Execution Steps (Beginner-proof)")

    st.markdown(
        f"""
**For {pick}:**
1) Place **BUY Stop** at **₹{r['entry_buy_above']}**  
2) Immediately place **STOP LOSS** at **₹{r['stop_loss']}**  
3) Place **TARGET SELL** at **₹{r['target']}**  
4) If entry not triggered today → **SKIP**  
5) Never risk more than **₹{int(r['max_loss_₹'])}** on this trade (that’s your max loss)
"""
    )

    st.divider()
    st.subheader("8) Copy Orders (Reference)")

    broker = st.radio("Format", ["Zerodha", "Upstox"], horizontal=True)

    if broker == "Zerodha":
        lines = [
            f"{x['symbol']},BUY,{int(x['qty'])},{x['entry_buy_above']},SL:{x['stop_loss']},TGT:{x['target']}"
            for _, x in df_show.iterrows()
        ]
        st.text_area("Copy-Paste", "\n".join(lines), height=140)
    else:
        lines = [
            f"{x['symbol']} | BUY | QTY {int(x['qty'])} | ENTRY {x['entry_buy_above']} | SL {x['stop_loss']} | TGT {x['target']}"
            for _, x in df_show.iterrows()
        ]
        st.text_area("Copy-Paste", "\n".join(lines), height=140)

    st.divider()
    st.subheader("9) Export + Journal")

    csv = df_show.to_csv(index=False).encode()
    st.download_button("⬇ Download Signals CSV", csv, "today_signals.csv")

    if st.button("📒 Add these trades to Paper Journal"):
        j = append_trades(df_show)
        st.success(f"Added {len(df_show)} trades to journal.")
        st.dataframe(j.tail(20), use_container_width=True)


# =========================================================
# PAPER JOURNAL
# =========================================================
with tab_journal:
    st.subheader("Paper Journal (Auto-fill + Pro Performance)")

    c1, c2 = st.columns([1, 2])
    if c1.button("⚡ Auto-Fill Using Live Prices"):
        with st.spinner("Checking live prices to fill entries/exits..."):
            auto_fill_open_trades()
        st.success("Auto-fill complete.")

    journal = load_journal()
    if journal.empty:
        st.info("Journal is empty. Add trades from Live tab.")
        st.stop()

    open_df = journal[journal["status"] == "OPEN"].copy()
    closed_df = journal[journal["status"] == "CLOSED"].copy()

    a, b, c = st.columns(3)
    a.metric("Open", len(open_df))
    b.metric("Closed", len(closed_df))
    stats = performance_stats()
    c.metric("Sharpe (paper)", stats["sharpe"])

    st.metric("Max Drawdown ₹ (paper)", stats["max_drawdown"])

    st.divider()
    st.subheader("Open Trades")
    if open_df.empty:
        st.info("No open trades.")
    else:
        st.dataframe(open_df, use_container_width=True)

    st.subheader("Closed Trades")
    if closed_df.empty:
        st.info("No closed trades yet.")
    else:
        st.dataframe(closed_df, use_container_width=True)

        pnl = pd.to_numeric(closed_df["pnl"], errors="coerce").fillna(0.0)

        st.subheader("PnL Histogram (distribution)")
        hist(pnl.values, "Closed Trade PnL Histogram")

        # equity & drawdown curves from closed pnl
        st.subheader("Equity Curve + Drawdown")
        eq = pnl.cumsum()
        roll_max = eq.cummax()
        dd = eq - roll_max

        st.line_chart(pd.DataFrame({"equity": eq.values}))
        st.line_chart(pd.DataFrame({"drawdown": dd.values}))
        st.caption("Drawdown is the drop from peak equity. Lower & shorter drawdowns are better.")

    st.divider()
    st.subheader("Export Journal CSV")
    st.download_button(
        "⬇ Download paper_journal.csv",
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
            st.warning("No trades generated by backtest.")
        else:
            st.subheader("Trade Log")
            st.dataframe(trades, use_container_width=True)

            st.subheader("Equity Curve")
            eq2 = trades[["equity"]].copy()
            eq2["i"] = range(len(eq2))
            st.line_chart(eq2.set_index("i"))

            if "exit_reason" in trades.columns:
                st.subheader("Exit Reasons")
                st.bar_chart(trades["exit_reason"].value_counts())


# =========================================================
# SETTINGS
# =========================================================
with tab_settings:
    st.subheader("Settings (UI controls / config reference)")

    st.caption("These sliders do not write back to config.yaml automatically. Edit config.yaml in repo to persist.")
    cfg["signals"]["top_stocks"] = st.slider("Top Stocks", 1, 5, int(cfg["signals"]["top_stocks"]))
    cfg["trade"]["risk_per_trade_pct"] = st.slider("Risk per Trade (%)", 0.001, 0.01, float(cfg["trade"]["risk_per_trade_pct"]), 0.001)
    cfg["trade"]["atr_mult"] = st.slider("ATR Multiplier", 1.0, 4.0, float(cfg["trade"]["atr_mult"]), 0.1)
    cfg["trade"]["take_profit_R"] = st.slider("Reward Ratio (R)", 1.0, 3.0, float(cfg["trade"]["take_profit_R"]), 0.1)

    st.info("Risk Gate is fixed at 1% of account. You can change RISK_GATE_PCT in app.py if needed.")


# =========================================================
# HELP / EXPLAIN
# =========================================================
with tab_help:
    st.subheader("What each feature is doing (so nothing is wasted)")

    st.markdown(
        """
### Candlestick charts
They show volatility and trend structure better than a single close line, so you can visually validate trade levels.

### Sector heatmap grid
Shows where signals are clustering. If multiple trades come from one sector, that is capital rotation (flow).

### Why this trade panel
Explains:
- ML confidence (ranking)
- Reward:Risk
- momentum/volatility
- market regime
This helps a beginner understand what they are doing.

### Risk gate (≤ 1%)
Prevents over-trading. If total risk exceeds 1% of your capital, the system automatically reduces trades or blocks.

### Today checklist
Forces discipline and prevents beginner mistakes. Trading is mostly execution + risk control.
        """
    )
