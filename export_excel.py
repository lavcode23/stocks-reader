import pandas as pd

def export_results_to_bytes(trade_plans: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, summary: dict) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        trade_plans.to_excel(w, index=False, sheet_name="TradePlans")
        trades.to_excel(w, index=False, sheet_name="Trades")
        equity.to_excel(w, index=False, sheet_name="EquityCurve")
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="Summary")
    buf.seek(0)
    return buf.read()
