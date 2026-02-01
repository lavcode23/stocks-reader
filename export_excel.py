import pandas as pd

def export_results(path: str, trades: pd.DataFrame, plans: pd.DataFrame, summary: dict):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        plans.to_excel(w, index=False, sheet_name="TradePlans")
        trades.to_excel(w, index=False, sheet_name="Trades")
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="Summary")
