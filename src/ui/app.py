
import streamlit as st
from bharat_sector_demand_hedged_weekly.sector import sector_scores, top_stocks

st.set_page_config(page_title="Sector Demand AI", layout="centered")

st.title("🇮🇳 Bharat Sector Demand – Weekly Prototype")

scores = sector_scores()
best_sector = max(scores, key=scores.get)

st.subheader("Sector Scores (Weekly Momentum)")
st.write(scores)

st.success(f"Top Sector This Week: {best_sector}")

stocks = top_stocks(best_sector)

st.subheader("Top Stocks")
for s in stocks:
    st.write(s)

st.info("This is a research prototype. Not financial advice.")
