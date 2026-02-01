import yfinance as yf
import pandas as pd
import pickle
from ml_ranker import make_dataset, train_model
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

sectors = cfg["universe"]["sectors"]
tickers = sorted({t for s in sectors.values() for t in s})

print("Downloading data...")
data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False)

cleaned = []

for t in tickers:
    if t in data:
        df = data[t].dropna()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if len(df) > 80:
            ds = make_dataset(df)
            if not ds.empty:
                cleaned.append(ds)

big = pd.concat(cleaned)

print("Training ML...")
model = train_model(big)

with open("ml_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved as ml_model.pkl")
