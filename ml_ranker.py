import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["ret5", "trend", "vol20"]

def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret1"] = d["Close"].pct_change(1)
    d["ret5"] = d["Close"].pct_change(5)
    d["ma5"] = d["Close"].rolling(5).mean()
    d["ma20"] = d["Close"].rolling(20).mean()
    d["trend"] = (d["ma5"] / d["ma20"] - 1.0)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["fwd5"] = d["Close"].shift(-5) / d["Close"] - 1.0
    d["y"] = (d["fwd5"] > 0).astype(int)
    return d.dropna()

def train_model(ds: pd.DataFrame):
    X = ds[FEATURES].values
    y = ds["y"].values
    m = LogisticRegression(max_iter=500)
    m.fit(X, y)
    return m

def predict_prob(model, row: pd.DataFrame) -> float:
    X = row[FEATURES].tail(1).values
    return float(model.predict_proba(X)[0,1])
