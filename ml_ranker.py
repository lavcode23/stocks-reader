import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_dataset(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["trend"] = (df["ma5"] / df["ma20"] - 1.0)
    df["vol20"] = df["ret1"].rolling(20).std()
    # label: next 5d return > 0
    df["fwd5"] = df["Close"].shift(-5) / df["Close"] - 1.0
    df["y"] = (df["fwd5"] > 0).astype(int)
    return df.dropna()

def train_ranker(df: pd.DataFrame, model_name: str = "logreg"):
    X = df[["ret5","trend","vol20"]].values
    y = df["y"].values
    if model_name == "rf":
        m = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
    else:
        m = LogisticRegression(max_iter=500)
    m.fit(X, y)
    return m

def predict_proba(model, df: pd.DataFrame) -> float:
    X = df[["ret5","trend","vol20"]].tail(1).values
    return float(model.predict_proba(X)[0,1])
