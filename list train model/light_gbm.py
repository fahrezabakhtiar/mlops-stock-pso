import os
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from config import TICKERS

def create_windows(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def train_lightgbm_model(ticker, window_size=5):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"))
    series = df['Close'].values
    X, y = create_windows(series, window_size)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_lightgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_lightgbm_mape.csv")
    pd.DataFrame([{"model": "lightgbm", "mape": mape}]).to_csv(mape_path, index=False)

if __name__ == "__main__":
    for ticker in TICKERS:
        train_lightgbm_model(ticker)