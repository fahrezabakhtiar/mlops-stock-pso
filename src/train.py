import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

def create_windows(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def train_random_forest(ticker, window_size=5):
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

    # MODEL: Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_random_forest_mape.csv")
    pd.DataFrame([{"model": "random_forest", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (Random Forest): {mape:.4f}, saved to {mape_path}")

if __name__ == "__main__":
    for ticker in ['BMRI', 'BBRI', 'BBCA']:
        train_random_forest(ticker)