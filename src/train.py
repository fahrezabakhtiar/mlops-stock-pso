# Tahapan Train: Melatih model prediksi harga saham untuk setiap ticker
# dan menyimpan model serta metrik evaluasinya (MAPE).

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from config import TICKERS  # Mengimpor daftar ticker dari config.py

def create_windows(series, window_size=5):
    """
    Membagi data time series menjadi window untuk supervised learning.

    Args:
        series (array): Deret harga saham (Close).
        window_size (int): Ukuran window (jumlah data sebelumnya untuk prediksi).

    Returns:
        tuple: X (fitur), y (target) dalam bentuk numpy array.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def train_linear_regression_model(ticker, window_size=5):
    """
    Melatih model Linear Regression untuk prediksi harga saham ticker tertentu.

    Args:
        ticker (str): Kode saham tanpa akhiran .JK.
        window_size (int): Jumlah langkah historis yang digunakan untuk prediksi.
    """
    # Menentukan direktori data dan model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Membaca data harga historis
    df = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"))
    series = df['Close'].values

    # Membuat window data
    X, y = create_windows(series, window_size)

    # Membagi data menjadi train dan test set (80-20 split)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Melatih model Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Melakukan prediksi dan evaluasi menggunakan MAPE
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    # Menyimpan model dalam format .pkl dengan penamaan sesuai konvensi pipeline
    model_path = os.path.join(MODELS_DIR, f"{ticker}_linear_regression_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    # Menyimpan nilai MAPE dalam file CSV agar dapat dibaca oleh tahapan berikutnya (evaluate/predict)
    mape_path = os.path.join(MODELS_DIR, f"{ticker}_linear_regression_mape.csv")
    pd.DataFrame([{"model": "linear_regression", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (Linear Regression): {mape:.4f}, saved to {mape_path}")

def train_random_forest_model(ticker, window_size=5):
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

    model = RandomForestRegressor(n_estimators=100, random_state=42)
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

def train_decision_tree_model(ticker, window_size=5):
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

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_decision_tree_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_decision_tree_mape.csv")
    pd.DataFrame([{"model": "decision_tree", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (Decision Tree): {mape:.4f}, saved to {mape_path}")

def train_catboost_model(ticker, window_size=5):
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

    model = CatBoostRegressor(iterations=100, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_catboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_catboost_mape.csv")
    pd.DataFrame([{"model": "catboost", "mape": mape}]).to_csv(mape_path, index=False)

def train_knn_model(ticker, window_size=5):
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

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_knn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_knn_mape.csv")
    pd.DataFrame([{"model": "knn", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (KNeighborsRegressor): {mape:.4f}, saved to {mape_path}")

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

def train_ridge_model(ticker, window_size=5):
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

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_ridge_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_ridge_mape.csv")
    pd.DataFrame([{"model": "ridge", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (Ridge): {mape:.4f}, saved to {mape_path}")

def train_svr_model(ticker, window_size=5):
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_svr_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_svr_mape.csv")
    pd.DataFrame([{"model": "svr", "mape": mape}]).to_csv(mape_path, index=False)
    print(f"MAPE for {ticker} (SVR): {mape:.4f}, saved to {mape_path}")

def train_xgboost_model(ticker, window_size=5):
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

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mape_path = os.path.join(MODELS_DIR, f"{ticker}_xgboost_mape.csv")
    pd.DataFrame([{"model": "xgboost", "mape": mape}]).to_csv(mape_path, index=False)


# Melatih model untuk seluruh ticker saat file ini dijalankan langsung
if __name__ == "__main__":
    for ticker in TICKERS:
        train_linear_regression_model(ticker)
        train_random_forest_model(ticker)
        train_decision_tree_model(ticker)
        train_catboost_model(ticker)
        train_knn_model(ticker)
        train_lightgbm_model(ticker)
        train_ridge_model(ticker)
        train_svr_model(ticker)
        train_xgboost_model(ticker)
