# Tahapan Train: Melatih model prediksi harga saham untuk setiap ticker
# dan menyimpan model serta metrik evaluasinya (MAPE).

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
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

# Melatih model untuk seluruh ticker saat file ini dijalankan langsung
if __name__ == "__main__":
    for ticker in TICKERS:
        train_linear_regression_model(ticker)