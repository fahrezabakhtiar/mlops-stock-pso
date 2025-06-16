# Tujuan: Menghasilkan prediksi harga penutupan selama 30 hari ke depan
# menggunakan model terbaik yang telah dipilih (berformat {ticker}_best_model.pkl).
# Output disimpan sebagai CSV di folder models/.

import os
import pandas as pd
import numpy as np
import pickle
from config import TICKERS  # Daftar ticker saham yang ingin diprediksi

def predict_next_days(ticker, window_size=5, n_days=30):
    """
    Menghasilkan prediksi harga saham selama n_days ke depan menggunakan model terbaik.

    Args:
        ticker (str): simbol saham (misal: 'BBCA', 'TLKM').
        window_size (int): ukuran jendela sliding window untuk prediksi.
        n_days (int): jumlah hari ke depan yang ingin diprediksi.

    Returns:
        pd.DataFrame: dataframe berisi tanggal dan prediksi harga saham.
    """
    # Menentukan direktori proyek dan path folder
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    
    # Memuat data harga penutupan (Close) dari file CSV
    df = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"))
    close = df['Close'].values

    # Memuat model terbaik hasil training sebelumnya
    best_model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
    if not os.path.exists(best_model_path):
        print(f"Best model for {ticker} not found!")
        return None

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    # Mengambil window terakhir dari data historis untuk memulai prediksi
    last_window = list(close[-window_size:])
    predictions = []

    # Melakukan prediksi iteratif untuk n_days ke depan
    for _ in range(n_days):
        # Bentuk input sesuai format model: array 2D [1, window_size]
        X_input = np.array(last_window).reshape(1, -1)
        pred = model.predict(X_input)[0]
        predictions.append(pred)

        # Update window dengan nilai prediksi baru
        last_window.pop(0)
        last_window.append(pred)

    # Menentukan path untuk menyimpan hasil prediksi
    forecast_path = os.path.join(MODELS_DIR, f"{ticker}_forecast_30d.csv")
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    
    # Membuat tanggal prediksi sebanyak n_days (hanya hari kerja)
    future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq="B")
    
    forecast_df = pd.DataFrame({
        "Date": future_days,
        "Forecast": predictions
    })

    # Simpan ke CSV
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Saved 30-days forecast for {ticker} to {forecast_path}")

    return forecast_df

if __name__ == "__main__":
    # Jalankan proses prediksi untuk seluruh ticker
    for ticker in TICKERS:
        predict_next_days(ticker)