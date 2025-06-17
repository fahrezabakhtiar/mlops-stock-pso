import os
import glob
import importlib.util
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from config import TICKERS

# Path ke root direktori proyek (naik satu level dari src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path ke folder berisi skrip-skrip model (misalnya: linear_regression.py, xgboost.py, dll.)
MODELS_SRC_DIR = os.path.join(ROOT_DIR, "list train model")

# Path ke folder output untuk file model (*.pkl) dan metrik evaluasi (*.csv)
OUTPUT_MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

def discover_model_scripts(src_dir):
    """
    Mengembalikan daftar file Python (.py) dalam folder model,
    mengecualikan file seperti __init__.py.
    """
    pattern = os.path.join(src_dir, "*.py")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("__")]
    return files

def import_model_module(path):
    """
    Mengimpor sebuah file Python sebagai modul dinamis.
    Mengembalikan objek modul yang bisa digunakan untuk memanggil fungsi di dalamnya.
    """
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def train_all_models(ticker, window_size=5):
    """
    Melatih semua model yang ada di folder 'list train model' untuk satu ticker.
    Menyimpan model hasil pelatihan dan metrik MAPE-nya.

    Args:
        ticker (str): simbol saham (misal: 'BBCA.JK', 'TLKM.JK').
        window_size (int): ukuran jendela (window) untuk membuat fitur time series.
    """
    # Membaca data harga penutupan (Close) saham dari file CSV
    data_path = os.path.join(ROOT_DIR, "data", "raw", f"{ticker}.csv")
    df = pd.read_csv(data_path)
    series = df['Close'].values

    # Membentuk fitur dan target berdasarkan window_size (sliding window)
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)

    # Membagi data menjadi train dan test (80:20)
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Melatih semua model dari skrip yang ditemukan
    for script_path in discover_model_scripts(MODELS_SRC_DIR):
        mod = import_model_module(script_path)
        model_name = os.path.splitext(os.path.basename(script_path))[0]

        # Asumsi: tiap file model punya fungsi bernama `build_and_train()`
        model, mape = mod.build_and_train(X_train, y_train, X_test, y_test)

        # Menyimpan model ke file pickle
        model_file = os.path.join(OUTPUT_MODELS_DIR, f"{ticker}_{model_name}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Menyimpan nilai MAPE ke file CSV
        mape_file = os.path.join(OUTPUT_MODELS_DIR, f"{ticker}_{model_name}_mape.csv")
        pd.DataFrame([{"model": model_name, "mape": mape}]).to_csv(mape_file, index=False)

        print(f"[{ticker}] Model {model_name} done. MAPE={mape:.4f}")

if __name__ == "__main__":
    # Melatih semua model untuk setiap ticker yang telah ditentukan
    for t in TICKERS:
        train_all_models(t)