# train.py
import os
import glob
import importlib.util
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from config import TICKERS

# Path ke folder yang berisi skrip-skrip model
MODELS_SRC_DIR = os.path.join(os.path.dirname(__file__), "list train model")
# Folder output untuk model dan MAPE
OUTPUT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

def discover_model_scripts(src_dir):
    """Kembalikan list path .py di folder model, kecuali __init__.py."""
    pattern = os.path.join(src_dir, "*.py")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("__")]
    return files

def import_model_module(path):
    """Import sebuah file .py sebagai modul, kembalikan namespace-nya."""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def train_all_models(ticker, window_size=5):
    # Baca data dan buat window (sama seperti sebelumnya)...
    df = pd.read_csv(f"data/raw/{ticker}.csv")
    series = df['Close'].values
    # -- buat fitur/target --
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Temukan dan jalankan tiap skrip model
    for script_path in discover_model_scripts(MODELS_SRC_DIR):
        mod = import_model_module(script_path)
        model_name = os.path.splitext(os.path.basename(script_path))[0]  # e.g. linear_regression_model
        # Asumsi: setiap modul punya fungsi build_and_train(X_train, y_train, X_test, y_test)
        # yang mengembalikan (trained_model, mape_score)
        model, mape = mod.build_and_train(X_train, y_train, X_test, y_test)

        # Simpan model .pkl
        model_file = os.path.join(OUTPUT_MODELS_DIR, f"{ticker}_{model_name}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Simpan MAPE
        mape_file = os.path.join(OUTPUT_MODELS_DIR, f"{ticker}_{model_name}_mape.csv")
        pd.DataFrame([{"model": model_name, "mape": mape}]).to_csv(mape_file, index=False)

        print(f"[{ticker}] Model {model_name} done. MAPE={mape:.4f}")

if __name__ == "__main__":
    for t in TICKERS:
        train_all_models(t)