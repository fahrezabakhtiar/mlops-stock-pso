# main.py
# Tujuan: Menjalankan pipeline utama untuk
# - Mengambil data (extract)
# - Melatih model
# - Memilih model terbaik
# - Melakukan prediksi 30 hari ke depan
# Output disimpan dalam folder models/

import sys
import os
import traceback
import subprocess

from extract import fetch_data
from config import TICKERS
from select_best_model import main as select_best_model_main
from predict import predict_next_days

# Setup environment path agar import antar modul tetap aman
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Fungsi untuk mengambil data saham dari sumber eksternal
def run_extract():
    print("=== EXTRACT ===")
    fetch_data(TICKERS)

# Fungsi untuk menjalankan training model
def run_train():
    print("\n=== TRAIN MODEL (train.py) ===")
    try:
        # Pastikan path ke train.py benar, dan tetap berjalan meskipun user run dari folder lain
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        result = subprocess.run([sys.executable, train_path], check=True)
    except subprocess.CalledProcessError as e:
        print("\n!!! TRAIN FAILED !!!")
        raise e  # Melempar kembali agar bisa ditangkap di try utama

# Fungsi untuk memilih model terbaik berdasarkan skor MAPE
def run_select_best_model():
    print("\n=== SELECT BEST MODEL ===")
    select_best_model_main()

# Fungsi untuk membuat prediksi 30 hari ke depan untuk setiap ticker
def run_predict():
    print("\n=== PREDICT (30 days forward) ===")
    for ticker in TICKERS:
        predict_next_days(ticker)

# Main Execution
if __name__ == "__main__":
    try:
        # Jalankan seluruh pipeline
        run_extract()
        run_train()
        run_select_best_model()
        run_predict()
        print("\n=== PIPELINE SELESAI! ===")
    except Exception as e:
        print("\n!!! PIPELINE GAGAL !!!")
        traceback.print_exc()  # Tampilkan detail error jika gagal
        sys.exit(1)