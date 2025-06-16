"""
Pipeline utama untuk forecasting harga saham.
Langkah-langkah:
1. Extract: Mengambil data dari sumber eksternal (fetch_data)
2. Train: Melatih model-model dari skrip di folder list train model
3. Select Best: Memilih model terbaik berdasarkan skor MAPE
4. Predict: Menghasilkan prediksi harga saham 30 hari ke depan

Semua output (model, metrik, prediksi) disimpan di folder models/
"""

import sys
import os
import traceback
import subprocess

from extract import fetch_data
from config import TICKERS
from select_best_model import main as select_best_model_main
from predict import predict_next_days

# Setup path agar modul dapat diimport dengan benar (meskipun dijalankan dari folder lain)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_extract():
    """Menjalankan proses pengambilan data saham mentah dari sumber eksternal."""
    print("=== EXTRACT ===")
    fetch_data(TICKERS)

def run_train():
    """
    Menjalankan proses pelatihan model dengan memanggil train.py sebagai subprocess.
    Subprocess digunakan untuk memisahkan eksekusi dan menjaga isolasi.
    """
    print("\n=== TRAIN MODEL (train.py) ===")
    try:
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        result = subprocess.run([sys.executable, train_path], check=True)
    except subprocess.CalledProcessError as e:
        print("\n!!! TRAIN FAILED !!!")
        raise e  # Biarkan error diteruskan ke blok try utama

def run_select_best_model():
    """Menjalankan pemilihan model terbaik berdasarkan skor MAPE terkecil."""
    print("\n=== SELECT BEST MODEL ===")
    select_best_model_main()

def run_predict():
    """Menghasilkan prediksi harga saham selama 30 hari ke depan untuk tiap ticker."""
    print("\n=== PREDICT (30 days forward) ===")
    for ticker in TICKERS:
        predict_next_days(ticker)

if __name__ == "__main__":
    try:
        # Jalankan pipeline end-to-end
        run_extract()
        run_train()
        run_select_best_model()
        run_predict()
        print("\n=== PIPELINE SELESAI! ===")
    except Exception as e:
        # Tangani error global dan tampilkan traceback
        print("\n!!! PIPELINE GAGAL !!!")
        traceback.print_exc()
        sys.exit(1)