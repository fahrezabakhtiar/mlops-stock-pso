import sys
import os
import traceback
import subprocess
from extract import fetch_data
from config import TICKERS
from select_best_model import main as select_best_model_main
from predict import predict_next_days

# Pastikan path root ada di sys.path (agar import tetap aman jika run dari root atau src)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_extract():
    print("=== EXTRACT ===")
    fetch_data(TICKERS)

def run_train():
    print("\n=== TRAIN MODEL (train.py) ===")
    try:
        # Path ke train.py yang pasti benar, apapun working directory kamu
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        result = subprocess.run([sys.executable, train_path], check=True)
    except subprocess.CalledProcessError as e:
        print("\n!!! TRAIN FAILED !!!")
        raise e

def run_select_best_model():
    print("\n=== SELECT BEST MODEL ===")
    select_best_model_main()

def run_predict():
    print("\n=== PREDICT (30 days forward) ===")
    for ticker in TICKERS:
        predict_next_days(ticker)

if __name__ == "__main__":
    try:
        run_extract()
        run_train()
        run_select_best_model()
        run_predict()
        print("\n=== PIPELINE SELESAI! ===")
    except Exception as e:
        print("\n!!! PIPELINE GAGAL !!!")
        traceback.print_exc()
        sys.exit(1)