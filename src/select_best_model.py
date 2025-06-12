# Tujuan: Memilih model terbaik (MAPE terkecil) untuk setiap ticker dan menyimpannya sebagai _best_model.pkl
# Juga menghasilkan ringkasan performa semua model yang telah dilatih.

import os
import glob
import pandas as pd
import shutil

def main():
    # Menentukan direktori utama dan lokasi folder model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    # Mengambil seluruh file *_mape.csv yang berisi hasil evaluasi model
    mape_files = glob.glob(os.path.join(MODELS_DIR, "*_mape.csv"))
    if not mape_files:
        print("Tidak ditemukan file MAPE di folder models/. Jalankan train terlebih dahulu.")
        return

    all_mape = []

    # Membaca dan menggabungkan seluruh file MAPE ke dalam satu dataframe
    for mape_path in mape_files:
        df = pd.read_csv(mape_path)
        basename = os.path.basename(mape_path)
        
        # Mengambil informasi ticker dan model_type dari nama file
        ticker = basename.split("_")[0]
        model_type = "_".join(basename.split("_")[1:-1])  # contoh: linear_regression

        df["ticker"] = ticker
        df["model_type"] = model_type
        all_mape.append(df)

    all_mape_df = pd.concat(all_mape, ignore_index=True)
    print("All model scores:")
    print(all_mape_df)

    # Menyimpan ringkasan seluruh performa model
    all_summary_path = os.path.join(MODELS_DIR, "all_models_mape_summary.csv")
    all_mape_df.to_csv(all_summary_path, index=False)
    print(f"\nAll models MAPE summary saved to {all_summary_path}")

    # Memilih model terbaik (MAPE terkecil) untuk setiap ticker
    best_rows = all_mape_df.loc[all_mape_df.groupby("ticker")["mape"].idxmin()].reset_index(drop=True)
    print("\nBest models per ticker:")
    print(best_rows)

    # Menyalin file model terbaik untuk setiap ticker sebagai {ticker}_best_model.pkl
    for _, row in best_rows.iterrows():
        ticker = row["ticker"]
        model_type = row["model_type"]
        pattern = f"{ticker}_{model_type}_model.pkl"
        model_path = os.path.join(MODELS_DIR, pattern)
        if os.path.exists(model_path):
            best_model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
            shutil.copyfile(model_path, best_model_path)
            print(f"Copied {model_path} -> {best_model_path}")
        else:
            print(f"Warning: Model file not found for {ticker} ({model_path})")

    # Menyimpan ringkasan model terbaik ke file CSV
    best_summary_path = os.path.join(MODELS_DIR, "best_models_summary.csv")
    best_rows.to_csv(best_summary_path, index=False)
    print(f"\nSummary disimpan di {best_summary_path}")

if __name__ == "__main__":
    main()