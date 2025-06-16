import os
import glob
import pandas as pd
import shutil

def main():
    # Menentukan direktori root proyek dan folder tempat menyimpan model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    # Mengambil semua file hasil evaluasi model dalam format CSV (berakhiran *_mape.csv)
    mape_files = glob.glob(os.path.join(MODELS_DIR, "*_mape.csv"))
    if not mape_files:
        print("Tidak ditemukan file MAPE di folder models/. Jalankan train terlebih dahulu.")
        return

    all_mape = []

    # Membaca semua file MAPE dan menggabungkannya ke dalam satu DataFrame
    for mape_path in mape_files:
        df = pd.read_csv(mape_path)
        basename = os.path.basename(mape_path)

        # Ekstraksi ticker dan model_type dari nama file (format: {ticker}_{model_type}_mape.csv)
        ticker = basename.split("_")[0]
        model_type = "_".join(basename.split("_")[1:-1])  # Menggabungkan bagian model_type yang terpisah dengan "_"

        # Menambahkan informasi ticker dan model ke dataframe
        df["ticker"] = ticker
        df["model_type"] = model_type
        all_mape.append(df)

    # Menggabungkan semua data MAPE dari berbagai model & ticker
    all_mape_df = pd.concat(all_mape, ignore_index=True)
    print("All model scores:")
    print(all_mape_df)

    # Menyimpan ringkasan performa semua model ke file CSV
    all_summary_path = os.path.join(MODELS_DIR, "all_models_mape_summary.csv")
    all_mape_df.to_csv(all_summary_path, index=False)
    print(f"\nAll models MAPE summary saved to {all_summary_path}")

    # Menentukan model terbaik untuk setiap ticker (berdasarkan nilai MAPE terkecil)
    best_rows = all_mape_df.loc[all_mape_df.groupby("ticker")["mape"].idxmin()].reset_index(drop=True)
    print("\nBest models per ticker:")
    print(best_rows)

    # Menyalin file model terbaik sebagai {ticker}_best_model.pkl
    for _, row in best_rows.iterrows():
        ticker = row["ticker"]
        model_type = row["model_type"]

        # Path file model asli dan file tujuan
        source_model_file = os.path.join(MODELS_DIR, f"{ticker}_{model_type}.pkl")
        best_model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")

        # Salin file model terbaik jika tersedia
        if os.path.exists(source_model_file):
            shutil.copyfile(source_model_file, best_model_path)
            print(f"Copied {source_model_file} -> {best_model_path}")
        else:
            print(f"Warning: Model file not found for {ticker} ({source_model_file})")

    # Menyimpan ringkasan model terbaik ke file CSV
    best_summary_path = os.path.join(MODELS_DIR, "best_models_summary.csv")
    best_rows.to_csv(best_summary_path, index=False)
    print(f"\nSummary disimpan di {best_summary_path}")

if __name__ == "__main__":
    main()