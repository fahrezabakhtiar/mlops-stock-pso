# Tahapan Extract: Mengunduh data historis harga saham dari Yahoo Finance
# dan menyimpannya dalam folder data/raw sebagai CSV.

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from config import TICKERS  # Mengimpor daftar ticker dari config.py

def fetch_data(tickers, start="2022-01-01", end=None):
    """
    Mengambil data harga penutupan saham dari Yahoo Finance
    dan menyimpannya dalam bentuk file CSV di direktori data/raw.

    Args:
        tickers (list): Daftar kode saham (tanpa .JK).
        start (str): Tanggal awal data dalam format YYYY-MM-DD.
        end (str, optional): Tanggal akhir data. Jika None, pakai tanggal hari ini.
    """

    # Menentukan direktori output relatif terhadap posisi file ini
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Jika tanggal akhir tidak diberikan, gunakan hari ini
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Iterasi setiap ticker dan simpan datanya
    for ticker in tickers:
        print(f"Fetching {ticker}...")

        # Unduh data historis saham dengan penyesuaian dividen dan stock split
        df = yf.download(ticker + ".JK", start=start, end=end, auto_adjust=True)

        # Jika data kosong, tampilkan peringatan
        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        # Jika kolom berupa MultiIndex, ratakan jadi kolom tunggal
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Ambil kolom harga penutupan (Close) dan hilangkan nilai kosong
        close_col = [col for col in df.columns if 'Close' in col][0]
        df = df[[close_col]].dropna()
        df = df.rename(columns={close_col: 'Close'})

        # Reset index dan tambahkan kolom Ticker
        df = df.reset_index()
        df['Ticker'] = ticker
        df = df[['Date', 'Ticker', 'Close']]

        # Simpan sebagai CSV di direktori data/raw
        out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

# Jika file ini dijalankan langsung, ambil data untuk semua ticker
if __name__ == "__main__":
    tickers = TICKERS
    fetch_data(tickers)