"""
Tahapan Extract:
Mengunduh data historis harga saham dari Yahoo Finance,
kemudian menyimpannya dalam bentuk CSV di folder data/raw.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from config import TICKERS  # Daftar ticker saham yang akan diambil

def fetch_data(tickers, start="2022-01-01", end=None):
    """
    Mengambil data harga penutupan saham dari Yahoo Finance
    dan menyimpannya dalam bentuk file CSV di direktori data/raw.

    Args:
        tickers (list): Daftar kode saham (string) yang akan diambil datanya.
        start (str): Tanggal awal pengambilan data, dalam format "YYYY-MM-DD".
        end (str, optional): Tanggal akhir data. Default: hari ini.
    """

    # Menentukan path direktori output
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Gunakan tanggal hari ini jika end tidak diberikan
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Iterasi setiap ticker untuk mengambil dan menyimpan datanya
    for ticker in tickers:
        print(f"Fetching {ticker}...")

        # Ambil data dari Yahoo Finance
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)

        # Skip jika tidak ada data yang ditemukan
        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        # Flatten MultiIndex jika kolom hasil download memiliki hierarki
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Ambil kolom harga penutupan (Close) dan hilangkan nilai kosong
        close_col = [col for col in df.columns if 'Close' in col][0]
        df = df[[close_col]].dropna()
        df = df.rename(columns={close_col: 'Close'})

        # Tambahkan kolom ticker dan reset index
        df = df.reset_index()
        df['Ticker'] = ticker
        df = df[['Date', 'Ticker', 'Close']]

        # Simpan ke file CSV
        out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

# Jika dijalankan langsung (bukan sebagai modul), ambil semua ticker dari config
if __name__ == "__main__":
    fetch_data(TICKERS)