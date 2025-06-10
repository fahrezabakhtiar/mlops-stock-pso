import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from config import TICKERS

def fetch_data(tickers, start="2022-01-01", end=None):
    # Dapatkan path ke root project (parent dari src)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)

    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker + ".JK", start=start, end=end, auto_adjust=True)
        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        close_col = [col for col in df.columns if 'Close' in col][0]
        df = df[[close_col]].dropna()
        df = df.rename(columns={close_col: 'Close'})
        df = df.reset_index()
        df['Ticker'] = ticker
        df = df[['Date', 'Ticker', 'Close']]
        out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    tickers = TICKERS
    fetch_data(tickers)