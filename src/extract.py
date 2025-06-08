import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def fetch_data(tickers, start="2022-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    os.makedirs("data/raw", exist_ok=True)

    for ticker in tickers:
        df = yf.download(ticker + ".JK", start=start, end=end, auto_adjust=True)

        # Jika kolomnya MultiIndex, datarkan
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Cari kolom 'Close' apapun bentuknya
        close_col = [col for col in df.columns if 'Close' in col][0]
        df = df[[close_col]].dropna()
        df = df.rename(columns={close_col: 'Close'})  # ganti nama jadi konsisten

        df = df.reset_index()
        df['Ticker'] = ticker
        df = df[['Date', 'Ticker', 'Close']]
        df.to_csv(f"data/raw/{ticker}.csv", index=False)