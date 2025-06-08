import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def fetch_data(tickers, start="2019-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    os.makedirs("data/raw", exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker + ".JK", start=start, end=end)
        df = df[['Close']].dropna()
        df.to_csv(f"data/raw/{ticker}.csv")