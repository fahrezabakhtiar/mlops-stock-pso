import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(path):
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df = df[['Close']].dropna()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_windows(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)