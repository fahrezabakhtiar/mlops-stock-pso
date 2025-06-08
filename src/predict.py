import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocess import load_and_clean, scale_data

def forecast_next_days(ticker, days=10):
    model = load_model(f"models/{ticker}_model.h5")
    df = load_and_clean(f"data/raw/{ticker}.csv")
    scaled, scaler = scale_data(df)
    window = scaled[-30:].reshape(1, 30, 1)
    predictions = []
    for _ in range(days):
        pred = model.predict(window)[0][0]
        predictions.append(pred)
        window = np.append(window[:,1:,:], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1))