import os
import pandas as pd
import numpy as np
import pickle

def predict_next_days(ticker, window_size=5, n_days=30):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    
    # Load closing price data
    df = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"))
    close = df['Close'].values

    # Load best model
    best_model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
    if not os.path.exists(best_model_path):
        print(f"Best model for {ticker} not found!")
        return None

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    # Mulai window dari data paling akhir
    last_window = list(close[-window_size:])
    predictions = []

    for _ in range(n_days):
        X_input = np.array(last_window).reshape(1, -1)
        pred = model.predict(X_input)[0]
        predictions.append(pred)
        # update window: buang data paling lama, tambahkan prediksi baru
        last_window.pop(0)
        last_window.append(pred)

    # Simpan ke CSV
    forecast_path = os.path.join(MODELS_DIR, f"{ticker}_forecast_30d.csv")
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq="D")
    forecast_df = pd.DataFrame({
        "Date": future_days,
        "Forecast": predictions
    })
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Saved 30-days forecast for {ticker} to {forecast_path}")
    return forecast_df

if __name__ == "__main__":
    for ticker in ['BMRI', 'BBRI', 'BBCA']:
        predict_next_days(ticker)
