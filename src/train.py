import os
import mlflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, SimpleRNN, Conv1D, Flatten, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocess import load_and_clean, scale_data, create_windows
from src.evaluate import mape_score

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_simple_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_bidirectional_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_best_model(ticker, max_epochs=100):
    # Load and preprocess data
    df = load_and_clean(f"data/raw/{ticker}.csv")
    scaled, scaler = scale_data(df)
    X, y = create_windows(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split train-test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define candidate models
     candidates = {
        "lstm": build_lstm_model((X.shape[1], 1)),
        "gru": build_gru_model((X.shape[1], 1)),
        "simple_rnn": build_simple_rnn_model((X.shape[1], 1)),
        "cnn": build_cnn_model((X.shape[1], 1)),
        "bidirectional_lstm": build_bidirectional_lstm_model((X.shape[1], 1))
    }

    best_model = None
    best_score = float("inf")
    best_name = ""

    mlflow.set_experiment(f"{ticker}_forecast")

    # Train and evaluate each model
     for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=max_epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            preds = model.predict(X_test)
            score = mape_score(y_test, preds)
            mlflow.log_metric("mape", score)

            if score < best_score:
                best_score = score
                best_model = model
                best_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    best_model.save(f"models/{ticker}_model.h5")
    print(f"Best model for {ticker} is '{best_name}' with MAPE: {best_score:.4f}")