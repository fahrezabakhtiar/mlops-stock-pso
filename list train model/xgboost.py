# xgboost.py
# Folder: list train model/
# Skrip standar untuk melatih model XGBoost Regressor
# Fungsi utama: build_and_train(X_train, y_train, X_test, y_test)

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error


def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Build and train an XGBoost Regressor model, then evaluate using MAPE.

    Args:
        X_train (np.ndarray): Training features, shape (n_samples, window_size).
        y_train (np.ndarray): Training targets, shape (n_samples,).
        X_test  (np.ndarray): Test features, shape (n_samples, window_size).
        y_test  (np.ndarray): Test targets, shape (n_samples,).

    Returns:
        model: Trained XGBRegressor instance.
        mape (float): Mean Absolute Percentage Error on test set.
    """
    # 1. Initialize and train model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # 2. Predict on test set
    preds = model.predict(X_test)

    # 3. Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape