# svr.py
# Folder: list train model/
# Skrip standar untuk melatih model Support Vector Regressor (SVR)
# Fungsi utama: build_and_train(X_train, y_train, X_test, y_test)

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Build and train an SVR model with RBF kernel, then evaluate using MAPE.

    Args:
        X_train (np.ndarray): Training features, shape (n_samples, window_size).
        y_train (np.ndarray): Training targets, shape (n_samples,).
        X_test  (np.ndarray): Test features, shape (n_samples, window_size).
        y_test  (np.ndarray): Test targets, shape (n_samples,).

    Returns:
        model: Trained SVR instance.
        mape (float): Mean Absolute Percentage Error on test set.
    """
    # 1. Standarisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 2. Inisialisasi dan latih model
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)

    # 3. Prediksi dan hitung MAPE
    preds = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape