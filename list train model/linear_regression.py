# linear_regression.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi menggunakan Linear Regression
# dari scikit-learn. Model ini merupakan baseline sederhana dan interpretatif
# untuk memprediksi data time series berbasis sliding window.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model LinearRegression
#   menggunakan data training dan testing.

# Dependencies:
# - numpy: Untuk representasi array numerik.
# - sklearn.linear_model.LinearRegression: Model regresi linier dari scikit-learn.
# - sklearn.metrics.mean_absolute_percentage_error: Digunakan untuk mengevaluasi performa
#   model menggunakan MAPE (Mean Absolute Percentage Error).

# Output:
# - Objek model LinearRegression yang telah dilatih.
# - Nilai MAPE (Mean Absolute Percentage Error) pada data testing.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Linear Regression dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek LinearRegression yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape