# knn.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi menggunakan K-Nearest Neighbors (KNN)
# dari scikit-learn. Cocok digunakan sebagai baseline sederhana dalam memprediksi data
# time series berbasis sliding window.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model KNN Regressor
#   dengan menggunakan data training dan testing.

# Dependencies:
# - numpy: Untuk pemrosesan array numerik.
# - sklearn.neighbors.KNeighborsRegressor: Model KNN untuk regresi.
# - sklearn.metrics.mean_absolute_percentage_error: Untuk menghitung MAPE
#   sebagai metrik evaluasi performa.

# Output:
# - Objek model KNeighborsRegressor yang telah dilatih.
# - Nilai MAPE (Mean Absolute Percentage Error) sebagai ukuran akurasi model.

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model K-Nearest Neighbors Regressor dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek KNeighborsRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape