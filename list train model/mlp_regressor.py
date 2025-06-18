# mlp_regressor.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk membangun dan melatih model regresi menggunakan
# Multi-Layer Perceptron Regressor (MLPRegressor) dari pustaka scikit-learn.
# Model ini menggunakan jaringan saraf tiruan dengan beberapa lapisan tersembunyi
# untuk mempelajari hubungan non-linier dalam data regresi.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model MLPRegressor
#   menggunakan data pelatihan dan pengujian. Evaluasi dilakukan menggunakan
#   metrik Mean Absolute Percentage Error (MAPE).

# Dependencies:
# - numpy: Untuk manipulasi array numerik.
# - sklearn.neural_network.MLPRegressor: Algoritma jaringan saraf regresi.
# - sklearn.metrics.mean_absolute_percentage_error: Metrik evaluasi akurasi model.

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Multi-Layer Perceptron Regressor dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek MLPRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape