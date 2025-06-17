# random_forest.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi menggunakan RandomForestRegressor
# dari scikit-learn. Model ini menggunakan ansambel decision tree untuk menghasilkan
# prediksi yang kuat terhadap noise dan overfitting.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model RandomForestRegressor
#   menggunakan data pelatihan dan pengujian.

# Dependencies:
# - numpy: Untuk array numerik.
# - sklearn.ensemble.RandomForestRegressor: Algoritma regresi random forest.
# - sklearn.metrics.mean_absolute_percentage_error: Metrik evaluasi akurasi model.

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Random Forest Regressor dan mengevaluasinya menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, shape (n_samples, window_size).
        y_train (np.ndarray): Target training, shape (n_samples,).
        X_test  (np.ndarray): Fitur testing, shape (n_samples, window_size).
        y_test  (np.ndarray): Target testing, shape (n_samples,).

    Returns:
        model: Objek RandomForestRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error pada data test.
    """
    # 1. Inisialisasi dan latih model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 2. Prediksi data test
    preds = model.predict(X_test)

    # 3. Evaluasi akurasi
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape