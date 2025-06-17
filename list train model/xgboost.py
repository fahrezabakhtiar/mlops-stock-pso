# xgboost.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model XGBoost Regressor, 
# yang merupakan algoritma berbasis pohon yang sangat efektif 
# untuk regresi dan prediksi deret waktu. XGBoost dikenal karena 
# kemampuannya menangani outlier, missing value, dan kompleksitas data tinggi.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model XGBoost 
#   berdasarkan data pelatihan dan pengujian.

# Dependencies:
# - numpy: Untuk representasi numerik array.
# - xgboost.XGBRegressor: Algoritma regresi dari library XGBoost.
# - sklearn.metrics.mean_absolute_percentage_error: Metrik evaluasi MAPE.

# Output:
# - model: Objek XGBRegressor yang telah dilatih.
# - mape: Nilai Mean Absolute Percentage Error pada data testing.

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
    Melatih model XGBoost Regressor dan mengevaluasinya menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, shape (n_samples, window_size).
        y_train (np.ndarray): Target training, shape (n_samples,).
        X_test  (np.ndarray): Fitur testing, shape (n_samples, window_size).
        y_test  (np.ndarray): Target testing, shape (n_samples,).

    Returns:
        model: Objek XGBRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error pada data test.
    """
    # 1. Inisialisasi dan latih model XGBoost
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape