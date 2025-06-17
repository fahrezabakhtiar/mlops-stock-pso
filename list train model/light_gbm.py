# light_gbm.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi menggunakan LightGBM Regressor
# (LGBMRegressor), sebuah model boosting berbasis gradient yang efisien dan cepat.
# Model ini cocok untuk digunakan pada dataset time series dengan pendekatan sliding window.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model LightGBM Regressor
#   berdasarkan data training dan testing yang telah disiapkan.

# Dependencies:
# - numpy: Untuk representasi array numerik.
# - lightgbm.LGBMRegressor: Model regresi dari library LightGBM.
# - sklearn.metrics.mean_absolute_percentage_error: Untuk evaluasi performa model
#   menggunakan metrik MAPE (Mean Absolute Percentage Error).

# Output:
# - Objek model LGBMRegressor yang telah dilatih.
# - Nilai MAPE (Mean Absolute Percentage Error) pada data test.

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model LightGBM Regressor dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek LGBMRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape