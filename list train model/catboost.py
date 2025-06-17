# catboost.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi CatBoost pada data time series
# menggunakan pendekatan sliding window. Skrip ini dirancang agar kompatibel dengan pipeline
# utama (train.py) melalui fungsi standar `build_and_train`.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model CatBoostRegressor
#   menggunakan data training dan testing yang telah disiapkan.

# Dependencies:
# - numpy: Untuk memproses array numerik.
# - catboost: Library gradient boosting dari Yandex, cocok untuk model tabular.
# - sklearn.metrics: Untuk menghitung akurasi prediksi menggunakan MAPE.

# Output:
# - Objek model CatBoostRegressor terlatih.
# - Nilai MAPE (Mean Absolute Percentage Error) untuk mengukur akurasi model.

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model CatBoostRegressor dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur untuk training, dimensi (n_samples, window_size).
        y_train (np.ndarray): Target untuk training, dimensi (n_samples,).
        X_test  (np.ndarray): Fitur untuk testing, dimensi (n_samples, window_size).
        y_test  (np.ndarray): Target untuk testing, dimensi (n_samples,).

    Returns:
        model: Objek CatBoostRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = CatBoostRegressor(iterations=100, verbose=0)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape