# lasso_regressor.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk membangun dan melatih model regresi menggunakan
# Lasso Regression dari pustaka scikit-learn.
# Model ini cocok untuk regresi linier dengan penalti L1, berguna untuk seleksi fitur
# dan mengurangi overfitting dalam data berukuran besar atau multikolinearitas.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model Lasso
#   menggunakan data pelatihan dan pengujian. Evaluasi dilakukan menggunakan
#   metrik Mean Absolute Percentage Error (MAPE).

# Dependencies:
# - numpy: Untuk manipulasi array numerik.
# - sklearn.linear_model.Lasso: Algoritma regresi linier dengan regularisasi L1.
# - sklearn.metrics.mean_absolute_percentage_error: Metrik evaluasi akurasi model.

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Lasso Regression dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek Lasso yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape