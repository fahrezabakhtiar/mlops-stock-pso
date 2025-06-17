# ridge.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi linear dengan regularisasi L2 
# menggunakan Ridge Regression dari scikit-learn. Ridge Regression efektif 
# dalam mengatasi multikolinearitas dan overfitting pada data.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model Ridge Regression
#   berdasarkan data pelatihan dan pengujian.

# Dependencies:
# - numpy: Untuk representasi array numerik.
# - sklearn.linear_model.Ridge: Algoritma regresi Ridge.
# - sklearn.metrics.mean_absolute_percentage_error: Metrik evaluasi akurasi model (MAPE).

# Output:
# - model: Objek Ridge yang telah dilatih.
# - mape: Nilai Mean Absolute Percentage Error pada data testing.

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Ridge Regression dan mengevaluasinya menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, shape (n_samples, window_size).
        y_train (np.ndarray): Target training, shape (n_samples,).
        X_test  (np.ndarray): Fitur testing, shape (n_samples, window_size).
        y_test  (np.ndarray): Target testing, shape (n_samples,).

    Returns:
        model: Objek Ridge Regression yang telah dilatih.
        mape (float): Mean Absolute Percentage Error pada data test.
    """
    # 1. Inisialisasi dan latih model Ridge dengan alpha default (1.0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # 2. Prediksi data test
    preds = model.predict(X_test)

    # 3. Evaluasi akurasi
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape