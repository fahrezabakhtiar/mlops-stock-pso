# decision_tree.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model regresi menggunakan DecisionTreeRegressor
# dari scikit-learn. Model ini merupakan baseline sederhana untuk memprediksi data
# time series dengan pendekatan sliding window.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model Decision Tree Regressor
#   berdasarkan data training dan testing yang telah disiapkan.

# Dependencies:
# - numpy: Untuk pemrosesan data numerik dalam bentuk array.
# - sklearn.tree.DecisionTreeRegressor: Model regresi berbasis pohon keputusan.
# - sklearn.metrics.mean_absolute_percentage_error: Untuk evaluasi performa model dengan MAPE.

# Output:
# - Objek model DecisionTreeRegressor terlatih.
# - Nilai MAPE (Mean Absolute Percentage Error) pada data test sebagai indikator akurasi.

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Decision Tree Regressor dan menghitung akurasi menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, ukuran (n_samples, window_size).
        y_train (np.ndarray): Target training, ukuran (n_samples,).
        X_test  (np.ndarray): Fitur testing, ukuran (n_samples, window_size).
        y_test  (np.ndarray): Target testing, ukuran (n_samples,).

    Returns:
        model: Objek DecisionTreeRegressor yang telah dilatih.
        mape (float): Mean Absolute Percentage Error (MAPE) pada data test.
    """
    # 1. Inisialisasi dan pelatihan model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 2. Prediksi pada data test
    preds = model.predict(X_test)

    # 3. Hitung MAPE
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape