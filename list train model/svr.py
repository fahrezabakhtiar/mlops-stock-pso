# svr.py
# Folder: list train model/
# Deskripsi:
# Skrip ini digunakan untuk melatih model Support Vector Regression (SVR) 
# menggunakan kernel RBF. SVR cocok untuk memodelkan hubungan non-linear 
# antara fitur dan target, terutama pada data deret waktu seperti harga saham.

# Fungsi utama:
# - build_and_train: Membangun, melatih, dan mengevaluasi model SVR 
#   berdasarkan data pelatihan dan pengujian.

# Dependencies:
# - numpy: Untuk representasi numerik array.
# - sklearn.svm.SVR: Model Support Vector Regression.
# - sklearn.preprocessing.StandardScaler: Untuk standarisasi fitur.
# - sklearn.metrics.mean_absolute_percentage_error: Evaluasi akurasi model (MAPE).

# Output:
# - model: Objek SVR yang telah dilatih.
# - mape: Nilai Mean Absolute Percentage Error pada data testing.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Melatih model Support Vector Regression (SVR) dengan kernel RBF dan 
    mengevaluasi performanya menggunakan MAPE.

    Args:
        X_train (np.ndarray): Fitur training, shape (n_samples, window_size).
        y_train (np.ndarray): Target training, shape (n_samples,).
        X_test  (np.ndarray): Fitur testing, shape (n_samples, window_size).
        y_test  (np.ndarray): Target testing, shape (n_samples,).

    Returns:
        model: Objek SVR yang telah dilatih.
        mape (float): Mean Absolute Percentage Error pada data test.
    """
    # 1. Standarisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 2. Inisialisasi dan latih model SVR dengan kernel RBF
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)

    # 3. Prediksi data test dan hitung MAPE
    preds = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, preds)

    return model, mape
