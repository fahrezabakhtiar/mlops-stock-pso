# MLOps Stock Forecasting Dashboard
![image](https://github.com/user-attachments/assets/57a98a20-299b-4680-9fd6-4496d3ca6582)

Merupakan project dashboard dan pipeline end-to-end untuk prediksi harga saham menggunakan pendekatan machine learning (time series), dilengkapi dengan CI/CD, evaluasi model otomatis, pemilihan model terbaik, dan visualisasi hasil prediksi (dashboarding) via Streamlit.

## ⚙️ Akses Apps
Kunjungi link berikut ini: https://stock-mlops-1c.streamlit.app/

## 🚀 Fitur Utama
- Ekstraksi data historis saham dari Yahoo Finance
- Pelatihan model (bisa disesuaikan setiap menambahkan model di folder `list train model`)
- Evaluasi MAPE otomatis dan seleksi model terbaik
- Prediksi 30 hari ke depan (business days)
- Dashboard interaktif via Streamlit (Pilih Ticker, Download Hasil Prediksi dengan Model Terbaik, Pilih Rentang Tanggal, Interaksi dengan Grafik dan Tabel, Download Hasil Evaluasi MAPE Semua Model)
- Pipeline otomatis via GitHub Actions

## ⚙️ Konfigurasi Fleksibel untuk Test Pipeline
### 📈 Ticker Saham (`config.py`)
Daftar ticker dapat diubah dengan bebas melalui `src/config.py`

### 🗓️ Rentang Data Historis ('extract.py')
Tanggal awal data (`start=`) bisa disesuaikan di `extract.py` sesuai kebutuhan, misalnya:
```python
fetch_data(tickers, start="2020-01-01")
```
### 🔄 Model Training (folder `list train model`)
Folder `list train model` dapat ditambah dengan model-model baru yang diinginkan. Setiap model yang ditambahkan akan otomatis dilatih + diregistrasi dan hasil MAPE-nya disimpan sebagai file `*_mape.csv`. Pipeline akan otomatis memilih model terbaik berdasarkan MAPE setiap kali CI/CD di-trigger oleh commit (push) di branch main pada path yang didefinisikan di `ci-cd.yml`.

### 📦 Daftar Model yang Sudah Tersedia
| Model                          | File Python                      |
|--------------------------------|----------------------------------|
| CatBoost                       | `catboost.py`          |
| Decision Tree Regressor       | `decision_tree.py`     |
| K-Nearest Neighbors (KNN)      | `knn.py`               |
| LightGBM                       | `light_gbm.py`          |
| Linear Regression              | `linear_regression.py` |
| Random Forest                  | `random_forest.py`                |
| Ridge Regression               | `ridge.py`             |
| Support Vector Regressor (SVR) | `svr.py`               |
| XGBoost                        | `xgboost.py`           |

## 🧰 Tools yang Digunakan
### Python Packages
- `yfinance` — ekstraksi data saham dari Yahoo Finance  
- `pandas`, `numpy` — manipulasi dan analisis data  
- `scikit-learn` — pelatihan dan evaluasi model machine learning
- `XGBoost` — library untuk Extreme Gradient Boosting
- `LightGBM` — library gradient boosting dari Microsoft
- `CatBoost` — library gradient boosting dari Yandex
- `plotly` — visualisasi interaktif  
- `streamlit` — UI dashboard prediksi

### Platform & DevOps
- `GitHub` — repositori kode dan version control  
- `GitHub Actions` — CI/CD pipeline otomatis  
- `VS Code` — editor pengembangan utama  

## ⚙️ Cara Menjalankan di Local
1. Clone repository

```bash 
git clone https://github.com/fahrezabakhtiar/mlops-stock-pso
```

2. Buat virtual environment 
2.1 Pastikan Python sudah terinstall
   
```bash 
python --version
```

Jika python tidak dikenali, coba:

```bash 
python3 --version
```

2.3 Buat Virtual Environment

Di root project:

```bash 
python -m venv venv
```

Atau jika pakai python3:

```bash 
python3 -m venv venv
```

Ini akan membuat folder venv/ berisi semua dependency terisolasi.

3. Aktifkan Virtual Environment
Windows:

```bash 
venv\Scripts\activate
```

macOS / Linux:

```bash 
source venv/bin/activate
```

Jika berhasil, terminal kamu akan muncul prefix seperti ini:

```bash 
(venv) user@machine:~/mlops-stock-pso$
```

4. Install Dependensi
Setelah venv aktif, install dengan:

```bash 
pip install -r requirements.txt
```

5. Anda terlebih dahulu dapat mengubah `config.py`, `extract.py`, dan menambahkan model baru di dalam folder `list train model` jika diinginkan, lalu jalankan pipeline end-to-end.
   
```bash 
python src/main.py
```

6. Jalankan Dashboard
   
```bash 
streamlit run app/dashboard.py

You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501 (bisa berubah jika dikonfigurasikan/ ada conflict port)
  Network URL: http://192.168.0.104:8501 (IP lokal bisa berubah tergantung koneksi dan DHCP)
```

![image](https://github.com/user-attachments/assets/c31118b2-84bf-4c1b-882f-269fff11ef1a)
![image](https://github.com/user-attachments/assets/151a9952-42de-42dc-b9ec-3055cd820519)

7. Menonaktifkan venv

Setelah selesai:
```bash 
deactivate
```

## ⚙️ Cara Menjalankan di Streamlit Cloud (Deployment Environment)

1. Push project ke GitHub
2. Pastikan repo Anda terdapat: app/dashboard.py

3. Kunjungi: https://streamlit.io/

  ![image](https://github.com/user-attachments/assets/f41aefa7-b812-4cb0-8650-d7ca28c3e626)

3. Sign in dengan GitHub dan klik "New app"

4. Pilih "Deploy a public app from GitHub"
  ![image](https://github.com/user-attachments/assets/a963c2f1-2011-4f36-b8dc-2c5d9a9b4278)

5. Isi:
* Repo GitHub Anda
* File utama (misal: app/dashboard.py)
* Branch: main
* App URL Anda

Klik Deploy!
  ![image](https://github.com/user-attachments/assets/c3606bbb-3d20-4b24-b2ff-d5f9e0315977)

6. Aplikasi berhasil di-deploy dan dapat diakses serta dibagikan menggunakan URL Anda.
![image](https://github.com/user-attachments/assets/3edf3edd-d1c1-4be7-b4d7-7c9a5a8ceded)
![image](https://github.com/user-attachments/assets/a7548c2f-7071-4c3a-a91b-c63f850fd515)

## 📦 Checklist Penting untuk Deployment:
 * ✅`requirements.txt` lengkap
 * ✅`dashboard.py` bisa di-run dengan streamlit run
 * ✅Tidak ada path hardcoded local
 * ✅Gunakan `config.py` untuk parameter yang fleksibel
 * ✅Hindari penulisan file di root saat di cloud (pakai folder `models/`, `data/`, dll)

## 💡 Kelebihan Deploy dengan Streamlit
* Gratis dan cepat
* Tidak perlu server pribadi
* Mendukung restart otomatis saat update GitHub

## 🏫 Anggota Kelompok
1. Andika Cahya Sutisna (5026221013)
2. Rizky Fahreza Bakhtiar (5026221075)
3. Satria Jati Kusuma (5026221165)
4. Muhammad Irsyad Fahmi (5026221187)
