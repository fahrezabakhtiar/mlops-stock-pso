# MLOps Stock Forecasting Dashboard (Jakarta Stock Exchange)
Merupakan project dashboard dan pipeline end-to-end untuk prediksi harga saham menggunakan pendekatan machine learning (time series), dilengkapi dengan CI/CD, evaluasi model otomatis, pemilihan model terbaik, dan visualisasi hasil prediksi (dashboarding) via Streamlit.

## 🚀 Fitur Utama
- Ekstraksi data historis saham dari Yahoo Finance (ticker `.JK`)
- Pelatihan model (bisa disesuaikan setiap commit code train.py)
- Evaluasi MAPE otomatis dan seleksi model terbaik
- Prediksi 30 hari ke depan (business days)
- Dashboard interaktif via Streamlit (Pilih Ticker, Download Hasil Prediksi dengan Model Terbaik, Pilih Rentang Tanggal, Interaksi dengan Grafik dan Tabel)
- Pipeline otomatis via GitHub Actions

## ⚙️ Konfigurasi Fleksibel untuk Test Pipeline
### 🔄 Model Training (`train.py`)
File `train.py` dapat disesuaikan dengan model apapun (Linear Regression, Random Forest, SVR, dll). Setiap model yang dilatih akan otomatis teregistrasi dan hasil MAPE-nya disimpan sebagai file `*_mape.csv`. Pipeline akan otomatis memilih model terbaik berdasarkan MAPE setiap kali script dijalankan atau CI/CD di-trigger oleh commit.

### 📈 Ticker Saham (`config.py`)
Daftar ticker dapat diubah dengan bebas melalui `src/config.py` selama merupakan kode saham dari **Bursa Efek Indonesia (dengan akhiran `.JK`)**, misalnya `BBRI`, `BMRI`, `BBCA`, dll.

### 🗓️ Rentang Data Historis
Tanggal awal data (`start=`) bisa disesuaikan di `extract.py` sesuai kebutuhan, misalnya:
```python
fetch_data(tickers, start="2020-01-01")

## 🧰 Tools yang Digunakan
### Python Packages
- `yfinance` — ekstraksi data saham dari Yahoo Finance  
- `pandas`, `numpy` — manipulasi dan analisis data  
- `scikit-learn` — pelatihan dan evaluasi model machine learning  
- `plotly` — visualisasi interaktif  
- `streamlit` — UI dashboard prediksi

### Platform & DevOps
- `GitHub` — repositori kode dan version control  
- `GitHub Actions` — CI/CD pipeline otomatis  
- `VS Code` — editor pengembangan utama  
