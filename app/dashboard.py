# dashboard.py
# Deskripsi:
# Dashboard Streamlit dinamis untuk menampilkan hasil prediksi saham 30 hari ke depan.

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
import re
from datetime import datetime, timedelta

# Setup path untuk akses modul dari folder src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Path ke folder model
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Fungsi untuk mendeteksi ticker yang tersedia berdasarkan file forecast
def get_available_tickers(model_dir):
    tickers = []
    for fname in os.listdir(model_dir):
        match = re.match(r"^(.+)_forecast_30d\.csv$", fname)
        if match:
            tickers.append(match.group(1))
    return sorted(tickers)

# Ambil daftar ticker yang tersedia
TICKERS = get_available_tickers(MODEL_DIR)

# Setup halaman utama
st.set_page_config(page_title="Stock Forecast Dashboard - Global Stock Exchange", layout="wide")
st.title("📈 Global Stock Exchange Forecasting Dashboard")
st.caption("30-Day Forecasting Powered by Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")

# Sidebar: pilih ticker
ticker = st.sidebar.selectbox("Pilih Ticker", TICKERS)

# Load file forecast
csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    st.subheader(f"📊 Prediksi Harga 30 Hari ke Depan ({ticker})")

    # Sidebar: filter tanggal
    today = datetime.today().date()
    end_date = today + timedelta(days=30)
    date_range = st.sidebar.date_input("Filter Tanggal", [today, end_date], min_value=today, max_value=end_date)
    filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

    # Visualisasi prediksi
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(
            filtered_df, 
            x="Date", 
            y="Forecast", 
            markers=True,
            labels={"Forecast": "Harga Prediksi", "Date": "Tanggal"}
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.download_button(
            "📥 Download Forecast CSV",
            df.to_csv(index=False).encode(),
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )
        st.markdown("#### 📊 Ringkasan Prediksi")
        st.metric("Rata-rata Prediksi", f"{filtered_df['Forecast'].mean():,.2f}")
        st.metric("Tertinggi", f"{filtered_df['Forecast'].max():,.2f}")
        st.markdown(f"📅 Total Hari Diprediksi: **{len(filtered_df)} hari**")

else:
    st.warning(f"❌ File prediksi `{ticker}_forecast_30d.csv` tidak ditemukan.")

# Footer
st.markdown("---")
st.caption("Made with Streamlit")
