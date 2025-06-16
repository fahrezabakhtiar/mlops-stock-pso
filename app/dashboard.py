# dashboard.py
# Deskripsi:
# Aplikasi dashboard interaktif menggunakan Streamlit untuk menampilkan hasil prediksi harga saham 
# 30 hari ke depan berdasarkan model terbaik serta membandingkan performa model dengan metrik MAPE.

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

# Menambahkan path src agar modul dapat diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import TICKERS  # Daftar ticker saham yang tersedia

# Path ke folder model
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# --- Header halaman utama ---
st.set_page_config(page_title="Stock Forecast Dashboard - Global Stock Exchange", layout="wide")
st.title("📈 Global Stock Exchange Forecasting Dashboard")
st.caption("30-Day Forecasting Powered by Machine Learning")

# --- Sidebar konfigurasi ---
st.sidebar.header("⚙️ Konfigurasi")
ticker = st.sidebar.selectbox("Pilih Ticker", TICKERS)

# --- Load dan tampilkan file hasil forecast ---
csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Konversi kolom tanggal

    st.subheader(f"Prediksi Harga 30 Hari ke Depan ({ticker})")

    # Sidebar: filter rentang tanggal
    from datetime import datetime, timedelta
    today = datetime.today().date()
    end_date = today + timedelta(days=30)
    date_range = st.sidebar.date_input("Filter Tanggal", [today, end_date], min_value=today, max_value=end_date)
    filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

    # Visualisasi grafik prediksi dan statistik prediktif
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
        # Tombol download CSV hasil prediksi
        st.download_button(
            "📥 Download Forecast CSV",
            df.to_csv(index=False).encode(),
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

        # Ringkasan statistik
        st.markdown("#### 📊 Ringkasan Prediksi")
        st.metric("Rata-rata Prediksi", f"{filtered_df['Forecast'].mean():,.2f}")
        st.metric("Tertinggi", f"{filtered_df['Forecast'].max():,.2f}")
        st.markdown(f"📅 Total Hari Diprediksi: **{len(filtered_df)} hari**")

else:
    st.warning(f"File prediksi `{ticker}_forecast_30d.csv` belum ditemukan di folder models/.")

# --- Menampilkan ringkasan akurasi MAPE semua model untuk ticker terpilih ---
mape_path = os.path.join(MODEL_DIR, "all_models_mape_summary.csv")
if os.path.exists(mape_path):
    mape_df = pd.read_csv(mape_path).round({"mape": 4})
    ticker_mapes = mape_df[mape_df['ticker'] == ticker].copy()
    ticker_mapes = ticker_mapes.sort_values("mape").reset_index(drop=True)

    # Menampilkan model terbaik berdasarkan MAPE
    best_row = ticker_mapes.iloc[0]
    best_model = best_row["model_type"].replace("_", " ").title()
    best_mape = best_row["mape"]

    st.info(f"Model Terbaik: {best_model} | MAPE: {best_mape:.4f}")
    st.markdown(f"### Model Terbaik: **{best_model}**")

    # Tabel perbandingan MAPE semua model
    st.markdown("### Perbandingan Akurasi Semua Model")
    st.dataframe(
        ticker_mapes[["model_type", "mape"]]
        .rename(columns={"model_type": "Model", "mape": "MAPE"})
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True if "hide_index" in st.dataframe.__code__.co_varnames else False
    )
else:
    st.warning("File all_models_mape_summary.csv belum ditemukan di folder models/.")

# --- Footer ---
st.markdown("---")
st.caption("Made with Streamlit")