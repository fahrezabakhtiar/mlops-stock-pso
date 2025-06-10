import streamlit as st
import pandas as pd
import os
import plotly.express as px

TICKERS = ['BMRI', 'BBRI', 'BBCA']
# Path absolut ke folder models di root project
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

st.set_page_config(page_title="Stock Forecast Dashboard", layout="centered")

st.title("Stock 30-Day Forecast Dashboard")

# === Bagian Forecast Per Ticker ===
st.header("Prediksi 30 Hari ke Depan")

ticker = st.selectbox("Pilih Ticker", TICKERS)

csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader(f"Forecast untuk {ticker}")
    
    # Line chart pakai plotly
    fig = px.line(df, x="Date", y="Forecast", title=f"Forecast 30 Hari ke Depan: {ticker}")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Lihat data tabel prediksi"):
        # Tampilkan tabel tanpa index
        try:
            st.dataframe(df, hide_index=True, use_container_width=True)
        except TypeError:
            st.dataframe(df.reset_index(drop=True), use_container_width=True)
else:
    st.warning(f"File prediksi `{ticker}_forecast_30d.csv` belum ditemukan di folder models/.")

st.markdown("---")

# === Bagian Summary Akurasi (MAPE) ===
st.header("Tabel Akurasi Semua Model (MAPE)")

mape_path = os.path.join(MODEL_DIR, "all_models_mape_summary.csv")

if os.path.exists(mape_path):
    mape_df = pd.read_csv(mape_path)
    ticker_mapes = mape_df[mape_df['ticker'] == ticker]
    st.subheader(f"Akurasi (MAPE) untuk {ticker}")
    try:
        st.dataframe(ticker_mapes, hide_index=True, use_container_width=True)
    except TypeError:
        st.dataframe(ticker_mapes.reset_index(drop=True), use_container_width=True)
    st.subheader("Bar Chart Akurasi per Model")
    fig2 = px.bar(ticker_mapes, x="model_type", y="mape", title=f"MAPE Tiap Model untuk {ticker}")
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Akurasi Seluruh Model & Ticker")
    try:
        st.dataframe(mape_df, hide_index=True, use_container_width=True)
    except TypeError:
        st.dataframe(mape_df.reset_index(drop=True), use_container_width=True)
else:
    st.warning("File all_models_mape_summary.csv belum ditemukan di folder models/.")

st.markdown("---")
st.caption("Built with Streamlit | Data MLOps Stock Forecast")
