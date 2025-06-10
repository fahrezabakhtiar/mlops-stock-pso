import streamlit as st
import pandas as pd
import os

# List ticker yang tersedia (bisa diedit)
TICKERS = ['BMRI', 'BBRI', 'BBCA']
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Stock Forecast Dashboard", layout="centered")

st.title("Stock 30-Day Forecast Dashboard")

# === Bagian Forecast Per Ticker ===
st.header("Prediksi 30 Hari ke Depan")

ticker = st.selectbox("Pilih Ticker", TICKERS)

csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader(f"Forecast untuk {ticker}")
    st.line_chart(df.set_index("Date")["Forecast"])
    with st.expander("Lihat data tabel prediksi"):
        st.dataframe(df, use_container_width=True)
else:
    st.warning(f"File prediksi `{ticker}_forecast_30d.csv` belum ditemukan di folder models/.")

st.markdown("---")

# === Bagian Summary Akurasi (MAPE) ===
st.header("Tabel Akurasi Semua Model (MAPE)")

mape_path = os.path.join(MODEL_DIR, "all_models_mape_summary.csv")

if os.path.exists(mape_path):
    mape_df = pd.read_csv(mape_path)
    # Optional filter by ticker
    ticker_mapes = mape_df[mape_df['ticker'] == ticker]
    st.subheader(f"Akurasi (MAPE) untuk {ticker}")
    st.dataframe(ticker_mapes, use_container_width=True)
    st.subheader("Bar Chart Akurasi per Model")
    st.bar_chart(ticker_mapes.set_index("model_type")["mape"])
    
    st.subheader("Akurasi Seluruh Model & Ticker")
    st.dataframe(mape_df, use_container_width=True)
else:
    st.warning("File all_models_mape_summary.csv belum ditemukan di folder models/.")

st.markdown("---")
st.caption("Built with Streamlit | Data MLOps Stock Forecast")
