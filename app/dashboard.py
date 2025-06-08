import streamlit as st
from src.predict import forecast_next_days
import pandas as pd
from datetime import datetime, timedelta

st.title("Prediksi Saham IDX")

ticker = st.selectbox("Pilih ticker", ["BMRI", "BBRI", "BBCA"])
days = st.slider("Jumlah hari ke depan untuk prediksi:", 1, 30, 10)  # max 30, default 10

try:
    data = forecast_next_days(ticker, days=days)
    df = pd.DataFrame(data, columns=["Predicted Close"])

    # Buat kolom tanggal prediksi, mulai dari besok
    start_date = datetime.today() + timedelta(days=1)
    df['Date'] = [start_date + timedelta(days=i) for i in range(days)]
    df = df.set_index('Date')

    st.line_chart(df["Predicted Close"])
    st.dataframe(df)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat prediksi: {e}")