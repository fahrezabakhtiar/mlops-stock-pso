import streamlit as st
from src.predict import forecast_next_days
import pandas as pd

st.title("Prediksi Saham IDX")
ticker = st.selectbox("Pilih ticker", ["BMRI", "BBRI", "BBCA"])
days = st.slider("Jumlah hari ke depan untuk prediksi:", 1, 10, 30)

try:
    data = forecast_next_days(ticker, days=days)
    df = pd.DataFrame(data, columns=["Predicted Close"])

    st.line_chart(df["Predicted Close"])
    st.dataframe(df)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat prediksi: {e}")