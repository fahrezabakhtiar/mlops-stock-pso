import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import TICKERS

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# --- Header ---
st.set_page_config(page_title="Stock Forecast Dashboard - Jakarta Stock Exchange", layout="wide")
st.title("📈 Jakarta Stock Forecasting Dashboard")
st.caption("30-Day Forecasting Powered by Machine Learning")

# --- Sidebar ---
st.sidebar.header("⚙️ Konfigurasi")
ticker = st.sidebar.selectbox("Pilih Ticker", TICKERS)

# --- Load Forecast CSV ---
csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    st.subheader(f"Prediksi Harga 30 Hari ke Depan ({ticker})")
    date_range = st.date_input("Filter Tanggal", [df['Date'].min(), df['Date'].max()])
    filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(filtered_df, x="Date", y="Forecast", markers=True,
                      labels={"Forecast": "Harga Prediksi", "Date": "Tanggal"})
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.download_button("📥 Download Forecast CSV", df.to_csv(index=False).encode(),
                           file_name=f"{ticker}_forecast.csv", mime="text/csv")
else:
    st.warning(f"File prediksi `{ticker}_forecast_30d.csv` belum ditemukan di folder models/.")

# --- Summary MAPE per Ticker ---
mape_path = os.path.join(MODEL_DIR, "all_models_mape_summary.csv")
if os.path.exists(mape_path):
    mape_df = pd.read_csv(mape_path)
    mape_df = mape_df.round({"mape": 4})
    ticker_mapes = mape_df[mape_df['ticker'] == ticker].copy()
    ticker_mapes = ticker_mapes.sort_values("mape").reset_index(drop=True)

    best_row = ticker_mapes.iloc[0]
    best_model = best_row["model_type"].replace("_", " ").title()
    best_mape = best_row["mape"]

    st.info(f"MAPE: {best_mape:.4f}")

    st.markdown(f"### Model Terbaik: **{best_model}**")
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

st.markdown("---")
st.caption("Made with ❤️ by MLOps Team (Reza, Satria, Dika, Irsyad) | Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))