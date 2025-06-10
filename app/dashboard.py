import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import TICKERS

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

st.set_page_config(page_title="Stock Forecast Dashboard -  Jakarta Stock Exchange", layout="centered")
st.title("Stock 30-Day Forecast Dashboard -  Jakarta Stock Exchange")

ticker = st.selectbox("Pilih Ticker", TICKERS)

# --- Plot Forecast ---
csv_path = os.path.join(MODEL_DIR, f"{ticker}_forecast_30d.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader(f"Prediksi Harga 30 Hari ke Depan ({ticker})")
    fig = px.line(df, x="Date", y="Forecast", markers=True,
                  labels={"Forecast": "Harga Prediksi", "Date": "Tanggal"},
                  title=None)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"File prediksi `{ticker}_forecast_30d.csv` belum ditemukan di folder models/.")

# --- Summary MAPE per Ticker ---
mape_path = os.path.join(MODEL_DIR, "all_models_mape_summary.csv")
if os.path.exists(mape_path):
    mape_df = pd.read_csv(mape_path)
    mape_df = mape_df.round({"mape": 4})  # biar rapih
    ticker_mapes = mape_df[mape_df['ticker'] == ticker].copy()
    ticker_mapes = ticker_mapes.sort_values("mape").reset_index(drop=True)

    # Model terbaik
    best_row = ticker_mapes.iloc[0]
    best_model = best_row["model_type"].replace("_", " ").title()
    best_mape = best_row["mape"]

    st.success(f"**Model Terbaik: {best_model}**\n\n**MAPE: {best_mape:.4f}**")

    st.markdown("### Perbandingan Akurasi Semua Model")
    styled_table = (
        ticker_mapes[["model_type", "mape"]]
        .rename(columns={"model_type": "Model", "mape": "MAPE"})
        .style
        .format({"MAPE": "{:.4f}"})
        .background_gradient(subset=["MAPE"], cmap="YlGn", axis=0)
        .set_properties(**{"text-align": "center"})
        .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
    )
    st.dataframe(ticker_mapes[["model_type", "mape"]]
                 .rename(columns={"model_type": "Model", "mape": "MAPE"})
                 .reset_index(drop=True),
                 use_container_width=True,
                 hide_index=True if "hide_index" in st.dataframe.__code__.co_varnames else False)

else:
    st.warning("File all_models_mape_summary.csv belum ditemukan di folder models/.")

st.caption("Built with Streamlit | Data MLOps Stock Forecast")