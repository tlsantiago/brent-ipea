# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Previsão Petróleo Brent", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=',', decimal=',')
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    return df.sort_values('data').set_index('data')

@st.cache_resource
def fit_sarimax(series, order=(1, 1, 1)):
    model = SARIMAX(series, order=order)
    return model.fit(disp=False)

st.title("📈 Preço do Petróleo Brent — Curto Prazo")
st.markdown(
    "Visualize os últimos 30 dias de preços históricos e a previsão para "
    "os próximos dias usando o modelo SARIMAX."
)

# Sidebar
st.sidebar.header("Configurações")
uploaded = st.sidebar.file_uploader("Carregar CSV com preços Brent", type=["csv"])
horizon = st.sidebar.slider("Dias de previsão", min_value=1, max_value=365, value=30, step=1)

if uploaded:
    # Carrega e prepara os dados
    df = load_data(uploaded)
    # Histórico dos últimos 30 dias
    df_last30 = df.last('30D')

    st.subheader("Histórico — últimos 30 dias")
    st.line_chart(df_last30['preco'])

    # Ajusta o modelo e gera forecast
    sarimax_res = fit_sarimax(df['preco'])
    forecast = sarimax_res.get_forecast(steps=horizon)
    y_pred = forecast.predicted_mean
    ci = forecast.conf_int()
    future_index = pd.date_range(
        start=df.index.max() + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    df_fore = pd.DataFrame({
        'previsto': y_pred.values,
        'ic_lower': ci.iloc[:, 0].values,
        'ic_upper': ci.iloc[:, 1].values
    }, index=future_index)

    st.subheader(f"Previsão — próximos {horizon} dias")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plota histórico e previsão
    ax.plot(df_last30.index, df_last30['preco'],
            label="Histórico (30 dias)", color='navy')
    ax.plot(df_fore.index, df_fore['previsto'],
            label="Previsão", color='orange', linestyle='--')
    ax.fill_between(df_fore.index,
                    df_fore['ic_lower'], df_fore['ic_upper'],
                    color='orange', alpha=0.2)

    # Configurações do eixo X para datas diárias
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # marca a cada 5 dias
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)

    ax.set_title("Histórico (últimos 30 dias) + Previsão SARIMAX")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

else:
    st.info("Faça o upload do CSV de preços Brent para iniciar o dashboard.")
