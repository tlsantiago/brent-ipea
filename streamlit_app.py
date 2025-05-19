import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Previsão Preço Petróleo Brent", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=',', decimal=',')
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df = df.sort_values('data').set_index('data')
    return df

@st.cache_resource
def fit_sarimax(series, order=(1, 1, 1)):
    model = SARIMAX(series, order=order)
    result = model.fit(disp=False)
    return result

st.title("📈 Preço do Petróleo Brent — Histórico e Previsão")
st.markdown(
    "Este dashboard permite visualizar a série histórica completa e a previsão para "
    "os próximos dias usando o modelo SARIMAX escolhido."
)

# Sidebar: upload e parâmetros
st.sidebar.header("Configurações")
uploaded = st.sidebar.file_uploader("Carregar CSV com preços Brent", type=["csv"])
horizon = st.sidebar.slider("Horizonte de previsão (dias)", min_value=30, max_value=365, value=180, step=30)

if uploaded:
    # carregar e preparar dados
    df = load_data(uploaded)

    # plot histórico completo
    st.subheader("Série Histórica Completa (1987 – hoje)")
    st.line_chart(df['preco'])

    # escolher a série de treino (toda a base)
    st.subheader(f"Previsão SARIMAX para os próximos {horizon} dias")
    sarimax = fit_sarimax(df['preco'])
    forecast = sarimax.get_forecast(steps=horizon)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
    future_index = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    df_fore = pd.DataFrame({
        'previsto': pred.values,
        'ic_lower': ci.iloc[:, 0].values,
        'ic_upper': ci.iloc[:, 1].values
    }, index=future_index)

    # plot histórico + previsão
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['preco'], label="Histórico", color='navy')
    ax.plot(df_fore.index, df_fore['previsto'], label="Previsão", color='orange', linestyle='--')
    ax.fill_between(df_fore.index, df_fore['ic_lower'], df_fore['ic_upper'], color='orange', alpha=0.2)
    ax.set_title("Histórico e Previsão de Preço do Petróleo Brent")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

else:
    st.info("Faça o upload do arquivo CSV de preços para iniciar o dashboard.")