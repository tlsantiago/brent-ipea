import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Previsão Petróleo Brent", layout="wide")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=',', decimal=',')
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    return df.sort_values('data').set_index('data')

@st.cache_resource
def fit_sarimax(series, order=(1, 1, 1)):
    model = SARIMAX(series, order=order)
    return model.fit(disp=False)

st.title("📈 Preço do Petróleo Brent — Curto Prazo")
st.markdown("Últimos 30 dias de histórico e previsão para os próximos dias via SARIMAX.")

uploaded = st.sidebar.file_uploader("CSV de preços Brent", type=["csv"])
horizon = st.sidebar.slider("Dias de previsão", 1, 365, 30)

if uploaded:
    df = load_data(uploaded)
    df_last30 = df.last('30D')
    st.subheader("Histórico — últimos 30 dias")
    st.line_chart(df_last30['preco'])

    sarimax_res = fit_sarimax(df['preco'])
    forecast = sarimax_res.get_forecast(steps=horizon)
    y_pred = forecast.predicted_mean
    ci = forecast.conf_int()
    future_idx = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon)

    df_fore = pd.DataFrame({
        'previsto': y_pred.values,
        'ic_lower': ci.iloc[:, 0].values,
        'ic_upper': ci.iloc[:, 1].values
    }, index=future_idx)

    st.subheader(f"Previsão — próximos {horizon} dias")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_last30.index, df_last30['preco'], label="Histórico", color='navy')
    ax.plot(df_fore.index, df_fore['previsto'], '--', label="Previsão", color='orange')
    ax.fill_between(df_fore.index, df_fore['ic_lower'], df_fore['ic_upper'], color='orange', alpha=0.2)

    for x, y in zip(df_fore.index, df_fore['previsto']):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)

    ax.set_title("Histórico (30 dias) + Previsão SARIMAX")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Faça o upload do CSV de preços Brent para iniciar.")
