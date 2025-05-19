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

st.title("📈 Preço do Petróleo Brent — Histórico e Previsão")

uploaded = st.sidebar.file_uploader("CSV de preços Brent", type=["csv"])
horizon = st.sidebar.slider("Dias de previsão", 1, 60, 30)

if uploaded:
    df = load_data(uploaded)

    # Série histórica completa
    st.subheader("Série Histórica Completa")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df['preco'], color='navy')
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Preço (USD)")
    ax1.grid(True)
    st.pyplot(fig1)

    # Últimos 30 dias + previsão
    df_last30 = df.last('30D')
    model = fit_sarimax(df['preco'])
    forecast = model.get_forecast(steps=horizon)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
    future_idx = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')

    df_fc = pd.DataFrame({
        'prev': pred.values,
        'low': ci.iloc[:, 0].values,
        'high': ci.iloc[:, 1].values
    }, index=future_idx)

    # gráfico
    st.subheader("Últimos 30 dias + Previsão")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_last30.index, df_last30['preco'], '-o', label="Histórico", color='navy')
    for x, y in zip(df_last30.index, df_last30['preco']):
        ax2.annotate(f"{y:.1f}", (x, y), xytext=(0, 4),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax2.plot(df_fc.index, df_fc['prev'], '--', label="Previsão", color='orange')
    mid = len(df_fc) // 2
    xm, ym = df_fc.index[mid], df_fc['prev'].iloc[mid]
    ax2.annotate(f"{ym:.1f}", (xm, ym), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax2.fill_between(df_fc.index, df_fc['low'], df_fc['high'], color='orange', alpha=0.2)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Preço (USD)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # tabela de previsão
    df_fc['margem_erro_pct'] = (df_fc['high'] - df_fc['low']) / 2 / df_fc['prev'] * 100
    df_table = (
        df_fc
        .reset_index()
        .rename(columns={'index': 'Data', 'prev': 'Previsão', 'margem_erro_pct': 'Margem de Erro (%)'})
    )
    st.subheader("Previsão em Tabela")
    st.table(df_table[['Data', 'Previsão', 'Margem de Erro (%)']])

else:
    st.info("Faça upload do CSV para iniciar.")
