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

    # Bloco de Seleção de Datas
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data Inicial", value=df.index.min().date(),
                                   min_value=df.index.min().date(), max_value=df.index.max().date())
    with col2:
        end_date = st.date_input("Data Final", value=df.index.max().date(),
                                 min_value=df.index.min().date(), max_value=df.index.max().date())
    if start_date > end_date:
        st.error("Data Inicial não pode ser maior que Data Final.")
    else:
        df_hist = df.loc[start_date:end_date]

        # Bloco da Série Histórica
        st.subheader("Série Histórica")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df_hist.index, df_hist['preco'], '-o', color='navy', markersize=4)
        ax1.set_xlabel("Data")
        ax1.set_ylabel("Preço (USD)")
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax1.grid(True)
        st.pyplot(fig1)

        # Bloco de Insights
        st.markdown("""
st.subheader("Principais fatores de oscilação dos preços no período avaliado:")
- **Pico de 2008:** em meados de 2008, o preço chegou a ~140 USD/bbl antes da crise financeira, caindo para ~40 USD no início de 2009.  
- **Queda de 2014–2016:** o boom do shale oil nos EUA e o excesso de oferta fizeram o preço recuar de ~110 USD para ~30 USD.  
- **Colapso de 2020:** durante a pandemia de COVID-19, a demanda caiu drasticamente, levando o preço a <20 USD em abril de 2020.  
- **Alta de 2022:** tensões geopolíticas após a invasão da Ucrânia e a recuperação econômica elevaram o preço acima de 120 USD.  
""")

        # Bloco dos Últimos 30 dias + Previsão
        df_last30 = df.last('30D')
        sarimax_res = fit_sarimax(df['preco'])
        forecast = sarimax_res.get_forecast(steps=horizon)
        pred = forecast.predicted_mean
        ci = forecast.conf_int()
        future_idx = pd.date_range(df.index.max() + pd.Timedelta(days=1),
                                   periods=horizon, freq='D')
        df_fc = pd.DataFrame({
            'prev': pred.values,
            'low': ci.iloc[:, 0].values,
            'high': ci.iloc[:, 1].values
        }, index=future_idx)

        st.subheader("Últimos 30 dias + Previsão")
        st.markdown("""
**Sobre o modelo de previsão**  
O modelo ARIMAX baseia‐se na ideia de que o preço de hoje reflete tanto o que aconteceu nos dias anteriores quanto as variações recentes do mercado. Ele identifica padrões de alta e de baixa — por exemplo, se o preço esteve subindo de forma constante ou sofreu quedas pontuais — e utiliza essas informações para projetar o que deve acontecer nos próximos dias.

- **Por que ARIMAX?**  
  - É um modelo simples e consolidado, amplamente usado em finanças e economia.  
  - Ajusta‐se bem a séries diárias de preços, capturando oscilações normais sem “torrar” extremos.  
  - Inclui um intervalo de confiança, mostrando a margem de erro esperada.  
  - Suas previsões são estáveis e fáceis de entender, mesmo para quem não é da área de dados.
  - Dos modelos avaliado, foi o modelo que demonstrou a menor taxa de erro médio na previsão.

Essa combinação de robustez, clareza e capacidade de lidar com séries temporais complexas fez do ARIMAX nossa melhor escolha para prever o preço do petróleo.
""")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df_last30.index, df_last30['preco'], '-o', label="Histórico", color='navy', markersize=4)
        for x, y in zip(df_last30.index, df_last30['preco']):
            ax2.annotate(f"{y:.1f}", (x, y), xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)
        ax2.plot(df_fc.index, df_fc['prev'], '--', label="Previsão", color='orange')
        mid = len(df_fc) // 2
        xm, ym = df_fc.index[mid], df_fc['prev'].iloc[mid]
        ax2.annotate(f"{ym:.1f}", (xm, ym), xytext=(0, 4), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)
        ax2.fill_between(df_fc.index, df_fc['low'], df_fc['high'], color='orange', alpha=0.2)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.xticks(rotation=45)
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Preço (USD)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Bloco da Tabela de Previsão
        df_fc['margem_erro_pct'] = (df_fc['high'] - df_fc['low']) / 2 / df_fc['prev'] * 100
        df_table = df_fc.reset_index().rename(
            columns={'index': 'Data', 'prev': 'Previsão', 'margem_erro_pct': 'Margem de Erro (%)'}
        )
        df_table['Data'] = df_table['Data'].dt.strftime('%Y-%m-%d')
        st.subheader("Previsão em Tabela")
        st.table(df_table[['Data', 'Previsão', 'Margem de Erro (%)']])

else:
    st.info("Faça upload do CSV para iniciar.")
