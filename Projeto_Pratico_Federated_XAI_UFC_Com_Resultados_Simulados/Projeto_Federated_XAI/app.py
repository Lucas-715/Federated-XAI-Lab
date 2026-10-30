"""Dashboard Streamlit para abrir RESULTADOS reais gerados pelo experimento."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Federated XAI Lab',layout='wide')
st.title('Federated XAI Lab')
st.caption('Exploração de desempenho e atribuições. Não interpreta dados sintéticos como evidência empírica.')
folder=Path(st.sidebar.text_input('Diretório da execução','results/demonstracao'))
metrics=folder/'metricas.csv'
if not metrics.exists():
    st.error('Não foram encontrados resultados. Execute: python -m flxai.experiment --mode synthetic --output-dir results/demonstracao')
    st.stop()
df=pd.read_csv(metrics)
source=str(df['source'].iloc[0])
if source=='synthetic':
    st.warning('DEMONSTRAÇÃO: resultados de dados sintéticos. Não são resultados científicos obtidos no UCI HAR.')
scenario=st.selectbox('Cenário',list(df.scenario.unique()))
view=df[df.scenario==scenario].set_index('method')
c1,c2=st.columns(2)
with c1:
    st.subheader('Macro-F1 no teste')
    st.bar_chart(view[['macro_f1']])
with c2:
    st.subheader('Acurácia no teste')
    st.bar_chart(view[['accuracy']])
st.dataframe(view.reset_index(),use_container_width=True,hide_index=True)

path=folder/'distribuicao_clientes.csv'
if path.exists():
    data=pd.read_csv(path); data=data[data.scenario==scenario]
    st.subheader('Distribuição das classes por cliente')
    st.bar_chart(data.pivot(index='client',columns='class',values='count'))

path=folder/'importancia_global.csv'
if path.exists():
    data=pd.read_csv(path);data=data[data.scenario==scenario]
    method=st.selectbox('Método para explicação',data.method.unique())
    show=data[data.method==method].sort_values('rank').head(15)
    st.subheader('Explicabilidade global — Integrated Gradients')
    st.bar_chart(show.set_index('feature')[['mean_abs_attribution']])
    st.caption('Agregação da importância absoluta sobre uma mesma amostra de teste. Atribuições não equivalem a causalidade.')

path=folder/'atribuicoes_locais.csv'
if path.exists():
    local=pd.read_csv(path)
    st.subheader('Exemplo de explicação local (FedAvg, IID)')
    index=int(st.number_input('Índice de exemplo',min_value=0,max_value=int(local['sample'].max()),value=0))
    subset=local[local['sample']==index]
    st.bar_chart(subset.set_index('feature')[['attribution']])

path=folder/'concordancia_top5.csv'
if path.exists():
    st.subheader('Concordância entre conjuntos Top-5')
    st.dataframe(pd.read_csv(path).query('scenario == @scenario'),hide_index=True)
st.markdown('Fonte dos dados: arquivos CSV da execução selecionada. Scripts e configurações constam do repositório.')

path=folder/'concordancia_entre_clientes.csv'
if path.exists():
    st.subheader('Concordância entre clientes')
    st.dataframe(pd.read_csv(path).query('scenario == @scenario'),hide_index=True)
    st.caption('Análise descritiva de atribuições nos dados locais de treino; não é avaliação de generalização.')
