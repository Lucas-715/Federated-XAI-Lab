"""Federated XAI Lab — Dashboard Cientifico em Streamlit.

Visualizacao reproduzivel de desempenho preditivo e explicabilidade com Integrated Gradients.
"""
from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Federated XAI Lab — UFC Russas', layout='wide', initial_sidebar_state='expanded')

st.title('Federated XAI Lab')
st.caption('Explicabilidade Local e Global em Aprendizado Federado Não-IID: Um Estudo com Dados de Sensores Distribuídos (UFC Campus Russas)')

# 1. Barra Lateral e Selecao de Execucao
st.sidebar.header('Configurações da Execução')

available_runs = sorted(glob.glob('results/uci_seed_*') + glob.glob('results/demonstracao*'))
if not available_runs:
    available_runs = sorted(glob.glob('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/uci_seed_*'))
if not available_runs:
    available_runs = ['results/uci_seed_42']

mode_choice = st.sidebar.radio('Modo de Visualização', ['Execução Individual', 'Síntese Multissemente (Repetições)'])

if mode_choice == 'Execução Individual':
    default_idx = 0
    selected_dir = st.sidebar.selectbox('Diretório da Execução', available_runs, index=default_idx)
    folder = Path(st.sidebar.text_input('Caminho personalizado', selected_dir))
    
    metrics_path = folder / 'metricas.csv'
    if not metrics_path.exists():
        st.error(f'Arquivo de métricas não encontrado em {folder}.')
        st.stop()
        
    df = pd.read_csv(metrics_path)
    source = str(df['source'].iloc[0]) if 'source' in df.columns else 'uci'
    seed = int(df['seed'].iloc[0]) if 'seed' in df.columns else 'N/A'
    
    if source == 'synthetic':
        st.warning('Demonstração com Dados Sintéticos: valores meramente ilustrativos.')
    else:
        st.success(f'Dataset Real: UCI Human Activity Recognition Using Smartphones (Semente {seed}).')
        
    scenarios = list(df['scenario'].unique())
    selected_scenario = st.sidebar.selectbox('Cenário de Distribuição', scenarios)
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Desempenho no Teste',
        'Curvas de Validação',
        'Matriz de Confusão',
        'Heterogeneidade de Clientes',
        'Explicabilidade (Global & Local)',
        'Concordância e Jaccard'
    ])
    
    # TAB 1: Desempenho no Teste
    with tab1:
        st.subheader(f'Métricas no Teste — {selected_scenario}')
        sub_df = df[df.scenario == selected_scenario]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Macro-F1 (FedAvg)', f"{sub_df[sub_df.method=='FedAvg']['macro_f1'].values[0]:.4f}" if 'FedAvg' in sub_df.method.values else 'N/A')
        with c2:
            st.metric('Macro-F1 (FedProx)', f"{sub_df[sub_df.method=='FedProx']['macro_f1'].values[0]:.4f}" if 'FedProx' in sub_df.method.values else 'N/A')
        with c3:
            st.metric('Macro-F1 (Centralizado)', f"{sub_df[sub_df.method=='Centralizado']['macro_f1'].values[0]:.4f}" if 'Centralizado' in sub_df.method.values else 'N/A')
            
        view = sub_df.set_index('method')
        st.bar_chart(view[['macro_f1', 'accuracy']])
        st.dataframe(view.reset_index()[['method', 'accuracy', 'macro_f1', 'loss', 'training_duration_seconds']], use_container_width=True, hide_index=True)
        
        det_path = folder / 'metricas_detalhadas.csv'
        if det_path.exists():
            st.subheader('Desempenho por Classe de Atividade')
            df_det = pd.read_csv(det_path)
            st.dataframe(df_det[df_det.scenario == selected_scenario], use_container_width=True, hide_index=True)

    # TAB 2: Curvas de Validacao
    with tab2:
        st.subheader(f'Evolução ao Longo das Rodadas — {selected_scenario}')
        hist_path = folder / 'historico_validacao.csv'
        if hist_path.exists():
            df_hist = pd.read_csv(hist_path)
            sub_hist = df_hist[df_hist.scenario == selected_scenario]
            
            c1, c2 = st.columns(2)
            with c1:
                st.write('**Macro-F1 de Validação por Rodada**')
                pv_f1 = sub_hist.pivot(index='round', columns='method', values='macro_f1')
                st.line_chart(pv_f1)
            with c2:
                if 'loss' in sub_hist.columns:
                    st.write('**Perda (Cross-Entropy) de Validação por Rodada**')
                    pv_loss = sub_hist.pivot(index='round', columns='method', values='loss')
                    st.line_chart(pv_loss)
            st.dataframe(sub_hist, use_container_width=True, hide_index=True)
        else:
            st.info('Histórico de validação não disponível.')

    # TAB 3: Matriz de Confusao
    with tab3:
        st.subheader(f'Matrizes de Confusão — {selected_scenario}')
        cm_path = folder / 'matrizes_confusao.json'
        if cm_path.exists():
            cms = json.loads(cm_path.read_text(encoding='utf-8'))
            methods = ['Centralizado', 'FedAvg', 'FedProx']
            cols = st.columns(3)
            for idx, m in enumerate(methods):
                key = f"{selected_scenario}_{m}"
                if key in cms:
                    with cols[idx]:
                        st.write(f"**{m}**")
                        cm_arr = np.array(cms[key])
                        # Ler classes do dataset_info
                        ds_info = folder / 'dataset_info.json'
                        classes = json.loads(ds_info.read_text(encoding='utf-8'))['classes'] if ds_info.exists() else [f'C{i}' for i in range(len(cm_arr))]
                        df_cm = pd.DataFrame(cm_arr, index=classes, columns=classes)
                        st.dataframe(df_cm, use_container_width=True)
        else:
            st.info('Matrizes de confusão não disponíveis para esta execução.')

    # TAB 4: Heterogeneidade de Clientes
    with tab4:
        st.subheader(f'Composição das Partições Locais — {selected_scenario}')
        dist_path = folder / 'distribuicao_clientes.csv'
        if dist_path.exists():
            df_dist = pd.read_csv(dist_path)
            sub_dist = df_dist[df_dist.scenario == selected_scenario]
            pivot_dist = sub_dist.pivot(index='client', columns='class_name' if 'class_name' in sub_dist.columns else 'class', values='count')
            st.bar_chart(pivot_dist)
            st.dataframe(sub_dist, use_container_width=True, hide_index=True)
            
            # Heterogeneidade info
            het_file = folder / 'analise_heterogeneidade_cenarios.json'
            if het_file.exists():
                het_data = json.loads(het_file.read_text(encoding='utf-8'))
                if selected_scenario in het_data:
                    st.json(het_data[selected_scenario])

    # TAB 5: Explicabilidade (Global & Local)
    with tab5:
        st.subheader(f'Explicabilidade Global — Integrated Gradients ({selected_scenario})')
        imp_path = folder / 'importancia_global.csv'
        if imp_path.exists():
            df_imp = pd.read_csv(imp_path)
            sub_imp = df_imp[df_imp.scenario == selected_scenario]
            method_xai = st.selectbox('Método para Explicar', sub_imp['method'].unique())
            top_n = st.slider('Quantidade de Características (Top-N)', min_value=5, max_value=30, value=15)
            
            show_imp = sub_imp[sub_imp.method == method_xai].sort_values('rank').head(top_n)
            st.bar_chart(show_imp.set_index('feature')[['mean_abs_attribution']])
            st.dataframe(show_imp, use_container_width=True, hide_index=True)
            
        loc_path = folder / 'atribuicoes_locais.csv'
        if loc_path.exists():
            st.subheader('Exemplo de Explicação Local (Amostra Individual)')
            df_loc = pd.read_csv(loc_path)
            if not df_loc.empty:
                max_s = int(df_loc['sample'].max())
                s_idx = st.number_input('Índice da Amostra', min_value=0, max_value=max_s, value=0)
                sub_loc = df_loc[df_loc['sample'] == s_idx]
                st.write(f"Classe Prevista: {sub_loc['predicted_class'].iloc[0] if 'predicted_class' in sub_loc.columns else 'N/A'}")
                st.bar_chart(sub_loc.sort_values(by='attribution', key=abs, ascending=False).head(15).set_index('feature')[['attribution']])

    # TAB 6: Concordancia e Jaccard
    with tab6:
        st.subheader('Concordância Top-5 entre Modelos (Jaccard Index)')
        c5_path = folder / 'concordancia_top5.csv'
        if c5_path.exists():
            df_c5 = pd.read_csv(c5_path)
            st.dataframe(df_c5[df_c5.scenario == selected_scenario], use_container_width=True, hide_index=True)
            
        st.subheader('Concordância Top-5 entre Clientes Locais (Inter-Client Jaccard)')
        cc_path = folder / 'concordancia_entre_clientes.csv'
        if cc_path.exists():
            df_cc = pd.read_csv(cc_path)
            st.dataframe(df_cc[df_cc.scenario == selected_scenario], use_container_width=True, hide_index=True)
            st.caption('Média das similaridades pareadas de Jaccard sobre dados locais de treinamento.')

else:
    # MODO SINTESE MULTISSEMENTE
    st.header('Síntese das Repetições com Dados Reais do UCI HAR')
    resumo_path = Path('results/resumo_experimentos_reais.csv')
    if not resumo_path.exists():
        resumo_path = Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/resumo_experimentos_reais.csv')
        
    if not resumo_path.exists():
        st.error('Resumo multissemente não encontrado. Execute o script scripts/analisar_experimentos_reais.py.')
        st.stop()
        
    df_res = pd.read_csv(resumo_path)
    st.subheader('Tabela Consolidada de Métricas (Média ± Desvio-Padrão)')
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    
    st.subheader('Gráfico Comparativo de Macro-F1 Médio')
    pv_mean = df_res.pivot(index='scenario', columns='method', values='macro_f1_media')
    st.bar_chart(pv_mean)
    
    # Testes Estatisticos Pareados
    stat_file = Path('results/teste_estatistico_pareado.json')
    if not stat_file.exists():
        stat_file = Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/teste_estatistico_pareado.json')
    if stat_file.exists():
        st.subheader('Testes Estatísticos Pareados (FedProx vs FedAvg)')
        st.json(json.loads(stat_file.read_text(encoding='utf-8')))
        
    # Estabilidade XAI
    xai_file = Path('results/estabilidade_xai.json')
    if not xai_file.exists():
        xai_file = Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/estabilidade_xai.json')
    if xai_file.exists():
        st.subheader('Estabilidade e Concordância de Explicabilidade')
        st.json(json.loads(xai_file.read_text(encoding='utf-8')))
