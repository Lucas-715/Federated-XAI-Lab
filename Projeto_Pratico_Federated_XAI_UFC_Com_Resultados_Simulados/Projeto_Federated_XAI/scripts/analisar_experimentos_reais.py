"""Consolidacao cientifica e analise estatistica das repeticoes reais do UCI HAR."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_statistical_analysis():
    # Coleta todas as execucoes reais
    seed_dirs = sorted(glob.glob('results/uci_seed_*'))
    if len(seed_dirs) < 3:
        # Tenta no diretorio do subprojeto
        seed_dirs = sorted(glob.glob('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/uci_seed_*'))
        
    print(f"Diretorios encontrados: {seed_dirs}")
    assert len(seed_dirs) >= 3, f"Necessario ao menos 3 execucoes reais. Encontrados: {len(seed_dirs)}"

    all_metrics = []
    all_concordance = []
    all_client_concordance = []
    all_importances = []
    
    for s_dir in seed_dirs:
        p = Path(s_dir)
        seed_num = int(p.name.split('_')[-1])
        
        df_m = pd.read_csv(p / 'metricas.csv')
        df_m['seed'] = seed_num
        all_metrics.append(df_m)
        
        if (p / 'concordancia_top5.csv').exists():
            df_c = pd.read_csv(p / 'concordancia_top5.csv')
            df_c['seed'] = seed_num
            all_concordance.append(df_c)
            
        if (p / 'concordancia_entre_clientes.csv').exists():
            df_cc = pd.read_csv(p / 'concordancia_entre_clientes.csv')
            df_cc['seed'] = seed_num
            all_client_concordance.append(df_cc)
            
        if (p / 'importancia_global.csv').exists():
            df_imp = pd.read_csv(p / 'importancia_global.csv')
            df_imp['seed'] = seed_num
            all_importances.append(df_imp)

    df_metrics = pd.concat(all_metrics, ignore_index=True)
    df_concordance = pd.concat(all_concordance, ignore_index=True) if all_concordance else pd.DataFrame()
    df_client_conc = pd.concat(all_client_concordance, ignore_index=True) if all_client_concordance else pd.DataFrame()
    df_importances = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    # 1. Resumo de Metricas (Media e Desvio-Padrao)
    summary = df_metrics.groupby(['scenario', 'method'], as_index=False).agg(
        n_sementes=('seed', 'nunique'),
        acuracia_media=('accuracy', 'mean'),
        acuracia_dp=('accuracy', 'std'),
        macro_f1_media=('macro_f1', 'mean'),
        macro_f1_dp=('macro_f1', 'std'),
        loss_media=('loss', 'mean'),
        loss_dp=('loss', 'std'),
        tempo_treino_medio_s=('training_duration_seconds', 'mean')
    )

    out_dirs = [Path('results'), Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results')]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(od / 'metricas_reais_todas_sementes.csv', index=False)
        summary.to_csv(od / 'resumo_experimentos_reais.csv', index=False)

    print("\n=== RESUMO DAS EXPERIMENTAÇÕES COM DADOS REAIS ===")
    print(summary.to_string(index=False))

    # 2. Testes Estatisticos Pareados (FedAvg vs FedProx por cenario)
    paired_tests = {}
    scenarios = df_metrics['scenario'].unique()
    
    for sc in scenarios:
        sub = df_metrics[df_metrics.scenario == sc]
        p_avg = sub[sub.method == 'FedAvg'].sort_values('seed')['macro_f1'].values
        p_prox = sub[sub.method == 'FedProx'].sort_values('seed')['macro_f1'].values
        
        diff = p_prox - p_avg
        t_stat, p_val_t = stats.ttest_rel(p_prox, p_avg)
        
        # Teste de Wilcoxon pareado caso n >= 5
        wilcoxon_stat, p_val_w = (None, None)
        if len(diff) >= 5 and np.any(diff != 0):
            try:
                res_w = stats.wilcoxon(p_prox, p_avg)
                wilcoxon_stat, p_val_w = float(res_w.statistic), float(res_w.pvalue)
            except Exception:
                pass
                
        paired_tests[sc] = {
            'n_pairs': len(diff),
            'diff_mean_fedprox_minus_fedavg': float(np.mean(diff)),
            'diff_std': float(np.std(diff, ddof=1)),
            't_statistic': float(t_stat),
            't_pvalue': float(p_val_t),
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_pvalue': p_val_w,
            'statistically_significant_005': bool(p_val_t < 0.05),
            'conclusion': 'Diferença estatisticamente significativa (p < 0.05)' if p_val_t < 0.05 else 'Sem diferença estatisticamente significativa (p >= 0.05)'
        }

    # Teste de impacto do grau Nao-IID (IID vs Nao-IID Severo no FedAvg)
    sub_avg = df_metrics[df_metrics.method == 'FedAvg']
    iid_f1 = sub_avg[sub_avg.scenario == 'IID'].sort_values('seed')['macro_f1'].values
    sev_f1 = sub_avg[sub_avg.scenario == 'Nao-IID severo'].sort_values('seed')['macro_f1'].values
    t_stat_het, p_val_het = stats.ttest_rel(sev_f1, iid_f1)
    
    paired_tests['heterogeneity_impact_fedavg_iid_vs_severo'] = {
        'n_pairs': len(iid_f1),
        'diff_mean_severo_minus_iid': float(np.mean(sev_f1 - iid_f1)),
        'diff_std': float(np.std(sev_f1 - iid_f1, ddof=1)),
        't_statistic': float(t_stat_het),
        't_pvalue': float(p_val_het),
        'statistically_significant_005': bool(p_val_het < 0.05),
        'conclusion': 'Degradação por heterogeneidade severa com significância estatística' if p_val_het < 0.05 and np.mean(sev_f1 - iid_f1) < 0 else 'Sem significância conclusiva'
    }

    for od in out_dirs:
        (od / 'teste_estatistico_pareado.json').write_text(json.dumps(paired_tests, indent=2, ensure_ascii=False), encoding='utf-8')

    # 3. Analise de Explicabilidade e Estabilidade
    xai_summary = {}
    if not df_concordance.empty:
        conc_summary = df_concordance.groupby(['scenario', 'comparison'], as_index=False).agg(
            jaccard_top5_medio=('jaccard_top5', 'mean'),
            jaccard_top5_dp=('jaccard_top5', 'std')
        )
        for od in out_dirs:
            conc_summary.to_csv(od / 'resumo_concordancia_top5_real.csv', index=False)
        xai_summary['concordancia_modelos'] = conc_summary.to_dict(orient='records')

    if not df_client_conc.empty:
        client_summary = df_client_conc.groupby(['scenario', 'method'], as_index=False).agg(
            jaccard_inter_clientes_medio=('mean_pairwise_top5_jaccard', 'mean'),
            jaccard_inter_clientes_dp=('mean_pairwise_top5_jaccard', 'std')
        )
        for od in out_dirs:
            client_summary.to_csv(od / 'resumo_concordancia_entre_clientes_real.csv', index=False)
        xai_summary['concordancia_entre_clientes'] = client_summary.to_dict(orient='records')

    # Estabilidade dos rankings de features entre sementes (Jaccard Top-10 entre sementes)
    stability_results = []
    if not df_importances.empty:
        for sc in scenarios:
            for m in ['Centralizado', 'FedAvg', 'FedProx']:
                sub_imp = df_importances[(df_importances.scenario == sc) & (df_importances.method == m)]
                seeds = sub_imp['seed'].unique()
                top10_by_seed = []
                for s in seeds:
                    top10 = set(sub_imp[sub_imp.seed == s].nsmallest(10, 'rank')['feature'].tolist())
                    top10_by_seed.append(top10)
                
                pairwise_jaccard = []
                for i in range(len(top10_by_seed)):
                    for j in range(i + 1, len(top10_by_seed)):
                        s1, s2 = top10_by_seed[i], top10_by_seed[j]
                        jacc = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0
                        pairwise_jaccard.append(jacc)
                
                stability_results.append({
                    'scenario': sc,
                    'method': m,
                    'top10_stability_jaccard_mean': float(np.mean(pairwise_jaccard)),
                    'top10_stability_jaccard_std': float(np.std(pairwise_jaccard, ddof=1))
                })
        
        df_stab = pd.DataFrame(stability_results)
        for od in out_dirs:
            df_stab.to_csv(od / 'estabilidade_rankings_xai.csv', index=False)
        xai_summary['estabilidade_rankings'] = stability_results

    for od in out_dirs:
        (od / 'estabilidade_xai.json').write_text(json.dumps(xai_summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # 4. Grafico Publicavel: Macro-F1 Medio com Barras de Erro (Desvio-Padrao)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pv_mean = summary.pivot(index='scenario', columns='method', values='macro_f1_media')
    pv_std = summary.pivot(index='scenario', columns='method', values='macro_f1_dp')
    
    # Reordenar cenarios se necessario
    order = [sc for sc in ['IID', 'Nao-IID moderado', 'Nao-IID severo', 'Participantes naturais'] if sc in pv_mean.index]
    pv_mean = pv_mean.reindex(order)
    pv_std = pv_std.reindex(order)
    
    pv_mean.plot(kind='bar', yerr=pv_std, capsize=4, ax=ax, colormap='tab10', edgecolor='black', linewidth=0.6, ylim=(0, 1.05), rot=10)
    ax.set_title(f'Desempenho no Teste UCI HAR: Macro-F1 (Média ± Desvio-Padrão, N={len(seed_dirs)} Sementes)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cenário de Distribuição Federada', fontsize=10)
    ax.set_ylabel('Macro-F1 Médio', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Método', loc='lower right', framealpha=0.95)
    fig.tight_layout()
    for od in out_dirs:
        fig.savefig(od / 'fig_resumo_macro_f1_real.png', dpi=250)
    plt.close(fig)

    # 5. Grafico Publicavel: Concordancia Top-5 Inter-Clientes
    if not df_client_conc.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        pv_cl_mean = client_summary.pivot(index='scenario', columns='method', values='jaccard_inter_clientes_medio')
        pv_cl_std = client_summary.pivot(index='scenario', columns='method', values='jaccard_inter_clientes_dp')
        pv_cl_mean = pv_cl_mean.reindex([sc for sc in order if sc in pv_cl_mean.index])
        pv_cl_std = pv_cl_std.reindex([sc for sc in order if sc in pv_cl_std.index])
        
        pv_cl_mean.plot(kind='bar', yerr=pv_cl_std, capsize=4, ax=ax, colormap='Accent', edgecolor='black', linewidth=0.6, ylim=(0, 1.05), rot=10)
        ax.set_title('Concordância Top-5 de Explicabilidade entre Clientes Locais (Índice de Jaccard Médio)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cenário de Distribuição', fontsize=10)
        ax.set_ylabel('Jaccard Top-5 Médio', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Método Federado', loc='lower right', framealpha=0.95)
        fig.tight_layout()
        for od in out_dirs:
            fig.savefig(od / 'fig_concordancia_inter_clientes_real.png', dpi=250)
        plt.close(fig)

    print("\nAnalise estatistica e graficos gerados com sucesso!")

if __name__ == '__main__':
    run_statistical_analysis()
