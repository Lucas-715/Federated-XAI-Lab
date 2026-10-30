"""Gera analise estatistica e graficos de heterogeneidade para os 4 cenarios federados no UCI HAR."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
from flxai.data import uci_har_dataset, partition_dirichlet, partition_natural_subjects

def compute_partition_statistics(dataset_path='data/UCI HAR Dataset', seed=42, clients=6):
    data = uci_har_dataset(dataset_path, seed=seed)
    n_train = len(data.y_train)
    n_classes = len(data.label_names)
    
    # Proporcao global de classes no treino
    global_counts = np.bincount(data.y_train, minlength=n_classes)
    global_props = global_counts / n_train
    
    scenarios = [
        ('Cenario A (IID, alpha=100)', 'dirichlet', 100.0),
        ('Cenario B (Nao-IID moderado, alpha=0.5)', 'dirichlet', 0.5),
        ('Cenario C (Nao-IID severo, alpha=0.15)', 'dirichlet', 0.15),
        ('Cenario D (Particionamento natural)', 'natural', None)
    ]
    
    detailed_rows = []
    scenario_metrics = {}
    partitions_by_scenario = {}
    
    for s_idx, (sc_name, kind, alpha) in enumerate(scenarios):
        if kind == 'natural':
            parts = partition_natural_subjects(data.group_train, clients=clients, seed=seed)
        else:
            parts = partition_dirichlet(data.y_train, clients=clients, alpha=alpha, seed=seed + s_idx, min_size=8)
            
        partitions_by_scenario[sc_name] = parts
        
        # Validacao de cobertura e disjuncao
        all_idx = np.concatenate(parts)
        assert len(all_idx) == n_train, f"Tamanho incorreto em {sc_name}"
        assert len(np.unique(all_idx)) == n_train, f"Indices duplicados em {sc_name}"
        assert set(all_idx) == set(range(n_train)), f"Indices nao cobrem todo o treino em {sc_name}"
        
        # Analise de heterogeneidade
        tvd_list = []
        entropy_list = []
        client_sizes = [len(p) for p in parts]
        
        matrix_counts = np.zeros((clients, n_classes), dtype=int)
        
        for c_id, subset in enumerate(parts):
            counts = np.bincount(data.y_train[subset], minlength=n_classes)
            matrix_counts[c_id] = counts
            props = counts / len(subset) if len(subset) > 0 else np.zeros(n_classes)
            
            # Total Variation Distance em relacao a distribuicao global
            tvd = 0.5 * np.sum(np.abs(props - global_props))
            tvd_list.append(float(tvd))
            
            # Entropia normalizada (0 = apenas 1 classe, 1 = perfeitamente uniforme)
            H = entropy(props + 1e-12) / np.log(n_classes)
            entropy_list.append(float(H))
            
            for cls_idx, cls_name in enumerate(data.label_names):
                detailed_rows.append({
                    'scenario': sc_name,
                    'client': c_id + 1,
                    'class_name': cls_name,
                    'class_index': cls_idx,
                    'count': int(counts[cls_idx]),
                    'proportion': float(props[cls_idx]),
                    'total_client_samples': int(len(subset)),
                    'tvd_to_global': float(tvd),
                    'entropy_normalized': float(H)
                })
                
        scenario_metrics[sc_name] = {
            'min_samples_client': int(min(client_sizes)),
            'max_samples_client': int(max(client_sizes)),
            'mean_samples_client': float(np.mean(client_sizes)),
            'std_samples_client': float(np.std(client_sizes)),
            'mean_total_variation_distance': float(np.mean(tvd_list)),
            'mean_normalized_entropy': float(np.mean(entropy_list)),
            'classes_per_client_min': int(min(np.count_nonzero(matrix_counts, axis=1))),
            'classes_per_client_max': int(max(np.count_nonzero(matrix_counts, axis=1)))
        }

    df = pd.DataFrame(detailed_rows)
    
    # Salva CSVs em results
    for out_dir in [Path('results'), Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results')]:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / 'distribuicao_clientes.csv', index=False)
        (out_dir / 'analise_heterogeneidade_cenarios.json').write_text(
            json.dumps(scenario_metrics, indent=2, ensure_ascii=False), encoding='utf-8'
        )

    # Gera visualizacao grafica dos 4 cenarios
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    axes = axes.flatten()
    
    # Compativel com Matplotlib 3.8+
    colors = plt.colormaps['tab10'].colors[:n_classes]
    
    for ax, (sc_name, _, _) in zip(axes, scenarios):
        sub_df = df[df.scenario == sc_name]
        pivot_df = sub_df.pivot(index='client', columns='class_name', values='count')[data.label_names]
        pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', edgecolor='black', linewidth=0.5)
        ax.set_title(f"{sc_name}\n(TVD Médio: {scenario_metrics[sc_name]['mean_total_variation_distance']:.3f} | H Médio: {scenario_metrics[sc_name]['mean_normalized_entropy']:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Cliente Simulado', fontsize=10)
        ax.set_ylabel('Quantidade de Amostras', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    for out_dir in [Path('results'), Path('Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results')]:
        fig.savefig(out_dir / 'fig_heterogeneidade_uci_4_cenarios.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print("Analise concluida com sucesso!")
    print(json.dumps(scenario_metrics, indent=2))

if __name__ == '__main__':
    compute_partition_statistics()
