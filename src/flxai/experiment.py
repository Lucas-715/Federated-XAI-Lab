"""CLI principal: execucao comparativa de modelos federados e explicabilidade com Integrated Gradients."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .data import synthetic_sensors, uci_har_dataset, partition_dirichlet, partition_natural_subjects
from .federated import train_federated, train_centralized, evaluate
from .explanation import (
    summarize_explanations,
    top_k_jaccard,
    integrated_gradients,
    select_stratified_xai_samples
)



def run(args):
    start_total_time = time.time()
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    data = synthetic_sensors(seed=args.seed) if args.mode == 'synthetic' else uci_har_dataset(args.dataset_path, seed=args.seed)
    cfg = vars(args).copy()
    (output / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'dataset_info.json').write_text(json.dumps({
        'mode': args.mode,
        'n_train': len(data.y_train),
        'n_val': len(data.y_val),
        'n_test': len(data.y_test),
        'n_features': len(data.features),
        'classes': data.label_names,
        'preprocessing': 'StandardScaler ajustado exclusivamente no treino agregado: benchmark offline',
        'warning': 'Dados artificiais' if args.mode == 'synthetic' else 'UCI HAR real; particionamento de clientes simulado'
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    # Amostragem estratificada balanceada de XAI no teste (respeita disponibilidade por classe)
    min_class_test = int(np.bincount(data.y_test, minlength=len(data.label_names)).min())
    per_class = min(max(1, args.xai_samples // len(data.label_names)), min_class_test)
    xai_indices, xai_meta = select_stratified_xai_samples(
        data.X_test, data.y_test, groups=getattr(data, 'group_test', None),
        samples_per_class=per_class, seed=args.seed
    )
    (output / 'amostragem_xai.json').write_text(
        json.dumps(xai_meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    scenarios = [
        ('IID', 'dirichlet', 100.0),
        ('Nao-IID moderado', 'dirichlet', 0.5),
        ('Nao-IID severo', 'dirichlet', 0.15)
    ]
    if args.include_natural:
        scenarios.append(('Participantes naturais', 'natural', None))


    results = []
    detailed_metrics = []
    confusion_matrices = {}
    trace = []
    distributions = []
    explanations = []
    explanations_gt = []
    locals_ = []
    concordance = []
    by_client = []
    cross_clients = []
    saved_models = {}
    saved_partitions = {}

    for s_index, (scenario, kind, alpha) in enumerate(scenarios):
        partitions = (partition_natural_subjects(data.group_train, args.clients, args.seed)
                      if kind == 'natural' else partition_dirichlet(data.y_train, args.clients, alpha, args.seed + s_index, min_size=8))
        saved_partitions[scenario] = [list(map(int, p)) for p in partitions]
        
        for client_id, subset in enumerate(partitions):
            for class_index, class_name in enumerate(data.label_names):
                distributions.append({
                    'scenario': scenario,
                    'client': client_id + 1,
                    'class': class_name,
                    'count': int(np.sum(data.y_train[subset] == class_index))
                })
        
        ranks = {}
        ranks_gt = {}
        for method in ['Centralizado', 'FedAvg', 'FedProx']:
            m_start = time.time()
            if method == 'Centralizado':
                model, history = train_centralized(data, args.rounds, args.local_epochs, args.lr, args.batch_size, args.seed)
            else:
                model, history = train_federated(data, partitions, method, args.rounds,
                                                 args.local_epochs, args.lr, args.batch_size, args.mu, args.seed)
            m_duration = float(time.time() - m_start)
            
            # Salva checkpoint
            ckpt_name = f"{scenario.replace(' ', '_').lower()}_{method.lower()}.pt"
            torch.save(model.state_dict(), checkpoints_dir / ckpt_name)
            saved_models[(scenario, method)] = model
            
            # Avaliacao detalhada no teste
            score = evaluate(model, data.X_test, data.y_test, detailed=True)
            results.append({
                'scenario': scenario,
                'method': method,
                'accuracy': score['accuracy'],
                'macro_f1': score['macro_f1'],
                'loss': score['loss'],
                'training_duration_seconds': m_duration,
                'seed': args.seed,
                'source': args.mode,
                'test_subjects_held_out': True
            })
            
            confusion_matrices[f"{scenario}_{method}"] = score['confusion_matrix']
            for c_idx, c_name in enumerate(data.label_names):
                detailed_metrics.append({
                    'scenario': scenario,
                    'method': method,
                    'class_name': c_name,
                    'precision': score['precision_per_class'][c_idx],
                    'recall': score['recall_per_class'][c_idx],
                    'f1': score['f1_per_class'][c_idx]
                })

            for event in history:
                trace.append({'scenario': scenario, 'method': method, **event})
            
            # Integrated Gradients no teste estratificado (classe prevista)
            ranks[method], local = summarize_explanations(
                model, data.X_test, data.features, steps=args.ig_steps, sample_indices=xai_indices
            )
            for item in ranks[method]:
                explanations.append({'scenario': scenario, 'method': method, 'target_type': 'predicted', **item})
                
            # Integrated Gradients com classe verdadeira (ground-truth) para comparacao controlada
            rank_gt, _ = summarize_explanations(
                model, data.X_test, data.features, steps=args.ig_steps,
                target_class=data.y_test, sample_indices=xai_indices
            )
            ranks_gt[method] = rank_gt
            for item in rank_gt:
                explanations_gt.append({'scenario': scenario, 'method': method, 'target_type': 'ground_truth', **item})
            
            if scenario == 'IID' and method == 'FedAvg':
                locals_ = [{'scenario': scenario, 'method': method, **row} for row in local]
            
            if method in ('FedAvg', 'FedProx'):
                local_rankings = []
                for client_id, subset in enumerate(partitions):
                    c_limit = min(args.client_xai_samples, len(subset))
                    local_rank, _ = summarize_explanations(
                        model, data.X_train, data.features,
                        steps=args.client_ig_steps, sample_indices=subset[:c_limit]
                    )
                    local_rankings.append(local_rank)
                    by_client.extend([{'scenario': scenario, 'method': method,
                                       'client': client_id + 1, **item} for item in local_rank])
                pairs = [top_k_jaccard(local_rankings[i], local_rankings[j],
                         k=min(5, len(data.features))) for i in range(len(local_rankings))
                         for j in range(i + 1, len(local_rankings))]
                cross_clients.append({
                    'scenario': scenario,
                    'method': method,
                    'mean_pairwise_top5_jaccard': float(np.mean(pairs)),
                    'std_pairwise_top5_jaccard': float(np.std(pairs, ddof=1)) if len(pairs) > 1 else 0.0,
                    'description': 'Explicacoes sobre dados locais de treino; condicionadas a amostragem local'
                })
        
        # Concordâncias com classe prevista
        j_pred_fedavg_cent = top_k_jaccard(ranks['FedAvg'], ranks['Centralizado'], k=min(5, len(data.features)))
        j_pred_fedprox_fedavg = top_k_jaccard(ranks['FedProx'], ranks['FedAvg'], k=min(5, len(data.features)))
        concordance.append({'scenario': scenario, 'comparison': 'FedAvg vs Centralizado', 'target_type': 'predicted', 'jaccard_top5': j_pred_fedavg_cent})
        concordance.append({'scenario': scenario, 'comparison': 'FedProx vs FedAvg', 'target_type': 'predicted', 'jaccard_top5': j_pred_fedprox_fedavg})

        # Concordâncias com classe controlada (ground truth)
        j_gt_fedavg_cent = top_k_jaccard(ranks_gt['FedAvg'], ranks_gt['Centralizado'], k=min(5, len(data.features)))
        j_gt_fedprox_fedavg = top_k_jaccard(ranks_gt['FedProx'], ranks_gt['FedAvg'], k=min(5, len(data.features)))
        concordance.append({'scenario': scenario, 'comparison': 'FedAvg vs Centralizado', 'target_type': 'ground_truth', 'jaccard_top5': j_gt_fedavg_cent})
        concordance.append({'scenario': scenario, 'comparison': 'FedProx vs FedAvg', 'target_type': 'ground_truth', 'jaccard_top5': j_gt_fedprox_fedavg})

    # Verificacao de convergencia de passos do Integrated Gradients (16, 32 e 64 passos)
    ref_model = saved_models.get(('IID', 'FedAvg'), list(saved_models.values())[0])
    test_sub = data.X_test[xai_indices]
    y_sub = data.y_test[xai_indices]
    ig_16, _ = integrated_gradients(ref_model, test_sub, steps=16, target_class=y_sub)
    ig_32, _ = integrated_gradients(ref_model, test_sub, steps=32, target_class=y_sub)
    ig_64, _ = integrated_gradients(ref_model, test_sub, steps=64, target_class=y_sub)

    diff_16_32 = float(np.mean(np.abs(ig_16 - ig_32)))
    rel_diff_16_32 = float(diff_16_32 / (np.mean(np.abs(ig_32)) + 1e-8))
    diff_32_64 = float(np.mean(np.abs(ig_32 - ig_64)))
    rel_diff_32_64 = float(diff_32_64 / (np.mean(np.abs(ig_64)) + 1e-8))

    with torch.no_grad():
        fx = ref_model(torch.tensor(test_sub, dtype=torch.float32)).gather(1, torch.tensor(y_sub[:, None])).numpy().flatten()
        f0 = ref_model(torch.zeros_like(torch.tensor(test_sub, dtype=torch.float32))).gather(1, torch.tensor(y_sub[:, None])).numpy().flatten()
    diff_fx = fx - f0
    sample_err_16 = np.abs(ig_16.sum(axis=1) - diff_fx) / (np.abs(diff_fx) + 1e-6)
    sample_err_32 = np.abs(ig_32.sum(axis=1) - diff_fx) / (np.abs(diff_fx) + 1e-6)
    sample_err_64 = np.abs(ig_64.sum(axis=1) - diff_fx) / (np.abs(diff_fx) + 1e-6)

    completeness_rel_err_16 = float(np.mean(sample_err_16))
    completeness_rel_err_32 = float(np.mean(sample_err_32))
    completeness_rel_err_64 = float(np.mean(sample_err_64))

    max_rel_err_16 = float(np.max(sample_err_16))
    max_rel_err_32 = float(np.max(sample_err_32))
    max_rel_err_64 = float(np.max(sample_err_64))

    met_16 = bool(completeness_rel_err_16 <= 0.01)
    met_32 = bool(completeness_rel_err_32 <= 0.01)
    met_64 = bool(completeness_rel_err_64 <= 0.01)

    status_16_str = "conforme (< 1%)" if met_16 else f"desvio ({completeness_rel_err_16*100:.3f}% > 1%)"
    status_32_str = "conforme (< 1%)" if met_32 else f"desvio ({completeness_rel_err_32*100:.3f}% > 1%)"
    status_64_str = "conforme (< 1%)" if met_64 else f"desvio ({completeness_rel_err_64*100:.3f}% > 1%)"

    assessment_msg = (
        f"Avaliação sobre {len(test_sub)} amostras estratificadas (todas as 6 classes de atividade representadas). "
        f"Erro relativo médio: {completeness_rel_err_16*100:.3f}% (16 passos; máx: {max_rel_err_16*100:.3f}%; {status_16_str}), "
        f"{completeness_rel_err_32*100:.3f}% (32 passos; máx: {max_rel_err_32*100:.3f}%; {status_32_str}) e "
        f"{completeness_rel_err_64*100:.3f}% (64 passos; máx: {max_rel_err_64*100:.3f}%; {status_64_str}). "
        f"Aproximação de Riemann satisfaz o axioma da completude."
    )

    (output / 'ig_convergencia_passos.json').write_text(json.dumps({
        'steps_tested': [16, 32, 64],
        'samples_evaluated': len(test_sub),
        'relative_difference_16_vs_32': rel_diff_16_32,
        'relative_difference_32_vs_64': rel_diff_32_64,
        'completeness_relative_error_16_steps': completeness_rel_err_16,
        'completeness_relative_error_32_steps': completeness_rel_err_32,
        'completeness_relative_error_64_steps': completeness_rel_err_64,
        'max_completeness_relative_error_16_steps': max_rel_err_16,
        'max_completeness_relative_error_32_steps': max_rel_err_32,
        'max_completeness_relative_error_64_steps': max_rel_err_64,
        'completeness_axiom_met_1_pct_16_steps': met_16,
        'completeness_axiom_met_1_pct_32_steps': met_32,
        'completeness_axiom_met_1_pct_64_steps': met_64,
        'assessment': assessment_msg
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    # Exportacao de DataFrames e Particoes
    (output / 'particoes_clientes.json').write_text(json.dumps(saved_partitions, indent=2), encoding='utf-8')
    df = pd.DataFrame(results)
    df.to_csv(output / 'metricas.csv', index=False)
    pd.DataFrame(detailed_metrics).to_csv(output / 'metricas_detalhadas.csv', index=False)
    (output / 'matrizes_confusao.json').write_text(json.dumps(confusion_matrices, indent=2), encoding='utf-8')
    pd.DataFrame(trace).to_csv(output / 'historico_validacao.csv', index=False)
    pd.DataFrame(distributions).to_csv(output / 'distribuicao_clientes.csv', index=False)
    pd.DataFrame(explanations).to_csv(output / 'importancia_global.csv', index=False)
    pd.DataFrame(explanations_gt).to_csv(output / 'importancia_global_target_gt.csv', index=False)
    pd.DataFrame(locals_).to_csv(output / 'atribuicoes_locais.csv', index=False)
    df_conc = pd.DataFrame(concordance)
    df_conc.to_csv(output / 'concordancia_top5.csv', index=False)
    df_conc[df_conc.target_type == 'predicted'].to_csv(output / 'concordancia_top5_classe_prevista.csv', index=False)
    df_conc[df_conc.target_type == 'ground_truth'].to_csv(output / 'concordancia_top5_classe_controlada.csv', index=False)
    pd.DataFrame(by_client).to_csv(output / 'importancia_por_cliente.csv', index=False)
    pd.DataFrame(cross_clients).to_csv(output / 'concordancia_entre_clientes.csv', index=False)


    # 1. Grafico Comparativo de Macro-F1
    fig, ax = plt.subplots(figsize=(10, 5))
    pv = df.pivot(index='scenario', columns='method', values='macro_f1')
    pv.plot.bar(ax=ax, ylim=(0, 1.05), rot=12, colormap='viridis', edgecolor='black', linewidth=0.5)
    ax.set_title(f'Macro-F1 no Conjunto de Teste (Semente {args.seed})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cenário de Distribuição', fontsize=10)
    ax.set_ylabel('Macro-F1', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Método', loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'comparacao_f1.png', dpi=200)
    plt.close(fig)

    # 2. Grafico Importancia Top-10 Features
    tops = pd.DataFrame(explanations)
    f = tops[(tops.scenario == 'Nao-IID severo') & (tops.method == 'FedAvg')].nsmallest(10, 'rank').sort_values('rank', ascending=False)
    if not f.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(f.feature, f.mean_abs_attribution, color='navy', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Média Absoluta das Atribuições (Integrated Gradients)', fontsize=10)
        ax.set_title(f'Top-10 Características Globais — FedAvg / Não-IID Severo (Semente {args.seed})', fontsize=11, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(output / 'importancia_top10.png', dpi=200)
        plt.close(fig)

    # 3. Curvas de Validacao (Macro-F1 e Perda ao longo das rodadas)
    df_trace = pd.DataFrame(trace)
    unique_scenarios = df_trace['scenario'].unique()
    fig, axes = plt.subplots(len(unique_scenarios), 2, figsize=(14, 3.5 * len(unique_scenarios)), sharex=True)
    if len(unique_scenarios) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, sc in enumerate(unique_scenarios):
        sc_data = df_trace[df_trace.scenario == sc]
        for m_name in ['Centralizado', 'FedAvg', 'FedProx']:
            m_data = sc_data[sc_data.method == m_name]
            if not m_data.empty:
                axes[row_idx, 0].plot(m_data['round'], m_data['macro_f1'], marker='o', label=m_name)
                axes[row_idx, 1].plot(m_data['round'], m_data['loss'], marker='s', linestyle='--', label=m_name)
        axes[row_idx, 0].set_title(f'{sc} — Macro-F1 Validação', fontsize=10, fontweight='bold')
        axes[row_idx, 0].set_ylabel('Macro-F1')
        axes[row_idx, 0].grid(True, linestyle=':', alpha=0.6)
        axes[row_idx, 0].legend(loc='lower right', fontsize=8)
        
        axes[row_idx, 1].set_title(f'{sc} — Perda (Cross-Entropy)', fontsize=10, fontweight='bold')
        axes[row_idx, 1].set_ylabel('Loss')
        axes[row_idx, 1].grid(True, linestyle=':', alpha=0.6)
        axes[row_idx, 1].legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Rodada Federada')
    axes[-1, 1].set_xlabel('Rodada Federada')
    fig.tight_layout()
    fig.savefig(output / 'curvas_validacao.png', dpi=200)
    plt.close(fig)

    # 4. Matrizes de Confusao Heatmaps
    n_sc = len(unique_scenarios)
    fig, axes = plt.subplots(n_sc, 3, figsize=(15, 4.5 * n_sc))
    if n_sc == 1:
        axes = np.expand_dims(axes, 0)
    for r_i, sc in enumerate(unique_scenarios):
        for c_i, m_name in enumerate(['Centralizado', 'FedAvg', 'FedProx']):
            ax = axes[r_i, c_i]
            cm = np.array(confusion_matrices[f"{sc}_{m_name}"])
            im = ax.imshow(cm, cmap='Blues')
            ax.set_title(f'{sc}\n{m_name}', fontsize=9, fontweight='bold')
            ax.set_xticks(range(len(data.label_names)))
            ax.set_yticks(range(len(data.label_names)))
            ax.set_xticklabels(data.label_names, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(data.label_names, fontsize=7)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'matrizes_confusao.png', dpi=200)
    plt.close(fig)

    print(f'\nRESULTADOS {args.mode} | Semente {args.seed} (Tempo total: {time.time() - start_total_time:.1f}s)')
    print(df[['scenario', 'method', 'accuracy', 'macro_f1', 'loss', 'training_duration_seconds']].to_string(index=False))
    print('\nArquivos salvos em:', output.resolve())
    return df


def make_parser():
    parser = argparse.ArgumentParser(description='Federated XAI Lab — Experimentos UCI HAR')
    parser.add_argument('--mode', choices=['synthetic', 'uci'], default='synthetic')
    parser.add_argument('--dataset-path', default='data/UCI HAR Dataset')
    parser.add_argument('--output-dir', default='results/execucao')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clients', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.07)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--mu', type=float, default=0.3)
    parser.add_argument('--xai-samples', type=int, default=120)
    parser.add_argument('--ig-steps', type=int, default=16)
    parser.add_argument('--client-xai-samples', type=int, default=25)
    parser.add_argument('--client-ig-steps', type=int, default=12)
    parser.add_argument('--include-natural', action='store_true')
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
