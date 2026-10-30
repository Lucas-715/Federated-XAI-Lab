"""Analise exaustiva de estabilidade de explicabilidade do Integrated Gradients.

Avalia a robustez dos rankings Top-5 em relacao a quantidade de passos de integracao
(16, 32 e 64 passos) em todos os 60 checkpoints treinados (5 sementes x 4 cenarios x 3 metodos).
Exporta:
  - results/estabilidade_ig_passos.csv (180 comparacoes pareadas)
  - results/resumo_estabilidade_ig_passos.csv (medias agregadas por cenario e metodo)
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from flxai.data import uci_har_dataset
from flxai.models import new_model
from flxai.explanation import (
    select_stratified_xai_samples,
    summarize_explanations,
    top_k_jaccard
)

def run_stability_analysis():
    root = Path(".")
    results_dir = root / "results"
    
    seeds = [42, 43, 44, 45, 46]
    records = []
    
    sc_map = {
        'iid': 'IID',
        'nao-iid_moderado': 'Nao-IID moderado',
        'nao-iid_severo': 'Nao-IID severo',
        'participantes_naturais': 'Participantes naturais'
    }

    print("=====================================================================")
    print("Iniciando Análise de Estabilidade de Rankings do Integrated Gradients")
    print("Avaliando 60 checkpoints x 3 comparações de resolução (180 pares)...")
    print("=====================================================================")

    for seed in seeds:
        seed_dir = results_dir / f"uci_seed_{seed}"
        ckpt_dir = seed_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"Aviso: Diretório {ckpt_dir} não encontrado. Pulando.")
            continue
            
        data = uci_har_dataset("data/UCI HAR Dataset", seed=seed)
        strat_idx, meta = select_stratified_xai_samples(
            data.X_test, data.y_test, groups=data.group_test,
            samples_per_class=20, seed=seed
        )
        X = data.X_test[strat_idx]
        y = data.y_test[strat_idx]
        
        for ckpt_file in sorted(ckpt_dir.glob("*.pt")):
            ckpt_name = ckpt_file.name
            parts = ckpt_name.replace(".pt", "").split("_")
            method = parts[-1].capitalize() if parts[-1] in ['fedavg', 'fedprox', 'centralizado'] else parts[-1]
            sc_part = "_".join(parts[:-1])
            sc = sc_map.get(sc_part, sc_part)
            
            model = new_model(561, 6, seed=seed)
            model.load_state_dict(torch.load(ckpt_file, map_location="cpu", weights_only=True))
            model.eval()
            
            # Atribuições e rankings em 16, 32 e 64 passos
            t16, _ = summarize_explanations(model, X, data.features, steps=16, target_class=y)
            t32, _ = summarize_explanations(model, X, data.features, steps=32, target_class=y)
            t64, _ = summarize_explanations(model, X, data.features, steps=64, target_class=y)
            
            f16 = [x['feature'] for x in t16[:5]]
            f32 = [x['feature'] for x in t32[:5]]
            f64 = [x['feature'] for x in t64[:5]]
            
            comparisons = [
                (16, 32, t16, t32, f16, f32),
                (32, 64, t32, t64, f32, f64),
                (16, 64, t16, t64, f16, f64),
            ]
            
            for sa, sb, ta, tb, fa, fb in comparisons:
                j = top_k_jaccard(ta, tb, k=5)
                order_match = (fa == fb)
                
                # Se os 5 atributos forem idênticos, calcula a correlação de Spearman na ordenação
                if j == 1.0:
                    ranks_a = {feat: idx for idx, feat in enumerate(fa)}
                    ranks_b = {feat: idx for idx, feat in enumerate(fb)}
                    order_a = [ranks_a[f] for f in fa]
                    order_b = [ranks_b[f] for f in fa]
                    spearman_val = 1.0 if order_match else float(spearmanr(order_a, order_b).statistic)
                else:
                    spearman_val = float('nan')
                    
                records.append({
                    'seed': seed,
                    'scenario': sc,
                    'method': method,
                    'checkpoint': ckpt_name,
                    'steps_a': sa,
                    'steps_b': sb,
                    'jaccard_top5': j,
                    'exact_order_match_top5': order_match,
                    'spearman_top5': spearman_val,
                    'top5_features_a': "; ".join(fa),
                    'top5_features_b': "; ".join(fb)
                })
        print(f"  -> Semente {seed} processada com sucesso.")

    df = pd.DataFrame(records)
    out_csv = results_dir / "estabilidade_ig_passos.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nArquivo granular exportado com {len(df)} registros: {out_csv}")
    
    # Resumo agregado por Cenário e Método
    summary = df.groupby(['scenario', 'method', 'steps_a', 'steps_b']).agg(
        n_comparacoes=('jaccard_top5', 'count'),
        jaccard_top5_medio=('jaccard_top5', 'mean'),
        jaccard_top5_min=('jaccard_top5', 'min'),
        taxa_ordem_identica=('exact_order_match_top5', 'mean')
    ).reset_index()
    
    out_summary = results_dir / "resumo_estabilidade_ig_passos.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8")
    print(f"Resumo agregado exportado para: {out_summary}")
    
    print("\n=== RESUMO GERAL DE ESTABILIDADE DOS RANKINGS TOP-5 ===")
    print(f"Total de comparações executadas: {len(df)}")
    print(f"Média Geral Jaccard Top-5:       {df['jaccard_top5'].mean():.4f} (DP: {df['jaccard_top5'].std(ddof=1):.4f})")
    print(f"Menor Jaccard observado:         {df['jaccard_top5'].min():.4f}")
    print(f"Taxa de Jaccard = 1.000:         {(df['jaccard_top5'] == 1.0).mean()*100:.2f}% ({(df['jaccard_top5'] == 1.0).sum()}/{len(df)})")
    print(f"Taxa de Ordem 100% Idêntica:     {df['exact_order_match_top5'].mean()*100:.2f}% ({df['exact_order_match_top5'].sum()}/{len(df)})")
    
    return df, summary

if __name__ == "__main__":
    run_stability_analysis()
