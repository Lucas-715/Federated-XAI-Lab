"""Teste de Integracao Ponta a Ponta do Pipeline Federated XAI Lab.

Garante que run() executa sem NameError ou falhas de fluxo,
e gera todos os arquivos esperados (CSVs, JSONs, PNGs e checkpoints .pt).
"""
import json
from pathlib import Path
import pytest
import pandas as pd

from flxai.experiment import run, make_parser


def test_full_pipeline_synthetic_execution(tmp_path):
    output_dir = tmp_path / "integration_output"
    
    parser = make_parser()
    args = parser.parse_args([
        "--mode", "synthetic",
        "--output-dir", str(output_dir),
        "--rounds", "2",
        "--clients", "3",
        "--local-epochs", "1",
        "--lr", "0.05",
        "--batch-size", "16",
        "--xai-samples", "12",
        "--ig-steps", "4",
        "--client-xai-samples", "6",
        "--client-ig-steps", "4",
        "--seed", "123"
    ])
    
    # 1. Execucao ponta a ponta
    df_result = run(args)
    assert isinstance(df_result, pd.DataFrame)
    assert not df_result.empty
    
    # 2. Verificacao de CSVs essenciais
    expected_csvs = [
        "metricas.csv",
        "metricas_detalhadas.csv",
        "historico_validacao.csv",
        "distribuicao_clientes.csv",
        "importancia_global.csv",
        "importancia_global_target_gt.csv",
        "atribuicoes_locais.csv",
        "concordancia_top5.csv",
        "concordancia_top5_classe_prevista.csv",
        "concordancia_top5_classe_controlada.csv",
        "importancia_por_cliente.csv",
        "concordancia_entre_clientes.csv",
    ]
    for csv_file in expected_csvs:
        csv_path = output_dir / csv_file
        assert csv_path.exists(), f"Arquivo CSV obrigatorio ausente: {csv_file}"
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) > 0, f"Arquivo CSV vazio: {csv_file}"

    # 3. Verificacao de JSONs
    expected_jsons = [
        "config.json",
        "dataset_info.json",
        "matrizes_confusao.json",
        "amostragem_xai.json",
        "particoes_clientes.json",
        "ig_convergencia_passos.json",
    ]
    for json_file in expected_jsons:
        json_path = output_dir / json_file
        assert json_path.exists(), f"Arquivo JSON obrigatorio ausente: {json_file}"
        content = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(content, (dict, list)), f"Conteudo invalido em {json_file}"

    # 4. Verificacao de Figuras geradas
    expected_pngs = [
        "comparacao_f1.png",
        "importancia_top10.png",
        "curvas_validacao.png",
        "matrizes_confusao.png",
    ]
    for png_file in expected_pngs:
        png_path = output_dir / png_file
        assert png_path.exists(), f"Figura obrigatoria ausente: {png_file}"
        assert png_path.stat().st_size > 500, f"Figura gerada corrompida ou vazia: {png_file}"

    # 5. Verificacao de Checkpoints (.pt)
    ckpts_dir = output_dir / "checkpoints"
    assert ckpts_dir.exists(), "Diretorio de checkpoints ausente"
    pts = list(ckpts_dir.glob("*.pt"))
    assert len(pts) > 0, "Nenhum arquivo de checkpoint .pt encontrado"
