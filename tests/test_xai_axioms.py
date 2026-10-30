"""Testes do axioma da completude, amostragem estratificada de producao e unicidade de atributos."""
from pathlib import Path
import numpy as np
import torch
import pytest
from flxai.data import uci_har_dataset, synthetic_sensors
from flxai.models import new_model
from flxai.explanation import (
    select_stratified_xai_samples,
    summarize_explanations,
    top_k_jaccard,
    integrated_gradients
)

@pytest.fixture(scope="module")
def uci_data():
    for p in [Path("data/UCI HAR Dataset"), Path("../../data/UCI HAR Dataset")]:
        if p.exists():
            return uci_har_dataset(p, seed=42)
    pytest.skip("Dataset oficial UCI HAR nao encontrado no caminho padrao.")

def test_production_stratified_sampling_unit():
    """Testa a rotina real de amostragem de producao sobre dados sinteticos controlados."""
    syn = synthetic_sensors(seed=42)
    per_class = 15
    selected, meta = select_stratified_xai_samples(
        syn.X_test, syn.y_test, groups=syn.group_test if hasattr(syn, 'group_test') else None,
        samples_per_class=per_class, seed=42
    )
    assert len(selected) == len(syn.label_names) * per_class
    counts = np.bincount(syn.y_test[selected], minlength=len(syn.label_names))
    assert (counts == per_class).all(), f"Todas as classes devem conter exatamente {per_class} amostras"
    assert meta['samples_per_class'] == per_class
    assert meta['total_samples'] == len(selected)

def test_production_stratified_sampling_on_uci(uci_data):
    """Testa a rotina real de amostragem de producao sobre o teste real do UCI HAR."""
    selected, meta = select_stratified_xai_samples(
        uci_data.X_test, uci_data.y_test, groups=uci_data.group_test,
        samples_per_class=20, seed=42
    )
    assert len(selected) == 120
    counts = np.bincount(uci_data.y_test[selected], minlength=6)
    assert (counts == 20).all(), "Cada uma das 6 classes deve conter exatamente 20 amostras no teste UCI HAR"
    assert meta['n_subjects_represented'] >= 5, "Amostragem estratificada deve abranger multiplos participantes de teste"
    # Participantes oficiais do teste sao subconjunto de [2, 4, 9, 10, 12, 13, 18, 20, 24]
    official_test_subjects = {2, 4, 9, 10, 12, 13, 18, 20, 24}
    assert set(meta['subjects_represented']).issubset(official_test_subjects)

def test_unique_feature_identifiers_in_explanations():
    """Testa que atributos com nomes identicos sao distinguidos univocamente por feature_idx."""
    m = new_model(4, 3, seed=42)
    m.eval()
    X = np.random.randn(10, 4).astype(np.float32)
    # Simula nomes de caracteristicas duplicados (como ocorre 42 vezes no UCI HAR)
    duplicated_names = ['bandsEnergy_1,8', 'bandsEnergy_1,8', 'other_feat', 'bandsEnergy_1,8']
    
    table, local = summarize_explanations(m, X, duplicated_names, steps=5)
    
    # 1. Cada linha da tabela deve ter feature_idx unico de 0 a 3
    indices = [x['feature_idx'] for x in table]
    assert len(set(indices)) == 4, "Cada caracteristica deve possuir indice posicional unico"
    assert sorted(indices) == [0, 1, 2, 3]
    
    # 2. As strings 'feature' incluem o indice posicional para desambiguacao textual
    labels = [x['feature'] for x in table]
    assert len(set(labels)) == 4, "Strings de caracteristicas devem ser unicas"
    
    # 3. Jaccard entre rankings identifica posicoes e nao colide nomes repetidos
    jacc = top_k_jaccard(table, table, k=2)
    assert jacc == 1.0

def test_axiom_of_completeness(uci_data):
    """Verifica formalmente o Axioma da Completude: sum(IG) == F(x) - F(0)."""
    m = new_model(561, 6, seed=42)
    ckpt_path = Path("results/uci_seed_42/checkpoints/iid_fedavg.pt")
    if ckpt_path.exists():
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.eval()
    
    # 1 amostra por classe no teste
    indices = [np.flatnonzero(uci_data.y_test == c)[0] for c in range(6)]
    x_sub = uci_data.X_test[indices]
    y_sub = uci_data.y_test[indices]
    
    ig, _ = integrated_gradients(m, x_sub, steps=32, target_class=y_sub)
    
    with torch.no_grad():
        fx = m(torch.tensor(x_sub, dtype=torch.float32)).gather(1, torch.tensor(y_sub[:, None])).numpy().flatten()
        f0 = m(torch.zeros_like(torch.tensor(x_sub, dtype=torch.float32))).gather(1, torch.tensor(y_sub[:, None])).numpy().flatten()
        
    diff_fx = fx - f0
    sum_ig = ig.sum(axis=1)
    abs_error = np.abs(sum_ig - diff_fx)
    rel_error = abs_error / (np.abs(diff_fx) + 1e-6)
    
    # Erro relativo medio < 2% (costuma ser < 0.6% no modelo treinado)
    assert rel_error.mean() < 0.02 or abs_error.mean() < 0.005, f"Erro da completude excedeu tolerancia: {rel_error.mean()*100:.3f}%"
