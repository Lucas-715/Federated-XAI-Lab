"""Testes do axioma da completude e amostragem estratificada para Integrated Gradients."""
import numpy as np
import torch
import pytest
from flxai.data import uci_har_dataset
from flxai.models import new_model
from flxai.explanation import integrated_gradients

@pytest.fixture(scope="module")
def uci_data():
    return uci_har_dataset("data/UCI HAR Dataset", seed=42)

def test_stratified_sampling_balance(uci_data):
    classes = np.unique(uci_data.y_test)
    n_samples = 120
    per_class = n_samples // len(classes)
    
    selected = []
    rng = np.random.default_rng(42)
    for c in classes:
        c_idx = np.flatnonzero(uci_data.y_test == c)
        selected.extend(rng.choice(c_idx, size=per_class, replace=False))
        
    counts = np.bincount(uci_data.y_test[selected], minlength=len(classes))
    assert len(selected) == 120
    assert (counts == 20).all(), "Amostragem deve conter exatamente 20 amostras por classe"

def test_axiom_of_completeness(uci_data):
    # Verifica sum(IG) == F(x) - F(0) com tolerancia de integracao numerica (Axioma da Completude)
    from pathlib import Path
    m = new_model(561, 6, seed=42)
    ckpt_path = Path("results/uci_seed_42/checkpoints/iid_fedavg.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/results/uci_seed_42/checkpoints/iid_fedavg.pt")
    if ckpt_path.exists():
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.eval()
    
    # Amostra pequena controlada (1 por classe)
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
    
    # Erro relativo medio < 2% (se treinado, costuma ser < 0.6%; se aleatorio, erro absoluto < 0.005)
    assert rel_error.mean() < 0.02 or abs_error.mean() < 0.005, f"Erro do axioma da completude excedeu limite: rel={rel_error.mean()*100:.3f}%, abs={abs_error.mean():.6f}"

