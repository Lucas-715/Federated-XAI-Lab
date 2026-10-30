"""Testes de integridade de particionamento dos cenarios federados."""
import numpy as np
import pytest
from flxai.data import uci_har_dataset, partition_dirichlet, partition_natural_subjects

@pytest.fixture(scope="module")
def uci_data():
    from pathlib import Path
    for p in [Path("data/UCI HAR Dataset"), Path("../../data/UCI HAR Dataset")]:
        if p.exists():
            return uci_har_dataset(p, seed=42)
    pytest.skip("Dataset oficial UCI HAR nao encontrado no caminho padrao.")


@pytest.mark.parametrize("scenario,kind,param", [
    ("IID", "dirichlet", 100.0),
    ("Nao-IID moderado", "dirichlet", 0.5),
    ("Nao-IID severo", "dirichlet", 0.15),
    ("Natural", "natural", None)
])
def test_partition_exhaustive_and_disjoint(uci_data, scenario, kind, param):
    clients = 6
    if kind == "natural":
        parts = partition_natural_subjects(uci_data.group_train, clients=clients, seed=42)
    else:
        parts = partition_dirichlet(uci_data.y_train, clients=clients, alpha=param, seed=42, min_size=8)
    
    assert len(parts) == clients
    all_indices = np.concatenate(parts)
    n_train = len(uci_data.y_train)
    
    # 1. Cobertura integral: soma total exatamente igual ao tamanho do treino
    assert len(all_indices) == n_train
    # 2. Disjuncao: nenhum indice duplicado entre clientes
    assert len(np.unique(all_indices)) == n_train
    # 3. Conjuntos cobrem exatamente 0..n_train-1
    assert set(all_indices) == set(range(n_train))
    # 4. Numero minimo de amostras por cliente atendido (> 8)
    for p in parts:
        assert len(p) >= 8

def test_no_test_samples_in_partitions(uci_data):
    # Assegura que indices particionados referem-se exclusivamente a X_train/y_train
    parts = partition_dirichlet(uci_data.y_train, clients=6, alpha=0.5, seed=42, min_size=8)
    train_len = len(uci_data.y_train)
    for p in parts:
        assert (p < train_len).all()
        assert (p >= 0).all()

def test_natural_subjects_exclusivity(uci_data):
    # Nenhum sujeito real de treino pode ser fatiado entre mais de um cliente
    parts = partition_natural_subjects(uci_data.group_train, clients=6, seed=42)
    sub_sets = [set(uci_data.group_train[p]) for p in parts]
    for i in range(len(sub_sets)):
        for j in range(i + 1, len(sub_sets)):
            assert len(sub_sets[i].intersection(sub_sets[j])) == 0
