"""Testes automatizados para integridade do dataset real UCI HAR."""
from pathlib import Path
import numpy as np
import pytest
from flxai.data import uci_har_dataset, partition_natural_subjects, partition_dirichlet

@pytest.fixture(scope="module")
def uci_data():
    # Tenta localizar a pasta oficial nos caminhos padrao
    for p in [Path("data/UCI HAR Dataset"), Path("../../data/UCI HAR Dataset")]:
        if p.exists():
            return uci_har_dataset(p, seed=42)
    pytest.skip("Dataset oficial UCI HAR nao encontrado no caminho padrao.")

def test_uci_shapes_and_features(uci_data):
    assert uci_data.X_train.shape[1] == 561
    assert uci_data.X_val.shape[1] == 561
    assert uci_data.X_test.shape[1] == 561
    assert len(uci_data.features) == 561
    assert len(uci_data.label_names) == 6

def test_uci_sample_counts(uci_data):
    total = len(uci_data.y_train) + len(uci_data.y_val) + len(uci_data.y_test)
    assert total == 10299
    assert len(uci_data.y_test) == 2947
    assert len(uci_data.y_train) + len(uci_data.y_val) == 7352

def test_no_nans_or_infs(uci_data):
    for arr in [uci_data.X_train, uci_data.X_val, uci_data.X_test]:
        assert np.isfinite(arr).all()
        assert not np.isnan(arr).any()

def test_label_ranges_and_completeness(uci_data):
    for y in [uci_data.y_train, uci_data.y_val, uci_data.y_test]:
        unique = np.unique(y)
        assert set(unique).issubset(set(range(6)))
        assert len(unique) == 6

def test_no_subject_leakage(uci_data):
    # O dataset uci_har_dataset ja contem assercoes internas, mas verificamos explicitamente
    assert len(uci_data.group_train) == len(uci_data.y_train)
    # Participantes de teste sao estritamente [2, 4, 9, 10, 12, 13, 18, 20, 24]
    test_official_subjects = {2, 4, 9, 10, 12, 13, 18, 20, 24}
    assert not set(uci_data.group_train).intersection(test_official_subjects)

def test_natural_subject_partitioning(uci_data):
    parts = partition_natural_subjects(uci_data.group_train, clients=6, seed=42)
    assert len(parts) == 6
    all_indices = np.concatenate(parts)
    assert len(all_indices) == len(uci_data.y_train)
    assert len(np.unique(all_indices)) == len(uci_data.y_train)
    # Nenhum sujeito deve aparecer em mais de um cliente
    client_subjects = [set(uci_data.group_train[p]) for p in parts]
    for i in range(6):
        for j in range(i + 1, 6):
            assert not client_subjects[i].intersection(client_subjects[j])

def test_dirichlet_partitioning_on_real_data(uci_data):
    parts = partition_dirichlet(uci_data.y_train, clients=6, alpha=0.5, seed=42, min_size=10)
    assert len(parts) == 6
    all_indices = np.concatenate(parts)
    assert len(all_indices) == len(uci_data.y_train)
    assert len(np.unique(all_indices)) == len(uci_data.y_train)
