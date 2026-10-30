"""Integrated Gradients deterministico, aplicado aos logits da classe prevista ou controlada.

Referencial: Sundararajan, Taly & Yan (2017). Nao produz valores SHAP.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch

def select_stratified_xai_samples(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
                                  samples_per_class: int = 20, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Seleciona amostras estratificadas balanceadas para todas as classes.
    
    Retorna:
        indices: Array 1D com os indices originais das amostras selecionadas.
        metadata: Metadados documentando a amostragem e os participantes representados.
    """
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    selected = []
    for c in classes:
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) < samples_per_class:
            raise ValueError(f"Classe {c} possui apenas {len(c_idx)} amostras, insuficiente para {samples_per_class}")
        chosen = rng.choice(c_idx, size=samples_per_class, replace=False)
        selected.extend(chosen)
    selected = np.array(selected, dtype=np.int64)
    
    class_distribution = {int(c): int((y[selected] == c).sum()) for c in classes}
    subjects = sorted([int(s) for s in np.unique(groups[selected])]) if groups is not None else []
    metadata = {
        'total_samples': int(len(selected)),
        'samples_per_class': int(samples_per_class),
        'num_classes': int(len(classes)),
        'class_distribution': class_distribution,
        'subjects_represented': subjects,
        'n_subjects_represented': len(subjects),
        'seed': int(seed)
    }
    return selected, metadata

def integrated_gradients(model, inputs, steps=20, baseline=None, target_class=None):
    model.eval()
    tensor = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    base = torch.zeros_like(tensor) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32)
    assert base.shape == tensor.shape
    with torch.no_grad():
        preds = model(tensor).argmax(dim=1)
    if target_class is None:
        labels = preds
    elif isinstance(target_class, int):
        labels = torch.full((len(tensor),), target_class, dtype=torch.long)
    else:
        labels = torch.as_tensor(target_class, dtype=torch.long)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
    # Media de gradientes em pontos entre baseline e entrada (aproximacao de Riemann).
    total = torch.zeros_like(tensor)
    for alpha in torch.linspace(0., 1., steps + 1)[1:]:
        interpolated = (base + alpha * (tensor - base)).detach().requires_grad_(True)
        logits = model(interpolated).gather(1, labels[:, None]).sum()
        grad = torch.autograd.grad(logits, interpolated)[0]
        total += grad.detach()
    return ((tensor - base) * (total / steps)).numpy(), labels.numpy()

def summarize_explanations(model, X, feature_names, max_samples=60, steps=20, target_class=None, sample_indices=None):
    if sample_indices is not None:
        subset = np.asarray(X[sample_indices], dtype=np.float32)
        orig_indices = np.asarray(sample_indices, dtype=np.int64)
        if target_class is not None and not isinstance(target_class, int):
            targets = target_class[sample_indices]
        else:
            targets = target_class
    else:
        limit = max_samples if max_samples is not None and max_samples < len(X) else len(X)
        subset = np.asarray(X[:limit], dtype=np.float32)
        orig_indices = np.arange(len(subset), dtype=np.int64)
        targets = target_class[:limit] if target_class is not None and not isinstance(target_class, int) else target_class

    ig, classes = integrated_gradients(model, subset, steps=steps, target_class=targets)
    ranking = np.mean(np.abs(ig), axis=0)
    order = np.argsort(ranking)[::-1]
    
    # Inclui identificador posicional unico para evitar ambiguidade de nomes repetidos no UCI HAR
    table = [{
        'feature_idx': int(i),
        'feature': f"{feature_names[i]} (idx {i})",
        'feature_name': str(feature_names[i]),
        'mean_abs_attribution': float(ranking[i]),
        'rank': int(k + 1)
    } for k, i in enumerate(order)]
    
    local = [{
        'sample': int(orig_indices[i]),
        'sample_idx': int(orig_indices[i]),
        'predicted_class': int(classes[i]),
        'feature_idx': int(j),
        'feature': f"{feature_names[j]} (idx {j})",
        'feature_name': str(feature_names[j]),
        'attribution': float(ig[i, j])
    } for i in range(len(subset)) for j in range(len(feature_names))]
    return table, local

def top_k_jaccard(rank_a, rank_b, k=5):
    # Utiliza identificador posicional unico feature_idx quando presente, fallback para feature
    a = set(x.get('feature_idx', x['feature']) for x in rank_a[:k])
    b = set(x.get('feature_idx', x['feature']) for x in rank_b[:k])
    return float(len(a & b) / len(a | b)) if (a or b) else 1.

