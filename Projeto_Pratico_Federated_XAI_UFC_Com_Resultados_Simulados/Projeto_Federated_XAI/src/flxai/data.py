"""Geracao sintetica, ingestao UCI HAR e particionamento federado."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_train: np.ndarray
    features: list[str]
    label_names: list[str]


def synthetic_sensors(seed=42, n_subjects=24, samples_per_subject=75):
    """Caracteristicas artificiais inspiradas em medidas de acelerometro/giroscopio.

    NAO representa medicoes reais, nem substitui avaliacao no UCI HAR.
    Cada participante possui deslocamento proprio (heterogeneidade de covariaveis).
    """
    rng = np.random.default_rng(seed)
    classes = ['parado', 'caminhando', 'correndo', 'subindo_escadas']
    features = [f'{sensor}_{stat}' for sensor in ('acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z')
                for stat in ('media','desvio')]
    centers = rng.normal(0, .45, (len(classes), len(features)))
    for c in range(len(classes)):
        centers[c, :3] += c * .65
        centers[c, 3:6] += c * .25
        centers[c, 6:9] += (c%2) * .8
        centers[c, 9:] += (c // 2) * .65
    X, y, group = [], [], []
    for subject in range(n_subjects):
        shift = rng.normal(0, .25, len(features))
        preferred = rng.dirichlet(np.ones(len(classes))*2)
        labels = rng.choice(len(classes), samples_per_subject, p=preferred)
        for cls in labels:
            sample = centers[cls] + shift + rng.normal(0, .55, len(features))
            X.append(sample)
            y.append(int(cls))
            group.append(subject)
    X = np.asarray(X, dtype=np.float32)
    y, group = np.asarray(y, dtype=np.int64), np.asarray(group, dtype=np.int64)
    # Separa PARTICIPANTES para nao repetir identidades entre treino/validacao/teste.
    outer = GroupShuffleSplit(n_splits=1, test_size=.21, random_state=seed)
    dev_idx, test_idx = next(outer.split(X, y, group))
    inner = GroupShuffleSplit(n_splits=1, test_size=.22, random_state=seed+1)
    train_local, val_local = next(inner.split(X[dev_idx], y[dev_idx], group[dev_idx]))
    train_idx, val_idx = dev_idx[train_local], dev_idx[val_local]
    return _scale_and_package(X, y, group, train_idx, val_idx, test_idx, features, classes)


def uci_har_dataset(root, seed=42):
    """Le arquivos oficiais do conjunto UCI HAR extraido em disco."""
    root = Path(root)
    if (root / 'UCI HAR Dataset').is_dir():
        root = root / 'UCI HAR Dataset'
    def load(split):
        base = root / split
        names = ['X_'+split+'.txt', 'y_'+split+'.txt', 'subject_'+split+'.txt']
        missing = [str(base/x) for x in names if not (base/x).is_file()]
        if missing:
            raise FileNotFoundError('Arquivos UCI HAR ausentes: ' + ', '.join(missing))
        X = np.loadtxt(base/names[0], dtype=np.float32)
        y = np.loadtxt(base/names[1], dtype=np.int64).reshape(-1)-1
        groups = np.loadtxt(base/names[2], dtype=np.int64).reshape(-1)
        return X, y, groups
    Xtr, ytr, gtr = load('train')
    Xte, yte, gte = load('test')
    # Validacao separada SOMENTE dos participantes oficiais do treinamento.
    inner = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    it, iv = next(inner.split(Xtr, ytr, gtr))
    feature_path = root/'features.txt'
    if feature_path.exists():
        features = [line.strip().split(' ',1)[1] if ' ' in line else line.strip()
                    for line in feature_path.read_text(encoding='utf-8').splitlines()]
    else:
        features = [f'feature_{i+1}' for i in range(Xtr.shape[1])]
    return _scale_and_package(np.vstack((Xtr,Xte)), np.hstack((ytr,yte)),
        np.hstack((gtr,gte)), it, iv, np.arange(len(ytr),len(ytr)+len(yte)),
        features, ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING'])


def _scale_and_package(X,y,groups,it,iv,ite,features,labels):
    # Escalonamento centralizado para benchmark OFFLINE: nao prova privacidade.
    scaler = StandardScaler().fit(X[it])
    XX = scaler.transform(X).astype(np.float32)
    assert not set(groups[it]).intersection(groups[iv])
    assert not set(groups[it]).intersection(groups[ite])
    assert not set(groups[iv]).intersection(groups[ite])
    return Dataset(XX[it],y[it],XX[iv],y[iv],XX[ite],y[ite],groups[it],features,labels)


def partition_dirichlet(y, clients=6, alpha=.5, seed=42, min_size=12):
    """Distribui indices por classe via Dirichlet, com tamanho minimo por cliente."""
    rng = np.random.default_rng(seed)
    y=np.asarray(y)
    if len(y) < clients*min_size:
        raise ValueError('Nao ha registros suficientes para o minimo por cliente.')
    labels = np.unique(y)
    for _ in range(1500):
        chunks=[[] for _ in range(clients)]
        for lab in labels:
            index = rng.permutation(np.flatnonzero(y == lab))
            proportions = rng.dirichlet(np.repeat(alpha, clients))
            boundaries = np.cumsum((proportions*len(index)).astype(int))[:-1]
            for i, part in enumerate(np.split(index, boundaries)):
                chunks[i].extend(part.tolist())
        if min(map(len,chunks))>=min_size:
            idx=[np.asarray(rng.permutation(v),dtype=np.int64) for v in chunks]
            all_idx=np.concatenate(idx)
            assert len(all_idx)==len(y) and len(np.unique(all_idx))==len(y)
            return idx
    raise RuntimeError('Particionamento sem tamanho minimo. Reduza clientes/min_size ou aumente alpha.')


def partition_natural_subjects(groups, clients=6, seed=42):
    rng=np.random.default_rng(seed)
    unique=rng.permutation(np.unique(groups))
    if len(unique)<clients:
        raise ValueError('Sao necessarios ao menos tantos participantes quanto clientes.')
    blocks=np.array_split(unique,clients)
    return [np.flatnonzero(np.isin(groups,b)) for b in blocks]
