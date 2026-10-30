"""FedAvg e FedProx simples, simulados sequencialmente em CPU."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time
from .models import new_model


def local_update(global_model, X, y, epochs=1, lr=.025, batch_size=64, mu=0., seed=42):
    model=copy.deepcopy(global_model)
    model.train()
    optimizer=torch.optim.SGD(model.parameters(),lr=lr)
    x=torch.tensor(X,dtype=torch.float32)
    target=torch.tensor(y,dtype=torch.long)
    anchor=[p.detach().clone() for p in global_model.parameters()]
    generator=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(x),generator=generator).split(batch_size):
            optimizer.zero_grad()
            objective=F.cross_entropy(model(x[batch]),target[batch])
            if mu>0:
                objective=objective+(mu/2.)*sum(((p-a)**2).sum() for p,a in zip(model.parameters(),anchor))
            objective.backward()
            optimizer.step()
    return model


def weighted_average(models, weights):
    weights=np.asarray(weights,dtype=np.float64)
    weights=weights/weights.sum()
    sd=[]
    for model in models:
        sd.append(model.state_dict())
    combined={name: sum((float(weights[i])*states[name] for i,states in enumerate(sd)))
              for name in sd[0].keys()}
    merged=copy.deepcopy(models[0]); merged.load_state_dict(combined)
    return merged


def evaluate(model, X, y, detailed=False):
    model.eval()
    t_x = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.long)
    with torch.no_grad():
        logits = model(t_x)
        loss = float(F.cross_entropy(logits, t_y).item())
        pred = logits.argmax(1).numpy()
    
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro', zero_division=0))
    res = {'accuracy': acc, 'macro_f1': f1, 'loss': loss}
    
    if detailed:
        prec_per_class = precision_score(y, pred, average=None, zero_division=0).tolist()
        rec_per_class = recall_score(y, pred, average=None, zero_division=0).tolist()
        f1_per_class = f1_score(y, pred, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y, pred).tolist()
        res.update({
            'precision_per_class': prec_per_class,
            'recall_per_class': rec_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        })
    return res


def train_federated(data, partitions, method, rounds=4, epochs=1, lr=.035,
                    batch_size=64, mu=.1, seed=42):
    start_time = time.time()
    model=new_model(data.X_train.shape[1],len(data.label_names),seed)
    history=[]
    for iteration in range(rounds):
        local_models=[]; sizes=[]
        for client_id, indices in enumerate(partitions):
            updated=local_update(model,data.X_train[indices],data.y_train[indices],
                epochs=epochs,lr=lr,batch_size=batch_size,
                mu=mu if method=='FedProx' else 0.,seed=seed+iteration*101+client_id)
            local_models.append(updated); sizes.append(len(indices))
        model=weighted_average(local_models,sizes)
        validation=evaluate(model,data.X_val,data.y_val)
        history.append({'round':iteration+1,**validation})
    model.training_duration_seconds = float(time.time() - start_time)
    return model, history


def train_centralized(data, rounds=4, epochs=1, lr=.035, batch_size=64,seed=42):
    start_time = time.time()
    model=new_model(data.X_train.shape[1],len(data.label_names),seed)
    history=[]
    # Quantidade de epocas igual a rounds*epochs, sem pretensao de equivalencia de FLOPs.
    x,y=data.X_train,data.y_train
    for iteration in range(rounds):
        model=local_update(model,x,y,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed+iteration*101)
        history.append({'round':iteration+1,**evaluate(model,data.X_val,data.y_val)})
    model.training_duration_seconds = float(time.time() - start_time)
    return model,history
