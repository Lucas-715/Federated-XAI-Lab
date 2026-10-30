"""Integrated Gradients deterministico, aplicado aos logits da classe prevista.

Referencial: Sundararajan, Taly & Yan (2017). Nao produz valores SHAP.
"""
import numpy as np
import torch

def integrated_gradients(model, inputs, steps=20, baseline=None):
    model.eval()
    tensor=torch.tensor(np.asarray(inputs),dtype=torch.float32)
    if tensor.ndim == 1:
        tensor=tensor.unsqueeze(0)
    base=torch.zeros_like(tensor) if baseline is None else torch.as_tensor(baseline,dtype=torch.float32)
    assert base.shape==tensor.shape
    with torch.no_grad():
        labels=model(tensor).argmax(dim=1)
    # Media de gradientes em pontos entre baseline e entrada (aproximacao de Riemann).
    total=torch.zeros_like(tensor)
    for alpha in torch.linspace(0.,1.,steps+1)[1:]:
        interpolated=(base+alpha*(tensor-base)).detach().requires_grad_(True)
        logits=model(interpolated).gather(1,labels[:,None]).sum()
        grad=torch.autograd.grad(logits,interpolated)[0]
        total+=grad.detach()
    return ((tensor-base)*(total/steps)).numpy(),labels.numpy()


def summarize_explanations(model,X,feature_names,max_samples=60,steps=20):
    subset=np.asarray(X[:max_samples],dtype=np.float32)
    ig,classes=integrated_gradients(model,subset,steps=steps)
    ranking=np.mean(np.abs(ig),axis=0)
    order=np.argsort(ranking)[::-1]
    table=[{'feature':str(feature_names[i]),'mean_abs_attribution':float(ranking[i]),
            'rank':int(k+1)} for k,i in enumerate(order)]
    local=[{'sample':int(i),'predicted_class':int(classes[i]),
            'feature':str(feature_names[j]),'attribution':float(ig[i,j])}
           for i in range(len(subset)) for j in range(len(feature_names))]
    return table, local


def top_k_jaccard(rank_a,rank_b,k=5):
    a=set(x['feature'] for x in rank_a[:k]); b=set(x['feature'] for x in rank_b[:k])
    return float(len(a&b)/len(a|b)) if (a or b) else 1.
