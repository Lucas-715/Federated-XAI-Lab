import numpy as np
import torch
from flxai.data import synthetic_sensors,partition_dirichlet
from flxai.models import new_model
from flxai.federated import weighted_average,train_federated,evaluate
from flxai.explanation import integrated_gradients

def test_groups_held_out():
    d=synthetic_sensors(seed=1)
    assert d.X_train.shape[0]>0 and d.X_test.shape[0]>0
    assert len(d.features)==d.X_train.shape[1]

def test_partitions_are_disjoint_and_exhaustive():
    y=np.tile(np.arange(4),150)
    chunks=partition_dirichlet(y,clients=5,alpha=.5,seed=8,min_size=5)
    everything=np.concatenate(chunks)
    assert len(everything)==len(y)
    assert np.array_equal(np.sort(everything),np.arange(len(y)))

def test_weighted_model_average():
    m1=new_model(3,2,1);m2=new_model(3,2,2)
    with torch.no_grad():
        for p in m1.parameters():p.fill_(1.)
        for p in m2.parameters():p.fill_(3.)
    avg=weighted_average([m1,m2],[1,3])
    for p in avg.parameters(): assert torch.allclose(p,torch.full_like(p,2.5))

def test_integrated_gradients_shapes():
    m=new_model(12,4,3)
    attrs,classes=integrated_gradients(m,np.ones((3,12),dtype=np.float32),steps=5)
    assert attrs.shape==(3,12) and classes.shape==(3,)
    assert np.isfinite(attrs).all()

def test_fedavg_trains_and_evaluates():
    d=synthetic_sensors(seed=2,n_subjects=12,samples_per_subject=30)
    parts=partition_dirichlet(d.y_train,clients=3,alpha=1,seed=2,min_size=5)
    model,h=train_federated(d,parts,'FedAvg',rounds=1,epochs=1,seed=2)
    assert len(h)==1 and 0<=evaluate(model,d.X_test,d.y_test)['macro_f1']<=1
