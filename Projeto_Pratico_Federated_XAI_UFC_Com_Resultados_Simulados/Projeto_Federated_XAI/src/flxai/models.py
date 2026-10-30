"""Arquitetura unica para comparacao justa entre estrategias de treino."""
import torch
from torch import nn

class SensorMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim,32), nn.ReLU(),
                                    nn.Linear(32,16), nn.ReLU(),
                                    nn.Linear(16,num_classes))
    def forward(self,x):
        return self.layers(x)


def new_model(input_dim, num_classes, seed=42):
    torch.manual_seed(seed)
    return SensorMLP(input_dim,num_classes)
