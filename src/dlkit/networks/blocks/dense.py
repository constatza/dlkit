import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import torch


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super(DenseBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        # self.fc2 = nn.Linear(out_features // 2, out_features)
        # self.fc3 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        # x = self.fc3(x)
        return x
