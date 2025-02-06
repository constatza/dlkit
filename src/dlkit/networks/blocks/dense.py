import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, batch_norm=True, activation=F.gelu
    ):
        super(DenseBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        # self.fc2 = nn.Linear(output_size, output_size)
        self.activation = activation
        self.bn = nn.BatchNorm1d(input_size) if batch_norm else nn.Identity()

    def forward(self, x):
        x = self.bn(x)
        # x = self.activation(x)
        x = self.fc1(x)
        return x
