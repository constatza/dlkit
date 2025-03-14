import abc

import torch.nn as nn

from dlkit.networks.blocks.network_types import OptimizerSchedulerNetwork
from dlkit.metrics.temporal import (
    mase,
    mase_with_derivative,
    mean_vectorized_scaled_error,
    derivative_mean_abs_loss,
    derivative_abs_error,
)
import torch.nn.functional as F
import torch


class CAE(OptimizerSchedulerNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["activation"])

    @abc.abstractmethod
    def encode(self, x):
        pass

    @abc.abstractmethod
    def decode(self, x):
        pass

    def forward(self, x):
        encoding = self.encode(x)
        return self.decode(encoding)

    @staticmethod
    def training_loss_func(x_hat, x):
        return mean_vectorized_scaled_error(x_hat, x)

    @staticmethod
    def test_loss_func(x_hat, x):
        return mase(x_hat, x)
