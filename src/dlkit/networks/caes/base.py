import abc

from dlkit.networks.blocks.network_types import OptimizerSchedulerNetwork
from dlkit.metrics import nmse_time_series_loss
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
        return nmse_time_series_loss(x_hat, x)

    @staticmethod
    def test_loss_func(x_hat, x):
        return nmse_time_series_loss(x_hat, x)
