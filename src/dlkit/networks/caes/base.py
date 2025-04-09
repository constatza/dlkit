import abc
from dlkit.networks.blocks.basic_network import BasicNetwork


class CAE(BasicNetwork):

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
    @abc.abstractmethod
    def training_loss_func(x_hat, x): ...

    @staticmethod
    @abc.abstractmethod
    def test_loss_func(x_hat, x): ...
