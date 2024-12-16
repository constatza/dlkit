import torch

from dlkit.metrics import normalized_rmse
from dlkit.networks.blocks import OptimizerSchedulerNetwork


class CAE(OptimizerSchedulerNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        encoding = self.encode(x)
        return self.decode(encoding)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss_func(x_hat, x)
        self.train_loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss_func(x_hat, x)
        self.val_loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.test_loss_func(x_hat, x)
        self.test_loss = loss
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        encoding = self.encode(x)
        predictions = self.decode(encoding)
        return predictions, encoding

    @staticmethod
    def training_loss_func(x_hat, x):
        return torch.nn.functional.huber_loss(x_hat, x)

    @staticmethod
    def test_loss_func(x_hat, x):
        return CAE.training_loss_func(x_hat, x)
