import abc

from torch import nn


class CAE(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, *args, **kwargs): ...

    @abc.abstractmethod
    def decode(self, *args, **kwargs): ...

    def forward(self, x):
        encoding = self.encode(x)
        return self.decode(encoding)

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        latent = self.encode(x)
        y = self.decode(latent)
        return {"predictions": y.detach(), "latent": latent.detach()}
