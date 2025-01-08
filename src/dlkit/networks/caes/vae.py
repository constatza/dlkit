from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule

from dlkit.networks.caes.simple1d import BasicCAE


class VAE(nn.Module):
    def __init__(self, cae, latent_size, activation=nn.GELU()):
        super(VAE, self).__init__()
        self.cae = cae
        self.latent_size = latent_size
        self.linear_mean = nn.Linear(latent_size, latent_size)
        self.linear_std = nn.Linear(latent_size, latent_size)
        self.linear_decode = nn.Linear(latent_size, latent_size)
        self.activation = activation

    def encode(self, x):
        x = self.cae.encode(x)
        x = self.activation(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.activation(z)
        z1 = self.activation(self.linear_decode(z))
        return self.cae.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE2(LightningModule):
    def __init__(
            self,
            cae,
            beta: float,
            activation=nn.functional.gelu,
    ):
        super(VAE2, self).__init__()
        hparams = cae.hparams
        self.cae = cae
        self.beta = beta
        self.linear_mean = nn.Linear(hparams.latent_size, hparams.latent_size)
        self.linear_std = nn.Linear(hparams.latent_size, hparams.latent_size)
        self.linear_decode = nn.Linear(hparams.latent_size, hparams.latent_size)
        self.activation = activation

    def encode(self, x):
        x = self.cae.encode(x)
        x = self.activation(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.activation(z)
        z1 = self.activation(self.linear_decode(z))
        return self.cae.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_recon, mu, logvar = self.forward(x)
        loss = self.loss(x, x_recon, mu, logvar)
        # Calculate loss (e.g., reconstruction loss + KL divergence)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_recon, mu, logvar = self.forward(x)
        loss = self.loss(x, x_recon, mu, logvar)
        # Calculate loss (e.g., reconstruction loss + KL divergence)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_recon, mu, logvar = self.forward(x)
        loss = self.loss(x, x_recon, mu, logvar)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx) -> Any:
        x = batch[0]
        latent_mean, _ = self.encode(x)
        return latent_mean

    def loss(self, x, x_recon, mu, logvar, kld_weight=0.05):
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        kl_div = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )
        loss = recon_loss + self.hparams.beta * kld_weight * kl_div
        return loss
        # return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "strict": True,
            },
        }

    def on_validation_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)


class LinearChannelDescentVAE(VAE2):

    def __init__(self, *args, beta: float, **kwargs):
        self.save_hyperparameters()
        base_cae = BasicCAE(*args, **kwargs)
        super(LinearChannelDescentVAE, self).__init__(
            cae=base_cae,
            beta=beta,
        )
        self.example_input_array = torch.randn(self.hparams.input_shape)


if __name__ == "__main__":
    pass
