from collections.abc import Sequence
import abc

from lightning import LightningModule

from dlkit.setup.optimizer import initialize_optimizer
from dlkit.setup.scheduler import initialize_scheduler
import torch
from dlkit.settings.classes import OptimizerSettings, SchedulerSettings, ModelSettings


class OptimizerSchedulerNetwork(LightningModule):

    def __init__(self, settings: ModelSettings):
        super().__init__()
        self.settings = settings
        self.optimizer_config = settings.optimizer
        self.scheduler_config = settings.scheduler
        self.val_loss = None
        self.train_loss = None
        self.test_loss = None

    def configure_optimizers(self):
        optimizer = initialize_optimizer(self.optimizer_config, self.parameters())
        scheduler = initialize_scheduler(self.scheduler_config, optimizer)
        # self.optimizer_config = None
        # self.scheduler_config = None
        if not scheduler:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.training_loss_func(y_hat, y)
        self.train_loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.training_loss_func(y_hat, y)
        self.val_loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.test_loss_func(y_hat, y)
        self.test_loss = loss
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        y_hat = self.forward(x)
        transform_chain = self.trainer.datamodule.transform_chain.to(self.device)
        predictions = transform_chain.inverse_transform(y_hat)
        return predictions.cpu()

    # def on_train_start(self) -> None:
    #     # transfer transforms to trainer device
    #     self.trainer.datamodule.dataset.transforms.to(self.device)

    def on_train_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log(
            "test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False
        )

    @staticmethod
    @abc.abstractmethod
    def training_loss_func(x_hat, x):
        pass

    @staticmethod
    @abc.abstractmethod
    def test_loss_func(x_hat, x):
        pass
