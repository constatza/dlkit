from lightning import LightningModule
from dlkit.setup.optimizer import initialize_optimizer
from dlkit.setup.scheduler import initialize_scheduler


class OptimizerSchedulerNetwork(LightningModule):

    def __init__(
        self,
        input_shape: tuple | list = None,
        output_shape: tuple | list = None,
        optimizer_config: dict = None,
        scheduler_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape or input_shape
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.val_loss = None
        self.train_loss = None
        self.test_loss = None

    def configure_optimizers(self):
        optimizer = initialize_optimizer(self.optimizer_config, self.parameters())
        scheduler = initialize_scheduler(self.scheduler_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

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
            "test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
