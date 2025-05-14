from collections.abc import Callable

import torch
from lightning import LightningModule, LightningDataModule
from loguru import logger

from dlkit.settings import ModelSettings, OptimizerSettings, SchedulerSettings
from dlkit.setup.optimizer import initialize_optimizer
from dlkit.setup.scheduler import initialize_scheduler
from dlkit.transforms.pipeline import Pipeline


class PipelineNetwork(LightningModule):
	settings: ModelSettings
	optimizer_settings: OptimizerSettings
	scheduler_settings: SchedulerSettings
	datamodule: LightningDataModule | None
	pipeline: Pipeline
	model: LightningModule
	train_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

	def __init__(
		self,
		settings: ModelSettings,
		model: LightningModule,
		pipeline: Pipeline,
	) -> None:
		super().__init__()
		self.settings = settings
		self.optimizer_config = settings.optimizer
		self.scheduler_config = settings.scheduler
		self.pipeline = pipeline
		self.model = model
		self.datamodule = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pipeline(x, which='features')
		x = self.model(x)
		return x

	def configure_optimizers(self):
		optimizer = initialize_optimizer(self.optimizer_config, self.parameters())
		scheduler = initialize_scheduler(self.scheduler_config, optimizer)
		if not scheduler:
			return {'optimizer': optimizer}
		return {
			'optimizer': optimizer,
			'lr_scheduler': {
				'scheduler': scheduler,
				'frequency': 1,
				'monitor': 'val_loss',
			},
		}

	def on_train_start(self):
		# Move pipeline (with buffers) to device
		self.pipeline = self.pipeline.to(self.device)
		# Fetch the ready train loader
		dl = self.trainer.datamodule.train_dataloader()
		x, y = next(iter(dl))
		if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
			x = x.to(self.device)
			y = y.to(self.device)
			self.pipeline.fit(x, y)  # fit-only-once on the training set
			logger.info('Pipeline fitted and moved to device.')
			return
		logger.warning('Unknown data type in train loader.')

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		y_true = self.pipeline(y, which='targets')
		loss = self.model.training_loss_func(y_hat, y_true)
		self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		y_true = self.pipeline(y, which='targets')
		loss = self.model.training_loss_func(y_hat, y_true)
		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		y_true = self.pipeline(y, which='targets')
		loss = self.model.test_loss_func(y_hat, y_true)
		self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
		return loss

	def predict_step(self, batch, batch_idx):
		x = batch[0]
		x = self.pipeline(x)
		possibly_multiple_arguments = self.model.predict_step((x,), batch_idx)
		main_prediction = possibly_multiple_arguments['predictions']
		possibly_multiple_arguments['predictions'] = self.pipeline.inverse_transform(
			main_prediction
		)
		return possibly_multiple_arguments

	def on_train_epoch_end(self) -> None:
		lr = self.trainer.optimizers[0].param_groups[0]['lr']
		self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True)
