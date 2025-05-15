from collections.abc import Callable, Sequence

import torch
from lightning import LightningModule, LightningDataModule
from loguru import logger

from dlkit.settings import ModelSettings, OptimizerSettings, SchedulerSettings
from dlkit.transforms.pipeline import Pipeline
from dlkit.utils.loading import init_class


class PipelineNetwork(LightningModule):
	settings: ModelSettings
	optimizer_settings: OptimizerSettings
	scheduler_settings: SchedulerSettings
	datamodule: LightningDataModule | None
	pipeline: Pipeline
	model: LightningModule
	train_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

	def __init__(self, settings: ModelSettings) -> None:
		super().__init__()
		super().save_hyperparameters(settings.model_dump())
		self.settings = settings
		self.pipeline = Pipeline(
			settings.input_shape,
			feature_transforms=settings.feature_transforms,
			target_transforms=settings.target_transforms,
			is_autoencoder=settings.is_autoencoder,
		)
		self.model = init_class(settings)
		self.datamodule = None
		self.training_loss_func = getattr(
			self.model, 'training_loss_func', init_class(self.settings.train_loss)
		)
		self.validation_loss_func = self.training_loss_func
		self.test_loss_func = getattr(
			self.model, 'test_loss_func', init_class(self.settings.test_loss)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pipeline(x, which='features')
		x = self.model(x)
		return x

	def configure_optimizers(self):
		optimizer = init_class(self.settings.optimizer, params=self.model.parameters())
		scheduler = init_class(self.settings.scheduler, optimizer=optimizer)
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
		loss = self.training_loss_func(y_hat, y_true)
		self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		y_true = self.pipeline(y, which='targets')
		loss = self.training_loss_func(y_hat, y_true)
		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		y_true = self.pipeline(y, which='targets')
		loss = self.test_loss_func(y_hat, y_true)
		self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
		return loss

	def predict_step(
		self, batch, batch_idx
	) -> torch.Tensor | tuple[torch.Tensor] | dict[str, torch.Tensor]:
		"""Predict step for the model.
		Important Assumption: the main prediction is the first element if the output is a list or tuple
		and the "predictions" key if the output is a dictionary.
		Args:
		    batch: The input batch.
		    batch_idx: The index of the batch.
		Returns:
		    The predictions from the model.
		"""
		x = batch[0]
		x = self.pipeline(x)

		predictions = self.model.predict_step((x,), batch_idx)
		if isinstance(predictions, torch.Tensor):
			return self._return_from_tensor(predictions)
		if isinstance(predictions, list | tuple):
			return self._return_from_sequence(predictions)
		if isinstance(predictions, dict):
			return self._return_from_dict(predictions)
		logger.error(f'Unexpected output type in predict_step: {type(predictions)}')
		raise ValueError(f'Unexpected output type in predict_step {type(predictions)}')

	def _return_from_dict(self, predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		predictions['predictions'] = self.pipeline.inverse_transform(
			predictions['predictions'], which='targets'
		)
		return predictions

	def _return_from_sequence(
		self, predictions: Sequence[torch.Tensor]
	) -> tuple[torch.Tensor, ...]:
		predictions_0 = self.pipeline.inverse_transform(predictions[0], which='targets')
		return (predictions_0,) + tuple(pred for pred in predictions[1:])  # new tuple

	def _return_from_tensor(self, predictions: torch.Tensor) -> torch.Tensor:
		return self.pipeline.inverse_transform(predictions, which='targets')

	def on_train_epoch_end(self) -> None:
		lr = self.trainer.optimizers[0].param_groups[0]['lr']
		self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True)

	@classmethod
	def load_from_checkpoint(cls, ckpt_path: str, **kwargs):
		# 1. pull out the saved hparams dict
		data = torch.load(ckpt_path)
		hparams = data.get('hyper_parameters') or data['hparams']
		# 2. rebuild ModelSettings
		settings = ModelSettings.model_validate(hparams)
		# 3. delegate to Lightning, injecting the Settings object
		return super().load_from_checkpoint(ckpt_path, settings=settings, **kwargs)
