import sys

import click
import torch
from lightning.pytorch import seed_everything
from loguru import logger
from pydantic import validate_call

from dlkit.datatypes.learning import TrainingState
from dlkit.io.settings import Settings, load_validated_settings
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.model import initialize_model
from dlkit.setup.trainer import initialize_trainer


@validate_call
def train(settings: Settings) -> TrainingState:
	"""Trains, tests, and predicts using the provided configuration.

	This function initializes the datamodule, trainer, and model using the
	provided configuration. It then executes the training, testing, and
	prediction steps. Finally, it saves the predictions to disk.

	Args:
	    settings: The configuration object for the training process.
	"""
	torch.set_float32_matmul_precision('medium')
	seed_everything(settings.seed)
	logger.info('Training started.')

	datamodule = initialize_datamodule(
		settings.DATA, settings.PATHS, datamodule_device=settings.TRAINER.accelerator
	)
	trainer = initialize_trainer(settings.TRAINER)

	# Initialize model with shapes derived from datamodule
	model = initialize_model(
		settings=settings.MODEL,
		settings_path=settings.PATHS.settings,
	)

	# Train and evaluate the model
	trainer.fit(model, datamodule=datamodule)
	trainer.test(model, datamodule=datamodule)
	trainer.predict(model, datamodule=datamodule)

	logger.info('Training completed.')
	return TrainingState(trainer=trainer, model=model, datamodule=datamodule)


@click.command('train', help='Trains, tests, and predicts using the provided configuration.')
@click.argument('config-path', type=str, default='./config.toml')
def train_cli(config_path: str = './config.toml'):
	settings = load_validated_settings(config_path)
	training_state = train(settings)
	return training_state


def main() -> None:
	"""Main function to parse configuration and trigger training."""
	try:
		train_cli()
	except KeyboardInterrupt:
		logger.warning('Training interrupted.')
	finally:
		sys.exit(0)


if __name__ == '__main__':
	main()
