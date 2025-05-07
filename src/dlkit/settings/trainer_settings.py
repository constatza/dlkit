from typing import Literal

from pydantic import DirectoryPath, Field

from .base_settings import BaseSettings


class CallbackSettings(BaseSettings):
	name: str | None = Field(
		default=None,
		description='Name of the callback.',
	)
	module_path: str = Field(
		default='lightning.pytorch.callbacks',
		description='Module path where the callback class is located.',
	)


class LoggerSettings(BaseSettings):
	name: str | None = Field(
		default=None,
		description='Name of the logger.',
	)
	module_path: str = Field(
		default='lightning.pytorch.loggers',
		description='Module path where the logger class is located.',
	)

	# save_dir: DirectoryPath | None = Field(
	#     None, description="Directory path where the logger should save the model."
	# )


class TrainerSettings(BaseSettings):
	"""TrainerSettings defines configuration options for training a model.

	Attributes:
	    max_epochs (int): Maximum number of epochs to train for. Defaults to 100.
	    gradient_clip_val (float | None): Value for gradient clipping, if any. Defaults to None.
	    fast_dev_run (bool | int): Flag for fast development run or number of batches to run in fast dev mode. Defaults to False.
	    default_root_dir (DirectoryPath | None): Default root directory for the model. Defaults to None.
	    enable_checkpointing (bool): Whether to enable checkpointing. Defaults to False.
	    callbacks (tuple[CallbackSettings, ...]): List of callbacks. Defaults to an empty tuple.
	    logger (LoggerSettings): Logger settings. Defaults to an instance of LoggerSettings.
	    accelerator (Literal["cpu", "cuda"]): Accelerator to use for training. Defaults to "cuda".
	"""

	max_epochs: int = Field(
		default=100,
		description='Maximum number of epochs to train for.',
	)
	gradient_clip_val: float | None = Field(
		default=None, description='Value for gradient clipping (if any).'
	)
	fast_dev_run: bool | int = Field(
		default=False,
		description='Flag for fast development run or number of batches to run in fast dev mode.',
	)
	default_root_dir: DirectoryPath | None = Field(
		default=None, description='Default root directory for the model.'
	)
	enable_checkpointing: bool = Field(
		default=False, description='Whether to enable checkpointing.'
	)
	callbacks: tuple[CallbackSettings, ...] = Field(
		default=tuple(), description='List of callbacks.'
	)

	logger: LoggerSettings = Field(default=LoggerSettings(), description='Logger settings.')

	accelerator: Literal['cpu', 'cuda'] = Field(
		default='cuda', description='Accelerator to use for training.'
	)
