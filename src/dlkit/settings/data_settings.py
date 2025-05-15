from pydantic import Field, NonNegativeFloat

from .base_settings import BaseSettings


class DataloaderSettings(BaseSettings):
	num_workers: int = Field(default=1, description='Number of worker processes.')
	batch_size: int = Field(default=64, description='Batch size.')
	shuffle: bool = Field(default=False, description='Whether to shuffle the training data set.')
	persistent_workers: bool = Field(default=True, description='Whether to use persistent workers.')
	pin_memory: bool = Field(default=True, description='Whether to pin memory.')


class DatasetSettings(BaseSettings):
	name: str = Field('NumpyDataset', description='Dataset name.')
	module_path: str = Field(
		default='dlkit.datasets',
		description='Module path where the dataset class is located.',
	)
	group_ids: list[str] | None = Field(
		default=None, description='Group IDs for pytorch forecasting TimeSeriesDataSet.'
	)


class DataModuleSettings(BaseSettings):
	name: str = Field(default='BaseDataModule', description='Datamodule name.')
	module_path: str = Field(
		default='dlkit.datamodules',
		description='Module path where the datamodule class is located.',
	)


class DataSettings(BaseSettings):
	module: DataModuleSettings = Field(DataModuleSettings(), description='Dataset settings.')
	dataloader: DataloaderSettings = Field(DataloaderSettings(), description='Dataloader settings.')
	dataset: DatasetSettings = Field(default=DatasetSettings(), description='Dataset settings.')
	targets_exist: bool = Field(
		default=True,
		description='Whether dataset has both features and targets.',
		frozen=False,
	)
	test_size: NonNegativeFloat = Field(
		default=0.15, description='Fraction of data used for testing.'
	)
	val_size: NonNegativeFloat = Field(
		default=0.15, description='Fraction of data used for validation.'
	)
