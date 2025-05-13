from typing import Literal

from dlkit.datamodules.base import BaseDataModule
from dlkit.datamodules.utils import get_or_create_idx_split
from dlkit.settings.general_settings import DataSettings, PathSettings
from dlkit.utils.import_utils import import_from_module


def initialize_datamodule(
	data_settings: DataSettings,
	paths: PathSettings,
	datamodule_device: Literal['cpu', 'cuda'] = 'cpu',
) -> BaseDataModule:
	"""Dynamically imports and sets up the datamodule based on the provided configuration.
	:return: LightningDataModule: The instantiated datamodule object.
	"""

	dataset = import_from_module(
		data_settings.dataset.name,
		module_prefix=data_settings.dataset.module_path,
	)

	dataset = dataset(
		**data_settings.dataset.to_dict_compatible_with(dataset),
		**paths.to_dict_compatible_with(dataset),
	)

	module = import_from_module(
		data_settings.module.name,
		module_prefix=data_settings.module.module_path,
	)

	idx_split = get_or_create_idx_split(
		n=len(dataset),
		filepath=paths.idx_split,
		save_dir=paths.input_dir,
		test_size=data_settings.test_size,
		val_size=data_settings.val_size,
	)

	datamodule_instance = module(
		dataset=dataset,
		settings=data_settings,
		idx_split=idx_split,
	)

	return datamodule_instance
