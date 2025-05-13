from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary

from dlkit.settings.general_settings import TrainerSettings
from dlkit.utils.import_utils import import_from_module


def initialize_trainer(settings: TrainerSettings) -> Trainer:
	callbacks = [ModelSummary(max_depth=2)]
	for callback in settings.callbacks:
		cb_class = import_from_module(callback.name, module_prefix=callback.module_path)
		callbacks.append(cb_class(**callback.to_dict_compatible_with(cb_class)))

	if settings.logger.name is not None:
		logger_class = import_from_module(
			class_name=settings.logger.name, module_prefix=settings.logger.module_path
		)
		logger = logger_class(**settings.logger.to_dict_compatible_with(logger_class))
	else:
		logger = False

	trainer = Trainer(
		**settings.to_dict_compatible_with(Trainer, exclude=('callbacks', 'name', 'logger')),
		callbacks=callbacks,
		logger=logger,
		num_sanity_val_steps=0,
	)
	return trainer
