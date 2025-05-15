from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary

from dlkit.settings.general_settings import TrainerSettings
from dlkit.utils.loading import init_class


def initialize_trainer(settings: TrainerSettings) -> Trainer:
	callbacks = [ModelSummary(max_depth=2)]
	for callback in settings.callbacks:
		callbacks.append(init_class(callback))

	if settings.logger.name is not None:
		logger = init_class(settings.logger)
	else:
		logger = False

	trainer = Trainer(
		**settings.to_dict_compatible_with(Trainer, exclude=('callbacks', 'name', 'logger')),
		callbacks=callbacks,
		logger=logger,
		num_sanity_val_steps=0,
	)
	return trainer
