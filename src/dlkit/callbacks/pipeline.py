from lightning.pytorch import Callback


class TransformFittingCallback(Callback):
	"""Fits the pipeline exactly once when training begins."""

	def __init__(self, pipeline: str = 'pipeline'):
		self.pipeline = pipeline
		self._fitted = False

	def on_train_start(self, trainer, pl_module):
		if self._fitted or not trainer.datamodule:
			return
		# Move pipeline (with buffers) to device
		pipeline = getattr(pl_module, self.pipeline)
		pipeline.to(pl_module.device)
		# Fetch the ready train loader
		dl = trainer.datamodule.train_dataloader()
		pipeline.fit(dl)  # fit-only-once on the training set
		self._fitted = True
