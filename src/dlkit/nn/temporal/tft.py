from lightning.pytorch import LightningModule
from pytorch_forecasting import TemporalFusionTransformer, MultiLoss, MASE, TimeSeriesDataSet

from dlkit.settings import ModelSettings


class TFT(LightningModule):
	def __init__(self, dataset: TimeSeriesDataSet, settings: ModelSettings):
		super().__init__()
		self.model = TemporalFusionTransformer.from_dataset(
			dataset,
			learning_rate=settings.learning_rate,
			hidden_size=settings.hidden_size,
			attention_head_size=settings.attention_head_size,
			dropout=settings.dropout,
			loss=MultiLoss(metrics=[MASE()]),
			log_interval=settings.log_interval,
			reduce_on_plateau_patience=settings.reduce_on_plateau_patience,
		)
