import abc

from lightning.pytorch import LightningModule


class CAE(LightningModule):
	def __init__(self):
		super().__init__()
		self.save_hyperparameters(ignore=['activation'])

	@abc.abstractmethod
	def encode(self, x):
		pass

	@abc.abstractmethod
	def decode(self, x):
		pass

	def forward(self, x):
		encoding = self.encode(x)
		return self.decode(encoding)

	def predict_step(self, batch, batch_idx):
		x = batch[0]
		latent = self.encode(x)
		y = self.decode(latent)
		return {'predictions': y.detach(), 'latent': latent.detach()}
