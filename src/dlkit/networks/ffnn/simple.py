from collections.abc import Sequence

import torch.nn as nn


class FeedForwardNN(nn.Module):
	def __init__(
		self,
		layers: Sequence[int],
		activation: nn.Module = nn.GELU(),
		layer_norm: bool = False,
		dropout: float = 0,
		batch_norm: bool = False,
	):
		super().__init__()
		self.num_layers = len(layers) - 1

		self.layers = nn.ModuleList()

		for i in range(self.num_layers - 1):
			self.layers.append(nn.Linear(layers[i], layers[i + 1]))
			if layer_norm:
				self.layers.append(nn.LayerNorm(layers[i + 1]))
			elif batch_norm:
				self.layers.append(nn.BatchNorm1d(layers[i + 1]))

			self.layers.append(activation)
			if dropout > 0:
				self.layers.append(nn.Dropout(dropout))

		self.layers.append(nn.Linear(layers[-2], layers[-1]))
		self.layers = nn.Sequential(*self.layers)

	def forward(self, x):
		return self.layers(x)

	def predict_step(self, batch, batch_idx):
		"""Predict step for the model."""
		x = batch[0]
		predictions = self.forward(x)
		return predictions


class ConstantHiddenSizeFFNN(FeedForwardNN):
	def __init__(
		self,
		input_size: int | None = None,
		output_size: int | None = None,
		hidden_size: int | None = None,
		num_layers: int | None = None,
		**kwargs,
	):
		layers = [input_size] + [hidden_size] * num_layers + [output_size]
		super().__init__(layers, **kwargs)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
			x = self.activation(x)
		return x
