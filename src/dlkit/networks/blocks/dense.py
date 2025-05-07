from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
	):
		"""Initializes a DenseBlock.

		Parameters:
		    in_features (int): Number of input features to the layer.
		    out_features (int): Number of output features from the layer.
		    activation (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to be used in the layer. Defaults to F.gelu.
		"""
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.layer_norm = nn.LayerNorm(in_features)
		self.fc1 = nn.Linear(in_features, out_features)
		self.activation = activation

	def forward(self, x):
		x = self.layer_norm(x)
		x = self.activation(x)
		x = self.fc1(x)
		return x
