import numpy as np
import torch
from lightning import LightningModule
from pydantic import ConfigDict, validate_call
from torch import nn
from torch.nn import Sequential

from dlkit.networks.blocks.convolutional import ConvolutionBlock1d
from dlkit.networks.blocks.latent import TensorToVectorBlock, VectorToTensorBlock
from dlkit.networks.caes.base import CAE


class BasicCAE(CAE):
	@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
	def __init__(
		self,
		input_shape: tuple,
		reduced_channels: int = 10,
		reduced_timesteps: int = 5,
		latent_size: int = 10,
		num_layers: int = 4,
		kernel_size: int = 5,
		lr: float = 1e-3,
		activation: nn.Module = nn.GELU(),
		*args,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.save_hyperparameters(
			'input_shape',
			'reduced_channels',
			'reduced_timesteps',
			'latent_size',
			'num_layers',
			'kernel_size',
			'lr',
			ignore=['activation'],
		)

		self.activation = activation
		self.input_shape = input_shape

		self.example_input_array = torch.randn(input_shape[-3:])

		channels = (
			np.linspace(input_shape[-2], reduced_channels, num_layers + 1).astype(int).tolist()
		)

		timesteps = (
			np.linspace(input_shape[-1], reduced_timesteps, num_layers + 1).astype(int).tolist()
		)

		# Instantiate feature extractor and latent encoder

		self.encoder = Encoder(
			input_shape=input_shape,
			latent_dim=latent_size,
			channels=channels,
			kernel_size=kernel_size,
			timesteps=timesteps,
		)

		# Instantiate latent decoder and feature decoder
		self.decoder = Decoder(
			latent_dim=latent_size,
			channels=channels,
			kernel_size=kernel_size,
			timesteps=timesteps,
			output_shape=input_shape,
		)

		self.smoothing_layer = nn.Sequential(
			nn.Conv1d(input_shape[1], input_shape[1], kernel_size=9, padding='same'),
			nn.GELU(),
			nn.Conv1d(input_shape[1], input_shape[1], kernel_size=9, padding='same'),
		)

	def encode(self, x):
		x = self.encoder(x)
		return x

	def decode(self, x):
		x = self.decoder(x)
		x = self.smoothing_layer(x)
		return x


class Encoder(LightningModule):
	def __init__(
		self,
		input_shape: tuple,
		latent_dim: int,
		channels: list[int],
		kernel_size: int = 3,
		timesteps: list[int] | None = None,
	):
		"""Complete encoder that compresses the input into a latent vector.

		Parameters:
		- input_shape (tuple): Shape of the input (batch_size, channels, timesteps).
		- latent_dim (int): Dimension of the latent vector.
		- channels (List[int]): List of channels for each layer.
		- kernel_size (int): Kernel size for convolutions.
		- timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
		- activation (nn.Module): Activation function for each block.
		"""
		super().__init__()
		self.save_hyperparameters(ignore=['activation'])

		num_layers = len(timesteps) - 1
		layers = []
		for i in range(num_layers):
			layers.append(
				ConvolutionBlock1d(
					in_channels=channels[i],
					out_channels=channels[i],
					in_timesteps=timesteps[i],
					kernel_size=kernel_size,
				)
			)
			layers.append(
				ConvolutionBlock1d(
					in_channels=channels[i],
					out_channels=channels[i + 1],
					in_timesteps=timesteps[i],
					kernel_size=kernel_size,
				)
			)
			layers.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

		self.feature_extractor = Sequential(*layers)

		self.feature_to_latent = TensorToVectorBlock(channels[-1], latent_dim)

	def forward(self, x):
		x = self.feature_extractor(x)
		x = self.feature_to_latent(x)
		return x


class Decoder(LightningModule):
	def __init__(
		self,
		latent_dim: int,
		channels: list[int],
		timesteps: list[int],
		kernel_size: int = 3,
		output_shape: tuple | None = None,
	):
		"""Complete decoder that reconstructs the input from a latent vector.

		Parameters:
		- latent_dim (int): Dimension of the latent vector input.
		- channels (List[int]): List of channels for each layer, in reverse order from the encoder.
		- kernel_size (int): Kernel size for transposed convolutions.
		- timesteps (List[int]): List of timesteps for adaptive upsampling.
		- activation (nn.Module): Activation function for each block.
		- output_shape (tuple): Target output shape (batch_size, channels, timesteps) to guarantee correct reconstruction.
		"""
		super().__init__()
		self.output_shape = output_shape
		channels = channels[::-1]
		timesteps = timesteps[::-1]
		self.latent_to_feature = VectorToTensorBlock(latent_dim, (channels[0], timesteps[0]))

		num_layers = len(timesteps) - 1
		# channels =
		layers = []
		for i in range(num_layers):
			layers.append(
				ConvolutionBlock1d(
					in_channels=channels[i],
					out_channels=channels[i],
					in_timesteps=timesteps[i],
					kernel_size=kernel_size,
				)
			)
			layers.append(
				ConvolutionBlock1d(
					in_channels=channels[i],
					out_channels=channels[i + 1],
					in_timesteps=timesteps[i],
					kernel_size=kernel_size,
				),
			)
			layers.append(nn.Upsample(timesteps[i + 1]))

		self.feature_decoder = Sequential(*layers)

	def forward(self, x):
		x = self.latent_to_feature(x)
		x = self.feature_decoder(x)
		return x
