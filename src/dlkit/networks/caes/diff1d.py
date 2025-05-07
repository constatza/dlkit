import torch
import torch.nn.functional as F
from pydantic import ConfigDict, validate_call
from torch import nn
from torch.nn import Sequential

from dlkit.networks.blocks.convolutional import ConvolutionBlock1d
from dlkit.networks.blocks.latent import TensorToVectorBlock, VectorToTensorBlock
from dlkit.networks.blocks.residual import SkipConnection
from dlkit.networks.caes.base import CAE
from dlkit.utils.math_utils import linear_interpolation_int


class DiffCAE1d(CAE):
	@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
	def __init__(
		self,
		input_shape: tuple,
		final_channels: int = 10,
		final_timesteps: int = 5,
		latent_size: int = 10,
		num_layers: int = 4,
		kernel_size: int = 5,
		activation: nn.Module = nn.GELU(),
		*args,
		**kwargs,
	):
		"""Initialize a `DiffCAE1d` instance.

		Parameters
		----------
		input_shape : tuple
		    Input shape of the data (batch_size, channels, timesteps).
		final_channels : int, optional
		    Number of channels in the final latent encoding, by default 10.
		final_timesteps : int, optional
		    Number of timesteps in the final latent encoding, by default 5.
		latent_size : int, optional
		    Size of the latent vector, by default 10.
		num_layers : int, optional
		    Number of layers in the encoder and decoder, by default 4.
		kernel_size : int, optional
		    Kernel size for convolutions, by default 5.
		activation : nn.Module, optional
		    Activation function for each block, by default nn.GELU().

		Returns:
		-------
		None
		"""
		super().__init__(*args, **kwargs)
		self.save_hyperparameters(
			ignore=['activation'],
		)

		self.activation = activation
		self.input_shape = input_shape

		self.example_input_array = torch.randn(2, *input_shape[1:])

		initial_channels = input_shape[-2]
		initial_time_steps = input_shape[-1]

		channels = linear_interpolation_int([initial_channels, final_channels], num_layers + 1)
		channels[1:] = channels[1:] + (channels[1:] % 2)
		channels = channels.tolist()

		timesteps = linear_interpolation_int(
			[initial_time_steps, final_timesteps], num_layers + 1
		).tolist()

		# Instantiate feature extractor and latent encoder
		encoder_channels = channels.copy()
		encoder_channels[0] *= 2

		self.encoder = SkipEncoder(
			channels=encoder_channels,
			timesteps=timesteps,
			latent_dim=latent_size,
			kernel_size=kernel_size,
		)

		# Instantiate latent decoder and feature decoder
		self.decoder = SkipDecoder(
			channels=channels,
			timesteps=timesteps,
			latent_dim=latent_size,
			kernel_size=kernel_size,
		)

	def encode(self, x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		dx = self.delta(x)
		x = torch.cat((x, dx), dim=1)
		x = self.encode(x)
		x = self.decode(x)
		return x

	@staticmethod
	def delta(x):
		dx = torch.diff(x, dim=-1, n=1)
		return F.pad(dx, (0, 1))


class SkipEncoder(nn.Module):
	def __init__(
		self,
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

		layers = []
		for i in range(len(timesteps) - 1):
			layers.append(
				SkipConnection(
					ConvolutionBlock1d(
						in_channels=channels[i],
						out_channels=channels[i + 1],
						in_timesteps=timesteps[i],
						kernel_size=kernel_size,
					),
					activation=F.gelu,
				)
			)
			layers.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

		self.feature_extractor = Sequential(*layers)

		self.feature_to_latent = TensorToVectorBlock(channels[-1], latent_dim)

	def forward(self, x):
		x = self.feature_extractor(x)
		# x = self.feature_to_latent(x)
		return x


class SkipDecoder(nn.Module):
	def __init__(
		self,
		latent_dim: int,
		channels: list[int],
		timesteps: list[int],
		kernel_size: int = 3,
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
		timesteps = timesteps[::-1]
		channels = channels[::-1]

		self.latent_to_feature = VectorToTensorBlock(
			latent_dim,
			(channels[0], timesteps[0]),
		)

		num_layers = len(timesteps) - 1
		layers = []
		for i in range(num_layers):
			layers.append(
				SkipConnection(
					ConvolutionBlock1d(
						in_channels=channels[i],
						out_channels=channels[i + 1],
						in_timesteps=timesteps[i],
						kernel_size=kernel_size,
					),
					activation=F.gelu,
				)
			)
			layers.append(nn.Upsample(size=timesteps[i + 1], mode='linear'))

		self.feature_decoder = Sequential(*layers)

		self.smoothing_layer = nn.Sequential(
			nn.GELU(),
			nn.Conv1d(channels[-1], channels[-1], kernel_size=kernel_size, padding='same'),
		)

	def forward(self, x):
		# x = self.latent_to_feature(x)
		x = self.feature_decoder(x)
		x = self.smoothing_layer(x)
		return x
