from pydantic import ConfigDict, validate_call

from dlkit.datatypes.dataset import Shape
from dlkit.nn.cae.base import CAE
from dlkit.nn.cae import SkipCAE1d


class LinearCAE1d(CAE):
    latent_channels: int
    latent_width: int
    latent_size: int
    num_layers: int
    kernel_size: int
    shape: Shape

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        shape: Shape,
        latent_channels: int = 5,
        latent_width: int = 10,
        latent_size: int = 10,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["activation"],
        )

        cae = SkipCAE1d(
            shape=shape,
            latent_channels=latent_channels,
            latent_width=latent_width,
            latent_size=latent_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=lambda x: x,
        )
