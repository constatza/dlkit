from dlkit.domain.nn.cae.skip1d import SkipCAE1d


class LinearCAE1d(SkipCAE1d):
    """1D Linear Convolutional Autoencoder.

    A CAE that uses an identity activation (no non-linearity) throughout.
    Equivalent to ``SkipCAE1d(activation=lambda x: x)``.

    Args:
        in_channels: Number of input channels.
        in_length: Length of input sequence.
        latent_channels: Number of latent channels. Defaults to 5.
        latent_width: Width of latent feature. Defaults to 10.
        latent_size: Size of latent vector. Defaults to 10.
        num_layers: Number of encoder/decoder layers. Defaults to 3.
        kernel_size: Convolution kernel size. Defaults to 3.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        in_length: int,
        latent_channels: int = 5,
        latent_width: int = 10,
        latent_size: int = 10,
        num_layers: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            in_length=in_length,
            latent_channels=latent_channels,
            latent_width=latent_width,
            latent_size=latent_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=lambda x: x,
        )
