from torch import nn


class LatentToFeature(nn.Module):
    def __init__(self, latent_dim: int, target_shape: tuple):
        """
        Converts latent vector into a feature map for the decoder.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - target_shape (tuple): Target shape as (channels, timesteps) for the feature map.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.fc = nn.Linear(latent_dim, target_shape[0] * target_shape[1])

    def forward(self, x):
        x = self.fc(x)
        return x.view(
            x.size(0), *self.target_shape
        )  # Reshape to (batch_size, channels, timesteps)


class FeatureToLatent(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int):
        """
        Converts the feature map into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the feature map (channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        """
        super().__init__()
        channels, timesteps = input_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * timesteps, latent_dim)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)
