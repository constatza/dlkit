"""ML pipeline role enum for DataEntry partitioning."""

from enum import StrEnum, auto


class DataRole(StrEnum):
    """ML pipeline role — determines which batch slot an entry occupies.

    Attributes:
        FEATURE: Model input routed via model_input field.
        TARGET: Ground truth compared against predictions.
        LATENT: Intermediate representation written during inference.
        AUXILIARY: Conditioning signal, metadata, graph-level features.
    """

    FEATURE = auto()
    TARGET = auto()
    LATENT = auto()
    AUXILIARY = auto()
