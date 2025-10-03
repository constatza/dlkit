from collections.abc import Sequence
from torch import Tensor
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.data import Data

# Small constant to prevent division by zero
EPSILON: float = 1e-14


def _min_max_scale(tensor: Tensor) -> Tensor:
    """Scale `tensor` to [0,1] based on its own min and max values.

    Args:
        tensor: Input tensor of arbitrary shape.

    Returns:
        A tensor of the same shape with values in [0,1].
    """
    # Compute per‑feature minima and maxima
    min_val = tensor.min(dim=0, keepdim=True).values  # torch.min
    max_val = tensor.max(dim=0, keepdim=True).values  # torch.max

    # Compute scale and clamp to avoid zero division
    scale = (max_val - min_val).clamp(min=EPSILON)  # torch.clamp
    return (tensor - min_val) / scale


class MinMaxTransform(BaseTransform):
    """Apply Min‑Max scaling to specified Data attributes.

    Each attribute in `attrs` is scaled independently to the [0,1] interval.
    """

    def __init__(self, attrs: Sequence[str] = ("x",)) -> None:
        # List of Data attribute names to scale (e.g., ["x", "edge_attr"])
        self.attrs = list(attrs)

    def forward(self, data: Data) -> Data:
        # Iterate over each named attribute
        for attr in self.attrs:
            if not hasattr(data, attr):
                continue
            tensor = getattr(data, attr)
            # Only scale if the attribute exists and is a Tensor
            if tensor is None or not isinstance(tensor, Tensor):
                continue
            # Perform min‑max scaling
            scaled = _min_max_scale(tensor)
            setattr(data, attr, scaled)

        return data  # Return the modified Data object

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs})"
