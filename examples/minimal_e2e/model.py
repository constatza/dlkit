from __future__ import annotations

import math

from torch import nn, Tensor


class SimpleNet(nn.Module):
    """A tiny MLP for tabular x->y regression.

    Accepts an optional shape mapping {"x": (in_features,), "y": (out_features,)}
    so it works seamlessly with DLKit's wrapper factories.
    """

    def __init__(
        self,
        *,
        shape: dict[str, tuple[int, ...]] | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        hidden_size: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        # Infer in/out dims from shape if provided
        if shape is not None:
            if in_features is None and "x" in shape and len(shape["x"]) >= 1:
                in_features = int(shape["x"][0])
            if out_features is None and "y" in shape and len(shape["y"]) >= 1:
                out_features = int(shape["y"][0])

        # Fallback defaults
        in_features = in_features or 4
        out_features = out_features or 1

        act_layer = nn.ReLU if activation.lower() == "relu" else nn.Tanh

        # Force a single-output head for simple regression
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Optional simple L2 loss for regression
        self.loss_function = nn.MSELoss()

        # Init weights lightly for stability on tiny dataflow
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        # Return shape (batch,) to match target vector shape for simple loss
        if out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out
