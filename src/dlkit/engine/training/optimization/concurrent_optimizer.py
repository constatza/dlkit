"""ConcurrentOptimizer: a real torch.optim.Optimizer wrapping multiple sub-optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

import torch
import torch.optim


class ConcurrentOptimizer(torch.optim.Optimizer):
    """Runs multiple sub-optimizers simultaneously on disjoint parameter sets.

    Wraps N sub-optimizers as a single ``torch.optim.Optimizer``. Each call to
    ``step()`` and ``zero_grad()`` delegates to all sub-optimizers in order.
    ``param_groups`` exposes the combined groups from all sub-optimizers.

    Typical use case: Muon on matrix parameters + AdamW on the rest, partitioned
    at construction time by the builder.

    Attributes:
        _sub_optimizers: The wrapped sub-optimizers.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        """Initialise the concurrent optimizer.

        Args:
            optimizers: Sub-optimizers to wrap. Each must have its ``param_groups``
                already populated (i.e. constructed with their parameter subsets).

        Raises:
            ValueError: If ``optimizers`` is empty.
        """
        if not optimizers:
            raise ValueError("ConcurrentOptimizer requires at least one sub-optimizer.")
        self._sub_optimizers = optimizers
        all_params: list[torch.Tensor] = [
            p for opt in optimizers for pg in opt.param_groups for p in pg["params"]
        ]
        super().__init__(all_params, defaults={})
        # Replace the base-class single param_group with the structured per-sub-optimizer groups.
        self.param_groups = [pg for opt in optimizers for pg in opt.param_groups]

    @property
    def sub_optimizers(self) -> list[torch.optim.Optimizer]:
        """Return the wrapped sub-optimizers.

        Returns:
            List of sub-optimizers.
        """
        return self._sub_optimizers

    @overload
    def step(self, closure: None = ...) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Step all sub-optimizers.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss.
                Forwarded to every sub-optimizer if provided.

        Returns:
            Loss value from the last sub-optimizer that returns one, or None.
        """
        loss: float | None = None
        for opt in self._sub_optimizers:
            result = opt.step(closure)
            if result is not None:
                loss = result
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all sub-optimizers.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        for opt in self._sub_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing — a list of per-sub-optimizer state dicts.

        Returns:
            Dict with key ``"sub_optimizers"`` containing a list of state dicts.
        """
        return {"sub_optimizers": [opt.state_dict() for opt in self._sub_optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from a checkpoint produced by ``state_dict()``.

        Args:
            state_dict: State dict from a previous ``state_dict()`` call.
        """
        for opt, sd in zip(self._sub_optimizers, state_dict["sub_optimizers"], strict=True):
            opt.load_state_dict(sd)
