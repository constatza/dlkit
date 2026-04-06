"""Structural protocols replacing hard Lightning imports in shared layer."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ITrainableModule(Protocol):
    """Structural protocol for a trainable model (Lightning-compatible)."""

    def parameters(self): ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any: ...


@runtime_checkable
class IDataModule(Protocol):
    """Structural protocol for a data module."""

    def setup(self, stage: str | None = None) -> None: ...

    def train_dataloader(self) -> Any: ...

    def val_dataloader(self) -> Any: ...
