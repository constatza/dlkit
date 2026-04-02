"""Shared model state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dlkit.shared.protocols import IDataModule, ITrainableModule


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelState:
    """Represents the state needed to execute or inspect a model."""

    model: ITrainableModule
    datamodule: IDataModule
    trainer: Any | None
    settings: Any
