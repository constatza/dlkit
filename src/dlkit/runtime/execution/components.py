"""Component bundles consumed by runtime execution services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.shared.shapes import ShapeSummary


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeComponents:
    """Concrete runtime components assembled for training or optimization."""

    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer | None
    shape_spec: ShapeSummary | None
    meta: dict[str, Any]
