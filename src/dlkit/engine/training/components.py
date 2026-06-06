"""Component bundles consumed by runtime execution services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.engine.artifacts import RuntimeArtifactManifest


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeComponents:
    """Concrete runtime components assembled for training or optimization."""

    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer | None
    meta: dict[str, Any]
    artifacts: RuntimeArtifactManifest = field(default_factory=RuntimeArtifactManifest)
