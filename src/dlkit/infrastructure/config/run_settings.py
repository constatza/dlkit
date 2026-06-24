"""Run settings — execution control for a single job invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.precision.strategy import PrecisionStrategy

RunType = Literal["train", "predict", "search"]


class RunSettings(BasicSettings):
    """Execution control: workflow type, seed, precision, and optional profile references.

    Profile references are validated at load time by load_job(). Each key must point
    to a TOML file whose top-level section matches the key name.

    Args:
        type: Workflow type (train, predict, search).
        seed: Global random seed.
        precision: Floating-point precision strategy.
        model: Optional path to a model profile TOML file.
        data: Optional path to a data profile TOML file.
        training: Optional path to a training profile TOML file.
        tracking: Optional path to a tracking profile TOML file.
    """

    type: RunType | None = None
    seed: int | None = None
    precision: PrecisionStrategy | None = None
    # Typed profile references — validated in load_job(), not here
    model: Path | None = None
    data: Path | None = None
    training: Path | None = None
    tracking: Path | None = None
