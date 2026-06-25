"""Run settings — execution control for a single job invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator

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

    def get_precision_strategy(self) -> PrecisionStrategy:
        """Return the configured precision strategy (satisfies PrecisionProvider protocol).

        Returns:
            PrecisionStrategy resolved from the precision field.

        Raises:
            NotImplementedError: If precision is None (signals the service to use its default).
        """
        if self.precision is None:
            raise NotImplementedError("No precision configured — use service default")
        return self.precision

    @field_validator("precision", mode="before")
    @classmethod
    def _coerce_precision(cls, v: object) -> PrecisionStrategy | None:
        """Coerce string/int precision values to PrecisionStrategy at validation time.

        Args:
            v: Raw precision value from init, TOML, or model_copy(update=...).

        Returns:
            PrecisionStrategy enum member, or None if not provided.

        Raises:
            ValueError: If the value cannot be mapped to a known PrecisionStrategy.
        """
        if v is None:
            return None
        if isinstance(v, PrecisionStrategy):
            return v
        try:
            return PrecisionStrategy(str(v).lower())
        except ValueError:
            raise ValueError(
                f"Invalid precision value {v!r}. Valid values: "
                + ", ".join(s.value for s in PrecisionStrategy)
            ) from None
