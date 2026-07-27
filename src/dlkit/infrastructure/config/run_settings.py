"""Run settings — execution control for a single job invocation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.precision.strategy import PrecisionStrategy

RunType = Literal["train", "predict", "search", "convergence", "multirun", "fit"]


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
        plots: Optional path to a plots profile TOML file.
    """

    type: RunType | None = None
    seed: int | None = None
    precision: PrecisionStrategy | None = None
    # Typed profile references — validated in load_job(), not here
    model: Path | None = None
    data: Path | None = None
    training: Path | None = None
    tracking: Path | None = None
    plots: Path | None = None

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

    def resolve_seed(self, *, default: int = 42) -> int:
        """Resolve the effective global random seed for this run.

        Single source of truth for seed defaulting, replacing ad hoc
        per-call-site seed-resolution helpers.

        Args:
            default: Seed to use when ``self.seed`` is unset.

        Returns:
            int: ``self.seed`` if set, otherwise ``default``.
        """
        return self.seed if self.seed is not None else default

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


@contextmanager
def apply_run_context(run: RunSettings, *, workers: bool = True) -> Iterator[int]:
    """Seed global RNG state and apply a run's precision override for a block.

    Kept as a standalone function rather than a ``RunSettings`` method: the
    class stays pure data plus pure resolution methods (``resolve_seed``,
    ``get_precision_strategy``); the side-effecting global-state mutation
    (RNG seeding, thread-local precision override) lives in this explicit
    action function instead — separation of actions vs. pure functions.

    Not thread/async-safe: mutates process-global RNG state. Safe for
    today's single-threaded CLI/script entrypoints; revisit if a
    concurrent-serving entrypoint is ever built.

    Args:
        run: RunSettings whose seed and precision are applied.
        workers: Whether to also seed dataloader worker processes
            (forwarded to ``lightning.pytorch.seed_everything``).

    Yields:
        int: The resolved seed that was applied.
    """
    from dlkit.infrastructure.precision.context import precision_override
    from dlkit.infrastructure.seeding.service import apply_global_seed

    seed = run.resolve_seed()
    apply_global_seed(seed, workers=workers)
    try:
        precision_strategy = run.get_precision_strategy()
    except NotImplementedError:
        yield seed
        return
    with precision_override(precision_strategy):
        yield seed
