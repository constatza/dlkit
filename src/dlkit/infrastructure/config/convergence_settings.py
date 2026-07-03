"""Convergence study configuration."""

from __future__ import annotations

from pydantic import Field, PositiveInt, model_validator

from .core.base_settings import BasicSettings


class ConvergenceSettings(BasicSettings):
    """Sample-size convergence study configuration.

    Specify either an explicit list of sample sizes (``sizes``) or a log-space
    formula (``min_samples`` + ``max_samples`` + ``steps``). The explicit list
    takes precedence when both are provided.

    Args:
        sizes: Explicit ordered list of training-set sizes to evaluate.
        min_samples: Minimum size for log-space generation.
        max_samples: Maximum size for log-space generation.
        steps: Number of log-space steps (geomspace). Default 8.
        repeats: Independent runs per size. Default 1 (DL models are expensive).
        target: Convergence threshold E_target. Omit for pure exploratory mode.
        target_metric: Metric key compared against ``target``.
        c: Multiplier for std in threshold: threshold = mu + c * sigma.
    """

    sizes: tuple[PositiveInt, ...] | None = Field(
        default=None, description="Explicit list of training-set sizes"
    )
    min_samples: PositiveInt | None = Field(
        default=None, description="Minimum size for log-space generation"
    )
    max_samples: PositiveInt | None = Field(
        default=None, description="Maximum size for log-space generation"
    )
    steps: PositiveInt = Field(default=8, description="Number of log-space steps")
    repeats: PositiveInt = Field(default=1, description="Independent runs per size")
    target: float | None = Field(default=None, description="Convergence threshold E_target")
    target_metric: str = Field(default="val/loss", description="Metric key for convergence check")
    c: float = Field(default=2.0, description="Multiplier: threshold = mu + c * sigma")

    @model_validator(mode="after")
    def _require_sizes(self) -> ConvergenceSettings:
        """Ensure at least one size specification is provided.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If neither ``sizes`` nor both ``min_samples`` and
                ``max_samples`` are provided.
        """
        if self.sizes is None and (self.min_samples is None or self.max_samples is None):
            raise ValueError("Provide either 'sizes' or both 'min_samples' and 'max_samples'.")
        return self

    def resolved_sizes(self) -> tuple[int, ...]:
        """Return the ordered list of training-set sizes to evaluate.

        Returns:
            Sorted, deduplicated sizes from the explicit list or log-space formula.
        """
        if self.sizes is not None:
            return self.sizes
        if self.min_samples is None or self.max_samples is None:
            raise ValueError("Provide either 'sizes' or both 'min_samples' and 'max_samples'.")
        import numpy as np

        return tuple(
            sorted({int(x) for x in np.geomspace(self.min_samples, self.max_samples, self.steps)})
        )
