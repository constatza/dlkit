from collections.abc import Sequence

from pydantic import BaseModel, Field


class BasicTypeSettings(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        validate_assignment = True


class IntDistribution(BasicTypeSettings):
    """Range of integers."""

    low: int
    high: int
    step: int = Field(default=1, description="Step size for the range.")


class FloatDistribution(BasicTypeSettings):
    """Range of floats."""

    low: float
    high: float
    step: float | None = Field(default=None, description="Step size for the range.")
    log: bool = Field(default=False, description="Whether to use log scale.")


class CategoricalDistribution[T](BasicTypeSettings):
    """Categorical distribution for hyperparameters."""

    choices: Sequence[T] = Field(
        ..., description="List of choices for the categorical distribution."
    )


IntHyper = int | IntDistribution | CategoricalDistribution[int] | tuple[int, ...]
FloatHyper = float | FloatDistribution | CategoricalDistribution[float] | tuple[float, ...]
StrHyper = str | CategoricalDistribution | CategoricalDistribution[str] | tuple[str, ...]
