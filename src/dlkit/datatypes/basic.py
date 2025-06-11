from collections.abc import Sequence

from pydantic import BaseModel, Field, ConfigDict


class BasicTypeSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )


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
    log: bool | None = Field(default=False, description="Whether to use log scale.")


class CategoricalDistribution[T](BasicTypeSettings):
    """Categorical distribution for hyperparameters."""

    choices: Sequence[T] = Field(
        ..., description="List of choices for the categorical distribution."
    )


class BoolDistribution(BasicTypeSettings):
    """Boolean distribution for hyperparameters."""

    choices: tuple[bool, bool] = Field(..., description="Tuple of 2 choices to toggle between.")


type IntHyperparameter = int | IntDistribution | CategoricalDistribution[int]
type FloatHyperparameter = float | FloatDistribution | CategoricalDistribution[float]
type StrHyperparameter = str | CategoricalDistribution[str]
type BoolHyperparameter = bool | BoolDistribution

type Hyperparameter = (
    BoolHyperparameter | IntHyperparameter | FloatHyperparameter | StrHyperparameter
)
