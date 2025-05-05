from pydantic import BaseModel, Field


class BasicTypeSettings(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        validate_assignment = True


class IntRange(BasicTypeSettings):
    """Range of integers."""

    low: int
    high: int
    step: int = Field(default=1, description="Step size for the range.")


class FloatRange(BasicTypeSettings):
    """Range of floats."""

    low: float
    high: float
    step: float = Field(default=1.0, description="Step size for the range.")
    log: bool = Field(default=False, description="Whether to use log scale.")


IntHyper = int | IntRange | tuple[int, ...]
FloatHyper = float | FloatRange | tuple[float, ...]
StrHyper = str | tuple[str, ...]
