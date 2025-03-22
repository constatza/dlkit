from pydantic import Field, BaseModel


class Type(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        validate_assignment = True


class IntRange(Type):
    low: int = Field(..., description="Minimum value")
    high: int = Field(..., description="Maximum value")
    step: int = Field(default=1, description="Step size (optional)")


class FloatRange(Type):
    low: float = Field(..., description="Minimum value")
    high: float = Field(..., description="Maximum value")
    step: float = Field(default=1.0, description="Step size (optional)")
    log: bool | None = Field(
        default=False, description="If true, sample on a log scale"
    )


IntHyper = int | IntRange | tuple[int, ...]
FloatHyper = float | FloatRange | tuple[float, ...]
StrHyper = str | tuple[str, ...]


class Shape(Type):

    features: tuple[int, ...] | None = Field(
        None, description="Input shape of the model."
    )
    targets: tuple[int, ...] | None = Field(
        None, description="Output shape of the model."
    )
