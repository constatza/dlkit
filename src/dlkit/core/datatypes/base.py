from typing import Annotated

from pydantic.functional_validators import AfterValidator


type IntHyperparameter = int | dict[str, int] | dict[str, tuple[int, ...]]
type FloatHyperparameter = float | dict[str, float | int] | dict[str, tuple[float, ...]]
type StrHyperparameter = str | dict[str, str] | dict[str, tuple[str, ...]]

type Hyperparameter = IntHyperparameter | FloatHyperparameter | StrHyperparameter


def _require_positive(v: FloatHyperparameter) -> FloatHyperparameter:
    """Reject non-positive plain-float values; dicts (search spaces) pass through.

    Args:
        v: Value to validate.

    Returns:
        FloatHyperparameter: The validated value, unchanged.

    Raises:
        ValueError: When ``v`` is a plain number ≤ 0.
    """
    if isinstance(v, (int, float)) and v <= 0:
        raise ValueError(f"Value must be positive, got {v}")
    return v


PositiveFloatHyperparameter = Annotated[FloatHyperparameter, AfterValidator(_require_positive)]
