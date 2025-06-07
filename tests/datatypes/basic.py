"""
test_distributions.py

Unit tests for IntDistribution, FloatDistribution, CategoricalDistribution,
BoolDistribution, and the Hyperparameter union, ensuring correct branch
selection and value checks based on input using pytest fixtures that
provide both input data and expected output types and values.
"""

import pytest
from pydantic import ValidationError, validate_call

from dlkit.datatypes.basic import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
    BoolDistribution,
    Hyperparameter,
)


@validate_call
def make_dist(dist: Hyperparameter) -> Hyperparameter:
    """Validate and return the distribution, falling back to FloatDistribution on floats or log."""
    return dist


def test_int_distribution():
    data = {"low": 0, "high": 10, "step": 2}
    obj = IntDistribution.model_validate(data)
    assert isinstance(obj, IntDistribution)
    assert obj.low == data["low"]
    assert obj.high == data["high"]
    assert obj.step == data["step"]


# ----- FloatDistribution tests -----


@pytest.mark.parametrize(
    "input_data, expected_step, expected_log",
    [
        ({"low": 1.5, "high": 10.0}, None, False),
        (
            {
                "low": -1.0,
                "high": 5.5,
            },
            None,
            False,
        ),
        ({"low": -1.0, "high": 5.5, "step": 0.5, "log": True}, 0.5, True),
    ],
)
def test_float_distribution(input_data, expected_step, expected_log):
    obj = FloatDistribution.model_validate(input_data)
    assert isinstance(obj, FloatDistribution)
    # reuse input values where they match
    assert obj.low == float(input_data["low"])
    assert obj.high == float(input_data["high"])
    # check defaults
    assert obj.step == expected_step
    assert obj.log == expected_log


# ----- Union fallback to FloatDistribution -----


@pytest.mark.parametrize(
    "input_data, expected_log",
    [
        ({"low": 1.0, "high": 10.0}, False),
    ],
)
def test_mixed_integer_with_float(input_data, expected_log):
    obj = make_dist(dist=input_data)
    assert isinstance(obj, FloatDistribution)
    assert obj.low == float(input_data["low"])
    assert obj.high == float(input_data["high"])
    assert obj.step is None
    assert obj.log == expected_log


# ----- CategoricalDistribution tests -----


@pytest.mark.parametrize(
    "choices",
    [
        [1, 2, 3],
        [0.1, 0.2, 0.3],
        ["a", "b", "c"],
    ],
)
def test_categorical_distribution(choices):
    obj = CategoricalDistribution.model_validate({"choices": choices})
    assert isinstance(obj, CategoricalDistribution)
    assert list(obj.choices) == choices


# ----- BoolDistribution tests -----


@pytest.mark.parametrize(
    "choices",
    [
        (True, False),
    ],
)
def test_bool_distribution(choices):
    obj = BoolDistribution.model_validate({"choices": choices})
    assert isinstance(obj, BoolDistribution)
    assert obj.choices == choices


# ----- Raw boolean through Hyperparameter -----


def test_raw_bool():
    result = make_dist(dist=True)
    assert result is True


# ----- Invalid input tests -----


@pytest.mark.parametrize(
    "dist",
    [
        {"low": 1.5, "high": 2},
        {"low": 1.0, "high": 2.1},
        {"low": 1, "high": 2, "step": 0.5},
    ],
)
def test_invalid_int_rejects(dist):
    with pytest.raises(ValidationError):
        IntDistribution.model_validate(dist)


def test_invalid_hyperparameter():
    with pytest.raises(ValidationError):
        make_dist(dist={"foo": "bar"})


# ----- Combined make_dist tests -----


@pytest.mark.parametrize(
    "input_data, expected_dist",
    [
        ({"low": 0, "high": 10, "step": 2}, IntDistribution),
        ({"low": 1.0, "high": 10}, FloatDistribution),
        (True, bool),
        ({"choices": [1, 2, 3]}, CategoricalDistribution),
        ({"choices": [True, False]}, BoolDistribution),
        ({"low": 0, "high": 10, "step": 2, "log": True}, FloatDistribution),
    ],
)
def test_make_dist_various(input_data, expected_dist):
    obj = make_dist(dist=input_data)
    assert isinstance(obj, expected_dist)
