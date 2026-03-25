"""Core settings test fixtures."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import strategies as st

from dlkit.tools.config.core.base_settings import (
    BasicSettings,
    ComponentSettings,
    HyperParameterSettings,
)


class MockBasicSettings(BasicSettings):
    """Mock implementation of BasicSettings for testing."""

    name: str
    value: int
    enabled: bool = True


class MockComponentSettings(ComponentSettings):
    """Mock implementation of ComponentSettings for testing."""

    param1: str = "default"
    param2: int = 42


class MockHyperParameterSettings(HyperParameterSettings):
    """Mock implementation of HyperParameterSettings for testing."""

    learning_rate: float | dict[str, Any] = 0.001
    batch_size: int | dict[str, Any] = 32
    optimizer_type: str | dict[str, Any] = "adam"


@pytest.fixture
def basic_settings_data() -> dict[str, Any]:
    """Sample dataflow for BasicSettings testing.

    Returns:
        Dict[str, Any]: Basic settings configuration
    """
    return {"name": "test_setting", "value": 100, "enabled": True}


@pytest.fixture
def component_settings_data() -> dict[str, Any]:
    """Sample dataflow for ComponentSettings testing.

    Returns:
        Dict[str, Any]: Component settings configuration
    """
    return {
        "name": "TestComponent",
        "module_path": "test.components",
        "param1": "custom_value",
        "param2": 999,
    }


@pytest.fixture
def hyperparameter_settings_data() -> dict[str, Any]:
    """Sample dataflow for HyperParameterSettings testing.

    Returns:
        Dict[str, Any]: Hyperparameter settings configuration
    """
    return {
        "learning_rate": {"low": 0.001, "high": 0.1, "log": True},
        "batch_size": {"choices": (16, 32, 64, 128)},
        "optimizer_type": "adam",
    }


@pytest.fixture
def build_context_data() -> dict[str, Any]:
    """Sample dataflow for BuildContext testing.

    Returns:
        Dict[str, Any]: Build context configuration
    """
    return {
        "mode": "testing",
        "device": "cpu",
        "random_seed": 123,
        "overrides": {"test_key": "test_value"},
    }


# Hypothesis strategies for property-based testing


@st.composite
def valid_name(draw):
    """Generate valid component names.

    Args:
        draw: Hypothesis draw function

    Returns:
        str: Valid component name
    """
    return draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=("_", ".")
            ),
        )
    )


@st.composite
def hyperparameter_int_spec(draw):
    """Generate integer hyperparameter specifications.

    Args:
        draw: Hypothesis draw function

    Returns:
        Dict[str, Any]: Integer hyperparameter spec
    """
    low = draw(st.integers(min_value=1, max_value=10))
    high = draw(st.integers(min_value=low + 1, max_value=100))
    step = draw(st.integers(min_value=1, max_value=min(5, high - low)))
    return {"low": low, "high": high, "step": step}


@st.composite
def hyperparameter_float_spec(draw):
    """Generate float hyperparameter specifications.

    Args:
        draw: Hypothesis draw function

    Returns:
        Dict[str, Any]: Float hyperparameter spec
    """
    low = draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False))
    high = draw(st.floats(min_value=low, max_value=1.0, allow_nan=False, allow_infinity=False))
    log = draw(st.booleans())
    return {"low": low, "high": high, "log": log}


@st.composite
def hyperparameter_categorical_spec(draw):
    """Generate categorical hyperparameter specifications.

    Args:
        draw: Hypothesis draw function

    Returns:
        Dict[str, Any]: Categorical hyperparameter spec
    """
    choices = draw(
        st.lists(st.one_of(st.integers(), st.text(min_size=1, max_size=20)), min_size=2, max_size=5)
    )
    return {"choices": tuple(choices)}
