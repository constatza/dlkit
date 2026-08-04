"""ConfigValidationError is a single canonical class shared across every import path."""

from __future__ import annotations

import pytest

from dlkit.common.errors import ConfigValidationError


def test_validators_reexports_the_canonical_class() -> None:
    from dlkit.infrastructure.config.validators import (
        ConfigValidationError as ValidatorsConfigValidationError,
    )

    assert ValidatorsConfigValidationError is ConfigValidationError


def test_config_errors_reexports_the_canonical_class() -> None:
    from dlkit.infrastructure.io.config_errors import (
        ConfigValidationError as ConfigErrorsConfigValidationError,
    )

    assert ConfigErrorsConfigValidationError is ConfigValidationError


def test_io_namespace_reexports_the_canonical_class() -> None:
    from dlkit.io import ConfigValidationError as IoConfigValidationError

    assert IoConfigValidationError is ConfigValidationError


def test_default_model_class_and_section_data() -> None:
    error = ConfigValidationError("bad config")

    assert error.model_class == ""
    assert error.section_data == {}


def test_model_class_and_section_data_are_keyword_only() -> None:
    error = ConfigValidationError(
        "bad config", model_class="TrainingJobConfig", section_data={"key": "value"}
    )

    assert error.model_class == "TrainingJobConfig"
    assert error.section_data == {"key": "value"}
    with pytest.raises(TypeError):
        ConfigValidationError("bad config", "TrainingJobConfig")  # ty: ignore[too-many-positional-arguments]


def test_is_a_value_error() -> None:
    assert isinstance(ConfigValidationError("bad config"), ValueError)
