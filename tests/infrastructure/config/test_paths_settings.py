"""Tests for the PATHS settings contract."""

from __future__ import annotations

import tomllib

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.paths_settings import PathsSettings
from dlkit.infrastructure.io.config import serialize_config_to_string


class TestPathsSettings:
    """Verify declared and extra PATHS values share one normalization contract."""

    def test_declared_and_extra_paths_normalize_identically_on_validate(self) -> None:
        paths = PathsSettings.model_validate(
            {
                "output_dir": r"C:\tmp\out",
                "processed_dir": r"C:\tmp\proc",
            }
        )

        assert paths.output_dir == "C:/tmp/out"
        assert paths.get_path("processed_dir") == "C:/tmp/proc"

    def test_extra_non_path_values_fail_clearly(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"PATHS\.retry_count must be a path-like string or None",
        ):
            PathsSettings.model_validate({"retry_count": 3})

    def test_serialization_round_trip_preserves_normalized_extra_paths(self) -> None:
        paths = PathsSettings.model_validate(
            {
                "output_dir": r"C:\tmp\out",
                "processed_dir": r"C:\tmp\proc",
            }
        )

        toml_str = serialize_config_to_string({"PATHS": paths})

        assert 'output_dir = "C:/tmp/out"' in toml_str
        assert 'processed_dir = "C:/tmp/proc"' in toml_str

        round_tripped = PathsSettings.model_validate(tomllib.loads(toml_str)["PATHS"])

        assert round_tripped.output_dir == paths.output_dir
        assert round_tripped.get_path("processed_dir") == paths.get_path("processed_dir")
