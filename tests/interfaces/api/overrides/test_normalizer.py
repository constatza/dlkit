"""Tests for OverrideNormalizer utility."""

from pathlib import Path

import pytest

from dlkit.interfaces.api.overrides.normalizer import OverrideNormalizer


class TestNormalizePath:
    """Test path normalization."""

    def test_normalize_path_from_string(self) -> None:
        """String paths should be converted to Path objects."""
        result = OverrideNormalizer.normalize_path("/tmp/test")
        assert isinstance(result, Path)
        assert result == Path("/tmp/test")

    def test_normalize_path_from_path(self) -> None:
        """Path objects should be passed through unchanged."""
        input_path = Path("/tmp/test")
        result = OverrideNormalizer.normalize_path(input_path)
        assert result == Path("/tmp/test")

    def test_normalize_path_from_none(self) -> None:
        """None values should be preserved."""
        result = OverrideNormalizer.normalize_path(None)
        assert result is None

    def test_normalize_path_relative(self) -> None:
        """Relative paths should be preserved as relative."""
        result = OverrideNormalizer.normalize_path("relative/path")
        assert isinstance(result, Path)
        assert result == Path("relative/path")
        assert not result.is_absolute()

    def test_normalize_path_expands_tilde(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Tilde paths should expand to HOME."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

        result = OverrideNormalizer.normalize_path("~/data")
        assert result == fake_home / "data"

    def test_normalize_root_dir_requires_absolute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Root dir normalization enforces absolute paths."""
        fake_home = Path("/home/tester")
        monkeypatch.setenv("HOME", str(fake_home))

        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

        result = OverrideNormalizer.normalize_path("home/tester/project", require_absolute=True)
        assert result == fake_home / "project"

    def test_normalize_root_dir_invalid_relative(self) -> None:
        """Relative root_dir values that cannot be coerced are discarded."""
        result = OverrideNormalizer.normalize_path("relative/project", require_absolute=True)
        assert result is None


class TestBuildOverridesDict:
    """Test override dictionary building."""

    def test_build_overrides_with_path_normalization(self) -> None:
        """Path fields should be automatically normalized to Path objects."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            root_dir="/tmp/root",
            output_dir="/tmp/output",
            data_dir="/tmp/data",
        )

        assert isinstance(result["checkpoint_path"], Path)
        assert isinstance(result["root_dir"], Path)
        assert isinstance(result["output_dir"], Path)
        assert isinstance(result["data_dir"], Path)

    def test_build_overrides_filters_none_values(self) -> None:
        """None values should be filtered from the result."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            root_dir=None,
            output_dir=None,
            batch_size=32,
            learning_rate=None,
        )

        assert "checkpoint_path" in result
        assert "batch_size" in result
        assert "root_dir" not in result
        assert "output_dir" not in result
        assert "learning_rate" not in result

    def test_build_overrides_preserves_non_path_values(self) -> None:
        """Non-path values should be passed through unchanged."""
        result = OverrideNormalizer.build_overrides_dict(
            batch_size=32,
            learning_rate=0.001,
            experiment_name="test_exp",
            mlflow=True,
        )

        assert result["batch_size"] == 32
        assert result["learning_rate"] == 0.001
        assert result["experiment_name"] == "test_exp"
        assert result["mlflow"] is True

    def test_build_overrides_handles_mixed_types(self) -> None:
        """Should handle mixed path and non-path overrides."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            batch_size=64,
            root_dir=None,
            experiment_name="mixed_test",
        )

        assert isinstance(result["checkpoint_path"], Path)
        assert result["batch_size"] == 64
        assert result["experiment_name"] == "mixed_test"
        assert "root_dir" not in result

    def test_build_overrides_root_dir_missing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Root dir values missing a leading slash should be coerced."""
        fake_home = Path("/home/tester")
        monkeypatch.setenv("HOME", str(fake_home))

        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

        result = OverrideNormalizer.build_overrides_dict(root_dir="home/tester/project")
        assert result["root_dir"] == fake_home / "project"

    def test_build_overrides_with_path_objects(self) -> None:
        """Path objects should be preserved, not converted."""
        input_path = Path("/tmp/model.ckpt")
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path=input_path,
            batch_size=32,
        )

        assert result["checkpoint_path"] == input_path

    def test_build_overrides_with_additional_overrides(self) -> None:
        """Additional overrides dict should be flattened into result."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            batch_size=32,
            additional_overrides={"custom_key": "custom_value", "another": 123},
        )

        assert isinstance(result["checkpoint_path"], Path)
        assert result["batch_size"] == 32
        assert result["custom_key"] == "custom_value"
        assert result["another"] == 123

    def test_build_overrides_additional_overrides_filters_none(self) -> None:
        """None values in additional_overrides should be filtered."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            additional_overrides={"keep_me": "value", "filter_me": None},
        )

        assert result["keep_me"] == "value"
        assert "filter_me" not in result

    def test_build_overrides_empty_dict(self) -> None:
        """Empty input should produce empty dict."""
        result = OverrideNormalizer.build_overrides_dict()
        assert result == {}

    def test_build_overrides_all_none_produces_empty(self) -> None:
        """All None values should produce empty dict."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path=None,
            root_dir=None,
            output_dir=None,
        )

        assert result == {}

    def test_build_overrides_additional_overrides_none_handled(self) -> None:
        """None additional_overrides should be handled gracefully."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            additional_overrides=None,
        )

        assert isinstance(result["checkpoint_path"], Path)
        assert "additional_overrides" not in result


class TestPathFields:
    """Test PATH_FIELDS constant."""

    def test_path_fields_contains_expected_keys(self) -> None:
        """PATH_FIELDS should contain all known path keys."""
        expected = {"checkpoint_path", "root_dir", "output_dir", "data_dir"}
        assert OverrideNormalizer.PATH_FIELDS == expected

    def test_path_fields_is_frozen(self) -> None:
        """PATH_FIELDS should be immutable."""
        with pytest.raises(AttributeError):
            OverrideNormalizer.PATH_FIELDS.add("new_field")  # type: ignore[attr-defined]


class TestIntegrationWithCommandInput:
    """Integration tests simulating command input usage."""

    def test_train_command_override_pattern(self) -> None:
        """Simulate train command override building."""
        # Simulate TrainCommandInput attributes
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            root_dir="/tmp/root",
            output_dir="/tmp/output",
            data_dir=None,
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            mlflow_host="localhost",
            mlflow_port=5000,
            experiment_name="test_exp",
            run_name="test_run",
            additional_overrides={"custom": "value"},
        )

        # All non-None values present
        assert isinstance(result["checkpoint_path"], Path)
        assert isinstance(result["root_dir"], Path)
        assert isinstance(result["output_dir"], Path)
        assert "data_dir" not in result
        assert result["epochs"] == 100
        assert result["batch_size"] == 32
        assert result["learning_rate"] == 0.001
        assert result["mlflow_host"] == "localhost"
        assert result["mlflow_port"] == 5000
        assert result["experiment_name"] == "test_exp"
        assert result["run_name"] == "test_run"
        assert result["custom"] == "value"

    def test_inference_command_override_pattern(self) -> None:
        """Simulate inference command override building."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path="/tmp/model.ckpt",
            root_dir="/tmp/root",
            output_dir=None,
            data_dir="/tmp/data",
            batch_size=16,
            additional_overrides=None,
        )

        assert isinstance(result["checkpoint_path"], Path)
        assert isinstance(result["root_dir"], Path)
        assert isinstance(result["data_dir"], Path)
        assert "output_dir" not in result
        assert result["batch_size"] == 16

    def test_optimization_command_override_pattern(self) -> None:
        """Simulate optimization command override building."""
        result = OverrideNormalizer.build_overrides_dict(
            checkpoint_path=None,
            root_dir="/tmp/root",
            output_dir="/tmp/output",
            data_dir="/tmp/data",
            trials=50,  # Conditional: only if != 100
            study_name="optuna_study",
            experiment_name="test_exp",
            run_name=None,
            additional_overrides={},
        )

        assert "checkpoint_path" not in result
        assert isinstance(result["root_dir"], Path)
        assert isinstance(result["output_dir"], Path)
        assert isinstance(result["data_dir"], Path)
        assert result["trials"] == 50
        assert result["study_name"] == "optuna_study"
        assert result["experiment_name"] == "test_exp"
        assert "run_name" not in result
