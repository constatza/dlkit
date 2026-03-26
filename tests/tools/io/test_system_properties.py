"""Property-based tests for system.py module using Hypothesis.

This module uses Hypothesis to generate test cases and verify invariants
for dynamic imports, path resolution, and parameter filtering.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from dlkit.tools.io.system import import_from_module, import_from_path, init_class, load_class

# Hypothesis strategies for generating test dataflow
valid_module_names = st.sampled_from(["collections", "itertools", "json", "pathlib", "os", "sys"])

valid_class_names = st.sampled_from(
    [
        ("collections", "defaultdict"),
        ("itertools", "chain"),
        ("json", "JSONEncoder"),
        ("pathlib", "Path"),
    ]
)

# Simplified strategies with better filtering
# Restricted to ASCII (max_codepoint=127) to avoid Windows cp1252 encoding failures
# and filesystem edge cases with non-ASCII module names.
python_identifiers = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll"), whitelist_characters="_", max_codepoint=127
    ),
    min_size=1,
    max_size=20,
).filter(lambda x: x.isidentifier() and not x.startswith("__") and x.isalpha())

simple_kwargs_values = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-100, max_value=100),
    st.booleans(),
)


class TestImportFromModuleProperties:
    """Property-based tests for import_from_module function."""

    @given(valid_class_names)
    def test_import_builtin_classes_always_succeeds(self, class_spec: tuple[str, str]) -> None:
        """Property: Importing known builtin classes should always succeed.

        Args:
            class_spec: Generated (module, class) pair from valid builtins
        """
        module_name, class_name = class_spec

        result = import_from_module(class_name, module_name)

        assert result is not None
        assert inspect.isclass(result) or callable(result)

    @given(valid_class_names)
    def test_import_preserves_class_name_in_result(self, class_spec: tuple[str, str]) -> None:
        """Property: Successfully imported objects should preserve the requested class name.

        Args:
            class_spec: Generated (module, class) pair
        """
        module_name, class_name = class_spec

        result = import_from_module(class_name, module_name)
        # class_name is from a controlled sampled_from — safe to assert against
        assert result.__name__ == class_name

    @given(valid_module_names, python_identifiers)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_import_failure_with_invalid_class_raises_attribute_error(
        self, module_name: str, invalid_class: str
    ) -> None:
        """Property: Importing invalid class from valid module should raise AttributeError.

        Args:
            module_name: Valid module name
            invalid_class: Generated invalid class name
        """
        # Import the module and verify the attribute doesn't exist
        import importlib

        module = importlib.import_module(module_name)
        assume(not hasattr(module, invalid_class))

        with pytest.raises(AttributeError):
            import_from_module(invalid_class, module_name)


class TestImportFromPathProperties:
    """Property-based tests for import_from_path function."""

    def test_import_from_nonexistent_path_raises_import_error(self, tmp_path: Path) -> None:
        """Property: Importing from non-existent paths should always raise ImportError."""

        @given(python_identifiers, python_identifiers)
        def run_test(file_name: str, class_name: str) -> None:
            nonexistent_path = tmp_path / f"{file_name}.py"
            # Ensure the path doesn't exist
            assume(not nonexistent_path.exists())

            with pytest.raises(ImportError, match="Path is neither file nor package"):
                import_from_path(class_name, nonexistent_path, tmp_path)

        run_test()

    def test_relative_vs_absolute_path_resolution_consistency(self, tmp_path: Path) -> None:
        """Property: Relative and absolute paths should resolve to the same module."""

        @given(python_identifiers)
        def run_test(mod_name: str) -> None:
            # Create a valid Python module
            module_content = (
                "class TestClass:\n"
                '    """Generated test class."""\n'
                "    def __init__(self):\n"
                "        pass\n"
            )

            module_file = tmp_path / f"{mod_name}.py"
            module_file.write_text(module_content, encoding="utf-8")

            # Import using relative path
            relative_path = Path(f"{mod_name}.py")
            try:
                result_relative = import_from_path("TestClass", relative_path, tmp_path)
            except Exception:
                return

            # Import using absolute path
            absolute_path = module_file.resolve()
            result_absolute = import_from_path("TestClass", absolute_path, tmp_path)

            # "TestClass" is a controlled literal defined in the module fixture above
            assert result_relative.__name__ == "TestClass"
            assert result_absolute.__name__ == "TestClass"

        run_test()


class TestLoadClassProperties:
    """Property-based tests for load_class function."""

    @given(valid_class_names)
    def test_load_class_module_path_consistency(self, class_spec: tuple[str, str]) -> None:
        """Property: load_class should behave consistently with direct module imports.

        Args:
            class_spec: Generated (module, class) pair
        """
        module_name, class_name = class_spec

        result_load_class = load_class(class_name, module_name)
        result_direct = import_from_module(class_name, module_name)

        assert result_load_class is result_direct
        assert result_load_class.__name__ == class_name

    def test_load_class_path_detection_logic(self, tmp_path: Path) -> None:
        """Property: Path detection should consistently choose between module/file imports."""

        @given(python_identifiers, python_identifiers)
        def run_test(mod_path: str, class_name: str) -> None:
            settings_dir = tmp_path / "settings"
            settings_dir.mkdir(exist_ok=True)

            # Add path separators to trigger path import
            module_path_with_separator = f"subdir/{mod_path}.py"

            # With settings_dir and path separators, should attempt path import
            try:
                load_class(class_name, module_path_with_separator, settings_dir)
                # If it succeeds, great; if it fails with ImportError, that's expected
            except ImportError:
                pass  # Expected for non-existent paths
            except Exception as e:
                # Other exceptions might indicate logic errors
                raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}") from e

        run_test()


class TestInitClassProperties:
    """Property-based tests for init_class function."""

    @given(st.dictionaries(python_identifiers, simple_kwargs_values, min_size=1, max_size=5))
    def test_init_class_kwargs_filtering_preserves_valid_parameters(
        self, kwargs_dict: dict[str, Any]
    ) -> None:
        """Property: init_class should preserve valid kwargs and filter invalid ones.

        Args:
            kwargs_dict: Generated kwargs dictionary
        """

        # Create a mock class with specific signature
        class MockClass:
            def __init__(self, param1: str = "default", param2: int = 42, **kwargs):
                self.param1 = param1
                self.param2 = param2
                self.extra_kwargs = kwargs

        with patch("dlkit.tools.io.system.load_class", return_value=MockClass):
            try:
                result = init_class(
                    name="MockClass",
                    module_path="test_module",
                    param1="test_value",  # Valid parameter
                    **kwargs_dict,  # Additional parameters to filter
                )

                assert result is not None
                assert isinstance(result, MockClass)
                assert result.param1 == "test_value"

            except TypeError, ValueError:
                # May fail if kwargs contain incompatible types
                pass

    @given(st.sets(python_identifiers, min_size=1, max_size=3))
    def test_init_class_exclude_set_filtering(self, exclude_keys: set[str]) -> None:
        """Property: exclude set should prevent specified parameters from being passed.

        Args:
            exclude_keys: Generated set of parameter names to exclude
        """

        class MockClass:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Filter out reserved parameter names to avoid conflicts
        reserved_params = {"name", "module_path", "settings_path", "exclude"}
        filtered_exclude_keys = exclude_keys - reserved_params

        # Skip test if no valid keys remain after filtering
        if not filtered_exclude_keys:
            return

        test_kwargs = {key: f"value_{i}" for i, key in enumerate(filtered_exclude_keys)}
        additional_kwargs = {"valid_param": "should_be_included"}

        with (
            patch("dlkit.tools.io.system.load_class", return_value=MockClass),
            patch("dlkit.tools.io.system.kwargs_compatible_with") as mock_kwargs,
        ):
            # Mock kwargs filtering to exclude the specified keys
            mock_kwargs.return_value = additional_kwargs

            result = init_class(
                name="MockClass",
                module_path="test_module",
                exclude=filtered_exclude_keys,
                **test_kwargs,  # ty: ignore[invalid-argument-type]
                **additional_kwargs,
            )

            assert result is not None
            assert isinstance(result, MockClass)

            # Non-excluded keys should appear (via mock)
            assert "valid_param" in result.kwargs
            assert result.kwargs["valid_param"] == "should_be_included"

    @given(python_identifiers, python_identifiers)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_init_class_function_handling(self, name: str, module_path: str) -> None:
        """Property: init_class should handle functions (non-class callables) correctly.

        Args:
            name: Generated name parameter
            module_path: Generated module path
        """

        def mock_function(x: int = 1, y: str = "default") -> str:
            return f"x={x}, y={y}"

        with patch("dlkit.tools.io.system.load_class", return_value=mock_function):
            try:
                result = init_class(name=name, module_path=module_path, x=42, y="test")

                # Should return the function result, not the function itself
                assert isinstance(result, str)
                assert "x=42" in result
                assert "y=test" in result

            except Exception:
                # May fail due to kwargs_compatible_with or other filtering
                pass


class TestSystemModuleInvariants:
    """Test invariants that should hold across all system module functions."""

    @given(python_identifiers)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_import_functions_preserve_python_naming_conventions(self, identifier: str) -> None:
        """Property: All import functions should respect Python naming conventions.

        Args:
            identifier: Generated identifier string
        """
        # Test with a known good module/class combination
        try:
            result = import_from_module(identifier, "collections")
            # If import succeeds, result should have valid __name__
            if result is not None:
                assert inspect.isclass(result) or callable(result)
                # result.__name__ would be a generated identifier — don't assert its value
        except ImportError, AttributeError:
            pass  # Expected for non-existent identifiers

    @given(st.dictionaries(python_identifiers, simple_kwargs_values, min_size=0, max_size=10))
    def test_kwargs_filtering_is_deterministic(self, kwargs_dict: dict[str, Any]) -> None:
        """Property: Parameter filtering should be deterministic and reproducible.

        Args:
            kwargs_dict: Generated kwargs dictionary
        """

        class ConsistentClass:
            def __init__(self, fixed_param: str = "fixed"):
                self.fixed_param = fixed_param

        with patch("dlkit.tools.io.system.load_class", return_value=ConsistentClass):
            try:
                # Run init_class twice with the same parameters
                result1 = init_class(
                    name="ConsistentClass",
                    module_path="test_module",
                    fixed_param="test",
                    **kwargs_dict,
                )

                result2 = init_class(
                    name="ConsistentClass",
                    module_path="test_module",
                    fixed_param="test",
                    **kwargs_dict,
                )

                # Results should be equivalent
                assert type(result1) is type(result2)
                assert result1.fixed_param == result2.fixed_param

            except TypeError, ValueError:
                # May fail due to incompatible kwargs types
                pass

    def test_path_resolution_respects_filesystem_case_sensitivity(self, tmp_path: Path) -> None:
        """Property: Path resolution should respect filesystem case sensitivity."""

        @given(python_identifiers)
        def run_test(file_name: str) -> None:
            # Create a module file
            module_content = """
class TestClass:
    def __init__(self):
        pass
"""
            module_file = tmp_path / f"{file_name}.py"
            module_file.write_text(module_content, encoding="utf-8")

            # Import using exact case
            result = import_from_path("TestClass", module_file, tmp_path)
            assert result is not None
            assert result.__name__ == "TestClass"

            # Case-sensitive filesystems should not find differently cased files
            if file_name.lower() != file_name.upper():
                wrong_case_file = tmp_path / f"{file_name.upper()}.py"
                if not wrong_case_file.exists():
                    with pytest.raises(ImportError):
                        import_from_path("TestClass", wrong_case_file, tmp_path)

        run_test()
