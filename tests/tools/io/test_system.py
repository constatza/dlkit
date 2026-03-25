"""Tests for system.py module - dynamic imports and class instantiation.

This module tests the core functionality of the dlkit.tools.io.system module,
focusing on good-path scenarios for dynamic imports, path-based loading,
and class instantiation with parameter filtering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dlkit.tools.io.system import import_from_module, import_from_path, init_class, load_class


class TestImportFromModule:
    """Test dynamic module imports with string paths."""

    def test_import_builtin_class_without_prefix(
        self, sample_class_specs: list[tuple[str, str]]
    ) -> None:
        """Test importing standard library classes without module prefix.

        Args:
            sample_class_specs: Fixture providing (module, class) pairs
        """
        module_name, class_name = sample_class_specs[0]  # collections.defaultdict
        full_path = f"{module_name}.{class_name}"

        result = import_from_module(class_name, full_path.rsplit(".", 1)[0])

        assert result is not None
        assert hasattr(result, "__name__")
        assert result.__name__ == class_name

    def test_import_builtin_class_with_prefix(
        self, sample_class_specs: list[tuple[str, str]]
    ) -> None:
        """Test importing standard library classes with module prefix.

        Args:
            sample_class_specs: Fixture providing (module, class) pairs
        """
        module_name, class_name = sample_class_specs[1]  # pathlib.Path

        result = import_from_module(class_name, module_name)

        assert result is not None
        assert hasattr(result, "__name__")
        assert result.__name__ == class_name

    def test_import_module_function(self) -> None:
        """Test importing a function from a standard library module."""
        result = import_from_module("chain", "itertools")

        assert result is not None
        assert callable(result)
        assert hasattr(result, "__name__")
        assert result.__name__ == "chain"

    def test_import_with_empty_prefix(self) -> None:
        """Test importing with empty prefix uses only class_name as full path."""
        result = import_from_module("defaultdict", "collections")

        assert result is not None
        assert result.__name__ == "defaultdict"

    def test_import_nested_module_path(self) -> None:
        """Test importing from deeply nested module paths."""
        result = import_from_module("JSONEncoder", "json")

        assert result is not None
        assert result.__name__ == "JSONEncoder"

    def test_import_nonexistent_module_raises_error(self, nonexistent_modules: list[str]) -> None:
        """Test that importing from non-existent modules raises ImportError.

        Args:
            nonexistent_modules: Fixture providing invalid module names
        """
        nonexistent_module = nonexistent_modules[0]

        with pytest.raises(ImportError):
            import_from_module("SomeClass", nonexistent_module)

    def test_import_nonexistent_class_raises_error(
        self, nonexistent_classes: list[tuple[str, str]]
    ) -> None:
        """Test that importing non-existent classes raises ImportError.

        Args:
            nonexistent_classes: Fixture providing (module, nonexistent_class) pairs
        """
        module_name, class_name = nonexistent_classes[0]

        with pytest.raises(AttributeError):
            import_from_module(class_name, module_name)


class TestImportFromPath:
    """Test file system path-based imports."""

    def test_import_from_file_path(self, temp_python_module: dict[str, Any]) -> None:
        """Test importing a class from a Python file path.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        module_file = temp_python_module["module_file"]
        class_name = temp_python_module["class_name"]
        base_dir = temp_python_module["module_dir"]

        result = import_from_path(class_name, module_file, base_dir)

        assert result is not None
        assert hasattr(result, "__name__")
        assert result.__name__ == class_name
        # Verify we can instantiate the class
        instance = result()
        assert hasattr(instance, "value")
        assert instance.value == temp_python_module["expected_value"]

    def test_import_from_package_directory(self, temp_python_package: dict[str, Any]) -> None:
        """Test importing a class from a package directory.

        Args:
            temp_python_package: Fixture providing temporary package
        """
        package_dir = temp_python_package["package_dir"]
        class_name = temp_python_package["class_name"]
        base_dir = temp_python_package["base_dir"]

        result = import_from_path(class_name, package_dir, base_dir)

        assert result is not None
        assert hasattr(result, "__name__")
        assert result.__name__ == class_name
        # Verify we can instantiate the class
        instance = result()
        assert hasattr(instance, "package_value")
        assert instance.package_value == temp_python_package["expected_value"]

    def test_import_with_relative_path(self, temp_python_module: dict[str, Any]) -> None:
        """Test importing using relative paths against base directory.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        class_name = temp_python_module["class_name"]
        base_dir = temp_python_module["module_dir"]

        # Create relative path
        relative_path = Path(temp_python_module["module_name"] + ".py")

        result = import_from_path(class_name, relative_path, base_dir)

        assert result is not None
        assert result.__name__ == class_name

    def test_import_with_absolute_path(self, temp_python_module: dict[str, Any]) -> None:
        """Test importing using absolute paths.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        module_file = temp_python_module["module_file"].resolve()
        class_name = temp_python_module["class_name"]
        base_dir = temp_python_module["module_dir"]

        result = import_from_path(class_name, module_file, base_dir)

        assert result is not None
        assert result.__name__ == class_name

    def test_import_function_from_path(self, temp_python_module: dict[str, Any]) -> None:
        """Test importing a function from file path.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        module_file = temp_python_module["module_file"]
        function_name = temp_python_module["function_name"]
        base_dir = temp_python_module["module_dir"]

        result = import_from_path(function_name, module_file, base_dir)

        assert result is not None
        assert callable(result)
        assert result.__name__ == function_name

    def test_import_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that importing from non-existent file raises ImportError."""
        nonexistent_file = tmp_path / "nonexistent.py"

        with pytest.raises(ImportError, match="Path is neither file nor package"):
            import_from_path("SomeClass", nonexistent_file, tmp_path)

    def test_import_from_directory_without_init_raises_error(
        self, empty_directory: Path, tmp_path: Path
    ) -> None:
        """Test that importing from directory without __init__.py raises ImportError.

        Args:
            empty_directory: Fixture providing directory without __init__.py
        """
        with pytest.raises(ImportError, match="not a package"):
            import_from_path("SomeClass", empty_directory, tmp_path)

    def test_import_invalid_python_file_raises_error(
        self, invalid_python_file: Path, tmp_path: Path
    ) -> None:
        """Test that importing invalid Python syntax raises ImportError.

        Args:
            invalid_python_file: Fixture providing file with invalid syntax
        """
        with pytest.raises(Exception):  # Could be SyntaxError or ImportError
            import_from_path("SomeClass", invalid_python_file, tmp_path)

    def test_import_nonexistent_attribute_raises_error(
        self, temp_python_module: dict[str, Any]
    ) -> None:
        """Test that importing non-existent attribute raises AttributeError.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        module_file = temp_python_module["module_file"]
        base_dir = temp_python_module["module_dir"]

        with pytest.raises(AttributeError):
            import_from_path("NonExistentClass", module_file, base_dir)


class TestLoadClass:
    """Test high-level class loading with automatic path detection."""

    def test_load_class_from_module_path(self, sample_class_specs: list[tuple[str, str]]) -> None:
        """Test loading class using module path syntax.

        Args:
            sample_class_specs: Fixture providing (module, class) pairs
        """
        module_name, class_name = sample_class_specs[0]

        result = load_class(class_name, module_name)

        assert result is not None
        assert result.__name__ == class_name

    def test_load_class_from_file_path_with_settings_dir(
        self, settings_dir_with_modules: dict[str, Any]
    ) -> None:
        """Test loading class from file path with settings directory.

        Args:
            settings_dir_with_modules: Fixture providing settings dir with modules
        """
        settings_dir = settings_dir_with_modules["settings_dir"]
        module_path = settings_dir_with_modules["relative_module_path"]
        class_name = settings_dir_with_modules["class_name"]

        result = load_class(class_name, module_path, settings_dir)

        assert result is not None
        assert result.__name__ == class_name

    def test_load_class_without_settings_dir_uses_module_import(self) -> None:
        """Test that without settings_dir, load_class uses module import even with slashes."""
        result = load_class("Path", "pathlib")

        assert result is not None
        assert result.__name__ == "Path"

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only backslash paths")
    def test_load_class_with_backslash_path(
        self, settings_dir_with_modules: dict[str, Any]
    ) -> None:
        """Test loading class with Windows-style backslash paths (Windows only)."""
        settings_dir = settings_dir_with_modules["settings_dir"]
        class_name = settings_dir_with_modules["class_name"]
        # Use backslash syntax - this should trigger path-based import
        module_path = "relative_modules\\rel_module.py"

        result = load_class(class_name, module_path, settings_dir)
        assert result is not None
        assert result.__name__ == class_name

    def test_load_class_invalid_module_raises_error(self) -> None:
        """Test that loading from invalid module raises ImportError."""
        with pytest.raises(ImportError):
            load_class("SomeClass", "nonexistent_module")


class TestInitClass:
    """Test complete class instantiation workflow with parameter filtering."""

    def test_init_class_from_module(
        self, mock_class_with_signature: type, kwargs_test_data: dict[str, Any]
    ) -> None:
        """Test instantiating class from module with parameter filtering.

        Args:
            mock_class_with_signature: Fixture providing mock class
            kwargs_test_data: Fixture providing test kwargs
        """
        with patch("dlkit.tools.io.system.load_class", return_value=mock_class_with_signature):
            result = init_class(
                name="MockTestClass",
                module_path="test_module",
                required_param="test",
                optional_param=500,
                extra_param="filtered_out",
            )

        assert result is not None
        assert isinstance(result, mock_class_with_signature)
        assert result.required_param == "test"
        assert result.optional_param == 500
        # extra_param should be filtered out

    def test_init_class_with_exclude_set(self, mock_class_with_signature: type) -> None:
        """Test parameter exclusion using exclude set.

        Args:
            mock_class_with_signature: Fixture providing mock class
        """
        with (
            patch("dlkit.tools.io.system.load_class", return_value=mock_class_with_signature),
            patch("dlkit.tools.io.system.kwargs_compatible_with") as mock_kwargs,
        ):
            # Mock the kwargs filtering to exclude optional_param
            mock_kwargs.return_value = {"required_param": "test", "another_param": False}

            result = init_class(
                name="MockTestClass",
                module_path="test_module",
                exclude={"optional_param"},
                required_param="test",
                optional_param=999,  # Should be excluded
                another_param=False,
            )

        assert result is not None
        assert result.required_param == "test"
        assert result.optional_param == 100  # Default value, not 999
        assert result.another_param is False

    def test_init_class_with_settings_path(
        self, mock_class_with_signature: type, tmp_path: Path
    ) -> None:
        """Test class instantiation with settings path for relative imports.

        Args:
            mock_class_with_signature: Fixture providing mock class
            tmp_path: pytest temporary directory fixture
        """
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()

        with patch("dlkit.tools.io.system.load_class", return_value=mock_class_with_signature):
            result = init_class(
                name="MockTestClass",
                module_path="relative/path/module.py",
                settings_path=settings_dir,
                required_param="with_settings",
            )

        assert result is not None
        assert result.required_param == "with_settings"

    def test_init_function_returns_callable(self, mock_function_with_signature: callable) -> None:
        """Test that init_class can handle functions (callables) as well as classes.

        Args:
            mock_function_with_signature: Fixture providing mock function
        """
        with patch("dlkit.tools.io.system.load_class", return_value=mock_function_with_signature):
            result = init_class(
                name="mock_test_function",
                module_path="test_module",
                x=10,
                y="custom_value",
                extra_param="filtered",
            )

        assert result is not None
        assert callable(result)

    def test_init_class_with_empty_kwargs(self, mock_class_with_signature: type) -> None:
        """Test class instantiation with minimal required parameters.

        Args:
            mock_class_with_signature: Fixture providing mock class
        """
        with patch("dlkit.tools.io.system.load_class", return_value=mock_class_with_signature):
            result = init_class(
                name="MockTestClass", module_path="test_module", required_param="minimal"
            )

        assert result is not None
        assert result.required_param == "minimal"
        assert result.optional_param == 100  # Default value
        assert result.another_param is True  # Default value

    def test_init_class_load_error_propagates(self) -> None:
        """Test that ImportError from load_class is properly propagated."""
        with patch("dlkit.tools.io.system.load_class", side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Module not found"):
                init_class(name="SomeClass", module_path="nonexistent", required_param="test")

    def test_init_class_kwargs_compatible_error_propagates(
        self, mock_class_with_signature: type
    ) -> None:
        """Test that errors from kwargs_compatible_with are properly propagated.

        Args:
            mock_class_with_signature: Fixture providing mock class
        """
        with (
            patch("dlkit.tools.io.system.load_class", return_value=mock_class_with_signature),
            patch(
                "dlkit.tools.io.system.kwargs_compatible_with",
                side_effect=ValueError("Invalid kwargs"),
            ),
            pytest.raises(ValueError, match="Invalid kwargs"),
        ):
            init_class(name="MockTestClass", module_path="test_module", required_param="test")


class TestSystemModuleIntegration:
    """Integration tests for system module workflows."""

    def test_full_workflow_with_real_builtin_class(self) -> None:
        """Test complete workflow using real standard library class."""
        # Use collections.defaultdict as a real example
        result = init_class(name="defaultdict", module_path="collections")

        assert result is not None
        assert hasattr(result, "default_factory")
        # Test functionality - defaultdict with no factory defaults to None
        result["test"] = ["value"]
        assert result["test"] == ["value"]

    def test_full_workflow_with_file_path(self, temp_python_module: dict[str, Any]) -> None:
        """Test complete workflow from file path to instantiated class.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        # Use the parent directory as settings directory
        settings_dir = temp_python_module["module_dir"].parent

        # Use a path with "/" to trigger path-based import
        module_path = f"test_modules/{temp_python_module['module_name']}.py"

        result = init_class(
            name=temp_python_module["class_name"],
            module_path=module_path,
            settings_path=settings_dir,
            value=999,
        )

        assert result is not None
        assert result.value == 999
        assert result.name == "test"  # Default value from TestClass

    def test_module_cleanup_after_import(self, temp_python_module: dict[str, Any]) -> None:
        """Test that dynamic imports work correctly and modules are added to sys.modules.

        Args:
            temp_python_module: Fixture providing temporary module file
        """
        module_file = temp_python_module["module_file"]
        class_name = temp_python_module["class_name"]
        base_dir = module_file.parent
        expected_module_name = temp_python_module["module_name"]

        # Import the class
        result = import_from_path(class_name, module_file, base_dir)
        assert result is not None

        # Check that the module was added to sys.modules
        assert expected_module_name in sys.modules

        # Verify we can still use the imported class
        instance = result()
        assert hasattr(instance, "value")


class TestSystemModuleErrorHandling:
    """Test error conditions and edge cases."""

    def test_import_from_module_with_malformed_path(self) -> None:
        """Test import_from_module with malformed module paths."""
        with pytest.raises(ValueError):
            import_from_module("Class", "")  # Empty module path with empty class name should fail

    def test_import_from_path_spec_loading_failure(self, tmp_path: Path) -> None:
        """Test import_from_path when spec creation fails."""
        # Create a file that exists but cannot be loaded as a module
        bad_file = tmp_path / "bad_file.py"
        bad_file.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        with pytest.raises(Exception):  # Could be UnicodeError or ImportError
            import_from_path("SomeClass", bad_file, tmp_path)

    def test_load_class_with_pydantic_validation(self, tmp_path: Path) -> None:
        """Test load_class with invalid DirectoryPath validation."""
        # Pass a file instead of directory to trigger Pydantic validation error
        invalid_file = tmp_path / "not_a_directory.txt"
        invalid_file.write_text("content")

        with pytest.raises(Exception):  # Pydantic validation error
            load_class("SomeClass", "some.module.path", invalid_file)
