"""Fixtures for testing I/O system utilities.

This module provides reusable fixtures for testing dynamic imports,
path resolution, and class instantiation across the test suite.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlkit.infrastructure.config.environment import env as global_environment
from dlkit.infrastructure.io.packs import write_array_pack


@pytest.fixture(autouse=True)
def _restore_global_environment_root() -> Any:
    """Save and restore global_environment.root_dir around each test.

    Tests in this package may temporarily set global_environment.root_dir = None
    to simulate a clean state. Without restoration the session-level root_dir
    set by conftest.pytest_configure leaks out and breaks later tests.

    Yields:
        None
    """
    saved = global_environment.root_dir
    yield
    global_environment.root_dir = saved


# ---------------------------------------------------------------------------
# Zarr dense pack fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_matrices_4x4() -> list[np.ndarray]:
    """Three 4×4 float32 dense matrices for zarr pack tests.

    Returns:
        List of three distinct 4×4 float32 numpy arrays.
    """
    rng = np.random.default_rng(0)
    return [rng.random((4, 4)).astype(np.float32) for _ in range(3)]


@pytest.fixture
def zarr_dense_pack(tmp_path: Path, dense_matrices_4x4: list[np.ndarray]) -> Path:
    """Save a zarr dense pack to ``tmp_path/zarr_pack`` and return the path.

    Args:
        tmp_path: pytest temporary directory.
        dense_matrices_4x4: Matrices to store.

    Returns:
        Path to the written zarr dense pack directory.
    """
    pack_path = tmp_path / "zarr_pack"
    with write_array_pack(pack_path, size=(4, 4)) as w:
        for matrix in dense_matrices_4x4:
            w.write_sample(matrix)
    return pack_path


# ---------------------------------------------------------------------------
# OOM / memory-bounded fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def matrix_stream() -> Generator[np.ndarray]:
    """Fixture returning a generator of 1_000 random (128, 128) float32 arrays.

    Each array is produced lazily — only one is alive at a time so the full
    65 MB dataset is never held in memory simultaneously.

    Returns:
        Generator yielding 1_000 float32 arrays of shape ``(128, 128)``.
    """

    def _gen() -> Generator[np.ndarray]:
        rng = np.random.default_rng(42)
        for _ in range(1_000):
            yield rng.random((128, 128)).astype(np.float32)

    return _gen()


@pytest.fixture(scope="session")
def pack_1k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped 1_000-sample 128×128 float32 zarr dense pack.

    Args:
        tmp_path_factory: pytest session-scoped temp path factory.

    Returns:
        Path to the written zarr dense pack directory.
    """
    path = tmp_path_factory.mktemp("packs") / "pack_1k"
    rng = np.random.default_rng(42)
    with write_array_pack(path, size=(128, 128)) as w:
        for _ in range(1_000):
            w.write_sample(rng.random((128, 128)).astype(np.float32))
    return path


# Test module constants
BUILTIN_MODULE_NAMES = [
    "collections",
    "os",
    "sys",
    "pathlib",
    "json",
    "itertools",
]

BUILTIN_CLASS_NAMES = [
    ("collections", "defaultdict"),
    ("pathlib", "Path"),
    ("json", "JSONEncoder"),
    ("itertools", "chain"),
]

NONEXISTENT_MODULE_NAMES = [
    "nonexistent_module",
    "fake.module.path",
    "missing_package.submodule",
]

NONEXISTENT_CLASS_NAMES = [
    ("collections", "NonExistentClass"),
    ("pathlib", "FakePathClass"),
    ("json", "MissingEncoder"),
]


@pytest.fixture
def sample_module_names() -> list[str]:
    """Provide sample module names for testing dynamic imports.

    Returns:
        List of valid Python module names that exist in the standard library.
    """
    return BUILTIN_MODULE_NAMES.copy()


@pytest.fixture
def sample_class_specs() -> list[tuple[str, str]]:
    """Provide sample (module, class) pairs for testing class imports.

    Returns:
        List of (module_name, class_name) tuples that exist in the standard library.
    """
    return BUILTIN_CLASS_NAMES.copy()


@pytest.fixture
def nonexistent_modules() -> list[str]:
    """Provide sample non-existent module names for error testing.

    Returns:
        List of module names that should not exist and cause ImportError.
    """
    return NONEXISTENT_MODULE_NAMES.copy()


@pytest.fixture
def nonexistent_classes() -> list[tuple[str, str]]:
    """Provide sample non-existent class names for error testing.

    Returns:
        List of (module_name, class_name) tuples where class doesn't exist.
    """
    return NONEXISTENT_CLASS_NAMES.copy()


@pytest.fixture
def temp_python_module(tmp_path: Path) -> dict[str, Any]:
    """Create a temporary Python module file for testing path-based imports.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary containing module path info and expected content.
    """
    module_dir = tmp_path / "test_modules"
    module_dir.mkdir()

    # Create a simple module with a test class
    module_content = '''
"""Test module for dynamic import testing."""

class TestClass:
    """A simple test class for import verification."""

    def __init__(self, value: int = 42, name: str = "test"):
        self.value = value
        self.name = name

    def get_info(self) -> str:
        return f"{self.name}: {self.value}"


def test_function(x: int = 1) -> int:
    """A simple test function."""
    return x * 2


TEST_CONSTANT = "test_value"
'''

    module_file = module_dir / "test_module.py"
    module_file.write_text(module_content)

    return {
        "module_dir": module_dir,
        "module_file": module_file,
        "module_name": "test_module",
        "class_name": "TestClass",
        "function_name": "test_function",
        "constant_name": "TEST_CONSTANT",
        "expected_value": 42,
        "expected_name": "test",
    }


@pytest.fixture
def temp_python_package(tmp_path: Path) -> dict[str, Any]:
    """Create a temporary Python package for testing package imports.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary containing package path info and expected content.
    """
    package_dir = tmp_path / "test_packages" / "sample_package"
    package_dir.mkdir(parents=True)

    # Create __init__.py
    init_content = '''
"""Sample package for testing."""

from .submodule import PackageClass

__all__ = ["PackageClass"]
'''
    (package_dir / "__init__.py").write_text(init_content)

    # Create a submodule
    submodule_content = '''
"""Submodule within the test package."""

class PackageClass:
    """A class defined in a package submodule."""

    def __init__(self, package_value: str = "package_test"):
        self.package_value = package_value

    def get_package_info(self) -> str:
        return f"Package: {self.package_value}"
'''
    (package_dir / "submodule.py").write_text(submodule_content)

    return {
        "package_dir": package_dir,
        "package_name": "sample_package",
        "class_name": "PackageClass",
        "expected_value": "package_test",
        "base_dir": tmp_path / "test_packages",
    }


@pytest.fixture
def invalid_python_file(tmp_path: Path) -> Path:
    """Create an invalid Python file for error testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to file with invalid Python syntax.
    """
    invalid_file = tmp_path / "invalid_module.py"
    invalid_file.write_text("This is not valid Python syntax !!@#$%")
    return invalid_file


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Create an empty directory without __init__.py for error testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to directory that's not a valid Python package.
    """
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def mock_class_with_signature() -> type:
    """Create a mock class with specific constructor signature for testing.

    Returns:
        Mock class with defined __init__ parameters.
    """

    class MockTestClass:
        """Mock class for testing parameter filtering."""

        def __init__(
            self, required_param: str, optional_param: int = 100, another_param: bool = True
        ):
            self.required_param = required_param
            self.optional_param = optional_param
            self.another_param = another_param

        def __repr__(self) -> str:
            return f"MockTestClass(required={self.required_param}, optional={self.optional_param}, another={self.another_param})"

    return MockTestClass


@pytest.fixture
def mock_function_with_signature() -> Callable[..., object]:
    """Create a mock function with specific signature for testing.

    Returns:
        Mock function with defined parameters.
    """

    def mock_test_function(x: int, y: str = "default", z: float = 3.14) -> str:
        """Mock function for testing parameter filtering."""
        return f"x={x}, y={y}, z={z}"

    return mock_test_function


@pytest.fixture
def kwargs_test_data() -> dict[str, Any]:
    """Provide test kwargs dataflow for parameter filtering tests.

    Returns:
        Dictionary with various parameter types for testing.
    """
    return {
        "required_param": "test_value",
        "optional_param": 200,
        "another_param": False,
        "extra_param": "should_be_filtered",
        "unknown_param": 999,
        "x": 42,
        "y": "custom",
        "z": 2.718,
        "invalid_extra": "not_used",
    }


@pytest.fixture
def settings_dir_with_modules(tmp_path: Path) -> dict[str, Any]:
    """Create a settings directory with Python modules for path resolution testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary with settings directory info and module paths.
    """
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()

    # Create relative module path
    rel_module_dir = settings_dir / "relative_modules"
    rel_module_dir.mkdir()

    rel_module_content = '''
class RelativeClass:
    """Class for testing relative path resolution."""

    def __init__(self, rel_value: str = "relative"):
        self.rel_value = rel_value
'''

    (rel_module_dir / "rel_module.py").write_text(rel_module_content)

    return {
        "settings_dir": settings_dir,
        "relative_module_path": "relative_modules/rel_module.py",
        "class_name": "RelativeClass",
        "expected_value": "relative",
    }


@pytest.fixture
def npz_single_array(tmp_path: Path) -> dict[str, Any]:
    """Create NPZ file with single array for auto-detection testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary with file path and expected array data.
    """
    data = np.ones((10, 5), dtype=np.float32)
    path = tmp_path / "single.npz"
    np.savez(path, data=data)

    return {"path": path, "array": data, "key": "data"}


@pytest.fixture
def npz_multi_array(tmp_path: Path) -> dict[str, Any]:
    """Create NPZ file with multiple arrays for key selection testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary with file path, array keys, and expected array data.
    """
    features = np.random.randn(10, 5).astype(np.float32)
    targets = np.random.randint(0, 2, (10, 1)).astype(np.int64)
    latent = np.zeros((10, 3), dtype=np.float32)

    path = tmp_path / "multi.npz"
    np.savez(path, features=features, targets=targets, latent=latent)

    return {
        "path": path,
        "features": features,
        "targets": targets,
        "latent": latent,
        "keys": ["features", "targets", "latent"],
    }


@pytest.fixture
def npz_empty(tmp_path: Path) -> Path:
    """Create empty NPZ file for edge case testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to empty NPZ file.
    """
    path = tmp_path / "empty.npz"
    np.savez(path)
    return path
