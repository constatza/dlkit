"""Architecture fitness functions for import rules and dependencies.

These tests enforce architectural boundaries and SOLID principles by validating
import patterns and dependencies across the codebase.
"""

import ast
import re
from pathlib import Path

import pytest


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import patterns in Python files."""

    def __init__(self):
        self.imports: list[tuple[str, str]] = []  # (module, name)
        self.from_imports: list[tuple[str, str]] = []  # (module, name)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(("", alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements."""
        if node.module:
            for alias in node.names:
                self.from_imports.append((node.module, alias.name))


def analyze_python_file(file_path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Analyze imports in a Python file.

    Args:
        file_path: Path to Python file to analyze

    Returns:
        Tuple of (imports, from_imports) lists
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return analyzer.imports, analyzer.from_imports
    except SyntaxError, UnicodeDecodeError:
        # Skip files that can't be parsed
        return [], []


def get_python_files(directory: Path, exclude_patterns: list[str] = None) -> list[Path]:
    """Get all Python files in a directory recursively.

    Args:
        directory: Root directory to search
        exclude_patterns: List of patterns to exclude

    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".venv", ".git", "node_modules"]

    python_files = []
    for file_path in directory.rglob("*.py"):
        # Skip excluded patterns
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue
        python_files.append(file_path)

    return python_files


class TestImportRules:
    """Architecture fitness functions for import rules."""

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture(scope="class")
    def test_files(self, project_root: Path) -> list[Path]:
        """Get all test files."""
        test_dir = project_root / "tests"
        return get_python_files(test_dir)

    @pytest.fixture(scope="class")
    def src_files(self, project_root: Path) -> list[Path]:
        """Get all source files."""
        src_dir = project_root / "src" / "dlkit"
        return get_python_files(src_dir)

    def test_no_direct_model_imports_in_tests(self, test_files: list[Path]) -> None:
        """Ensure tests don't directly import concrete model classes.

        This enforces the Dependency Inversion Principle by ensuring tests
        depend on abstractions (protocols/factories) rather than concrete classes.
        """
        violations = []
        prohibited_patterns = [
            # Specific patterns for the old BaseModel that was removed
            r"from dlkit\.core\.models\.nn\.base import BaseModel",
            r"import.*BaseModel.*from.*dlkit",
        ]

        # Allow certain legitimate imports
        allowed_exceptions = [
            "conftest.py",  # Test utilities and factories
            "test_import_rules.py",  # This file needs to check patterns
            "test_abc_shape_architecture.py",  # Tests the new architecture directly
        ]

        for file_path in test_files:
            # Skip allowed exceptions
            if any(exception in str(file_path) for exception in allowed_exceptions):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                for pattern in prohibited_patterns:
                    if re.search(pattern, content):
                        violations.append(str(file_path))
                        break
            except OSError, UnicodeDecodeError:
                continue

        assert not violations, (
            f"Tests should not directly import removed BaseModel class. "
            f"Use factory patterns or protocols instead. Violations: {violations}"
        )

    def test_integration_tests_use_high_level_api(self, project_root: Path) -> None:
        """Ensure integration tests use dlkit.train/infer/optimize APIs.

        This enforces proper architectural layering by ensuring integration
        tests exercise the complete command pattern flow.
        """
        integration_dir = project_root / "tests" / "integration"
        if not integration_dir.exists():
            pytest.skip("No integration tests directory found")

        integration_files = get_python_files(integration_dir)
        violations = []

        # Allow certain tests that test low-level components directly
        allowed_exceptions = [
            "test_custom_metrics_integration.py",  # Tests metrics/config integration, not full workflow
            "conftest.py",
            "__init__.py",
        ]

        for file_path in integration_files:
            # Skip allowed exceptions
            if any(exception in str(file_path) for exception in allowed_exceptions):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for high-level API usage
                high_level_apis = [
                    r"dlkit\.train\s*\(",
                    r"dlkit\.infer\s*\(",
                    r"dlkit\.optimize\s*\(",
                    r"dlkit\.load_model\s*\(",  # stateful inference API
                ]

                has_api_usage = any(re.search(pattern, content) for pattern in high_level_apis)
                has_test_functions = re.search(r"def test_", content)

                if has_test_functions and not has_api_usage:
                    violations.append(str(file_path))
            except OSError, UnicodeDecodeError:
                continue

        assert not violations, (
            f"Integration tests must use high-level APIs "
            f"(dlkit.train/infer/optimize/load_model). "
            f"Files without high-level API usage: {violations}"
        )

    def test_no_circular_dependencies(self, src_files: list[Path]) -> None:
        """Ensure no circular dependencies exist in source code.

        This enforces good architectural design by preventing circular
        dependencies that violate the Dependency Inversion Principle.
        """
        # Build dependency graph
        dependencies: dict[str, set[str]] = {}

        for file_path in src_files:
            # Convert file path to module name
            relative_path = file_path.relative_to(file_path.parents[3])  # Remove up to src/
            module_name = str(relative_path.with_suffix(""))
            module_name = module_name.replace("/", ".")

            dependencies[module_name] = set()

            imports, from_imports = analyze_python_file(file_path)

            # Analyze imports for dlkit modules
            for _, name in imports:
                if name.startswith("dlkit"):
                    dependencies[module_name].add(name)

            for module, _ in from_imports:
                if module and module.startswith("dlkit"):
                    dependencies[module_name].add(module)

        # Detect cycles using DFS
        def has_cycle(graph: dict[str, set[str]]) -> list[str]:
            WHITE, GRAY, BLACK = 0, 1, 2
            colors = dict.fromkeys(graph, WHITE)

            def dfs(node: str, path: list[str]) -> list[str]:
                if colors[node] == GRAY:
                    # Found cycle
                    cycle_start = path.index(node)
                    return path[cycle_start:] + [node]
                if colors[node] == BLACK:
                    return []

                colors[node] = GRAY
                for neighbor in graph.get(node, set()):
                    if neighbor in graph:  # Only check if neighbor exists in our graph
                        cycle = dfs(neighbor, path + [node])
                        if cycle:
                            return cycle
                colors[node] = BLACK
                return []

            for node in graph:
                if colors[node] == WHITE:
                    cycle = dfs(node, [])
                    if cycle:
                        return cycle
            return []

        # Known exceptions for function-level imports that break circular dependencies
        known_exceptions = {
            (
                "dlkit.tools.io.config",
                "dlkit.tools.config.general_settings",
                "dlkit.tools.io.config",
            )
        }

        cycle = has_cycle(dependencies)
        if cycle:
            cycle_tuple = tuple(cycle)
            if cycle_tuple not in known_exceptions:
                assert False, f"Circular dependency detected: {' -> '.join(cycle)}"
            # If it's a known exception, continue (the circular dependency is broken by function-level imports)

    def test_domain_layer_isolation(self, src_files: list[Path]) -> None:
        """Ensure domain layer doesn't depend on infrastructure concerns.

        This enforces the Dependency Inversion Principle by ensuring the domain
        layer (core models, services) doesn't import infrastructure components.
        """
        violations = []

        # Define domain and infrastructure patterns
        domain_patterns = [
            r"src/dlkit/core/",
            r"src/dlkit/interfaces/api/domain/",
        ]

        infrastructure_patterns = [
            r"import.*torch\.",
            r"import.*lightning",
            r"import.*mlflow",
            r"import.*optuna",
            r"from.*torch\.",
            r"from.*lightning",
            r"from.*mlflow",
            r"from.*optuna",
        ]

        for file_path in src_files:
            # Check if this is a domain file
            is_domain = any(re.search(pattern, str(file_path)) for pattern in domain_patterns)
            if not is_domain:
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for infrastructure imports
                for pattern in infrastructure_patterns:
                    if re.search(pattern, content):
                        violations.append((str(file_path), pattern))
                        break
            except OSError, UnicodeDecodeError:
                continue

        # Allow some exceptions for necessary integrations
        allowed_exceptions = [
            # torch is allowed in model implementations
            ("core/models", "torch"),
            # torch is allowed in data modules (tensor operations)
            ("core/datamodules", "torch"),
            # torch is allowed in datasets (tensor creation)
            ("core/datasets", "torch"),
            # torch is allowed in datatypes (tensor collation and Batch handling)
            ("core/datatypes", "torch"),
            # torch is allowed in transforms (tensor transformations)
            ("core/training/transforms", "torch"),
            # lightning is allowed in domain models (Lightning integration)
            ("interfaces/api/domain/models", "lightning"),
            # lightning is allowed in core models (Lightning base classes)
            ("core/models", "lightning"),
            # lightning is allowed in shape specs (checkpoint handling)
            ("core/shape_specs", "lightning"),
            # mlflow is allowed in callbacks (experiment tracking)
            ("core/training/callbacks", "mlflow"),
            # lightning is allowed in callbacks (callback base classes)
            ("core/training/callbacks", "lightning"),
            # precision service can import torch for dtype handling
            ("precision_service", "torch"),
        ]

        filtered_violations = []
        for file_path, pattern in violations:
            is_exception = any(
                exception_path in file_path and exception_import in pattern
                for exception_path, exception_import in allowed_exceptions
            )
            if not is_exception:
                filtered_violations.append((file_path, pattern))

        assert not filtered_violations, (
            f"Domain layer should not depend on infrastructure concerns. "
            f"Violations: {filtered_violations}"
        )
