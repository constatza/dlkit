"""Comprehensive tests for tilde_expansion module.

This test suite provides full coverage of the tilde expansion functionality,
including good-path scenarios, edge cases, and property-based testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dlkit.core.datatypes.tilde_expansion import (
    _expand_tilde_in_path,
    _expand_tilde_in_string,
    _expand_tilde_in_url,
    create_tilde_expanding_validator,
    expand_tilde_in_value,
)

# Constants for test data
HOME_PATH = "/test/home/user"
SAMPLE_VALIDATOR_RETURN = "validated_result"


class TestExpandTildeInValue:
    """Test the main public API function expand_tilde_in_value."""

    def test_expand_string_with_tilde(
        self, sample_paths: dict[str, str], expected_path_expansions: dict[str, str]
    ) -> None:
        """Test that string values with tildes are expanded correctly.

        Args:
            sample_paths: Sample path strings fixture
            expected_path_expansions: Expected expansion results fixture
        """
        with patch.object(Path, "home", return_value=Path("/mock/home/user")):
            for input_path in sample_paths.values():
                if "~" in input_path and input_path in expected_path_expansions:
                    result = expand_tilde_in_value(input_path)
                    expected = expected_path_expansions[input_path]
                    assert result == expected, f"Failed to expand {input_path}"
                # Paths not in expected_path_expansions pass through unchanged

    def test_non_string_values_unchanged(self, non_string_inputs: list[Any]) -> None:
        """Test that non-string values pass through unchanged.

        Args:
            non_string_inputs: List of non-string test values
        """
        for input_value in non_string_inputs:
            result = expand_tilde_in_value(input_value)
            assert result is input_value, f"Non-string value {input_value} should remain unchanged"

    def test_string_without_tilde_unchanged(self) -> None:
        """Test that strings without tildes pass through unchanged."""
        test_strings = [
            "/home/user/file.txt",
            "relative/path/file.txt",
            "http://localhost:5000",
            "file:///absolute/path",
            "",
            " ",
        ]

        for test_string in test_strings:
            result = expand_tilde_in_value(test_string)
            assert result == test_string, (
                f"String without tilde should remain unchanged: {test_string}"
            )

    def test_url_expansion(
        self, sample_urls: dict[str, str], expected_url_expansions: dict[str, str]
    ) -> None:
        """Test URL expansion through the main API function.

        Args:
            sample_urls: Sample URL strings fixture
            expected_url_expansions: Expected URL expansion results fixture
        """
        with patch.object(Path, "home", return_value=Path("/mock/home/user")):
            for input_url in sample_urls.values():
                if "~" in input_url:
                    result = expand_tilde_in_value(input_url)
                    expected = expected_url_expansions.get(input_url, input_url)
                    assert result == expected, f"Failed to expand URL {input_url}"


class TestExpandTildeInString:
    """Test the internal _expand_tilde_in_string function."""

    def test_routes_to_url_expansion(self, mock_home_path: str) -> None:
        """Test that URLs are routed to URL-specific expansion logic."""
        test_url = "sqlite:///~/database.db"

        with patch.object(Path, "home", return_value=Path(mock_home_path)):
            result = _expand_tilde_in_string(test_url)
            expected = f"sqlite:///{mock_home_path}/database.db"
            assert result == expected

    def test_routes_to_path_expansion(self, mock_home_path: str) -> None:
        """Test that regular paths are routed to path-specific expansion logic."""
        test_path = "~/documents/file.txt"

        with patch.object(Path, "home", return_value=Path(mock_home_path)):
            result = _expand_tilde_in_string(test_path)
            expected = f"{mock_home_path}/documents/file.txt"
            assert result == expected

    def test_no_tilde_returns_unchanged(self) -> None:
        """Test that strings without tildes return unchanged."""
        test_string = "/home/user/file.txt"
        result = _expand_tilde_in_string(test_string)
        assert result == test_string


class TestExpandTildeInUrl:
    """Test URL-specific tilde expansion logic."""

    def test_non_url_returns_unchanged(self) -> None:
        """Test that non-URL strings return unchanged."""
        non_url = "/regular/path/~/file.txt"
        with pytest.raises(ValueError):
            _expand_tilde_in_url(non_url, HOME_PATH)

    def test_url_with_leading_slash_tilde(self) -> None:
        """Test URL with /~/ pattern (scheme:///~/path)."""
        url = "sqlite:///~/database.db"
        result = _expand_tilde_in_url(url, HOME_PATH)
        expected = f"sqlite:///{HOME_PATH}/database.db"
        assert result == expected

    def test_url_with_tilde_start(self) -> None:
        """Test URL with ~/  pattern (scheme://~/path)."""
        url = "file://~/documents/file.txt"
        result = _expand_tilde_in_url(url, HOME_PATH)
        expected = f"file:///{HOME_PATH.lstrip('/')}/documents/file.txt"
        assert result == expected

    def test_url_with_middle_tilde(self) -> None:
        """Test URL with tilde in middle of path (scheme://host/~/path)."""
        url = "http://localhost/~/api/data"
        result = _expand_tilde_in_url(url, HOME_PATH)
        expected = f"http://localhost/{HOME_PATH}/api/data"
        assert result == expected

    def test_url_no_tilde_unchanged(self) -> None:
        """Test URL without tilde remains unchanged."""
        url = "http://localhost:5000/api/data"
        result = _expand_tilde_in_url(url, HOME_PATH)
        assert result == url

    def test_url_with_complex_scheme(self) -> None:
        """Test URL with complex schemes and multiple path segments."""
        test_cases = [
            (
                "postgresql://user:pass@host:5432/~/db",
                f"postgresql://user:pass@host:5432/{HOME_PATH}/db",
            ),
            ("s3://bucket-name/~/data/file.json", f"s3://bucket-name/{HOME_PATH}/data/file.json"),
            (
                "ftp://ftp.example.com/~/uploads/file.zip",
                f"ftp://ftp.example.com/{HOME_PATH}/uploads/file.zip",
            ),
        ]

        for input_url, expected in test_cases:
            result = _expand_tilde_in_url(input_url, HOME_PATH)
            assert result == expected, f"Failed for URL: {input_url}"

    def test_url_edge_cases(self) -> None:
        """Test edge cases for URL expansion."""
        edge_cases = [
            ("scheme://", "scheme://"),  # Empty path
            ("file:///", "file:///"),  # Just root path
            ("http://~server/path", "http://~server/path"),  # Tilde in host (should not expand)
        ]

        for input_url, expected in edge_cases:
            result = _expand_tilde_in_url(input_url, HOME_PATH)
            assert result == expected, f"Failed for edge case: {input_url}"


class TestExpandTildeInPath:
    """Test file path-specific tilde expansion logic."""

    def test_tilde_start_pattern(self) -> None:
        """Test ~/path pattern (tilde at start)."""
        path = "~/documents/file.txt"
        result = _expand_tilde_in_path(path, HOME_PATH)
        expected = f"{HOME_PATH}/documents/file.txt"
        assert result == expected

    def test_root_tilde_pattern(self) -> None:
        """Test /~/path pattern - NOT expanded (invalid), passes through."""
        path = "/~/documents/file.txt"
        result = _expand_tilde_in_path(path, HOME_PATH)
        assert result == path  # Pass through unchanged, will fail naturally

    def test_middle_tilde_pattern(self) -> None:
        """Test prefix/~/path pattern - NOT expanded (invalid), passes through."""
        path = "data/~/backup/file.txt"
        result = _expand_tilde_in_path(path, HOME_PATH)
        assert result == path  # Pass through unchanged, will fail naturally

    def test_multiple_tilde_pattern_is_invalid(self) -> None:
        """Multiple tildes - NOT expanded (invalid), passes through."""
        path = "data/~/backup/~/file.txt"
        result = _expand_tilde_in_path(path, HOME_PATH)
        assert result == path  # Pass through unchanged, will fail naturally

    def test_just_tilde(self) -> None:
        """Test lone tilde expansion."""
        path = "~"
        result = _expand_tilde_in_path(path, HOME_PATH)
        assert result == HOME_PATH

    def test_just_root_tilde(self) -> None:
        """Test /~ pattern - NOT expanded (invalid), passes through."""
        path = "/~"
        result = _expand_tilde_in_path(path, HOME_PATH)
        assert result == path  # Pass through unchanged, will fail naturally

    def test_multiple_tildes_pass_through(self) -> None:
        """Multiple tildes - NOT expanded (invalid), passes through."""
        result1 = _expand_tilde_in_path("~/data/~/backup/~/file.txt", HOME_PATH)
        # Only first ~/ is expanded, rest passes through
        assert result1 == f"{HOME_PATH}/data/~/backup/~/file.txt"

        result2 = _expand_tilde_in_path("data/~/backup/~/file.txt", HOME_PATH)
        # No expansion, passes through
        assert result2 == "data/~/backup/~/file.txt"

    def test_no_expansion_cases(self) -> None:
        """Test cases where no expansion should occur."""
        no_expansion_cases = [
            "/home/user/file.txt",  # No tilde
            "relative/path/file.txt",  # No tilde
            "~file.txt",  # Tilde not followed by slash
            "",  # Empty string
        ]

        for path in no_expansion_cases:
            result = _expand_tilde_in_path(path, HOME_PATH)
            assert result == path, f"Path should remain unchanged: {path}"

    def test_tilde_in_filename_passes_through(self) -> None:
        """Tilde in middle of filename - passes through unchanged."""
        result = _expand_tilde_in_path("file~.txt", HOME_PATH)
        assert result == "file~.txt"  # Pass through, will fail naturally if path doesn't exist


class TestCreateTildeExpandingValidator:
    """Test the validator decorator factory."""

    def test_decorator_expands_tilde_then_validates(self, mock_validator_func: Mock) -> None:
        """Test that decorator expands tildes before calling base validator.

        Args:
            mock_validator_func: Mock validator function fixture
        """
        decorated_validator = create_tilde_expanding_validator(mock_validator_func)
        input_value = "~/test/path"

        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = decorated_validator(input_value)

            # Verify tilde was expanded before calling validator
            expected_expanded = f"{HOME_PATH}/test/path"
            mock_validator_func.assert_called_once_with(expected_expanded)
            assert result == mock_validator_func.return_value

    def test_decorator_passes_non_string_unchanged(self, mock_validator_func: Mock) -> None:
        """Test that decorator passes non-string values unchanged.

        Args:
            mock_validator_func: Mock validator function fixture
        """
        decorated_validator = create_tilde_expanding_validator(mock_validator_func)
        input_value = 42

        result = decorated_validator(input_value)

        mock_validator_func.assert_called_once_with(input_value)
        assert result == mock_validator_func.return_value

    def test_decorator_preserves_validator_signature(self) -> None:
        """Test that decorator preserves original validator's behavior."""

        def original_validator(value: str) -> str:
            return f"validated_{value}"

        decorated_validator = create_tilde_expanding_validator(original_validator)

        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = decorated_validator("~/test")
            expected_input = f"{HOME_PATH}/test"
            expected_output = f"validated_{expected_input}"
            assert result == expected_output

    def test_decorator_handles_validator_exceptions(self) -> None:
        """Test that decorator properly propagates validator exceptions."""

        def failing_validator(value: str) -> str:
            raise ValueError(f"Invalid value: {value}")

        decorated_validator = create_tilde_expanding_validator(failing_validator)

        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            with pytest.raises(ValueError, match="Invalid value:"):
                decorated_validator("~/test")


class TestTildeExpansionProperties:
    """Property-based tests using Hypothesis."""

    @given(st.text())
    def test_non_tilde_strings_unchanged(self, text: str) -> None:
        """Test that strings without tildes are always unchanged.

        Args:
            text: Random text string from Hypothesis
        """
        # Skip strings containing tilde to test only non-tilde cases
        if "~" not in text:
            result = expand_tilde_in_value(text)
            assert result == text

    @given(st.one_of(st.integers(), st.floats(), st.booleans(), st.none(), st.binary()))
    def test_non_string_types_unchanged(self, value: Any) -> None:
        """Test that non-string types always pass through unchanged.

        Args:
            value: Random non-string value from Hypothesis
        """
        result = expand_tilde_in_value(value)
        assert result is value

    @given(
        st.sampled_from([
            "~/test",
            "/~/test",
            "~",
            "/~",
            "sqlite:///~/db.sqlite",
            "file:///~/docs/file.txt",
        ])
    )
    def test_tilde_expansion_idempotent(self, text_with_tilde: str) -> None:
        """Test that tilde expansion is idempotent (expanding twice gives same result).

        Args:
            text_with_tilde: Known tilde patterns from Hypothesis
        """
        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            first_expansion = expand_tilde_in_value(text_with_tilde)
            second_expansion = expand_tilde_in_value(first_expansion)
            assert first_expansion == second_expansion

    @given(
        st.sampled_from(["~/", "~"]).flatmap(
            lambda prefix: st.text(alphabet=st.characters(blacklist_characters="~")).map(
                lambda suffix: prefix + suffix
            )
        )
    )
    def test_valid_tilde_patterns_expand(self, path_with_tilde: str) -> None:
        """Test that VALID tilde patterns (~ and ~/) get expanded.

        Invalid patterns like /~/ pass through unchanged.

        Args:
            path_with_tilde: Path with tilde pattern from Hypothesis
        """
        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = expand_tilde_in_value(path_with_tilde)
            # Only ~ and ~/ are expanded
            if path_with_tilde == "~":
                assert result == HOME_PATH
            elif path_with_tilde.startswith("~/"):
                assert result.startswith(HOME_PATH) and not result.startswith("~/")

    @given(
        st.sampled_from([
            "sqlite:///~/db.sqlite",
            "file:///~/docs/file.txt",
            "http://localhost/~/api",
            "https://example.com/~/data/file.json",
            "ftp://server/~/uploads/file.zip",
        ])
    )
    def test_url_scheme_preserved(self, url_with_tilde: str) -> None:
        """Test that URL schemes are always preserved during expansion.

        Args:
            url_with_tilde: URL containing tilde from Hypothesis
        """
        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = expand_tilde_in_value(url_with_tilde)

            # Extract schemes from original and result
            original_scheme = url_with_tilde.split("://", maxsplit=1)[0]
            result_scheme = result.split("://")[0]

            # Scheme should be preserved
            assert result_scheme == original_scheme
            # Tilde should be expanded
            assert "~" not in result or "~" not in result.split("://")[1]  # No tilde in path part


class TestIntegrationWithPydantic:
    """Test integration scenarios with Pydantic validation."""

    def test_simple_tilde_path_annotation(self, tmp_path: Path) -> None:
        """Test SimpleTildePath annotation expands tildes correctly.

        Args:
            tmp_path: pytest temporary directory fixture
        """
        # Import here to avoid circular dependencies during module import
        from pydantic import TypeAdapter

        from dlkit.core.datatypes import SimpleTildePath

        adapter = TypeAdapter(SimpleTildePath)

        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = adapter.validate_python("~/test/file.txt")
            expected = f"{HOME_PATH}/test/file.txt"
            assert result == expected

    def test_simple_mlflow_uri_annotation(self, tmp_path: Path) -> None:
        """Test SimpleMLflowURI annotation expands tildes correctly.

        Args:
            tmp_path: pytest temporary directory fixture
        """
        # Import here to avoid circular dependencies during module import
        from pydantic import TypeAdapter

        from dlkit.core.datatypes import SimpleMLflowURI

        adapter = TypeAdapter(SimpleMLflowURI)

        with patch.object(Path, "home", return_value=Path(HOME_PATH)):
            result = adapter.validate_python("sqlite:///~/mlflow.db")
            expected = f"sqlite:///{HOME_PATH}/mlflow.db"
            assert result == expected


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_home_path_resolution_error(self) -> None:
        """Test behavior when Path.home() raises an exception."""
        with patch.object(Path, "home", side_effect=RuntimeError("Cannot resolve home")):
            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Cannot resolve home"):
                expand_tilde_in_value("~/test/path")

    def test_validator_with_home_error(self, mock_validator_func: Mock) -> None:
        """Test decorator behavior when Path.home() fails.

        Args:
            mock_validator_func: Mock validator function fixture
        """
        decorated_validator = create_tilde_expanding_validator(mock_validator_func)

        with patch.object(Path, "home", side_effect=OSError("Home not accessible")):
            with pytest.raises(OSError, match="Home not accessible"):
                decorated_validator("~/test/path")

            # Validator should not be called if home resolution fails
            mock_validator_func.assert_not_called()


class TestNormalizeUrlPath:
    """Tests for URL path component normalization."""

    def test_adds_leading_slash(self) -> None:
        """URL paths always get leading slash."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("path/to/file") == "/path/to/file"

    def test_preserves_leading_slash(self) -> None:
        """Existing leading slash is preserved."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("/path/to/file") == "/path/to/file"

    def test_converts_backslashes(self) -> None:
        """Backslashes converted to forward slashes."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("\\path\\to\\file") == "/path/to/file"

    def test_collapses_triple_slashes(self) -> None:
        """Triple slashes collapsed to single."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("///path") == "/path"

    def test_handles_mixed_slashes(self) -> None:
        """Mixed forward and backslashes normalized."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("path\\to/file") == "/path/to/file"

    def test_empty_path(self) -> None:
        """Empty path gets leading slash."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("") == "/"

    def test_relative_path_gets_slash(self) -> None:
        """Relative paths get leading slash for URL usage."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_url_path

        assert _normalize_url_path("relative") == "/relative"


class TestNormalizeFilePath:
    """Tests for file system path normalization."""

    def test_unix_absolute_path(self) -> None:
        """Unix absolute paths preserved."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        assert _normalize_file_path("/home/user/file") == "/home/user/file"

    def test_windows_absolute_path(self) -> None:
        """Windows absolute paths preserved with drive letter."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("C:\\Users\\test\\file")
        # pathlib normalizes to forward slashes but keeps drive letter
        assert result == "C:/Users/test/file"

    def test_windows_absolute_with_forward_slashes(self) -> None:
        """Windows paths with forward slashes work correctly."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("C:/Users/test/file")
        assert result == "C:/Users/test/file"

    def test_converts_backslashes(self) -> None:
        """Backslashes converted to forward slashes."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("path\\to\\file")
        assert "\\" not in result
        assert result == "path/to/file"

    def test_relative_paths_unchanged(self) -> None:
        """Relative paths remain relative."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("relative/path")
        assert result == "relative/path"

    def test_mixed_slashes(self) -> None:
        """Mixed slashes normalized."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("path\\to/file")
        assert result == "path/to/file"

    def test_windows_unc_path(self) -> None:
        """Windows UNC paths handled correctly."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("\\\\server\\share\\file")
        # UNC paths start with //
        assert result.startswith("//")

    def test_dot_segments_normalized(self) -> None:
        """Dot segments in paths partially normalized."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("path/./to/../file")
        # pathlib normalizes away . but keeps .. (doesn't fully resolve)
        assert result == "path/to/../file"

    def test_empty_path(self) -> None:
        """Empty path becomes current directory."""
        from dlkit.core.datatypes.tilde_expansion import _normalize_file_path

        result = _normalize_file_path("")
        assert result == "."
