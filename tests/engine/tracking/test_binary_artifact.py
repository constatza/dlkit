"""Tests for log_binary_artifact — binary-safe MLflow artifact staging.

MlflowClient.log_text always writes through a UTF-8 text file handle, which
corrupts non-text bytes (e.g. PNG). log_binary_artifact stages content on disk
in binary mode and uploads it via the binary-safe log_artifact path instead.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dlkit.engine.tracking.binary_artifact import log_binary_artifact

RUN_ID = "run-1"


@pytest.fixture
def mock_client() -> MagicMock:
    """MLflow client mock capturing log_artifact/log_text calls.

    Returns:
        MagicMock standing in for MlflowClient.
    """
    return MagicMock()


@pytest.fixture
def png_like_bytes() -> bytes:
    """Byte content that is not valid UTF-8, mimicking real PNG magic bytes.

    Returns:
        Bytes containing the PNG signature plus the full byte range.
    """
    return b"\x89PNG\r\n\x1a\n" + bytes(range(256))


def test_staged_file_content_matches_input_bytes(
    mock_client: MagicMock, png_like_bytes: bytes
) -> None:
    """The bytes written to the staged temp file exactly match the input content.

    Args:
        mock_client: Mock MlflowClient.
        png_like_bytes: Non-UTF-8-decodable content to stage.
    """
    written: dict[str, bytes] = {}

    def _capture(run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        written["content"] = Path(local_path).read_bytes()

    mock_client.log_artifact.side_effect = _capture

    log_binary_artifact(mock_client, RUN_ID, png_like_bytes, "plots/loss_curve.png")

    assert written["content"] == png_like_bytes


def test_uses_log_artifact_not_log_text(mock_client: MagicMock, png_like_bytes: bytes) -> None:
    """Binary content is uploaded via log_artifact, never log_text.

    Args:
        mock_client: Mock MlflowClient.
        png_like_bytes: Non-UTF-8-decodable content to stage.
    """
    log_binary_artifact(mock_client, RUN_ID, png_like_bytes, "plots/loss_curve.png")

    mock_client.log_artifact.assert_called_once()
    mock_client.log_text.assert_not_called()


def test_splits_artifact_dir_from_filename(mock_client: MagicMock, png_like_bytes: bytes) -> None:
    """artifact_file's directory becomes artifact_path; the rest becomes the filename.

    Args:
        mock_client: Mock MlflowClient.
        png_like_bytes: Non-UTF-8-decodable content to stage.
    """
    log_binary_artifact(mock_client, RUN_ID, png_like_bytes, "plots/loss_curve.png")

    call = mock_client.log_artifact.call_args
    assert call.args[0] == RUN_ID
    assert Path(call.args[1]).name == "loss_curve.png"
    assert call.kwargs["artifact_path"] == "plots"


def test_no_subdirectory_passes_none_artifact_path(
    mock_client: MagicMock, png_like_bytes: bytes
) -> None:
    """A bare filename with no directory component results in artifact_path=None.

    Args:
        mock_client: Mock MlflowClient.
        png_like_bytes: Non-UTF-8-decodable content to stage.
    """
    log_binary_artifact(mock_client, RUN_ID, png_like_bytes, "manifest.png")

    call = mock_client.log_artifact.call_args
    assert call.kwargs["artifact_path"] is None
