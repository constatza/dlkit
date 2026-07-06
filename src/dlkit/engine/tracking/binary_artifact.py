"""Binary-safe artifact staging for MLflow.

``MlflowClient.log_text`` always writes through a UTF-8 text file handle,
which corrupts non-text bytes (e.g. PNG). This module stages bytes on disk
in binary mode and uploads them via the binary-safe ``log_artifact`` path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow import MlflowClient


def log_binary_artifact(
    client: MlflowClient,
    run_id: str,
    content: bytes,
    artifact_file: str,
) -> None:
    """Stage bytes to a temp file and upload via MlflowClient.log_artifact.

    Args:
        client: MLflow client instance.
        run_id: Active run ID for logging operations.
        content: Raw bytes to persist as an artifact (e.g. PNG image data).
        artifact_file: Destination path within the run artifact store,
            e.g. ``"plots/loss_curve.png"``.
    """
    artifact_path = Path(artifact_file)
    artifact_dir = artifact_path.parent.as_posix()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / artifact_path.name
        tmp_path.write_bytes(content)
        client.log_artifact(
            run_id, str(tmp_path), artifact_path=None if artifact_dir == "." else artifact_dir
        )
