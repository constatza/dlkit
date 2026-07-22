"""API functions module."""

from dlkit.common import MultiRunResult
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint

from .config import generate_template, validate_config, validate_template
from .core import build_inference_datamodule, converge, optimize, train
from .execution import execute
from .model_logged import (
    LoggedModelRecord,
    build_logged_model_uri,
    load_logged_model,
    search_logged_models,
)
from .model_registry import (
    build_registered_model_uri,
    download_checkpoint_artifact,
    download_run_split,
    find_child_run_ids,
    find_latest_run_id,
    get_model_version,
    has_checkpoint_artifact,
    list_model_versions,
    load_registered_model,
    register_logged_model,
    search_registered_models,
    set_registered_model_alias,
    set_registered_model_version_tag,
    set_registered_model_version_tags,
)

__all__ = [
    # Core workflow functions
    "converge",
    "train",
    "optimize",
    "build_inference_datamodule",
    # Configuration functions
    "validate_config",
    "generate_template",
    "validate_template",
    # Unified execution function
    "execute",
    # Model registry helpers
    "search_registered_models",
    "list_model_versions",
    "get_model_version",
    "has_checkpoint_artifact",
    "download_run_split",
    "register_logged_model",
    "set_registered_model_alias",
    "set_registered_model_version_tag",
    "set_registered_model_version_tags",
    "build_registered_model_uri",
    "load_registered_model",
    "LoggedModelRecord",
    "search_logged_models",
    "build_logged_model_uri",
    "load_logged_model",
    # Run-based checkpoint selection helpers
    "find_latest_run_id",
    "find_child_run_ids",
    "download_checkpoint_artifact",
    "RunCheckpoint",
    "LatestRunCheckpoint",
    "MultiRunResult",
]
