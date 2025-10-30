"""Pure functions for server configuration validation and defaults.

This module contains pure functions for validating server configurations
and determining configuration defaults. All functions are side-effect-free
and follow functional programming principles.
"""

from __future__ import annotations

from typing import Any


def validate_mlflow_config(mlflow_cfg: Any) -> None:
    """Pure function to validate MLflow configuration.

    Args:
        mlflow_cfg: MLflow configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if mlflow_cfg is None:
        raise ValueError("MLFLOW configuration is missing")

    if not getattr(mlflow_cfg, "enabled", False):
        raise ValueError("MLflow is not enabled")


def get_host_variants(host: str, port: int) -> list[str]:
    """Pure function to get all equivalent host:port combinations.

    Args:
        host: Server hostname
        port: Server port

    Returns:
        List of equivalent host:port strings
    """
    variants = [f"{host}:{port}"]

    # Add localhost variants for local addresses
    if host == "localhost":
        variants.append(f"127.0.0.1:{port}")
    elif host == "127.0.0.1":
        variants.append(f"localhost:{port}")
    elif host not in ("127.0.0.1", "localhost"):
        variants.extend([f"127.0.0.1:{port}", f"localhost:{port}"])

    # Remove duplicates while preserving order
    return list(dict.fromkeys(variants))


def should_use_default_storage(server_config: Any, overrides: dict[str, Any]) -> bool:
    """Pure function to determine if default storage setup is needed.

    Args:
        server_config: Server configuration object (could be MLflowServerSettings or MLflowServerContext)
        overrides: CLI parameter overrides

    Returns:
        True if default storage setup is needed
    """
    # If user provided explicit storage options, respect them
    if (
        overrides.get("backend_store_uri") is not None
        or overrides.get("artifacts_destination") is not None
    ):
        return False

    # Get the actual settings object - could be MLflowServerContext wrapping settings
    # or direct MLflowServerSettings
    actual_config = server_config
    if (
        hasattr(server_config, "_server_config")
        and server_config.__class__.__module__ == "dlkit.interfaces.servers.mlflow_adapter"
    ):
        # This is a MLflowServerContext, get the wrapped settings
        actual_config = server_config._server_config

    # If config already has storage configured, use it
    backend_uri = getattr(actual_config, "backend_store_uri", None)
    artifacts_dest = getattr(actual_config, "artifacts_destination", None)

    # Check if either setting is configured (not None and not empty string)
    if (
        backend_uri is not None
        and str(backend_uri).strip() != ""
        or artifacts_dest is not None
        and str(artifacts_dest).strip() != ""
    ):
        return False

    return True
