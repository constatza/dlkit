"""Explicit path context for API/CLI operations.

This module provides an immutable PathContext dataclass that replaces the
thread-local PathOverrideContext system with explicit dependency injection.

Design Philosophy:
- Explicit over Implicit: Pass context as parameter, not thread-local
- Immutability: Frozen dataclass, created once
- Async-compatible: No threading.local() dependencies
- Debuggable: Context visible in stack traces and debugger
- Type-safe: Comprehensive type hints throughout

Migration Note:
    This is Phase 0 of the path context refactoring. The new PathContext
    exists alongside the old thread-local system. All new code should use
    PathContext, but old code continues to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlkit.tools.config.environment import DLKitEnvironment
from dlkit.tools.config.general_settings import GeneralSettings


@dataclass(frozen=True)
class PathContext:
    """Explicit path context for API/CLI operations.

    Immutable context holding path overrides that flow explicitly through
    the application. Replaces thread-local PathOverrideContext.

    Design Rationale:
        - Frozen dataclass: Immutable, hashable, clear semantics
        - Explicit parameters: Visible in debugger and function signatures
        - Factory methods: Centralized creation logic
        - No thread-local: Async-compatible, debugger-friendly

    Precedence (when resolving paths):
        1. Explicit context fields (root_dir, output_dir, etc.)
        2. DLKitEnvironment (from DLKIT_ROOT_DIR env or SESSION.root_dir)
        3. Current working directory

    Attributes:
        root_dir: Optional root directory override for the session
        output_dir: Optional output directory override (for predictions, artifacts)
        data_dir: Optional data directory override (for datasets)
        checkpoints_dir: Optional checkpoints directory override (for model weights)

    Example:
        >>> # From CLI args
        >>> ctx = PathContext.from_cli_args(root_dir="/project")
        >>> ctx.root_dir
        PosixPath('/project')

        >>> # From settings
        >>> settings = GeneralSettings(...)
        >>> ctx = PathContext.from_settings(settings)

        >>> # Merging contexts (other takes precedence)
        >>> ctx1 = PathContext(root_dir=Path("/base"))
        >>> ctx2 = PathContext(output_dir=Path("/output"))
        >>> merged = ctx1.merge(ctx2)
        >>> merged.root_dir, merged.output_dir
        (PosixPath('/base'), PosixPath('/output'))
    """

    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    checkpoints_dir: Path | None = None

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> PathContext:
        """Create PathContext from override dictionary.

        This factory method is useful when deserializing from JSON/TOML
        or when receiving path overrides as a dictionary from CLI/API.

        Args:
            overrides: Dictionary with path overrides. Recognized keys:
                - root_dir: Root directory path (str or Path)
                - output_dir: Output directory path (str or Path)
                - data_dir: Data directory path (str or Path)
                - checkpoints_dir: Checkpoints directory path (str or Path)
                All keys are optional. Non-path keys are ignored.

        Returns:
            Immutable PathContext instance with paths converted to Path objects.

        Example:
            >>> overrides = {"root_dir": "/project", "output_dir": "/output"}
            >>> ctx = PathContext.from_dict(overrides)
            >>> ctx.root_dir
            PosixPath('/project')
        """
        return cls(
            root_dir=Path(overrides["root_dir"]) if overrides.get("root_dir") else None,
            output_dir=Path(overrides["output_dir"]) if overrides.get("output_dir") else None,
            data_dir=Path(overrides["data_dir"]) if overrides.get("data_dir") else None,
            checkpoints_dir=Path(overrides["checkpoints_dir"])
            if overrides.get("checkpoints_dir")
            else None,
        )

    @classmethod
    def from_settings(cls, settings: GeneralSettings) -> PathContext:
        """Create PathContext from settings (SESSION.root_dir).

        Extracts the root_dir from settings.SESSION.root_dir if present,
        otherwise returns an empty context. This is useful for defensive
        context creation from loaded configuration.

        Args:
            settings: GeneralSettings instance with optional SESSION.root_dir.
                If SESSION section or root_dir field is missing, returns
                empty context.

        Returns:
            PathContext with root_dir from settings, or empty context if
            SESSION.root_dir is not present.

        Example:
            >>> settings = GeneralSettings(SESSION=SessionSettings(root_dir="/project"))
            >>> ctx = PathContext.from_settings(settings)
            >>> ctx.root_dir
            PosixPath('/project')

            >>> # Settings without SESSION.root_dir
            >>> settings = GeneralSettings()
            >>> ctx = PathContext.from_settings(settings)
            >>> ctx.root_dir is None
            True
        """
        root_dir = getattr(getattr(settings, "SESSION", None), "root_dir", None)
        if root_dir:
            return cls(root_dir=Path(root_dir))
        return cls()

    @classmethod
    def from_cli_args(
        cls,
        root_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        checkpoints_dir: Path | str | None = None,
    ) -> PathContext:
        """Create PathContext from CLI arguments.

        This factory method is designed for CLI adapters that receive
        path overrides as command-line arguments (e.g., --root-dir).

        Args:
            root_dir: Root directory override (str or Path). Can be None.
            output_dir: Output directory override (str or Path). Can be None.
            data_dir: Data directory override (str or Path). Can be None.
            checkpoints_dir: Checkpoints directory override (str or Path).
                Can be None.

        Returns:
            PathContext with provided overrides converted to Path objects.
            None values are preserved (no override).

        Example:
            >>> # From CLI args like: --root-dir /project --output-dir /output
            >>> ctx = PathContext.from_cli_args(root_dir="/project", output_dir="/output")
            >>> ctx.root_dir, ctx.output_dir
            (PosixPath('/project'), PosixPath('/output'))
        """
        return cls(
            root_dir=Path(root_dir) if root_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
            data_dir=Path(data_dir) if data_dir else None,
            checkpoints_dir=Path(checkpoints_dir) if checkpoints_dir else None,
        )

    @classmethod
    def empty(cls) -> PathContext:
        """Create empty PathContext (no overrides).

        This is useful for testing or when you want to create a context
        that relies entirely on fallback resolution (DLKitEnvironment, CWD).

        Returns:
            PathContext with all fields set to None.

        Example:
            >>> ctx = PathContext.empty()
            >>> ctx.root_dir is None
            True
            >>> ctx.has_root_override()
            False
        """
        return cls()

    def has_root_override(self) -> bool:
        """Check if context has explicit root_dir override.

        This is useful for defensive checks to determine if the context
        should take precedence over environment/config-based defaults.

        Returns:
            True if root_dir is explicitly set, False otherwise.

        Example:
            >>> ctx = PathContext(root_dir=Path("/project"))
            >>> ctx.has_root_override()
            True

            >>> ctx = PathContext.empty()
            >>> ctx.has_root_override()
            False
        """
        return self.root_dir is not None

    def merge(self, other: PathContext) -> PathContext:
        """Merge with another context (other takes precedence).

        Creates a new PathContext by combining fields from self and other,
        with other's fields taking priority when both are set.

        This is useful for layering contexts, e.g., combining config-based
        defaults with CLI overrides.

        Args:
            other: PathContext to merge (higher priority). Fields set in
                this context will override corresponding fields in self.

        Returns:
            New PathContext with merged values. For each field:
            - If other.field is not None, use it
            - Otherwise, use self.field

        Example:
            >>> base = PathContext(root_dir=Path("/base"), output_dir=Path("/base/output"))
            >>> override = PathContext(output_dir=Path("/custom/output"))
            >>> merged = base.merge(override)
            >>> merged.root_dir, merged.output_dir
            (PosixPath('/base'), PosixPath('/custom/output'))
        """
        return PathContext(
            root_dir=other.root_dir if other.root_dir else self.root_dir,
            output_dir=other.output_dir if other.output_dir else self.output_dir,
            data_dir=other.data_dir if other.data_dir else self.data_dir,
            checkpoints_dir=other.checkpoints_dir
            if other.checkpoints_dir
            else self.checkpoints_dir,
        )


def resolve_root_dir(
    path_context: PathContext | None = None,
    env: DLKitEnvironment | None = None,
) -> Path:
    """Resolve root directory with explicit precedence.

    This is the core resolution function for determining the root directory
    for DLKit operations. It implements the three-layer precedence system.

    Precedence Rules:
        1. path_context.root_dir (explicit API/CLI override) - HIGHEST
        2. env.get_root_path() (from DLKIT_ROOT_DIR or SESSION.root_dir)
        3. Path.cwd() (current working directory) - LOWEST (fallback)

    Args:
        path_context: Explicit path context (may be None). If provided and
            has root_dir set, takes highest precedence.
        env: DLKitEnvironment instance (may be None). If provided, used as
            fallback after path_context.

    Returns:
        Resolved root directory path (always absolute, resolved).

    Example:
        >>> # With explicit context (highest priority)
        >>> ctx = PathContext(root_dir=Path("/project"))
        >>> resolve_root_dir(path_context=ctx)
        PosixPath('/project')

        >>> # Fallback to environment
        >>> env = DLKitEnvironment()  # Reads DLKIT_ROOT_DIR
        >>> resolve_root_dir(env=env)
        PosixPath('/home/user/projects/dlkit')

        >>> # Fallback to CWD
        >>> resolve_root_dir()
        PosixPath('/current/working/directory')
    """
    # Priority 1: Explicit context
    if path_context and path_context.root_dir:
        return path_context.root_dir.resolve()

    # Priority 2: Environment
    if env:
        return env.get_root_path().resolve()

    # Priority 3: CWD fallback
    return Path.cwd().resolve()


def resolve_component_path(
    component_path: str,
    path_context: PathContext | None = None,
    env: DLKitEnvironment | None = None,
) -> Path:
    """Resolve component path (e.g., 'output/mlruns') with context.

    This function resolves standard DLKit component paths (output, data,
    checkpoints, mlruns, etc.) using the path context and environment.

    Resolution Strategy:
        1. Check for direct component overrides in path_context
           (e.g., output_dir for "output")
        2. Check for path prefix overrides in path_context
           (e.g., output_dir for "output/mlruns")
        3. Fall back to standard resolution: root_dir / component_path

    Args:
        component_path: Component path string. Common values:
            - "output": Output directory
            - "output/mlruns": MLflow tracking directory
            - "output/predictions": Predictions directory
            - "data": Data directory
            - "checkpoints": Checkpoints directory
            - Any relative path under root
        path_context: Explicit path context (may be None). Used to check
            for component-specific overrides.
        env: DLKitEnvironment instance (may be None). Used for root
            directory fallback.

    Returns:
        Resolved absolute path for the component.

    Example:
        >>> # Direct component override
        >>> ctx = PathContext(output_dir=Path("/custom/output"))
        >>> resolve_component_path("output", path_context=ctx)
        PosixPath('/custom/output')

        >>> # Prefixed component path
        >>> resolve_component_path("output/mlruns", path_context=ctx)
        PosixPath('/custom/output/mlruns')

        >>> # Standard resolution (no override)
        >>> ctx = PathContext(root_dir=Path("/project"))
        >>> resolve_component_path("data", path_context=ctx)
        PosixPath('/project/data')

        >>> # Checkpoints with override
        >>> ctx = PathContext(checkpoints_dir=Path("/models"))
        >>> resolve_component_path("checkpoints", path_context=ctx)
        PosixPath('/models')
    """
    # Check for direct component overrides in context
    if path_context:
        # Direct output directory override
        if component_path == "output" and path_context.output_dir:
            return path_context.output_dir.resolve()

        # Prefixed output paths (e.g., "output/mlruns")
        if component_path.startswith("output/") and path_context.output_dir:
            relative_path = component_path[7:]  # Remove "output/" prefix
            return (path_context.output_dir / relative_path).resolve()

        # Direct data directory override
        if component_path == "data" and path_context.data_dir:
            return path_context.data_dir.resolve()

        # Checkpoints override (matches "checkpoints" or contains "checkpoint")
        if "checkpoint" in component_path and path_context.checkpoints_dir:
            return path_context.checkpoints_dir.resolve()

    # Fallback to standard resolution: root / component
    root_dir = resolve_root_dir(path_context, env)
    return (root_dir / component_path).resolve()
