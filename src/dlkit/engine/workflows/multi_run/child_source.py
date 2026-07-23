"""Closed sum type describing how a RunSpec is produced, plus its expansion.

Each :data:`ChildSource` variant is a different *recipe* for producing one or
more :class:`~dlkit.engine.workflows.multi_run.spec.RunSpec` instances:
loading TOML file(s) from disk, globbing a directory of TOML variants, or
wrapping already-loaded, already-validated settings. Adding a new way to
source children means adding a new dataclass and one more ``match`` arm in
:func:`expand_child_sources` — existing sources are never touched (OCP).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dlkit.common.errors import ConfigValidationError
from dlkit.engine.workflows.factories.build_strategy import WorkflowSettings

from .spec import RunSpec

# run.type value reserved for nested multirun sweeps. Nested sweeps-of-sweeps
# are out of scope by design, not an oversight — see _reject_nested_multirun.
_MULTIRUN_RUN_TYPE = "multirun"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplicitFileSource:
    """One child sourced from one or more TOML files merged left-to-right.

    Args:
        id: Stable identifier for the resulting ``RunSpec``.
        label: Human-readable label; also used as the child's MLflow run name.
        files: TOML file paths to merge, job-file-wins order (later files in
            the sequence take priority), matching ``load_job()``'s merge order.
        run_type: Optional ``run.type`` override, same semantics as
            ``load_job()``'s ``run_type`` parameter.
        patches: Optional patch mapping applied via ``settings.patch(...)``
            after loading.
        tags: MLflow tags for the resulting child's run.
        params: Opaque caller-supplied parameters threaded onto the ``RunSpec``.
        metadata: Opaque caller-supplied metadata threaded onto the ``RunSpec``.
    """

    id: str
    label: str
    files: tuple[Path, ...]
    run_type: str | None = None
    patches: dict[str, object] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class GlobSource:
    """Many children sourced from every TOML file matching a glob pattern.

    Each matched file is expanded exactly like an ``ExplicitFileSource`` with
    a single file. Child ids/labels are derived as ``f"{id_prefix}{stem}"``
    where ``stem`` is the matched file's ``Path.stem`` — deterministic and
    collision-resistant across re-globs of a changing directory (unlike a
    positional index).

    Args:
        id_prefix: Prefix prepended to each matched file's stem to form its
            child id/label.
        pattern: Glob pattern (e.g. ``"jobs/variants/*.toml"``), resolved
            relative to ``base_dir``.
        base_dir: Directory the glob pattern is resolved against.
        run_type: Optional ``run.type`` override applied to every match.
        tags: MLflow tags applied to every resulting child's run.
        params: Opaque caller-supplied parameters applied to every child.
        metadata: Opaque caller-supplied metadata applied to every child.
    """

    id_prefix: str
    pattern: str
    base_dir: Path
    run_type: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedSettingsSource:
    """One child from already-validated, already-loaded settings.

    Args:
        id: Stable identifier for the resulting ``RunSpec``.
        label: Human-readable label; also used as the child's MLflow run name.
        settings: Already-validated workflow settings — today's only real case.
        tags: MLflow tags for the resulting child's run.
        params: Opaque caller-supplied parameters threaded onto the ``RunSpec``.
        metadata: Opaque caller-supplied metadata threaded onto the ``RunSpec``.
    """

    id: str
    label: str
    settings: WorkflowSettings
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


type ChildSource = ExplicitFileSource | GlobSource | LoadedSettingsSource


def expand_child_sources(sources: Sequence[ChildSource]) -> tuple[RunSpec, ...]:
    """Expand a sequence of child sources into concrete RunSpecs.

    Args:
        sources: Ordered child sources; a single ``GlobSource`` can expand
            into many ``RunSpec``s.

    Returns:
        One ``RunSpec`` per resolved child, in expansion order.

    Raises:
        ConfigValidationError: Two resolved children share an ``id``, a
            ``GlobSource`` pattern matches zero files, or a resolved child's
            settings declare ``run.type == "multirun"`` (nested sweeps-of-sweeps
            are out of scope).
    """
    specs: list[RunSpec] = []
    for source in sources:
        match source:
            case ExplicitFileSource():
                specs.append(_expand_explicit_file_source(source))
            case GlobSource():
                specs.extend(_expand_glob_source(source))
            case LoadedSettingsSource():
                specs.append(
                    RunSpec(
                        id=source.id,
                        label=source.label,
                        settings=source.settings,
                        run_name=source.label,
                        tags=source.tags,
                        params=source.params,
                        metadata=source.metadata,
                    )
                )

    resolved = tuple(specs)
    _validate_specs(resolved)
    return resolved


def _expand_explicit_file_source(source: ExplicitFileSource) -> RunSpec:
    """Load, patch, and wrap one ExplicitFileSource into a RunSpec.

    Args:
        source: File-backed child source.

    Returns:
        The resulting RunSpec.
    """
    from dlkit.infrastructure.config.factories import load_job
    from dlkit.infrastructure.config.job_config import MultiRunJobConfig

    loaded = load_job(list(source.files), run_type=source.run_type)
    if isinstance(loaded, MultiRunJobConfig):
        raise ConfigValidationError(
            f"Child {source.id!r} resolves to run.type={_MULTIRUN_RUN_TYPE!r}; "
            "nested multirun sweeps are not supported."
        )
    settings = loaded
    if source.patches:
        settings = settings.patch(source.patches)
    return RunSpec(
        id=source.id,
        label=source.label,
        settings=settings,
        run_name=source.label,
        tags=source.tags,
        params=source.params,
        metadata=source.metadata,
    )


def _expand_glob_source(source: GlobSource) -> tuple[RunSpec, ...]:
    """Resolve a GlobSource's pattern and expand each match into a RunSpec.

    Args:
        source: Glob-backed child source.

    Returns:
        One RunSpec per matched file, in lexical (sorted path) order.

    Raises:
        ConfigValidationError: The pattern matches zero files.
    """
    matches = sorted(source.base_dir.glob(source.pattern))
    if not matches:
        raise ConfigValidationError(
            f"GlobSource pattern {source.pattern!r} under {source.base_dir} matched no files."
        )

    return tuple(
        _expand_explicit_file_source(
            ExplicitFileSource(
                id=f"{source.id_prefix}{path.stem}",
                label=f"{source.id_prefix}{path.stem}",
                files=(path,),
                run_type=source.run_type,
                tags=source.tags,
                params=source.params,
                metadata=source.metadata,
            )
        )
        for path in matches
    )


def _validate_specs(specs: tuple[RunSpec, ...]) -> None:
    """Fail loudly on duplicate ids or nested-multirun children.

    Args:
        specs: Fully-resolved RunSpecs to validate.

    Raises:
        ConfigValidationError: A duplicate ``id`` is found, or a child's
            settings declare ``run.type == "multirun"``.
    """
    seen_ids: set[str] = set()
    for spec in specs:
        if spec.id in seen_ids:
            raise ConfigValidationError(f"Duplicate multirun child id: {spec.id!r}")
        seen_ids.add(spec.id)
        _reject_nested_multirun(spec)


def _reject_nested_multirun(spec: RunSpec) -> None:
    """Raise if a child's settings resolve to a nested multirun sweep.

    Nested sweeps-of-sweeps are out of scope by design: a multirun child
    must be a leaf workflow (train/optimize/converge), not another sweep.

    Args:
        spec: Resolved child to check.

    Raises:
        ConfigValidationError: The child's ``run.type`` is ``"multirun"``.
    """
    run_type = getattr(getattr(spec.settings, "run", None), "type", None)
    if run_type == _MULTIRUN_RUN_TYPE:
        raise ConfigValidationError(
            f"Child {spec.id!r} resolves to run.type={_MULTIRUN_RUN_TYPE!r}; "
            "nested multirun sweeps are not supported."
        )
