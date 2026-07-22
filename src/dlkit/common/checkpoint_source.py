"""Sum type for selecting which MLflow run (if any) a checkpoint comes from.

Lives in ``common`` — not ``interfaces.inference`` — so that any layer
(including, later, training's checkpoint-resume path) could adopt it without
requiring the type to move. These dataclasses are plain data: no MLflow
import, no I/O. Actual resolution to a downloaded local Path happens in
``interfaces.inference`` / ``engine.tracking``, which are allowed to import
MLflow-coupled tracking code per tach.toml.

Always resolves to the run's best checkpoint — there is no "which file
within the run" axis, only "which run".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCheckpoint:
    """Resolve the best checkpoint from a caller-named MLflow run.

    Args:
        run_id: Exact MLflow run id to pull the checkpoint from.
    """

    run_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LatestRunCheckpoint:
    """Resolve the best checkpoint from the temporally-latest run in an experiment scope.

    Args:
        experiment_name: Experiment to search; None defers to
            settings.experiment.name.
    """

    experiment_name: str | None = None


type CheckpointSource = RunCheckpoint | LatestRunCheckpoint
