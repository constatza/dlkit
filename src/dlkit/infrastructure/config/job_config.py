"""Job config — top-level discriminated union of training, inference, and search jobs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dlkit.common.results import FailurePolicy
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings
from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.plot_settings import PlotSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.search_settings import SearchSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


class JobConfig(BasicSettings):
    """Base job config. All workflow types share this structure.

    Args:
        run: Execution control (type, seed, precision, profile references).
        experiment: Experiment identity and registration metadata.
        model: Model class selector and hyperparameters.
        data: Dataset, dataloader, and DataModule configuration.
        training: Training pipeline configuration.
        search: HPO search configuration.
        tracking: Tracking backend connection.
        plots: Plot artifact generation settings (opt-in, disabled by default).
    """

    run: RunSettings
    experiment: ExperimentSettings | None = None
    model: ModelComponentSettings | None = None
    data: DataSettings | None = None
    training: TrainingSettings | None = None
    search: SearchSettings | None = None
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    plots: PlotSettings = Field(default_factory=PlotSettings)


class TrainingJobConfig(JobConfig):
    """Validated training job: model, data, and training sections are required.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
        training: Training pipeline configuration (required).
    """

    model: ModelComponentSettings
    data: DataSettings
    training: TrainingSettings


class InferenceJobConfig(JobConfig):
    """Validated inference job: only model is required (data optional for batch predict).

    Args:
        model: Model class selector and hyperparameters (required).
        split: Which labeled split ``evaluate()`` compares predictions against.
        device: Inference device (``"auto"``, ``"cpu"``, ``"cuda:0"``, ...). Free-form,
            unlike ``training.trainer.accelerator``'s restricted Lightning literal.
    """

    model: ModelComponentSettings
    split: Literal["test", "predict"] = "test"
    device: str = "auto"

    @model_validator(mode="after")
    def _checkpoint_required(self) -> InferenceJobConfig:
        """Ensure a checkpoint path is provided for inference.

        Returns:
            Self, if validation passes.

        Raises:
            ValueError: If model.checkpoint is None.
        """
        if self.model.checkpoint is None:
            raise ValueError("InferenceJobConfig requires model.checkpoint to be set.")
        return self


class FitJobConfig(JobConfig):
    """Validated one-shot fit job: model and data are required; training stays unset.

    For models whose entire "training" is one deterministic, non-gradient
    call (e.g. a thin-SVD into a ``register_buffer``) — no epochs, optimizer,
    or loss, so ``training`` is intentionally left unset rather than required.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
    """

    model: ModelComponentSettings
    data: DataSettings


class SearchJobConfig(JobConfig):
    """Validated HPO job: same as training plus a non-empty search section.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
        training: Training pipeline configuration (required).
        search: HPO search configuration (required, must have non-empty space).
    """

    model: ModelComponentSettings
    data: DataSettings
    training: TrainingSettings
    search: SearchSettings

    @model_validator(mode="after")
    def _space_required(self) -> SearchJobConfig:
        """Ensure the search space is non-empty.

        Returns:
            Self, if validation passes.

        Raises:
            ValueError: If search.space is empty.
        """
        if not self.search.space:
            raise ValueError("SearchJobConfig requires at least one entry in search.space.")
        return self


class ConvergenceJobConfig(JobConfig):
    """Validated convergence job: model, data, training, and convergence sections required.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
        training: Training pipeline configuration (required).
        convergence: Sample-size convergence study configuration (required).
    """

    model: ModelComponentSettings
    data: DataSettings
    training: TrainingSettings
    convergence: ConvergenceSettings


class ChildEntryConfig(BasicSettings):
    """One ``[[multirun.runs]]`` entry: a child run source plus opaque metadata.

    ``files`` is either an explicit list of TOML paths merged left-to-right
    (job-file-wins order, matching ``load_job()``'s merge semantics) or a
    single glob-pattern string (containing ``*``, ``?``, or ``[``) matched
    against files under the multirun config's own directory. Relative paths
    in either form are resolved by ``load_job()`` before this model is
    validated — by the time a ``ChildEntryConfig`` exists, ``files`` holds
    only absolute path(s)/pattern.

    Args:
        id: Stable identifier for this child within the sweep. Must be
            unique across the sweep (enforced at expansion time, not here).
        label: Human-readable label; also used as the child's MLflow run
            name. Defaults to ``id`` when left blank.
        files: Explicit TOML file path(s) merged left-to-right, or a single
            glob pattern string.
        run_type: Optional ``run.type`` override for the child's own
            settings, same semantics as ``load_job()``'s ``run_type`` param.
        patches: Optional patch mapping applied to the child's settings via
            ``settings.patch(...)`` after loading, same semantics as
            ``ExplicitFileSource.patches``. Ignored for glob-sourced children
            (a single patch dict can't sensibly apply to every glob match).
        tags: MLflow tags applied to the child's run.
        params: Opaque caller-supplied parameters threaded onto the
            resulting ``RunSpec`` — never interpreted by DLKit.
        metadata: Opaque caller-supplied metadata threaded onto the
            resulting ``RunSpec`` — never interpreted by DLKit.
    """

    id: str
    label: str = ""
    files: str | list[str]
    run_type: str | None = None
    patches: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildDefaults(BasicSettings):
    """Optional ``[multirun.defaults]`` table merged into every child entry.

    Explicit, opt-in sharing: nothing here applies unless a sweep author
    writes this table, and each ``ChildEntryConfig``'s own values always win
    over these on conflict — same precedence direction
    ``MultiRunOrchestrator._run_one`` already uses merging a child's tags
    over its settings-baked ones.

    Args:
        patches: Default patch mapping merged under each child's own
            ``patches`` (child keys win).
        tags: Default MLflow tags merged under each child's own ``tags``.
        params: Default opaque params merged under each child's own
            ``params``.
        metadata: Default opaque metadata merged under each child's own
            ``metadata``.
    """

    patches: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiRunJobConfig(BasicSettings):
    """Validated multirun sweep job: parent run identity plus child run sources.

    Authored in TOML under a single ``[multirun]`` table (parent identity
    fields plus ``[[multirun.runs]]`` child entries); this model hoists that
    table's keys onto itself so callers read ``settings.experiment_name``,
    ``settings.parent_run_name``, ``settings.failure_policy``, and
    ``settings.runs`` directly, without an extra ``.multirun`` indirection.

    Args:
        run: Execution control (``run.type`` must be ``"multirun"``).
        experiment_name: MLflow experiment name shared by the parent and
            every child run.
        parent_run_name: MLflow name for the parent sweep run.
        parent_tags: MLflow tags applied to the parent sweep run.
        failure_policy: How the sweep reacts when a child run fails.
        defaults: Optional ``[multirun.defaults]`` table merged into every
            child entry (child values win). Unset by default — nothing is
            shared across children unless a sweep author explicitly writes
            this table.
        runs: Ordered child run sources.
    """

    run: RunSettings
    experiment_name: str
    parent_run_name: str
    parent_tags: dict[str, str] = Field(default_factory=dict)
    failure_policy: FailurePolicy = "fail_fast"
    defaults: ChildDefaults = Field(default_factory=ChildDefaults)
    runs: list[ChildEntryConfig]

    @model_validator(mode="before")
    @classmethod
    def _hoist_multirun_section(cls, data: Any) -> Any:
        """Lift ``[multirun]`` table keys onto the top-level payload.

        Args:
            data: Raw payload, typically the ``load_job()``-merged dict with
                a nested ``"multirun"`` table.

        Returns:
            Payload with ``multirun``'s keys merged at the top level (top-level
            keys win on conflict), or the original payload unchanged if it is
            not a dict or carries no ``"multirun"`` table.
        """
        if not isinstance(data, dict):
            return data
        multirun_section = data.get("multirun")
        if not isinstance(multirun_section, dict):
            return data
        merged: dict[str, Any] = dict(multirun_section)
        merged.update({k: v for k, v in data.items() if k != "multirun"})
        return merged


# Sum type of every validated JobConfig subtype. Single source of truth for
# the union spelled out repeatedly across load_job()'s overloads, execute()'s
# dispatch signature, and similar call sites — update here, not at each site.
type AnyJobConfig = (
    TrainingJobConfig
    | InferenceJobConfig
    | SearchJobConfig
    | FitJobConfig
    | ConvergenceJobConfig
    | MultiRunJobConfig
)
