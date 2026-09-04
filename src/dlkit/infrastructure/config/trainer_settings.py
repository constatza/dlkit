from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlkit.infrastructure.config.run_settings import RunSettings

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from pydantic import DirectoryPath, Field

from dlkit.infrastructure.config import BuildContext, FactoryProvider

# Import moved to method level to avoid circular imports
from dlkit.infrastructure.utils.logging_config import get_logger, should_enable_progress_bar

from .core.base_settings import ComponentSettings

loguru_logger = get_logger(__name__)


class CallbackSettings(ComponentSettings):
    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default=None, description="Name of the callback"
    )
    module_path: str | None = Field(
        default="lightning.pytorch.callbacks",
        description="Module path where the callback class is located.",
    )


class LoggerSettings(ComponentSettings):
    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default=None, description="Name of the logger."
    )
    module_path: str | None = Field(
        default="lightning.pytorch.loggers",
        description="Module path where the logger class is located.",
    )


class TrainerSettings(ComponentSettings):
    """TrainerSettings defines configuration options for training a model.

    Attributes:
        max_epochs (int): Maximum number of epochs to train for. Defaults to 100.
        gradient_clip_val (float | None): Value for gradient clipping, if any. Defaults to None.
        fast_dev_run (bool | int): Flag for fast development run or number of batches to run in fast dev mode. Defaults to False.
        overfit_batches (int | float): Overfit on this many/fraction of training batches, for capacity sanity checks. Defaults to 0.
        default_root_dir (DirectoryPath | None): Default root directory for the model. Defaults to None.
        enable_checkpointing (bool): Whether to enable checkpointing. Defaults to False.
        callbacks (tuple[CallbackSettings, ...]): List of callbacks. Defaults to an empty tuple.
        logger (LoggerSettings): Logger settings. Defaults to an instance of LoggerSettings.
        accelerator (Literal["cpu", "cuda"]): Accelerator to use for training. Defaults to "cuda".
        strategy (str | None): Lightning distributed strategy (e.g. "ddp"). Defaults to None.
        devices (int | list[int] | Literal["auto"] | None): Devices per node. Defaults to None (derived).
        num_nodes (int | None): Number of nodes. Defaults to None (derived).
    """

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="Trainer", description="Name of the trainer."
    )
    module_path: str | None = Field(
        default="lightning.pytorch",
        description="Module path where the trainer class is located.",
    )

    max_epochs: int = Field(
        default=100,
        ge=1,
        description="Maximum number of epochs to train for. Must be at least 1.",
    )
    gradient_clip_val: float | None = Field(
        default=None, description="Value for gradient clipping (if any)."
    )
    fast_dev_run: bool | int = Field(
        default=False,
        description="Flag for fast development run or number of batches to run in fast dev mode.",
    )
    overfit_batches: int | float = Field(
        default=0,
        description=(
            "Overfit on this many (int) or this fraction (float) of batches. "
            "Applies the same count to train and val, each on its own data "
            "(not shared between them), repeating those same batch(es) every "
            "epoch with shuffling disabled; does not affect the test split. "
            "Sanity check for model capacity."
        ),
    )
    default_root_dir: DirectoryPath | None = Field(
        default=None, description="Default root directory for the model."
    )
    enable_checkpointing: bool = Field(
        default=False, description="Whether to enable checkpointing."
    )
    callbacks: tuple[CallbackSettings, ...] = Field(
        default=tuple(), description="List of callbacks."
    )

    logger: LoggerSettings = Field(default=LoggerSettings(), description="Logger settings.")

    accelerator: Literal["cpu", "gpu", "auto", "tpu"] = Field(
        default="auto", description="Accelerator to use for training."
    )

    # Precision parameter for Lightning integration
    precision: str | int | None = Field(
        default=None,
        description="Lightning precision parameter. If None, uses session precision strategy.",
    )

    strategy: str | None = Field(
        default=None,
        description=(
            "Lightning distributed strategy (e.g. 'ddp', 'fsdp', 'deepspeed'). "
            "Passed straight through to Trainer — DLKit does not select or "
            "validate strategies itself. If None, Lightning's own default "
            "applies ('ddp' when multiple devices/nodes are resolved, "
            "single-device otherwise)."
        ),
    )
    devices: int | list[int] | Literal["auto"] | None = Field(
        default=None,
        description=(
            "Lightning devices parameter. If None, derived from the resolved "
            "compute environment (session.compute — see "
            "dlkit.infrastructure.compute), or Lightning's own default."
        ),
    )
    num_nodes: int | None = Field(
        default=None,
        description=(
            "Lightning num_nodes parameter. If None, derived from the "
            "resolved compute environment (session.compute), or Lightning's "
            "own default (1)."
        ),
    )

    def build(self, session: RunSettings | None = None) -> Trainer:
        """Build PyTorch Lightning Trainer with precision resolution.

        Args:
            session: Optional RunSettings, used as precision provider and as
                     the source of compute topology (session.compute). If not
                     provided, uses global default precision and default
                     (auto-detected) compute topology.

        Returns:
            Configured PyTorch Lightning Trainer instance.
        """
        # Import here to avoid circular imports

        # Build callbacks via factory
        callbacks: list[Callback] = []
        for callback in self.callbacks:
            cb = FactoryProvider.create_component(callback, BuildContext(mode="training"))
            callbacks.append(cb)
            loguru_logger.debug("Added trainer callback '{}'", callback.name)

        # Build logger via factory if configured
        if self.logger.name:
            lightning_logger = FactoryProvider.create_component(
                self.logger, BuildContext(mode="training")
            )
        else:
            lightning_logger = False

        # Resolve precision parameter using precision service
        from dlkit.infrastructure.precision.service import get_precision_service

        precision_service = get_precision_service()
        lightning_precision = self.precision
        if lightning_precision is None:
            # Use session precision strategy if not explicitly set
            # Pass session as provider so precision service can read session.precision
            lightning_precision = precision_service.get_lightning_precision(provider=session)
            loguru_logger.debug("Using session precision strategy: {}", lightning_precision)
        else:
            loguru_logger.debug("Using explicit trainer precision: {}", lightning_precision)

        enable_progress_bar = should_enable_progress_bar()

        # Resolve node/device topology with three-tier precedence:
        # 1. This trainer's own explicit devices/num_nodes (mirrors Trainer's
        #    own constructor directly, like accelerator/strategy/precision).
        # 2. The compute environment's required fields (LSF/MPI/Kubeflow only
        #    — those environments can't auto-derive, so their settings
        #    classes require devices/num_nodes; see infrastructure.compute).
        # 3. Auto-detected topology (local/SLURM/torchrun/etc.).
        # session.compute lives on RunSettings, not here, because "which
        # environment is this job running under" is job-wide config (like
        # precision/seed), not trainer-construction config.
        from dlkit.infrastructure.compute import resolve_compute_environment
        from dlkit.infrastructure.config.compute_settings import AutoComputeSettings

        compute = session.compute if session is not None else AutoComputeSettings()
        topology = resolve_compute_environment(compute.environment)
        # Only LSFComputeSettings/MPIComputeSettings/KubeflowComputeSettings
        # declare devices/num_nodes at all (as required fields); every other
        # environment class has neither, so getattr's default applies.
        environment_devices = getattr(compute, "devices", None)
        environment_num_nodes = getattr(compute, "num_nodes", None)

        devices = self.devices
        if devices is None:
            devices = environment_devices if environment_devices is not None else topology.devices
        num_nodes = self.num_nodes
        if num_nodes is None:
            num_nodes = (
                environment_num_nodes if environment_num_nodes is not None else topology.num_nodes
            )

        overrides: dict[str, Any] = {
            "callbacks": callbacks,
            "logger": lightning_logger,
            "precision": lightning_precision,
            "enable_model_summary": False,
            "enable_progress_bar": enable_progress_bar,
        }
        if devices is not None:
            overrides["devices"] = devices
        if num_nodes is not None:
            overrides["num_nodes"] = num_nodes
        if topology.cluster_environment is not None:
            overrides["plugins"] = [topology.cluster_environment]

        # Build Trainer via factory with explicit overrides
        return FactoryProvider.create_component(
            self,
            BuildContext(mode="training", overrides=overrides),
        )
