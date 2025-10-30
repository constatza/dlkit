from typing import Literal

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelSummary
from lightning.pytorch.loggers import Logger
from pydantic import DirectoryPath, Field

from .core.base_settings import ComponentSettings
from .core import BuildContext, FactoryProvider

# Import moved to method level to avoid circular imports
from loguru import logger as loguru_logger


class CallbackSettings(ComponentSettings[Callback]):
    name: str | None = Field(default=None, description="Name of the callback")
    module_path: str = Field(
        default="lightning.pytorch.callbacks",
        description="Module path where the callback class is located.",
    )


class LoggerSettings(ComponentSettings[Logger]):
    name: str | None = Field(default=None, description="Name of the logger.")
    module_path: str = Field(
        default="lightning.pytorch.loggers",
        description="Module path where the logger class is located.",
    )


class TrainerSettings(ComponentSettings[Trainer]):
    """TrainerSettings defines configuration options for training a model.

    Attributes:
        max_epochs (int): Maximum number of epochs to train for. Defaults to 100.
        gradient_clip_val (float | None): Value for gradient clipping, if any. Defaults to None.
        fast_dev_run (bool | int): Flag for fast development run or number of batches to run in fast dev mode. Defaults to False.
        default_root_dir (DirectoryPath | None): Default root directory for the model. Defaults to None.
        enable_checkpointing (bool): Whether to enable checkpointing. Defaults to False.
        callbacks (tuple[CallbackSettings, ...]): List of callbacks. Defaults to an empty tuple.
        logger (LoggerSettings): Logger settings. Defaults to an instance of LoggerSettings.
        accelerator (Literal["cpu", "cuda"]): Accelerator to use for training. Defaults to "cuda".
    """

    name: str = Field(default="Trainer", description="Name of the trainer.")
    module_path: str = Field(
        default="lightning.pytorch",
        description="Module path where the trainer class is located.",
    )

    max_epochs: int = Field(
        default=100,
        description="Maximum number of epochs to train for.",
    )
    gradient_clip_val: float | None = Field(
        default=None, description="Value for gradient clipping (if any)."
    )
    fast_dev_run: bool | int = Field(
        default=False,
        description="Flag for fast development run or number of batches to run in fast dev mode.",
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

    def build(self, session: "SessionSettings | None" = None) -> Trainer:
        """Build PyTorch Lightning Trainer with precision resolution.

        Args:
            session: Optional SessionSettings to use as precision provider.
                     If not provided, will use global default precision.

        Returns:
            Configured PyTorch Lightning Trainer instance.
        """
        # Import here to avoid circular imports
        from dlkit.tools.config.session_settings import SessionSettings

        # Build callbacks via factory
        callbacks: list[Callback] = [ModelSummary(max_depth=2)]
        for callback in self.callbacks:
            cb = FactoryProvider.create_component(callback, BuildContext(mode="training"))
            callbacks.append(cb)
            loguru_logger.info(f"Added callback: {callback.name}")

        # Build logger via factory if configured
        if self.logger.name:
            lightning_logger = FactoryProvider.create_component(
                self.logger, BuildContext(mode="training")
            )
        else:
            lightning_logger = False

        # Resolve precision parameter using precision service
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        lightning_precision = self.precision
        if lightning_precision is None:
            # Use session precision strategy if not explicitly set
            # Pass session as provider so precision service can read session.precision
            lightning_precision = precision_service.get_lightning_precision(provider=session)
            loguru_logger.info(f"Using session precision strategy: {lightning_precision}")
        else:
            loguru_logger.info(f"Using explicit trainer precision: {lightning_precision}")

        # Build Trainer via factory with explicit overrides
        return FactoryProvider.create_component(
            self,
            BuildContext(
                mode="training",
                overrides={
                    "callbacks": callbacks,
                    "logger": lightning_logger,
                    "precision": lightning_precision,
                },
            ),
        )
