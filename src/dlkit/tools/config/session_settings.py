"""Session settings for controlling execution mode and top-level configuration.

Simplified: use a single boolean flag `inference` to distinguish inference vs.
training flows. Detailed training/inference configuration lives at the
GeneralSettings top level (TRAINING, DATAMODULE, DATASET, etc.).
"""

from pydantic import Field, model_validator
from dlkit.core.datatypes.secure_uris import SecurePath

from .core.base_settings import BasicSettings
from .precision import PrecisionStrategy


class SessionSettings(BasicSettings):
    """Top-level session configuration controlling execution mode and global settings.

    This replaces the old RunSettings with a more technical name and cleaner
    separation between modes. It controls the overall execution flow and
    provides mode-specific configuration.

    Model configuration is not stored here. Use top-level [MODEL] in GeneralSettings
    for a shallow hierarchy.

    Args:
        name: Name of the session for identification
        mode: Execution mode (training, inference, testing)
        training: (Removed) Training config is defined at GeneralSettings.TRAINING
        inference: (Removed) Inference config is handled at GeneralSettings and MODEL.checkpoint
        seed: Random seed for reproducibility
        precision: Precision for computation
    """

    name: str = Field(default="session", description="Name of the session")
    inference: bool = Field(default=False, description="Run in inference mode when true")

    # Global session settings
    seed: int = Field(default=1, description="Random seed for reproducibility")
    precision: PrecisionStrategy = Field(
        default=PrecisionStrategy.FULL_32,
        description="Precision strategy for computation (default: full 32-bit)",
    )
    # Formal root directory field for path resolution (optional)
    root_dir: SecurePath | None = Field(
        default=None,
        description=(
            "Optional root directory for resolving relative paths. When provided via CLI or config,"
            " it drives all standard locations under <root>/output."
        ),
    )

    @model_validator(mode="after")
    def validate_mode_configuration(self):
        """Ensure appropriate configuration is provided for the selected mode.

        Note: The checkpoint requirement for inference is enforced in
        GeneralSettings.validate_inference_checkpoint(), where MODEL is available.
        This validator is intentionally minimal to avoid cross-model coupling.
        """
        return self

    @property
    def is_training_mode(self) -> bool:
        """True if running training (not inference)."""
        return not self.inference

    @property
    def is_inference_mode(self) -> bool:
        """True if running inference."""
        return self.inference

    @property
    def is_testing_mode(self) -> bool:
        """Testing mode is not used in the simplified model."""
        return False

    # Mode-specific configs are managed at GeneralSettings level; no per-mode payload here.

    def get_precision_strategy(self) -> PrecisionStrategy:
        """Get the precision strategy for this session.

        Implements PrecisionProvider protocol for centralized precision management.

        Returns:
            PrecisionStrategy configured for this session.
        """
        return self.precision
