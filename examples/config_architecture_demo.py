"""Demo showing the new SOLID-compliant configuration architecture."""

from pathlib import Path
from typing import Protocol

# Import the new SOLID-compliant configuration system
from dlkit.tools.config import (
    # Protocols for dependency inversion
    TrainingSettingsProtocol,
    InferenceSettingsProtocol,
    SettingsLoaderProtocol,
    # Concrete implementations
    load_training_settings,
    load_inference_settings,
    create_settings_loader,
)


def demo_training_workflow(config_path: Path) -> None:
    """Demonstrate training workflow with partial loading.

    This function shows how training commands can load only the
    sections they need, following ISP and improving performance.
    """
    print("\n=== Training Workflow Demo ===")

    # Load only training-specific sections
    # Loads: SESSION, MODEL, DATAMODULE, DATASET, TRAINING, MLFLOW (optional), OPTUNA (optional), PATHS, EXTRAS
    # Excludes: Nothing (training needs all potential sections)
    settings = load_training_settings(config_path)

    print(f"Is training: {settings.is_training}")
    print(f"Has training config: {settings.has_training_config}")
    print(f"MLflow enabled: {settings.mlflow_enabled}")
    print(f"Optuna enabled: {settings.optuna_enabled}")

    if settings.has_training_config:
        training_config = settings.get_training_config()
        print(f"Training epochs: {getattr(training_config, 'epochs', 'not set')}")


def demo_inference_workflow(config_path: Path) -> None:
    """Demonstrate inference workflow with minimal loading.

    This function shows how inference commands load only what
    they need, excluding training/optimization sections per ISP.
    """
    print("\n=== Inference Workflow Demo ===")

    # Load only inference-specific sections
    # Loads: SESSION, MODEL, DATAMODULE, DATASET, PATHS, EXTRAS
    # Excludes: TRAINING, MLFLOW, OPTUNA (per Interface Segregation Principle)
    settings = load_inference_settings(config_path)

    print(f"Is inference: {settings.is_inference}")
    print(f"Has data config: {settings.has_data_config}")
    print(f"Checkpoint path: {settings.checkpoint_path}")

    # Note: These would raise AttributeError because inference settings
    # don't have training-specific methods (ISP compliance)
    # print(f"MLflow enabled: {settings.mlflow_enabled}")  # ❌ Not available
    # print(f"Has training config: {settings.has_training_config}")  # ❌ Not available


def demo_dependency_inversion() -> None:
    """Demonstrate dependency inversion with protocols.

    This shows how components can depend on abstractions
    rather than concrete settings classes.
    """
    print("\n=== Dependency Inversion Demo ===")

    def process_training_settings(settings: TrainingSettingsProtocol) -> str:
        """Function that depends on protocol, not concrete class."""
        workflow_info = f"Training workflow (MLflow: {settings.mlflow_enabled})"
        if settings.has_training_config:
            return f"{workflow_info} - Ready for training"
        return f"{workflow_info} - Missing training config"

    def process_inference_settings(settings: InferenceSettingsProtocol) -> str:
        """Function that depends on protocol, not concrete class."""
        checkpoint = settings.checkpoint_path
        return f"Inference workflow (checkpoint: {checkpoint is not None})"

    # These functions work with any implementation of the protocols
    print("Functions that depend on protocols can work with any implementation")
    print("This enables testability and flexibility")


def demo_factory_pattern(config_path: Path) -> None:
    """Demonstrate factory pattern for extensibility.

    This shows how the factory pattern makes it easy to
    add new workflow types without modifying existing code (OCP).
    """
    print("\n=== Factory Pattern Demo ===")

    # Create a factory instance
    loader = create_settings_loader()

    # Use factory methods for different workflows
    workflow_types = ["training", "inference", "general"]

    for workflow_type in workflow_types:
        try:
            # Use appropriate loader method based on workflow type
            if workflow_type == "training":
                settings = loader.load_training_settings(config_path)
            elif workflow_type == "inference":
                settings = loader.load_inference_settings(config_path)
            else:  # general
                settings = loader.load_general_settings(config_path)
            print(f"{workflow_type.capitalize()} settings loaded successfully")
            print(f"  - Type: {type(settings).__name__}")
            print(f"  - Is training: {settings.is_training}")
            print(f"  - Has data config: {settings.has_data_config}")
        except Exception as e:
            print(f"{workflow_type.capitalize()} settings failed: {e}")


def demo_command_integration() -> None:
    """Show how commands can integrate with the new architecture.

    This demonstrates the patterns that commands should follow
    to benefit from the SOLID principles.
    """
    print("\n=== Command Integration Pattern ===")

    class ModernTrainCommand:
        """Example of how commands should use the new settings architecture."""

        def __init__(self, settings_loader: SettingsLoaderProtocol) -> None:
            # Dependency injection following DIP
            self._settings_loader = settings_loader

        def execute(self, config_path: Path) -> str:
            # Load only what we need (ISP)
            settings = self._settings_loader.load_training_settings(config_path)

            # Use protocol methods (DIP)
            if not settings.has_training_config:
                return "❌ Missing training configuration"

            training_config = settings.get_training_config()
            status = f"✅ Training ready (epochs: {getattr(training_config, 'epochs', 'default')})"

            if settings.mlflow_enabled:
                status += " with MLflow tracking"

            return status

    class ModernInferenceCommand:
        """Example of how inference commands benefit from minimal loading."""

        def __init__(self, settings_loader: SettingsLoaderProtocol) -> None:
            self._settings_loader = settings_loader

        def execute(self, config_path: Path) -> str:
            # Load only inference sections (performance benefit)
            settings = self._settings_loader.load_inference_settings(config_path)

            if not settings.checkpoint_path:
                return "❌ Missing checkpoint path for inference"

            return f"✅ Inference ready (checkpoint: {settings.checkpoint_path})"

    # Demonstrate usage
    loader = create_settings_loader()
    train_cmd = ModernTrainCommand(loader)
    infer_cmd = ModernInferenceCommand(loader)

    print("Commands using dependency injection and protocol interfaces:")
    print(f"  - Train command type: {type(train_cmd).__name__}")
    print(f"  - Inference command type: {type(infer_cmd).__name__}")
    print("  - Both depend on SettingsLoaderProtocol (DIP)")
    print("  - Each loads only required sections (ISP)")


def main() -> None:
    """Run all architecture demos."""
    print("DLKit SOLID Configuration Architecture Demo")
    print("=" * 50)

    # Note: This demo assumes you have a config file available
    # For actual testing, you'd need a real config file
    config_path = Path("config.toml")  # Placeholder

    if not config_path.exists():
        print(f"⚠️  Config file {config_path} not found")
        print("This demo shows the architecture patterns without actual file loading")
        print()

    # Run architecture pattern demos
    demo_dependency_inversion()
    demo_command_integration()

    # These would require actual config files:
    if config_path.exists():
        demo_training_workflow(config_path)
        demo_inference_workflow(config_path)
        demo_factory_pattern(config_path)
    else:
        print("\n💡 To see full demos with config loading:")
        print("   1. Create a config.toml file")
        print("   2. Run this script again")


if __name__ == "__main__":
    main()