"""Factory for optimization services with proper dependency injection.

This factory creates optimization services using clean dependency injection
and follows SOLID principles. It replaces the mixed-concern factory pattern
with proper separation of concerns.
"""

from __future__ import annotations

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.utils.logging_config import get_logger

from .infrastructure import (
    InMemoryStudyRepository,
    MLflowTrackingAdapter,
    NullConfigurationPersister,
    NullTrackingAdapter,
    OptunaStudyRepository,
    TOMLConfigurationPersister,
)
from .services import (
    OptimizationOrchestrator,
    StudyManager,
    TrialExecutor,
)
from .value_objects import (
    IConfigurationPersistence,
    IExperimentTracker,
    IStudyRepository,
    OptimizationDirection,
)

# Settings union accepted by optimization factory methods
type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig

logger = get_logger(__name__)


class OptimizationServiceFactory:
    """Factory for creating optimization services with dependency injection.

    This factory follows SOLID principles:
    - Single Responsibility: Only creates optimization services
    - Open/Closed: New implementations can be added without modification
    - Liskov Substitution: All implementations are interchangeable
    - Interface Segregation: Uses focused protocols
    - Dependency Inversion: Depends on abstractions, not concretions

    It replaces the mixed-concern factory pattern with clean DI.
    """

    def __init__(
        self,
        build_factory: BuildFactory | None = None,
        study_repository: IStudyRepository | None = None,
        experiment_tracker: IExperimentTracker | None = None,
        config_persister: IConfigurationPersistence | None = None,
    ):
        """Initialize factory with optional dependency overrides.

        Args:
            build_factory: Training component factory
            study_repository: Study persistence implementation
            experiment_tracker: Experiment tracking implementation
            config_persister: Configuration persistence implementation
        """
        self._build_factory = build_factory or BuildFactory()
        self._study_repository_override = study_repository
        self._experiment_tracker_override = experiment_tracker
        self._config_persister_override = config_persister

    def create_optimization_orchestrator(
        self, settings: _WorkflowSettings
    ) -> OptimizationOrchestrator:
        """Create optimization orchestrator with proper dependency injection.

        Args:
            settings: Configuration settings

        Returns:
            Freshly configured optimization orchestrator

        Raises:
            WorkflowError: If required dependencies cannot be created
        """
        try:
            # Create dependencies
            study_repository = self.create_study_repository(settings)
            experiment_tracker = self.create_experiment_tracker(settings)
            config_persister = self.create_config_persister(settings)

            # Create services
            study_manager = StudyManager(study_repository)
            trial_executor = TrialExecutor(self._build_factory)

            # Create orchestrator with all dependencies
            orchestrator = OptimizationOrchestrator(
                study_manager=study_manager,
                trial_executor=trial_executor,
                experiment_tracker=experiment_tracker,
                config_persister=config_persister,
            )

            logger.debug("Optimization orchestrator created successfully")
            return orchestrator

        except Exception as e:
            logger.error("Failed to create optimization orchestrator: {}", e)
            raise WorkflowError(
                f"Orchestrator creation failed: {e}", {"stage": "orchestrator_creation"}
            ) from e

    def create_optimization_strategy(self, settings: _WorkflowSettings):
        """Create optimization strategy that implements IOptimizationStrategy.

        Args:
            settings: Configuration settings

        Returns:
            Strategy implementing IOptimizationStrategy interface
        """
        from .strategy import OptimizationStrategy

        return OptimizationStrategy(self, settings)

    def create_study_manager(self, settings: _WorkflowSettings) -> StudyManager:
        """Create study manager service.

        Args:
            settings: Configuration settings

        Returns:
            Configured study manager
        """
        repository = self.create_study_repository(settings)
        return StudyManager(repository)

    def create_trial_executor(self) -> TrialExecutor:
        """Create trial executor service.

        Returns:
            Configured trial executor
        """
        return TrialExecutor(self._build_factory)

    def create_study_repository(self, settings: _WorkflowSettings) -> IStudyRepository:
        """Create study repository based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Study repository implementation
        """
        if self._study_repository_override:
            return self._study_repository_override

        # Check if Optuna is enabled and available
        optuna_config = getattr(settings, "OPTUNA", None)
        if optuna_config and getattr(optuna_config, "enabled", False):
            try:
                return OptunaStudyRepository()
            except WorkflowError as e:
                logger.warning("Failed to create Optuna repository: {}; using in-memory", e)

        # Fall back to in-memory repository for testing/development
        return InMemoryStudyRepository()

    def create_experiment_tracker(self, settings: _WorkflowSettings) -> IExperimentTracker | None:
        """Create experiment tracker based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Experiment tracker implementation
        """
        if self._experiment_tracker_override:
            return self._experiment_tracker_override

        # Check if MLflow tracking is configured (presence of section enables it)
        mlflow_config = getattr(settings, "MLFLOW", None)
        logger.debug("MLflow config present: {}", mlflow_config is not None)

        if mlflow_config:
            from dlkit.engine.tracking.naming import (
                determine_experiment_name,
            )

            experiment_name = determine_experiment_name(settings, mlflow_config)
            logger.info(
                "Creating MLflow tracking adapter for optimization experiment '{}'",
                experiment_name,
            )
            session = getattr(settings, "SESSION", None)
            root_dir = getattr(session, "root_dir", None) if session is not None else None
            return MLflowTrackingAdapter(
                mlflow_settings=mlflow_config,
                session_name=experiment_name,
                root_dir=root_dir,
            )

        # Use null tracker by default when MLflow is not enabled
        logger.debug("Using NullTrackingAdapter (MLflow not enabled)")
        return NullTrackingAdapter()

    def create_config_persister(
        self, settings: _WorkflowSettings
    ) -> IConfigurationPersistence | None:
        """Create configuration persister based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Configuration persister implementation or None
        """
        if self._config_persister_override:
            return self._config_persister_override

        # Check if configuration persistence is enabled
        # For now, default to TOML persistence if optimization is enabled
        optuna_config = getattr(settings, "OPTUNA", None)
        if optuna_config and getattr(optuna_config, "enabled", False):
            # Check for explicit persistence configuration
            persistence_config = getattr(optuna_config, "persistence", None)
            if persistence_config and not getattr(persistence_config, "enabled", True):
                return NullConfigurationPersister()

            return TOMLConfigurationPersister()

        # Use null persister when optimization is disabled
        return NullConfigurationPersister()

    @staticmethod
    def extract_optimization_config(settings: _WorkflowSettings) -> dict:
        """Extract optimization configuration from settings.

        Args:
            settings: Configuration settings

        Returns:
            Optimization configuration dictionary
        """
        optuna_config = getattr(settings, "OPTUNA", None)
        if not optuna_config:
            raise WorkflowError("OPTUNA configuration not found", {"stage": "config_extraction"})

        if not getattr(optuna_config, "enabled", False):
            raise WorkflowError(
                "OPTUNA is not enabled in configuration", {"stage": "config_extraction"}
            )

        # Extract optimization parameters
        config = {
            "n_trials": getattr(optuna_config, "n_trials", 10),
            "direction": OptimizationDirection.MINIMIZE
            if getattr(optuna_config, "direction", "minimize") == "minimize"
            else OptimizationDirection.MAXIMIZE,
        }

        # Determine study name (study gets run name)
        from dlkit.engine.tracking.naming import determine_study_name

        config["study_name"] = determine_study_name(settings, optuna_config)

        # Extract sampler configuration
        if hasattr(optuna_config, "sampler") and optuna_config.sampler:
            sampler_params = optuna_config.sampler.get_init_kwargs()

            # Inject SESSION.seed if sampler seed is not specified
            if "seed" not in sampler_params or sampler_params.get("seed") is None:
                session_config = getattr(settings, "SESSION", None)
                if session_config and hasattr(session_config, "seed"):
                    sampler_params["seed"] = session_config.seed

            config["sampler_config"] = {
                "type": optuna_config.sampler.name,
                "params": sampler_params,
            }

        # Extract pruner configuration
        if hasattr(optuna_config, "pruner") and optuna_config.pruner:
            config["pruner_config"] = {
                "type": optuna_config.pruner.name,
                "params": optuna_config.pruner.get_init_kwargs(),
            }

        # Extract storage configuration
        if hasattr(optuna_config, "storage") and optuna_config.storage:
            config["storage_config"] = {
                "url": str(optuna_config.storage),
                "load_if_exists": getattr(optuna_config, "load_if_exists", True),
            }

        return config
