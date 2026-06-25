"""Factory for optimization services with proper dependency injection.

This factory creates optimization services using clean dependency injection
and follows SOLID principles. It replaces the mixed-concern factory pattern
with proper separation of concerns.
"""

from __future__ import annotations

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from .infrastructure import (
    InMemoryStudyRepository,
    MLflowTrackingAdapter,
    NullOptimizationBackendSession,
    NullTrackingAdapter,
    OptunaOptimizationBackendSession,
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
    IOptimizationBackendSession,
    IStudyRepository,
    OptimizationDirection,
)

logger = get_logger(__name__)


def _optuna_enabled(settings: SearchJobConfig) -> bool:
    """Return whether Optuna-backed optimization is enabled.

    For ``SearchJobConfig`` instances (new-style config), Optuna is always
    enabled — the presence of a search section implies it.  For duck-typed
    legacy settings that expose an ``OPTUNA`` attribute (used in tests),
    the ``enabled`` flag controls the decision.

    Args:
        settings: A SearchJobConfig or duck-typed legacy settings object.

    Returns:
        True when Optuna should be used for this workflow.
    """
    if isinstance(settings, SearchJobConfig):
        return True
    optuna_cfg = getattr(settings, "OPTUNA", None)
    if optuna_cfg is None:
        return False
    return bool(getattr(optuna_cfg, "enabled", False))


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
        optimization_backend_session: IOptimizationBackendSession | None = None,
        experiment_tracker: IExperimentTracker | None = None,
        config_persister: IConfigurationPersistence | None = None,
    ):
        """Initialize factory with optional dependency overrides.

        Args:
            build_factory: Training component factory
            study_repository: Study persistence implementation
            optimization_backend_session: Backend runtime coordination implementation
            experiment_tracker: Experiment tracking implementation
            config_persister: Configuration persistence implementation
        """
        self._build_factory = build_factory or BuildFactory()
        self._study_repository_override = study_repository
        self._optimization_backend_session_override = optimization_backend_session
        self._experiment_tracker_override = experiment_tracker
        self._config_persister_override = config_persister

    def create_optimization_orchestrator(
        self, settings: SearchJobConfig
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
            optimization_backend_session = self.create_optimization_backend_session(
                settings,
                study_repository,
            )
            experiment_tracker = self.create_experiment_tracker(settings)
            config_persister = self.create_config_persister(settings)

            # Create services
            study_manager = StudyManager(study_repository)
            trial_executor = TrialExecutor(self._build_factory)

            # Create orchestrator with all dependencies
            orchestrator = OptimizationOrchestrator(
                study_manager=study_manager,
                trial_executor=trial_executor,
                optimization_backend_session=optimization_backend_session,
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

    def create_optimization_strategy(self, settings: SearchJobConfig):
        """Create optimization strategy that implements IOptimizationStrategy.

        Args:
            settings: Configuration settings

        Returns:
            Strategy implementing IOptimizationStrategy interface
        """
        from .strategy import OptimizationStrategy

        return OptimizationStrategy(self, settings)

    def create_study_manager(self, settings: SearchJobConfig) -> StudyManager:
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

    def create_study_repository(self, settings: SearchJobConfig) -> IStudyRepository:
        """Create study repository based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Study repository implementation
        """
        if self._study_repository_override:
            return self._study_repository_override

        # Use Optuna repository for optimization workflows
        if _optuna_enabled(settings):
            try:
                return OptunaStudyRepository()
            except WorkflowError as e:
                logger.warning("Failed to create Optuna repository: {}; using in-memory", e)

        # Fall back to in-memory repository for testing/development
        return InMemoryStudyRepository()

    def create_optimization_backend_session(
        self,
        settings: SearchJobConfig,
        repository: IStudyRepository,
    ) -> IOptimizationBackendSession:
        """Create runtime backend coordination session for optimization execution."""
        if self._optimization_backend_session_override:
            return self._optimization_backend_session_override

        if _optuna_enabled(settings) and isinstance(repository, OptunaStudyRepository):
            return OptunaOptimizationBackendSession(repository.study_registry)

        return NullOptimizationBackendSession()

    def create_experiment_tracker(self, settings: SearchJobConfig) -> IExperimentTracker | None:
        """Create experiment tracker based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Experiment tracker implementation
        """
        if self._experiment_tracker_override:
            return self._experiment_tracker_override

        # MLflow tracking is enabled when tracking.backend == "mlflow"
        is_mlflow = settings.tracking.backend == "mlflow"
        logger.debug("MLflow tracking enabled: {}", is_mlflow)

        if is_mlflow:
            from dlkit.engine.tracking.naming import (
                determine_experiment_name,
            )

            experiment_name = determine_experiment_name(settings)
            logger.info(
                "Creating MLflow tracking adapter for optimization experiment '{}'",
                experiment_name,
            )
            return MLflowTrackingAdapter(
                mlflow_settings=settings.tracking,
                session_name=experiment_name,
            )

        # Use null tracker by default when MLflow is not enabled
        logger.debug("Using NullTrackingAdapter (MLflow not enabled)")
        return NullTrackingAdapter()

    def create_config_persister(
        self, settings: SearchJobConfig
    ) -> IConfigurationPersistence | None:
        """Create configuration persister based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Configuration persister implementation or None
        """
        if self._config_persister_override:
            return self._config_persister_override

        return TOMLConfigurationPersister()

    @staticmethod
    def extract_optimization_config(settings: SearchJobConfig) -> dict:
        """Extract optimization configuration from settings.

        Args:
            settings: SearchJobConfig instance with search section.

        Returns:
            Optimization configuration dictionary.
        """
        from dlkit.engine.tracking.naming import determine_study_name

        search = settings.search
        config: dict = {
            "n_trials": search.n_trials,
            "direction": OptimizationDirection.MINIMIZE
            if search.direction == "minimize"
            else OptimizationDirection.MAXIMIZE,
            "study_name": determine_study_name(settings, None),
        }
        if search.sampler:
            config["sampler_config"] = {
                "type": search.sampler.name,
                "params": {
                    "seed": search.sampler.seed
                    if search.sampler.seed is not None
                    else settings.run.seed,
                },
            }
        if search.pruner:
            config["pruner_config"] = {
                "type": search.pruner.name,
                "params": {},
            }
        if search.storage:
            config["storage_config"] = {
                "url": str(search.storage),
                "load_if_exists": True,
            }
        return config
