"""Server adapters for managing external services like MLflow."""

from .factory import (
    ServerFactory,
    create_mlflow_adapter,
    server_factory,
)
from .health_checker import CompositeHealthChecker, HTTPHealthChecker
from .mlflow_adapter import MLflowServerAdapter, MLflowServerContext
from .process_manager import SubprocessManager
from .config_normalizer import ServerConfigNormalizer
from .config_applier import ServerConfigApplier
from .storage_ensurer import ServerStorageEnsurer
from .protocols import (
    ContextualServerAdapter,
    HealthChecker,
    ProcessManager,
    ServerAdapter,
    ServerInfo,
    ServerStatus,
)
from .server_management_service import ServerManagementService
from .application_service import ServerApplicationService
from .domain_protocols import (
    ServerTracker,
    ProcessKiller,
    StorageSetup,
    UserInteraction,
    FileSystemOperations,
    ServerContextFactory,
)
from .infrastructure_adapters import MLflowContextFactory

__all__ = [
    # Factory
    "ServerFactory",
    "server_factory",
    "create_mlflow_adapter",
    # Protocols
    "ServerAdapter",
    "ContextualServerAdapter",
    "ProcessManager",
    "HealthChecker",
    "ServerInfo",
    "ServerStatus",
    # Implementations
    "MLflowServerAdapter",
    "MLflowServerContext",
    "SubprocessManager",
    "HTTPHealthChecker",
    "CompositeHealthChecker",
    # Configuration Services
    "ServerConfigNormalizer",
    "ServerConfigApplier",
    "ServerStorageEnsurer",
    # Services
    "ServerManagementService",
    "ServerApplicationService",
    # Domain protocols
    "ServerTracker",
    "ProcessKiller",
    "StorageSetup",
    "UserInteraction",
    "FileSystemOperations",
    "ServerContextFactory",
    "MLflowContextFactory",
]
