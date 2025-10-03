"""Server factory for dependency injection and configuration."""

from __future__ import annotations

from typing import Any, Literal

from .health_checker import HTTPHealthChecker, CompositeHealthChecker
from .mlflow_adapter import MLflowServerAdapter
from .process_manager import SubprocessManager
from .protocols import HealthChecker, ProcessManager, ServerAdapter


class ServerFactory:
    """Factory for creating server adapters with appropriate dependencies."""

    def __init__(self) -> None:
        """Initialize server factory."""
        self._process_managers: dict[str, ProcessManager] = {}
        self._health_checkers: dict[str, HealthChecker] = {}

    def create_process_manager(
        self,
        manager_type: Literal["subprocess"] = "subprocess",
        **kwargs: Any,
    ) -> ProcessManager:
        """Create a process manager instance.

        Args:
            manager_type: Type of process manager to create
            **kwargs: Configuration options for the process manager

        Returns:
            ProcessManager instance
        """
        if manager_type == "subprocess":
            return SubprocessManager(**kwargs)
        else:
            raise ValueError(f"Unknown process manager type: {manager_type}")

    def create_health_checker(
        self,
        checker_type: Literal["http", "composite"] = "http",
        **kwargs: Any,
    ) -> HealthChecker:
        """Create a health checker instance.

        Args:
            checker_type: Type of health checker to create
            **kwargs: Configuration options for the health checker

        Returns:
            HealthChecker instance
        """
        if checker_type == "http":
            return HTTPHealthChecker(**kwargs)
        elif checker_type == "composite":
            checkers = kwargs.get("checkers", [])
            if not checkers:
                # Default to HTTP checker if no checkers provided
                checkers = [HTTPHealthChecker()]
            return CompositeHealthChecker(*checkers)
        else:
            raise ValueError(f"Unknown health checker type: {checker_type}")

    def create_server_adapter(
        self,
        adapter_type: Literal["mlflow"] = "mlflow",
        process_manager: ProcessManager | None = None,
        health_checker: HealthChecker | None = None,
        **kwargs: Any,
    ) -> ServerAdapter:
        """Create a server adapter with dependencies.

        Args:
            adapter_type: Type of server adapter to create
            process_manager: Process manager to inject (creates default if None)
            health_checker: Health checker to inject (creates default if None)
            **kwargs: Additional configuration for the adapter

        Returns:
            ServerAdapter instance with injected dependencies
        """
        if adapter_type == "mlflow":
            # Create dependencies if not provided
            if process_manager is None:
                process_manager = self.create_process_manager("subprocess")

            if health_checker is None:
                health_checker = self.create_health_checker("http")

            return MLflowServerAdapter(
                process_manager=process_manager,
                health_checker=health_checker,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown server adapter type: {adapter_type}")

    def get_cached_process_manager(
        self,
        key: str,
        manager_type: Literal["subprocess"] = "subprocess",
        **kwargs: Any,
    ) -> ProcessManager:
        """Get a cached process manager or create a new one.

        Args:
            key: Cache key for the process manager
            manager_type: Type of process manager to create if not cached
            **kwargs: Configuration options for new process managers

        Returns:
            ProcessManager instance (cached or new)
        """
        if key not in self._process_managers:
            self._process_managers[key] = self.create_process_manager(manager_type, **kwargs)
        return self._process_managers[key]

    def get_cached_health_checker(
        self,
        key: str,
        checker_type: Literal["http", "composite"] = "http",
        **kwargs: Any,
    ) -> HealthChecker:
        """Get a cached health checker or create a new one.

        Args:
            key: Cache key for the health checker
            checker_type: Type of health checker to create if not cached
            **kwargs: Configuration options for new health checkers

        Returns:
            HealthChecker instance (cached or new)
        """
        if key not in self._health_checkers:
            self._health_checkers[key] = self.create_health_checker(checker_type, **kwargs)
        return self._health_checkers[key]

    def clear_cache(self) -> None:
        """Clear all cached instances."""
        self._process_managers.clear()
        self._health_checkers.clear()


# Global factory instance for convenience
server_factory = ServerFactory()


def create_mlflow_adapter(
    process_manager: ProcessManager | None = None,
    health_checker: HealthChecker | None = None,
    **kwargs: Any,
) -> MLflowServerAdapter:
    """Convenience function to create an MLflow server adapter.

    Args:
        process_manager: Process manager to inject
        health_checker: Health checker to inject
        **kwargs: Additional configuration for the adapter

    Returns:
        MLflowServerAdapter with injected dependencies
    """
    from typing import cast

    return cast(
        MLflowServerAdapter,
        server_factory.create_server_adapter(
            "mlflow",
            process_manager=process_manager,
            health_checker=health_checker,
            **kwargs,
        ),
    )
