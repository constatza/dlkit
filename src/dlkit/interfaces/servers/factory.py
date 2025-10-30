"""Server factory for dependency injection and configuration."""

from __future__ import annotations

from typing import Any, Literal

from .health_checker import HTTPHealthChecker, CompositeHealthChecker
from .mlflow_adapter import MLflowServerAdapter
from .process_manager import SubprocessManager
from .protocols import HealthChecker, ProcessManager, ServerAdapter


class ServerFactory:
    """Factory for creating server adapters with appropriate dependencies."""

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
