"""Command dispatcher for routing and executing commands."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.utils.error_handling import raise_error
from dlkit.tools.utils.logging_config import get_logger

from .base import BaseCommand

logger = get_logger(__name__, "dispatcher")


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandRegistry:
    """Registry for command implementations.

    Provides centralized command registration and lookup
    following the Registry pattern.
    """

    commands: dict[str, type[BaseCommand[Any, Any, Any]]]

    def register(
        self, name: str, command_class: type[BaseCommand[Any, Any, Any]]
    ) -> CommandRegistry:
        """Register a command implementation.

        Args:
            name: Command identifier
            command_class: Command implementation class
        """
        logger.debug("Registering command '{}' with {}", name, command_class.__name__)
        return replace(self, commands={**self.commands, name: command_class})

    def get(self, name: str) -> type[BaseCommand[Any, Any, Any]] | None:
        """Get command class by name.

        Args:
            name: Command identifier

        Returns:
            Command class or None if not found
        """
        return self.commands.get(name)

    def list_commands(self) -> list[str]:
        """List all registered command names."""
        return list(self.commands.keys())


class CommandDispatcher:
    """Dispatcher for executing commands with proper error handling.

    Implements the Command pattern dispatcher with:
    - Command registration and lookup
    - Execution timing and metadata
    - Consistent error handling
    - Request/response logging
    """

    def __init__(self) -> None:
        """Initialize command dispatcher."""
        self.registry = CommandRegistry(commands={})

    def register_command(self, name: str, command_class: type[BaseCommand[Any, Any, Any]]) -> None:
        """Register a command for execution.

        Args:
            name: Command identifier
            command_class: Command implementation class
        """
        logger.debug("Registering command '{}' in dispatcher", name)
        self.registry = self.registry.register(name, command_class)

    def execute(
        self, command_name: str, input_data: Any, settings: BaseSettingsProtocol, **kwargs: Any
    ) -> Any:
        """Execute a registered command.

        Args:
            command_name: Name of command to execute
            input_data: Command input dataflow
            settings: DLKit configuration
            **kwargs: Additional command parameters

        Returns:
            Command execution result

        Raises:
            WorkflowError: On execution failure
        """
        logger.debug("Executing command: %s", command_name)
        try:
            # Look up command
            command_class = self.registry.get(command_name)
            if command_class is None:
                available_commands = self.registry.list_commands()
                logger.error(
                    "Command '{}' not found. Available commands: {}",
                    command_name,
                    ", ".join(available_commands),
                )
                raise_error(
                    f"Unknown command: {command_name}. Available: {', '.join(available_commands)}"
                )

            # Create command instance
            logger.debug(
                "Creating command '{}' with handler {}",
                command_name,
                command_class.__name__,
            )
            command_instance = command_class(command_name)

            # Execute command
            logger.debug("Starting command '{}'", command_name)
            result = command_instance.execute(input_data, settings, **kwargs)

            # Log successful execution
            logger.debug(
                "Command '{}' completed with result type {}",
                command_name,
                type(result).__name__,
            )

            return result

        except Exception as e:
            raise_error(f"Command execution failed: {command_name}", e)

    def list_available_commands(self) -> list[str]:
        """List all available commands."""
        return self.registry.list_commands()


# Global command dispatcher instance
_dispatcher = CommandDispatcher()


def get_dispatcher() -> CommandDispatcher:
    """Get the global command dispatcher instance."""
    return _dispatcher
