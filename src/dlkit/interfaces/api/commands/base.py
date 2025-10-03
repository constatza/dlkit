"""Base command classes implementing the Command pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol


class BaseCommand[TInput, TOutput](ABC):
    """Abstract base class for all commands.

    Implements the Command pattern with:
    - Single responsibility per command
    - Dependency injection via constructor
    - Direct exception raising for errors
    """

    def __init__(self, command_name: str) -> None:
        """Initialize base command.

        Args:
            command_name: Human-readable command identifier
        """
        self.command_name = command_name

    @abstractmethod
    def execute(self, input_data: TInput, settings: BaseSettingsProtocol, **kwargs: Any) -> TOutput:
        """Execute the command.

        Args:
            input_data: Command-specific input dataflow
            settings: DLKit configuration
            **kwargs: Additional command parameters

        Returns:
            Command execution result

        Raises:
            DLKitError: On execution failure
        """

    @abstractmethod
    def validate_input(self, input_data: TInput, settings: BaseSettingsProtocol) -> None:
        """Validate command input before execution.

        Args:
            input_data: Input dataflow to validate
            settings: Configuration to validate against

        Raises:
            DLKitError: On validation failure
        """

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.command_name}')"
