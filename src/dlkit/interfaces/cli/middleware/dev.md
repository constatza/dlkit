# CLI Middleware Module

## Overview
The CLI middleware module provides centralized error handling and presentation for DLKit command-line interface. It translates domain-specific errors into user-friendly, actionable messages with contextual suggestions and beautiful Rich formatting. This module implements the middleware pattern to intercept and process errors before displaying them to users.

## Architecture & Design Patterns
- **Middleware Pattern**: Intercepts errors between API layer and CLI presentation
- **Error Translation**: Converts technical errors to user-friendly messages
- **Contextual Suggestions**: Provides actionable next steps based on error type
- **Rich Formatting**: Beautiful terminal output with panels, colors, and formatting
- **Separation of Concerns**: Error handling separate from business logic

Key architectural decisions:
- All DLKitError subclasses handled uniformly
- Context-aware suggestions based on error type and context
- Clean error formatting with Rich panels
- Debugging information for unexpected errors
- Graceful keyboard interrupt handling

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `handle_api_error` | Function | Handle DLKit domain errors | `None` |
| `format_validation_error` | Function | Format Pydantic validation errors | `str` |
| `handle_keyboard_interrupt` | Function | Handle Ctrl+C gracefully | `None` |
| `handle_unexpected_error` | Function | Handle unexpected exceptions | `None` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_get_error_suggestions` | Function | Generate contextual suggestions | `list[str]` |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.domain`: Error types (`DLKitError`, `ConfigurationError`, `WorkflowError`, etc.)

### External Dependencies
- `rich`: Terminal formatting (Console, Panel, Text)

## Key Components

### Component 1: `handle_api_error`

**Purpose**: Handle DLKit API errors with formatted display, context information, and actionable suggestions.

**Parameters**:
- `error: DLKitError` - Domain error from API layer
- `console: Console` - Rich console for output

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit.interfaces.api.domain.errors import ConfigurationError
from dlkit.interfaces.cli.middleware.error_handler import handle_api_error

console = Console()

try:
    # API operation that may fail
    result = train(settings)
except ConfigurationError as e:
    handle_api_error(e, console)
    raise typer.Exit(1)
```

**Output Format**:
```
╭─ ❌ Error ────────────────────────────╮
│ ConfigurationError                    │
│ Configuration file not found          │
│                                       │
│ Context:                              │
│   config_path: /path/to/config.toml   │
│                                       │
│ 💡 Suggestions:                       │
│   • Check your configuration file...  │
│   • Validate configuration: dlkit...  │
│   • Create a template: dlkit config...│
│   • Verify file exists: /path/to/...  │
╰───────────────────────────────────────╯
```

**Implementation Notes**:
- Displays error type name prominently
- Shows error message in red
- Includes all context fields if available
- Generates contextual suggestions via `_get_error_suggestions()`
- Rich panel with red border for visual emphasis

---

### Component 2: `_get_error_suggestions`

**Purpose**: Generate helpful, actionable suggestions based on error type and context.

**Parameters**:
- `error: DLKitError` - Domain error to analyze

**Returns**: `list[str]` - List of suggestion strings

**Error Type Handling**:
- **ConfigurationError**: Validation, template creation, file verification
- **StrategyError**: Strategy validation, plugin enablement, compatibility checks
- **PluginError**: Plugin configuration, dependency installation
- **ModelStateError**: Model/dataflow configuration checks
- **WorkflowError**: Log inspection, resource verification, verbose mode

**Example**:
```python
from dlkit.interfaces.api.domain.errors import ConfigurationError
from dlkit.interfaces.cli.middleware.error_handler import _get_error_suggestions

error = ConfigurationError("Invalid TOML syntax", context={"config_path": "config.toml"})

suggestions = _get_error_suggestions(error)
# Returns:
# [
#     "Check your configuration file syntax and formatting",
#     "Validate configuration: dlkit config validate <config_file>",
#     "Create a template: dlkit config create --output config.toml",
#     "Verify file exists: config.toml"
# ]
```

**Implementation Notes**:
- Context-aware: uses error.context for specific suggestions
- Extensible: easy to add new error types
- Actionable: all suggestions include specific commands
- Progressive: general suggestions first, specific ones last

---

### Component 3: `format_validation_error`

**Purpose**: Format Pydantic validation errors in user-friendly format by simplifying technical messages.

**Parameters**:
- `error: Exception` - Validation exception (typically Pydantic)

**Returns**: `str` - Formatted error message

**Example**:
```python
from pydantic import ValidationError
from dlkit.interfaces.cli.middleware.error_handler import format_validation_error

try:
    settings = Settings(**invalid_data)
except ValidationError as e:
    formatted = format_validation_error(e)
    console.print(formatted)
```

**Implementation Notes**:
- Cleans up Pydantic's verbose error messages
- Extracts field names and constraint violations
- Simplifies "field required", "type_error", "value_error" messages
- Falls back to original message if not recognized

---

### Component 4: `handle_keyboard_interrupt`

**Purpose**: Handle keyboard interrupts (Ctrl+C) gracefully with user-friendly message.

**Parameters**:
- `console: Console` - Rich console for output

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit.interfaces.cli.middleware.error_handler import handle_keyboard_interrupt

console = Console()

try:
    long_running_operation()
except KeyboardInterrupt:
    handle_keyboard_interrupt(console)
    raise typer.Exit(0)  # Clean exit
```

**Output Format**:
```
╭─ ⚠️ Interrupted ──────────────────────╮
│ Operation cancelled by user           │
╰───────────────────────────────────────╯
```

**Implementation Notes**:
- Yellow warning color (not error red)
- Brief, non-alarming message
- Used for clean user-initiated exits

---

### Component 5: `handle_unexpected_error`

**Purpose**: Handle unexpected errors with debugging information and troubleshooting steps.

**Parameters**:
- `error: Exception` - Unexpected exception
- `console: Console` - Rich console for output

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit.interfaces.cli.middleware.error_handler import handle_unexpected_error

console = Console()

try:
    risky_operation()
except Exception as e:
    handle_unexpected_error(e, console)
    raise typer.Exit(1)
```

**Output Format**:
```
╭─ 🐛 Unexpected Error ─────────────────╮
│ Unexpected Error                      │
│ KeyError: 'missing_key'               │
│                                       │
│ 💡 Debug Steps:                       │
│   • Run with --verbose for detailed...│
│   • Check log files in output dir...  │
│   • Validate configuration: dlkit...  │
│   • Report issue with full error...   │
╰───────────────────────────────────────╯
```

**Implementation Notes**:
- Shows exception type and message
- Provides generic debugging steps
- Encourages verbose mode and logging
- Suggests bug reporting for persistent issues

## Usage Patterns

### Common Use Case 1: CLI Command Error Handling
```python
from rich.console import Console
from dlkit.interfaces.api import train
from dlkit.interfaces.api.domain.errors import DLKitError
from dlkit.interfaces.cli.middleware.error_handler import handle_api_error

console = Console()

try:
    result = train(settings)
    console.print("🎉 Training completed successfully!")
except DLKitError as e:
    handle_api_error(e, console)
    raise typer.Exit(1)
except Exception as e:
    handle_unexpected_error(e, console)
    raise typer.Exit(1)
```

### Common Use Case 2: Keyboard Interrupt Handling
```python
from dlkit.interfaces.cli.middleware.error_handler import handle_keyboard_interrupt

try:
    with Progress() as progress:
        task = progress.add_task("Training...", total=None)
        result = train(settings)
except KeyboardInterrupt:
    handle_keyboard_interrupt(console)
    raise typer.Exit(130)  # Standard Ctrl+C exit code
```

### Common Use Case 3: Validation Error Formatting
```python
from dlkit.interfaces.cli.middleware.error_handler import format_validation_error

try:
    settings = GeneralSettings.from_toml(config_path)
except ValidationError as e:
    formatted_msg = format_validation_error(e)
    console.print(f"[red]{formatted_msg}[/red]")
    raise typer.Exit(1)
```

## Error Handling

**Handled Error Types**:
- `ConfigurationError`: Configuration file/validation errors
- `StrategyError`: Strategy selection/execution errors
- `PluginError`: Plugin loading/configuration errors
- `ModelStateError`: Model initialization/state errors
- `WorkflowError`: Workflow execution errors
- `KeyboardInterrupt`: User-initiated cancellation
- `Exception`: Catch-all for unexpected errors

**Error Context Usage**:
All DLKitError exceptions include context dict for debugging:
```python
error.context = {
    "config_path": "/path/to/config.toml",
    "plugin": "optuna",
    "strategy": "mlflow",
    "available_modes": ["training", "inference"],
}
```

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/cli/middleware/test_error_handler.py`
- Integration tests: End-to-end CLI error scenarios

### Key Test Scenarios
1. **Configuration errors**: Missing files, invalid TOML, validation failures
2. **Strategy errors**: Invalid strategy names, missing plugins
3. **Plugin errors**: Plugin not enabled, dependency missing
4. **Workflow errors**: Training/optimization failures
5. **Keyboard interrupts**: Graceful Ctrl+C handling
6. **Unexpected errors**: Non-DLKit exceptions
7. **Suggestion generation**: Context-aware recommendations

### Fixtures Used
- `mock_console`: Captured Rich console output
- `sample_errors`: Various DLKitError instances
- `validation_errors`: Pydantic validation exceptions

## Performance Considerations
- Minimal overhead: only activates on error paths
- No I/O during error handling (pure formatting)
- Efficient Rich rendering with lazy formatting
- Context dict shallow copy (no deep inspection)

## Future Improvements / TODOs
- [ ] Multilingual error messages (i18n support)
- [ ] Error code system for documentation links
- [ ] Interactive error recovery prompts
- [ ] Error analytics: track common errors for UX improvements
- [ ] Stack trace capture with privacy filtering
- [ ] Colored diffs for configuration errors
- [ ] Integration with online documentation (error code → docs URL)
- [ ] Suggestion ranking based on error context similarity

## Related Modules
- `dlkit.interfaces.api.domain.errors`: Error type definitions
- `dlkit.interfaces.cli.commands`: CLI commands that use error handling
- `dlkit.interfaces.cli.adapters`: Configuration adapter that raises ConfigurationError
- `dlkit.tools.config`: Configuration loading that may raise validation errors

## Change Log
- **2025-10-03**: Comprehensive CLI middleware documentation created
- **2024-10-02**: Added strategy-specific suggestions for WorkflowError
- **2024-09-30**: Enhanced validation error formatting
- **2024-09-24**: Initial error handling middleware with Rich formatting
