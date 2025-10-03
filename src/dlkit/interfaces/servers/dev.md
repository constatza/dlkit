# Server Management Module

## Overview
The server management module provides a SOLID-compliant architecture for managing external service servers (primarily MLflow tracking server). It implements dependency inversion, separation of concerns, and clean architecture principles to manage server lifecycle, health monitoring, process management, and storage setup through well-defined protocols and adapters.

## Architecture & Design Patterns
- **Dependency Inversion Principle (DIP)**: Protocol-based abstractions (`ServerAdapter`, `ProcessManager`, `HealthChecker`)
- **Clean Architecture**: Domain/Application/Infrastructure separation with domain protocols
- **Factory Pattern**: `ServerFactory` for creating configured adapters
- **Adapter Pattern**: `MLflowServerAdapter` adapts MLflow CLI to server protocol
- **Context Manager Pattern**: `MLflowServerContext` for guaranteed cleanup
- **Service Layer Pattern**: `ServerApplicationService` orchestrates workflows
- **Single Responsibility Principle**: Each component has focused concerns

Key architectural decisions:
- Protocols over concrete dependencies (testable, extensible)
- Separation of user interaction from business logic (CLI vs API concerns)
- Explicit dependency injection via factory methods
- Context managers for resource lifecycle management
- Process tracking for multi-server coordination

## Module Structure

### Public API (Protocols)
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ServerAdapter` | Protocol | Server lifecycle management interface | N/A |
| `ContextualServerAdapter` | Protocol | Server adapter with context manager support | N/A |
| `ProcessManager` | Protocol | Process lifecycle management interface | N/A |
| `HealthChecker` | Protocol | Server health monitoring interface | N/A |
| `ServerInfo` | Data Class | Server instance information | N/A |
| `ServerStatus` | Data Class | Server health status | N/A |

### Application Services
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ServerApplicationService` | Class | High-level server workflow orchestration | N/A |
| `ServerManagementService` | Class | Core server management business logic | N/A |

### Implementations
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `MLflowServerAdapter` | Class | MLflow server adapter implementation | N/A |
| `MLflowServerContext` | Class | Context manager for MLflow servers | N/A |
| `SubprocessManager` | Class | Subprocess-based process manager | N/A |
| `HTTPHealthChecker` | Class | HTTP-based health checker | N/A |
| `CompositeHealthChecker` | Class | Multi-strategy health checker | N/A |

### Factories
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ServerFactory` | Class | Server adapter factory with DI | `ServerAdapter` |
| `create_mlflow_adapter` | Function | Create configured MLflow adapter | `MLflowServerAdapter` |
| `create_default_mlflow_adapter` | Function | Create default MLflow adapter | `MLflowServerAdapter` |

### Domain Protocols
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ServerTracker` | Protocol | Server instance tracking | N/A |
| `ProcessKiller` | Protocol | Process termination | N/A |
| `StorageSetup` | Protocol | Storage initialization | N/A |
| `UserInteraction` | Protocol | User prompts (CLI concern) | N/A |
| `FileSystemOperations` | Protocol | File system operations | N/A |

## Dependencies

### Internal Dependencies
- `dlkit.tools.config.mlflow_settings`: MLflow configuration models
- `dlkit.tools.io.locations`: Standard path resolution
- `dlkit.interfaces.cli.adapters.config_adapter`: Configuration loading (CLI only)

### External Dependencies
- `subprocess`: Process management
- `requests`: HTTP health checks
- `psutil`: Process monitoring and termination
- `pathlib`: Path handling

## Key Components

### Component 1: `ServerApplicationService`

**Purpose**: High-level application service orchestrating server management workflows with configuration loading, storage setup, and server tracking.

**Constructor Parameters**:
- `server_adapter: ServerAdapter | None = None` - Server adapter implementation
- `server_management: ServerManagementService | None = None` - Management service
- `server_config: Any = None` - Server configuration for timeout settings

**Key Methods**:
- `start_server(config_path, host, port, backend_store_uri, artifacts_destination) -> ServerInfo` - Start server with config/overrides
- `stop_server(host, port, force) -> tuple[bool, list[str]]` - Stop server at host:port
- `check_server_status(host, port) -> ServerStatus` - Check server health
- `get_server_configuration_info(config_path) -> dict[str, Any]` - Get config details

**Returns**: Varies by method

**Raises**:
- `RuntimeError`: If server cannot be started

**Example**:
```python
from dlkit.interfaces.servers import ServerApplicationService
from pathlib import Path

# Initialize service
app_service = ServerApplicationService()

# Start server with config
server_info = app_service.start_server(
    config_path=Path("config.toml"),
    host="0.0.0.0",
    port=8080
)
print(f"Server started at {server_info.url}")

# Check server status
status = app_service.check_server_status("0.0.0.0", 8080)
print(f"Server running: {status.is_running}")

# Stop server
success, messages = app_service.stop_server("0.0.0.0", 8080)
for msg in messages:
    print(msg)
```

**Implementation Notes**:
- Loads configuration from file or creates defaults
- Delegates storage setup to `ServerManagementService`
- Tracks started servers for later management
- Untracks servers on successful stop
- Separates user interaction (CLI handles prompts)
- Returns structured results for presentation layer

---

### Component 2: `MLflowServerAdapter` / `MLflowServerContext`

**Purpose**: Concrete implementation of `ContextualServerAdapter` for MLflow tracking server with subprocess management and health monitoring.

**MLflowServerAdapter Methods**:
- `start_server(server_config, **overrides) -> ServerInfo` - Start MLflow server
- `stop_server(server_info) -> bool` - Stop MLflow server
- `check_server(host, port) -> ServerStatus` - Check server health
- `get_server_url(host, port) -> str` - Build server URL

**MLflowServerContext Usage**:
```python
from dlkit.interfaces.servers import MLflowServerContext
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

config = MLflowServerSettings(
    host="localhost",
    port=5000,
    backend_store_uri="sqlite:///mlflow.db",
    artifacts_destination="./mlruns/artifacts"
)

# Context manager ensures cleanup
with MLflowServerContext(config) as server_info:
    print(f"Server running at {server_info.url}")
    # Use server...
# Server automatically stopped on exit
```

**Implementation Notes**:
- Uses `SubprocessManager` for process lifecycle
- Uses `HTTPHealthChecker` for health monitoring
- Waits for server ready before returning from `start_server()`
- Timeout configuration via `server_config.startup_timeout`
- Process tracking for graceful shutdown
- Context manager guarantees cleanup even on exceptions

---

### Component 3: `ServerInfo` and `ServerStatus`

**Purpose**: Data classes encapsulating server instance information and health status.

**ServerInfo**:
- `process: Any` - Process handle
- `url: str` - Full server URL (e.g., "http://localhost:5000")
- `host: str` - Server hostname
- `port: int` - Server port
- `pid: int | None` - Process ID

**ServerStatus**:
- `is_running: bool` - Whether server is responding
- `url: str` - Server URL checked
- `response_time: float | None` - Response time in seconds
- `error_message: str | None` - Error if not running

**Example**:
```python
# Creating ServerInfo
server_info = ServerInfo(
    process=subprocess_handle,
    url="http://localhost:5000",
    host="localhost",
    port=5000,
    pid=12345
)

# Creating ServerStatus
status = ServerStatus(
    is_running=True,
    url="http://localhost:5000",
    response_time=0.123
)
```

---

### Component 4: `HTTPHealthChecker`

**Purpose**: Health checker implementation using HTTP requests to verify server availability.

**Methods**:
- `check_health(url, timeout=5.0) -> ServerStatus` - Single health check
- `wait_for_health(url, timeout=10.0, poll_interval=0.5) -> bool` - Wait for server ready

**Example**:
```python
from dlkit.interfaces.servers import HTTPHealthChecker

checker = HTTPHealthChecker()

# Quick health check
status = checker.check_health("http://localhost:5000", timeout=2.0)
if status.is_running:
    print(f"Server healthy (response time: {status.response_time:.3f}s)")

# Wait for server to become ready
if checker.wait_for_health("http://localhost:5000", timeout=30.0):
    print("Server is ready!")
else:
    print("Server failed to start within timeout")
```

**Implementation Notes**:
- Uses `/health` endpoint for MLflow servers
- Measures response time for monitoring
- Configurable timeouts and poll intervals
- Graceful error handling with descriptive messages
- Support for composite health checks via `CompositeHealthChecker`

---

### Component 5: `SubprocessManager`

**Purpose**: Process manager implementation using Python's subprocess module for server process lifecycle.

**Methods**:
- `start_process(config) -> subprocess.Popen` - Start server subprocess
- `stop_process(process) -> bool` - Terminate subprocess gracefully
- `is_process_running(process) -> bool` - Check process status

**Example**:
```python
from dlkit.interfaces.servers import SubprocessManager
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

manager = SubprocessManager()
config = MLflowServerSettings(host="localhost", port=5000)

# Start process
process = manager.start_process(config)
print(f"Started process PID: {process.pid}")

# Check if running
if manager.is_process_running(process):
    print("Process is running")

# Stop process
if manager.stop_process(process):
    print("Process stopped successfully")
```

**Implementation Notes**:
- Constructs MLflow CLI command from config
- Redirects stdout/stderr to subprocess.PIPE
- Graceful termination with SIGTERM
- Forceful kill with SIGKILL if graceful fails
- Platform-specific process handling (Windows vs Unix)

---

### Component 6: `ServerFactory`

**Purpose**: Factory for creating server adapters with dependency injection.

**Methods**:
- `create_adapter(server_type, config) -> ServerAdapter` - Create adapter for server type

**Example**:
```python
from dlkit.interfaces.servers import ServerFactory, server_factory
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

config = MLflowServerSettings(host="localhost", port=5000)

# Using singleton factory
adapter = server_factory.create_adapter("mlflow", config)

# Using custom factory
custom_factory = ServerFactory()
adapter = custom_factory.create_adapter("mlflow", config)

# Start server via adapter
server_info = adapter.start_server(config)
```

**Implementation Notes**:
- Singleton pattern via `server_factory` global instance
- Extensible: easy to add new server types
- Dependency injection: factory creates all dependencies
- Configured adapters ready to use

---

### Component 7: Domain Protocols for Infrastructure

**Purpose**: Protocol definitions separating domain concerns from infrastructure adapters.

**Protocols**:
- `ServerTracker`: Track running server instances (host:port → PID mapping)
- `ProcessKiller`: Terminate processes by PID
- `StorageSetup`: Initialize MLflow storage directories
- `UserInteraction`: Prompt users for input (CLI-specific)
- `FileSystemOperations`: Create directories, check existence

**Usage**:
```python
from dlkit.interfaces.servers.domain_protocols import (
    ServerTracker,
    StorageSetup,
    FileSystemOperations
)

# Infrastructure implementations injected at runtime
class ServerManagementService:
    def __init__(
        self,
        tracker: ServerTracker,
        storage_setup: StorageSetup,
        filesystem: FileSystemOperations
    ):
        self.tracker = tracker
        self.storage_setup = storage_setup
        self.filesystem = filesystem
```

**Implementation Notes**:
- Enables testing with mocks (no real file I/O or processes)
- Separates domain logic from platform-specific code
- Follows Interface Segregation Principle (focused protocols)
- Implementations in `infrastructure_adapters.py`

## Usage Patterns

### Common Use Case 1: Starting MLflow Server Programmatically
```python
from dlkit.interfaces.servers import create_default_mlflow_adapter
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

# Create adapter
adapter = create_default_mlflow_adapter()

# Configure server
config = MLflowServerSettings(
    host="localhost",
    port=5000,
    backend_store_uri="sqlite:///mlflow.db",
    artifacts_destination="./mlruns/artifacts"
)

# Start server
server_info = adapter.start_server(config)
print(f"Server running at {server_info.url} (PID: {server_info.pid})")

# Stop when done
adapter.stop_server(server_info)
```

### Common Use Case 2: Context Manager for Automatic Cleanup
```python
from dlkit.interfaces.servers import MLflowServerContext
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

config = MLflowServerSettings(host="localhost", port=5000)

with MLflowServerContext(config) as server:
    # Server automatically started
    print(f"Using server at {server.url}")

    # Run training with MLflow tracking
    result = train_model()
# Server automatically stopped on exit
```

### Common Use Case 3: Application Service for Complete Workflows
```python
from dlkit.interfaces.servers import ServerApplicationService
from pathlib import Path

service = ServerApplicationService()

# Start from config file with overrides
server_info = service.start_server(
    config_path=Path("config.toml"),
    host="0.0.0.0",  # Override config
    port=8080        # Override config
)

# Check status
status = service.check_server_status("0.0.0.0", 8080)
print(f"Server healthy: {status.is_running}")

# Stop server
success, messages = service.stop_server("0.0.0.0", 8080)
```

### Common Use Case 4: Health Monitoring
```python
from dlkit.interfaces.servers import HTTPHealthChecker

checker = HTTPHealthChecker()

# Wait for server to be ready (e.g., after startup)
url = "http://localhost:5000"
if checker.wait_for_health(url, timeout=30.0, poll_interval=1.0):
    print("Server is ready to accept requests!")

    # Periodic health checks
    status = checker.check_health(url, timeout=2.0)
    if status.is_running:
        print(f"Server response time: {status.response_time:.3f}s")
```

## Error Handling

**Exceptions Raised**:
- `RuntimeError`: Server startup failures, process management errors
- `TimeoutError`: Health check timeouts, server startup timeouts
- `subprocess.SubprocessError`: Process spawning failures
- `requests.RequestException`: HTTP health check failures
- `OSError`: File system operations (storage setup)

**Error Handling Pattern**:
```python
try:
    server_info = adapter.start_server(config)
except RuntimeError as e:
    logger.error(f"Failed to start server: {e}")
    # Attempt cleanup
    try:
        adapter.stop_server(server_info)
    except Exception:
        pass
    raise
except TimeoutError as e:
    logger.error(f"Server startup timeout: {e}")
    # Kill process if started
    raise
```

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/servers/test_domain_functions.py`
- Unit tests: `tests/interfaces/servers/test_server_management.py`
- Unit tests: `tests/interfaces/servers/test_solid_adapters.py`
- Integration tests: `tests/integration/test_mlflow_server_behavior.py`

### Key Test Scenarios
1. **Server lifecycle**: Start, health check, stop
2. **Context manager**: Automatic cleanup on exit/exception
3. **Process management**: Subprocess creation, termination
4. **Health checking**: HTTP checks, timeouts, retries
5. **Configuration**: Loading, overrides, defaults
6. **Error handling**: Startup failures, cleanup on error
7. **Storage setup**: Directory creation, permissions

### Fixtures Used
- `mock_process_manager`: Mocked process operations
- `mock_health_checker`: Mocked health checks
- `tmp_mlruns_dir`: Temporary MLflow storage
- `mlflow_config`: Test server configurations

## Performance Considerations
- Health checks: Configurable timeouts (default 5s)
- Poll intervals: Adjustable for balance between responsiveness and load
- Process cleanup: Graceful termination before forceful kill
- Server tracking: In-memory dict (fast lookup)
- Startup timeout: Default 30s, configurable per environment

## Future Improvements / TODOs
- [ ] Support for additional server types (TensorBoard, Weights & Biases)
- [ ] Distributed server management (multiple hosts)
- [ ] Server clustering and load balancing
- [ ] Persistent server tracking (survive process restarts)
- [ ] Advanced health checks (custom endpoints, multiple checks)
- [ ] Server metrics collection (uptime, request counts)
- [ ] Auto-restart on failure
- [ ] Server configuration validation before startup

## Related Modules
- `dlkit.tools.config.mlflow_settings`: MLflow configuration models
- `dlkit.interfaces.cli.commands.server`: CLI commands using these services
- `dlkit.runtime.workflows.strategies.tracking`: MLflow integration for training
- `dlkit.tools.io.locations`: Standard path resolution for storage

## Change Log
- **2025-10-03**: Comprehensive server management documentation created
- **2024-10-02**: Added composite health checker for multi-strategy checks
- **2024-09-30**: Refactored to clean architecture with domain protocols
- **2024-09-24**: Initial server management with SOLID principles
