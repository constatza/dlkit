"""Server management commands for DLKit CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dlkit.interfaces.servers import ServerApplicationService
from dlkit.interfaces.servers.protocols import ServerStatus


# Create server command group
app = typer.Typer(
    name="server",
    help="🖥️ Server management commands for MLflow and other services",
    no_args_is_help=True,
)

console = Console()


def _print_server_launch_summary(server_info) -> None:
    """Render launch summary and success panel for a running server."""

    console.print("🚀 Starting MLflow server...")
    console.print(f"  Host: {server_info.host}")
    console.print(f"  Port: {server_info.port}")
    console.print(f"  URL: {server_info.url}")

    if server_info.pid:
        console.print(
            f"📝 Tracking server {server_info.host}:{server_info.port} (PID: {server_info.pid})"
        )

    success_text = Text()
    success_text.append("✅ MLflow server started successfully!\n\n", style="bold green")
    success_text.append(f"URL: {server_info.url}\n", style="cyan")
    success_text.append(f"Host: {server_info.host}\n", style="white")
    success_text.append(f"Port: {server_info.port}\n", style="white")
    if server_info.pid:
        success_text.append(f"PID: {server_info.pid}\n", style="yellow")

    success_panel = Panel.fit(success_text, title="Server Started", border_style="green")
    console.print(success_panel)


def _handle_server_mode(
    app_service: ServerApplicationService, server_info, detach: bool
) -> None:
    """Handle attached and detached server modes with graceful shutdown."""

    if detach or not getattr(server_info, "process", None):
        console.print("🔗 Server is running in the background")
        return

    console.print("\n💡 Press Ctrl+C to stop the server")
    try:
        process = getattr(server_info.process, "process", None)
        if process:
            process.wait()
        else:
            console.print("⚠️ Server started but process handle not available for monitoring")
    except KeyboardInterrupt:
        console.print("\n🛑 Stopping server...")
        try:
            success, messages = app_service.stop_server(server_info.host, server_info.port)
            if success:
                console.print("✅ Server stopped successfully")
            else:
                for msg in messages:
                    console.print(f"❌ {msg}")
        except Exception as exc:  # pragma: no cover - defensive logging path
            console.print(f"❌ Error stopping server: {exc}")


def _ensure_storage_setup_at_cli_level(
    config_path: Path | None,
    host: str | None,
    port: int | None,
    backend_store_uri: str | None,
    artifacts_destination: str | None,
) -> None:
    """Handle storage setup at CLI level before calling API.

    This separates user interaction (CLI concern) from business logic (API concern).
    """
    from dlkit.interfaces.servers.domain_functions import (
        should_use_default_storage,
        get_default_mlruns_path,
    )
    from dlkit.interfaces.servers.application_service import ServerApplicationService
    from dlkit.interfaces.servers.infrastructure_adapters import (
        TyperUserInteraction,
        StandardFileSystemOperations,
    )

    # Load configuration to check if storage setup is needed
    app_service = ServerApplicationService()
    server_config = app_service._load_server_configuration(
        config_path, host, port, backend_store_uri, artifacts_destination
    )

    overrides = app_service._build_overrides_dict(
        host, port, backend_store_uri, artifacts_destination
    )

    # Check if we need to handle storage setup
    if not should_use_default_storage(server_config, overrides):
        return  # No storage setup needed

    mlruns_path = get_default_mlruns_path()
    file_system = StandardFileSystemOperations()

    if file_system.directory_exists(mlruns_path):
        return  # Storage already exists

    # Handle missing storage with user interaction at CLI level
    user_interaction = TyperUserInteraction()

    user_interaction.show_message("\n💾 MLflow Storage Setup")
    user_interaction.show_message("MLflow needs storage locations for experiments and artifacts.")
    user_interaction.show_message(f"Default location: {mlruns_path}")
    user_interaction.show_message("\nOptions:")
    user_interaction.show_message("  • Create default storage directory")
    user_interaction.show_message("  • Use --backend-store-uri to specify database location")
    user_interaction.show_message("  • Use --artifacts-destination to specify artifacts location")
    user_interaction.show_message("  • Use a configuration file with [MLFLOW.server] section")

    create_storage = user_interaction.confirm_action(
        "Create default storage directory and continue?", default=True
    )

    if create_storage:
        file_system.create_directory(mlruns_path)
        user_interaction.show_message(f"✅ Created storage directory: {mlruns_path}")
    else:
        user_interaction.show_message("Server startup cancelled")
        raise typer.Exit(1)


@app.command("start")
def start_server(
    config_path: Annotated[
        Path | None, typer.Argument(help="Optional path to configuration file")
    ] = None,
    host: Annotated[
        str | None, typer.Option("--host", "-h", help="Override server hostname")
    ] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Override server port")] = None,
    backend_store_uri: Annotated[
        str | None, typer.Option("--backend-store-uri", help="Override backend store URI")
    ] = None,
    artifacts_destination: Annotated[
        str | None, typer.Option("--artifacts-destination", help="Override artifacts destination")
    ] = None,
    detach: Annotated[
        bool, typer.Option("--detach", "-d", help="Start server in background (detached mode)")
    ] = False,
) -> None:
    """Start MLflow tracking server with optional config and overrides.

    Behavior:
    - With CONFIG: reads [MLFLOW.server] for defaults and applies CLI overrides.
    - Without CONFIG: uses sensible defaults (host=localhost, port=5000) and any overrides.

    Examples:
      dlkit server start
      dlkit server start --host 0.0.0.0 --port 8080
      dlkit server start config.toml --backend-store-uri sqlite:///mlflow.db --detach
    """
    try:
        # Display configuration loading
        if config_path is not None:
            typer.echo(f"📖 Loading configuration from: {config_path}")

        # Handle storage setup at CLI level if needed
        _ensure_storage_setup_at_cli_level(
            config_path, host, port, backend_store_uri, artifacts_destination
        )

        # Use application service for business logic (no user interaction needed)
        app_service = ServerApplicationService()
        server_info = app_service.start_server(
            config_path, host, port, backend_store_uri, artifacts_destination
        )

        _print_server_launch_summary(server_info)
        _handle_server_mode(app_service, server_info, detach)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error starting server: {e}[/red]")
        raise typer.Exit(1)


@app.command("stop")
def stop_server(
    host: Annotated[str, typer.Option("--host", "-h", help="Server hostname")] = "localhost",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 5000,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force stop even if server is not responding")
    ] = False,
) -> None:
    """Stop MLflow tracking server.

    Note: This command can only stop servers that are not running in external processes.
    For servers started with --detach or external tools, you'll need to stop them manually.

    Examples:
        dlkit server stop
        dlkit server stop --port 8080
        dlkit server stop --host 0.0.0.0 --port 8080 --force
    """
    try:
        console.print(f"🛑 Attempting to stop MLflow server at {host}:{port}")

        # Use application service for business logic
        app_service = ServerApplicationService()
        success, messages = app_service.stop_server(host, port, force)

        # Display status messages (presentation logic)
        for message in messages:
            if message.startswith("🔍") or message.startswith("✓") or message.startswith("⚠️"):
                console.print(f"  {message}")
            else:
                console.print(message)

        if success:
            console.print("📝 Removed from tracking")
            console.print("✅ MLflow server stopped successfully")
        else:
            console.print("⚠️ Could not stop all server processes")
            console.print(f"To manually stop MLflow server at {host}:{port}:")
            console.print("  • Use process management tools: pkill -f 'mlflow server'")
            console.print("  • Or kill specific processes if needed")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error stopping server: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def check_server_status(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Server hostname to check")
    ] = "localhost",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port to check")] = 5000,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed status information")
    ] = False,
) -> None:
    """Check MLflow server status and connectivity.

    Examples:
        dlkit server status
        dlkit server status --port 8080
        dlkit server status --host mlflow.company.com --port 80 --verbose
    """
    try:
        console.print(f"🔍 Checking MLflow server status at {host}:{port}")

        # Use application service for business logic
        app_service = ServerApplicationService()
        status = app_service.check_server_status(host, port)

        # Display status information (presentation logic)
        _display_server_status_table(status, host, port, verbose)

        # Set exit code based on status
        if not status.is_running:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error checking server status: {e}[/red]")
        raise typer.Exit(1)


def _display_server_status_table(status: ServerStatus, host: str, port: int, verbose: bool) -> None:
    """Helper function to display server status table."""

    table = Table(title=f"Server Status: {host}:{port}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green" if status.is_running else "red")

    # Basic status
    table.add_row("Status", "🟢 Running" if status.is_running else "🔴 Not Running")
    table.add_row("URL", status.url)

    if status.is_running and status.response_time:
        table.add_row("Response Time", f"{status.response_time:.3f}s")
    elif not status.is_running and status.error_message:
        table.add_row("Error", status.error_message)

    console.print(table)

    if verbose and status.is_running:
        console.print("\n📊 Additional Information:")
        console.print(f"  • Health endpoint: {status.url}/health")
        console.print(f"  • Web UI: {status.url}")
        console.print(f"  • API docs: {status.url}/docs")


@app.command("info")
def show_server_info(
    config_path: Annotated[
        Path | None, typer.Argument(help="Path to configuration file (optional)")
    ] = None,
) -> None:
    """Show server configuration information from settings.

    Examples:
        dlkit server info
        dlkit server info config.toml
    """
    try:
        if config_path:
            # Use typer.echo to avoid rich wrapping of long paths in tests
            typer.echo(f"📖 Loading configuration from: {config_path}")

            # Use application service for business logic
            app_service = ServerApplicationService()
            config_info = app_service.get_server_configuration_info(config_path)

            # Display server configuration (presentation logic)
            info_text = Text()
            info_text.append("🖥️ MLflow Server Configuration\n\n", style="bold blue")

            if config_info["configured"]:
                server_config = config_info["server"]
                client_config = config_info["client"]

                info_text.append("Server Settings:\n", style="bold")
                info_text.append(f"  Host: {server_config['host']}\n")
                info_text.append(f"  Port: {server_config['port']}\n")
                info_text.append(f"  Backend Store: {server_config['backend_store']}\n")
                info_text.append(f"  Artifacts: {server_config['artifacts']}\n")

                info_text.append("\nClient Settings:\n", style="bold")
                info_text.append(f"  Tracking URI: {client_config['tracking_uri']}\n")
                info_text.append(f"  Experiment: {client_config['experiment']}\n")
            else:
                info_text.append(f"{config_info['message']}\n", style="yellow")
                if "not configured" in config_info["message"]:
                    info_text.append(
                        "Add [MLFLOW] section with enabled = true to enable\n", style="dim"
                    )

            info_panel = Panel.fit(info_text, title="Server Configuration", border_style="blue")
            console.print(info_panel)

            # Exit with error code if configuration loading failed
            if (
                not config_info["configured"]
                and "Error loading configuration:" in config_info["message"]
            ):
                raise typer.Exit(1)

        else:
            console.print("💡 Server Management Commands Available:")
            console.print("  • dlkit server start <config>     - Start MLflow server")
            console.print("  • dlkit server stop              - Stop MLflow server")
            console.print("  • dlkit server status            - Check server status")
            console.print("  • dlkit server info <config>     - Show server config")
            console.print("\nUse --help with any command for detailed options")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error showing server info: {e}[/red]")
        raise typer.Exit(1)
