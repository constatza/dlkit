"""Process management adapter implementing single responsibility principle."""

from __future__ import annotations

import psutil

from .domain_protocols import ProcessKiller, ServerTracker
from .process_inspection import is_mlflow_process, matches_host_port


class PsutilProcessKiller(ProcessKiller):
    """Process killing implementation using psutil (SRP: Only handles process termination)."""

    def __init__(self, server_tracker: ServerTracker) -> None:
        """Initialize with dependency injection.

        Args:
            server_tracker: Server tracking implementation
        """
        self._server_tracker = server_tracker

    def stop_server_processes(
        self, host: str, port: int, force: bool = False
    ) -> tuple[bool, list[str]]:
        """Stop server processes for given host:port.

        Args:
            host: Server hostname
            port: Server port
            force: Whether to force kill processes

        Returns:
            Tuple of (success, status_messages)
        """
        if psutil is None:
            return False, ["psutil package not available - cannot stop processes automatically"]

        try:
            messages = []

            # First try tracked processes
            tracked_pids = self._server_tracker.get_tracked_pids(host, port)
            mlflow_processes = []

            if tracked_pids:
                messages.append(f"Found {len(tracked_pids)} tracked server(s) for {host}:{port}")
                mlflow_processes = self._validate_tracked_processes(tracked_pids, messages)

            # If no valid tracked processes, scan for MLflow processes
            if not mlflow_processes:
                messages.append(f"Scanning for MLflow server processes on {host}:{port}...")
                mlflow_processes = self._scan_for_mlflow_processes(host, port)

            if not mlflow_processes:
                messages.append(f"No MLflow server processes found for {host}:{port}")
                return True, messages

            success = self._terminate_processes(mlflow_processes, force, messages)
            return success, messages

        except Exception as e:
            return False, [f"Error stopping MLflow processes: {e}"]

    def _validate_tracked_processes(
        self, tracked_pids: list[int], messages: list[str]
    ) -> list[int]:
        """Validate that tracked PIDs are still MLflow servers.

        Args:
            tracked_pids: List of tracked process IDs
            messages: List to append status messages to

        Returns:
            List of valid MLflow process IDs
        """

        valid_processes = []

        for pid in tracked_pids:
            try:
                proc = psutil.Process(pid)
                cmdline = proc.cmdline()

                if is_mlflow_process(cmdline):
                    valid_processes.append(pid)
                    messages.append(f"  ✓ Verified tracked process {pid} is MLflow server")
                else:
                    messages.append(f"  Tracked process {pid} is no longer MLflow server")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                messages.append(f"  ⚠️ Tracked process {pid} no longer exists")

        return valid_processes

    def _scan_for_mlflow_processes(self, host: str, port: int) -> list[int]:
        """Scan for MLflow server processes on specified host:port.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            List of MLflow process IDs
        """

        mlflow_processes = []

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if not cmdline:
                    continue

                # Must be MLflow process and match host:port
                if is_mlflow_process(cmdline) and matches_host_port(cmdline, host, port):
                    mlflow_processes.append(proc.info["pid"])

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return mlflow_processes

    def _terminate_processes(self, pids: list[int], force: bool, messages: list[str]) -> bool:
        """Terminate the specified processes.

        Args:
            pids: List of process IDs to terminate
            force: Whether to force kill if graceful termination fails
            messages: List to append status messages to

        Returns:
            True if all processes were terminated successfully
        """

        messages.append(f"Found {len(pids)} MLflow server process(es) to stop")

        # Try graceful shutdown first
        terminated_processes = []
        for pid in pids:
            try:
                messages.append(f"  Stopping process {pid}...")
                proc = psutil.Process(pid)
                proc.terminate()
                terminated_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                messages.append(f"  ⚠️ Could not terminate process {pid}: {e}")

        # Wait for graceful termination
        if terminated_processes:
            messages.append("⏳ Waiting for processes to terminate gracefully...")
            gone, alive = psutil.wait_procs(terminated_processes, timeout=5)

            # Force kill if needed
            if alive:
                messages.append(f"🔨 Force killing {len(alive)} remaining processes...")
                for proc in alive:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                gone2, still_alive = psutil.wait_procs(alive, timeout=2)

                if still_alive:
                    messages.append(f"❌ Could not stop {len(still_alive)} processes")
                    return False

        # Final verification
        remaining = 0
        for pid in pids:
            try:
                psutil.Process(pid)
                remaining += 1
            except psutil.NoSuchProcess:
                pass

        if remaining > 0:
            messages.append(f"❌ {remaining} processes are still running")
            return False

        return True
