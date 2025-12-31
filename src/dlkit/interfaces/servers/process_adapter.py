"""Process management adapter implementing single responsibility principle."""

from __future__ import annotations

import psutil

from dlkit.tools.utils.logging_config import get_logger
from .domain_protocols import ProcessKiller, ServerTracker
from .process_inspection import is_mlflow_process, matches_host_port

logger = get_logger(__name__)


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
    ) -> bool:
        """Stop server processes for given host:port.

        Args:
            host: Server hostname
            port: Server port
            force: Whether to force kill processes

        Returns:
            True if stopped successfully or no processes found (idempotent)
            False if operational failure (processes exist but won't stop)

        Raises:
            RuntimeError: If exceptional failure
        """
        try:
            # First try tracked processes
            tracked_pids = self._server_tracker.get_tracked_pids(host, port)
            mlflow_processes = []

            if tracked_pids:
                logger.info(f"Found {len(tracked_pids)} tracked server(s) for {host}:{port}")
                mlflow_processes = self._validate_tracked_processes(tracked_pids)

            # If no valid tracked processes, scan for MLflow processes
            if not mlflow_processes:
                logger.debug(f"Scanning for MLflow server processes on {host}:{port}")
                mlflow_processes = self._scan_for_mlflow_processes(host, port)

            if not mlflow_processes:
                logger.info(f"No MLflow server processes found for {host}:{port}")
                return True  # Idempotent - already stopped

            success = self._terminate_processes(mlflow_processes, force)
            return success

        except Exception as e:
            logger.error(f"Error stopping MLflow processes: {e}", exc_info=True)
            raise RuntimeError(f"Error stopping MLflow processes: {e}") from e

    def _validate_tracked_processes(
        self, tracked_pids: list[int]
    ) -> list[int]:
        """Validate that tracked PIDs are still MLflow servers.

        Args:
            tracked_pids: List of tracked process IDs

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
                    logger.debug(f"Verified tracked process {pid} is MLflow server")
                else:
                    logger.debug(f"Tracked process {pid} is no longer MLflow server")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.debug(f"Tracked process {pid} no longer exists")

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

    def _terminate_processes(self, pids: list[int], force: bool) -> bool:
        """Terminate the specified processes.

        Args:
            pids: List of process IDs to terminate
            force: Whether to force kill if graceful termination fails

        Returns:
            True if all processes were terminated successfully
        """
        logger.info(f"Stopping {len(pids)} MLflow server process(es)")

        # Try graceful shutdown first
        terminated_processes = []
        for pid in pids:
            try:
                logger.debug(f"Stopping process {pid}")
                proc = psutil.Process(pid)
                proc.terminate()
                terminated_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not terminate process {pid}: {e}")

        # Wait for graceful termination
        if terminated_processes:
            logger.debug("Waiting for processes to terminate gracefully")
            gone, alive = psutil.wait_procs(terminated_processes, timeout=5)

            # Force kill if needed
            if alive:
                logger.warning(f"Force killing {len(alive)} remaining processes")
                for proc in alive:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                gone2, still_alive = psutil.wait_procs(alive, timeout=2)

                if still_alive:
                    logger.error(f"Could not stop {len(still_alive)} processes")
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
            logger.error(f"{remaining} processes are still running")
            return False

        logger.info("All processes stopped successfully")
        return True
