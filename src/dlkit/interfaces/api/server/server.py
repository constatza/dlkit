import os
import threading
from subprocess import Popen, PIPE
from time import monotonic, sleep
from typing import Any

import requests
from loguru import logger

from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.utils.system_utils import mkdir_for_local
from dlkit.tools.utils.subprocess import stop_process_tree

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
_DEFAULT_INIT_TIMEOUT: float = 60.0  # seconds, configurable via config if you like
_DEFAULT_POLL_INTERVAL: float = 0.5  # seconds
_HEALTH_ENDPOINT: str = "/health"


def _drain_pipe(pipe, prefix: str) -> None:
    """Continuously read from a pipe so the child never blocks."""
    if pipe is None:
        return
    for line in iter(pipe.readline, b""):
        logger.debug(f"{prefix}: {line.decode(errors='replace').rstrip()}")
    pipe.close()


class ServerProcess:
    """Manage lifecycle of an MLflow server subprocess with proper health checks.

    Ensures:
    - Local directories exist
    - Server is up (HTTP 200 on /health) before returning
    - Clean shutdown with stop_process_tree

    Attributes:
        process: The subprocess handle if we spawned one, else None.
        is_active: False if we detected a server already running and skipped spawn.
        url: Base URL of the server.
    """

    def __init__(
        self,
        config: MLflowServerSettings,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        init_timeout: float = _DEFAULT_INIT_TIMEOUT,
    ) -> None:
        self._config = config
        self._poll_interval = poll_interval
        self._init_timeout = init_timeout
        self.process: Popen | None = None
        self._started_by_me: bool = False
        self.enabled = True
        self.url = f"{config.scheme}://{config.host}:{config.port}"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "ServerProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._started_by_me and self.process:
            # Optional "keep alive" sleep
            if self._config.keep_alive_interval:
                sleep(self._config.keep_alive_interval)
            self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start MLflow server if not already running.

        Raises:
            RuntimeError: If startup fails or server is unreachable after timeout.
        """
        if self.process is not None:
            raise RuntimeError("Server already started in this instance.")

        self._ensure_local_storage()

        if self._is_server_running():
            logger.info(f"Server already running at: {self.url}")
            self.enabled = False
            return

        popen_kwargs = {
            "stdout": PIPE,
            "stderr": PIPE,
            "shell": False,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        try:
            self.process = Popen(self._config.command, **popen_kwargs)
            self._started_by_me = True

            # Drain logs to avoid deadlock (on Windows especially)
            threading.Thread(
                target=_drain_pipe, args=(self.process.stdout, "mlflow-stdout"), daemon=True
            ).start()
            threading.Thread(
                target=_drain_pipe, args=(self.process.stderr, "mlflow-stderr"), daemon=True
            ).start()

            self._wait_until_healthy()

            logger.success(f"Started server with PID={self.process.pid}")
            logger.info(f"Backend store URI: {self._config.backend_store_uri}")
            logger.info(f"Artifacts destination: {self._config.artifacts_destination}")
            logger.info(f"Listening at: {self.url}")

        except Exception:
            # Ensure no orphan process on failure
            logger.exception("Failed to start MLflow server")
            self.stop()
            raise

    def stop(self) -> None:
        """Terminate the MLflow server process tree."""
        if self.process is None:
            return
        pid = self.process.pid
        stop_process_tree(pid, timeout=self._config.shutdown_timeout)
        self.process = None
        self._started_by_me = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _is_server_running(self) -> bool:
        """Return True if the /health endpoint returns HTTP 200."""
        url = f"{self.url}{_HEALTH_ENDPOINT}"
        try:
            resp = requests.get(url, timeout=self._poll_interval)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _wait_until_healthy(self) -> None:
        """Poll until server returns 200 or timeout; raise if not healthy."""
        deadline = monotonic() + self._init_timeout
        backoff = self._poll_interval

        while monotonic() < deadline:
            if self._is_server_running():
                return
            sleep(backoff)
            # Optional: exponential backoff but cap it
            backoff = min(backoff * 1.5, 10.0)

        # If we reach here, startup failed
        raise RuntimeError(
            f"MLflow server failed health check within {self._init_timeout:.1f}s at {self.url}"
        )

    def _ensure_local_storage(self) -> None:
        """Create local storage directories for backend & artifacts if needed."""
        local_hosts = {"localhost", "127.0.0.1", "::1", None}
        cfg = self._config
        for attr in ("backend_store_uri", "artifacts_destination"):
            uri = getattr(cfg, attr)
            if uri is None:
                continue
            is_file = getattr(uri, "scheme", None) == "file"
            host_local = getattr(uri, "host", None) in local_hosts
            if is_file or host_local:
                try:
                    mkdir_for_local(uri)
                except Exception as e:
                    logger.error(f"Failed to create local directory for {attr}: {e}", exc_info=True)
                    raise
