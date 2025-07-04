import os
from signal import SIGTERM
import subprocess
from subprocess import Popen
from time import sleep

import requests
from loguru import logger
from pydantic import validate_call


from dlkit.settings import MLflowServerSettings
from dlkit.utils.system_utils import mkdir_for_local

WINDOWS_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


class ServerProcess:
    """A class to manage the MLflow server process.

    Attributes:
            config (MLflowServerSettings): The configuration for the MLflow server.
            proc (Popen | None): The subprocess object for the MLflow server.

    Methods:
            start() -> None: Starts the MLflow server process.
            stop() -> None: Stops the MLflow server process.
            __enter__() -> ServerProcess: Context manager entry method.
            __exit__(exc_type, exc, tb) -> None: Context manager exit method.
    """

    def __init__(self, config: MLflowServerSettings) -> None:
        self._config = config
        self.process: Popen | None = None
        self.is_active = (
            True  # Flag to indicate if the server is actively running in case of an existing server
        )

    def __enter__(self) -> "ServerProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process:
            sleep(self._config.keep_alive_interval)
        self.stop()

    def start(self) -> None:
        """Spawn the subprocess as a new process group."""
        if self.process is not None:
            raise RuntimeError("Server already started.")

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = WINDOWS_NEW_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        # Start the server
        if _is_server_running(self._config.host, self._config.port):
            logger.info(
                f"Server already running at {self._config.scheme}://{self._config.host}:{self._config.port}"
            )
            self.is_active = False
            return
        self.process = start_server(self._config, **popen_kwargs)
        logger.info(f"Started server PID={self.process.pid}")
        logger.info(
            f"Server running at {self._config.scheme}://{self._config.host}:{self._config.port}"
        )

    def stop(self) -> None:
        """Attempt graceful terminate, then force‐kill after timeout."""
        if not self.process:
            return
        proc: Popen = self.process
        try:
            print(f"Terminating server PID={proc.pid}...")
            # Graceful
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), SIGTERM)
            else:
                from signal import CTRL_BREAK_EVENT

                proc.send_signal(CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            proc.wait(timeout=self._config.shutdown_timeout)
        except Exception as e:
            # Force‐kill
            proc.kill()
            logger.error(e)
            raise e
        finally:
            self.process = None
            logger.info("Server stopped.")


def _is_server_running(host: str, port: int, timeout: float = 2.0, schema: str = "http") -> bool:
    """Query the MLflow /health endpoint to verify liveness.

    Args:
        host: Hostname for the MLflow server.
        port: Port number for the MLflow server.
        timeout: Seconds to wait for the HTTP response.

    Returns:
        True if the server responds with HTTP 200 OK, False otherwise.
    """
    url = f"{schema}://{host}:{port}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@validate_call
def start_server(config: MLflowServerSettings, **kwargs) -> Popen:
    """Starts the MLflow server as a subprocess.

    Args:
        config:
            host (str): Host address for the MLflow server.
            port (int): Port for the MLflow server.
            backend_store_uri (str): Backend store URI for MLflow tracking.
            artifact_root (str): Default artifact root directory or URI.
    """
    # Check if the backend store URI and artifacts destination are local
    local_hosts = ["localhost", "127.0.0.1", "::1", None]

    if config.backend_store_uri.host in local_hosts:
        mkdir_for_local(config.backend_store_uri)
    if config.artifacts_destination.host in local_hosts:
        mkdir_for_local(config.artifacts_destination)

    # Start in a new session on Unix
    mlflow_process = Popen(config.command, **kwargs)
    logger.info(f"Backend store URI: {config.backend_store_uri}")
    logger.info(f"Artifacts destination: {config.artifacts_destination}")
    return mlflow_process
