import os
from signal import CTRL_BREAK_EVENT, SIGTERM
from subprocess import CREATE_NEW_PROCESS_GROUP, PIPE, Popen
from time import sleep

import requests
from loguru import logger
from pydantic import FileUrl, validate_call

from dlkit.settings import MLflowServerSettings
from dlkit.utils.system_utils import mkdir_for_local


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
        self._proc: Popen | None = None
        self.is_active = (
            True  # Flag to indicate if the server is actively running in case of an existing server
        )

    def __enter__(self) -> "ServerProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc:
            sleep(self._config.keep_alive_interval)
        self.stop()

    def start(self) -> None:
        """Spawn the subprocess as a new process group."""
        if self._proc is not None:
            raise RuntimeError("Server already started.")
        kwargs = {
            "stdout": PIPE,
            "stderr": PIPE,
        }
        # UNIX: new session → new process group; Windows: CREATE_NEW_PROCESS_GROUP
        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid
        else:
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        # Start the server
        if _is_server_running(self._config.host, self._config.port):
            logger.info(f"Server already running at {self._config.host}:{self._config.port}")
            self.is_active = False
            return
        self._proc = start_server(self._config, **kwargs)
        logger.info(f"Started server PID={self._proc.pid}")

    def stop(self) -> None:
        """Attempt graceful terminate, then force‐kill after timeout."""
        if not self._proc:
            return
        proc = self._proc
        print(f"Stopping server PID={proc.pid}...")
        try:
            # Graceful
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), SIGTERM)
            else:
                proc.send_signal(CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            proc.wait(timeout=self._config.shutdown_timeout)
        except Exception as e:
            # Force‐kill
            proc.kill()
            proc.wait()
            logger.error("Unexpected error .Server process killed.")
            raise e
        finally:
            self._proc = None
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
def checks_before_start(artifacts_destination: FileUrl, backend_store_uri: FileUrl) -> None:
    """Start the MLflow server with validated configuration.

    Args:
            artifacts_destination (FileUrl): The destination for MLflow artifacts.
        backend_store_uri (FileUrl): The URI for the backend store.
    """
    mkdir_for_local(artifacts_destination)
    mkdir_for_local(backend_store_uri)
    logger.info(f"Backend store URI: {backend_store_uri}")
    logger.info(f"Artifacts destination: {artifacts_destination}")


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
    if isinstance(config.backend_store_uri, FileUrl):
        if config.backend_store_uri.scheme == "file":
            checks_before_start(config.artifacts_destination, config.backend_store_uri)
        else:
            logger.info("Backend store URI is not local, no need to create directories.")
    # Start in a new session on Unix
    mlflow_process = Popen(config.command, shell=False, **kwargs)
    return mlflow_process
