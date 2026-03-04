from __future__ import annotations

from pathlib import Path
import pathlib
import os
import shutil
import socket
from collections.abc import Iterator

from _pytest.tmpdir import TempPathFactory
import psutil
import pytest

from dlkit.tools.io import load_config
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.environment import env as global_environment

_ORIGINAL_HOME_ENV = os.environ.get("HOME")
_ORIGINAL_DLKIT_ROOT_DIR = os.environ.get("DLKIT_ROOT_DIR")
_ORIGINAL_PATH_HOME = pathlib.Path.home
_ORIGINAL_ENV_ROOT = global_environment.root_dir
_TEST_SESSION_ROOT: Path | None = None
_TEST_HOME_DIR: Path | None = None
_TEST_ARTIFACTS_DIR: Path | None = None


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Find an available TCP port on the given host.

    Returns a port number that was free at the time of asking by binding to
    port 0 and querying the assigned port, then immediately releasing it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _kill_processes_on_port(port: int) -> None:
    """Best-effort cleanup of any processes listening on a port."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections():
                if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


@pytest.fixture()
def free_port() -> Iterator[int]:
    """Provide a free localhost TCP port and ensure cleanup after test.

    Many server tests need an available port; using this fixture avoids clashes
    on shared CI runners. After the test, any listener on the port is terminated
    as a safety net to prevent leaks across tests.
    """
    port = _find_free_port()
    try:
        yield port
    finally:
        _kill_processes_on_port(port)


@pytest.fixture()
def free_ports() -> Iterator[tuple[int, int]]:
    """Provide two distinct free ports for tests that need multiple servers."""
    p1 = _find_free_port()
    p2 = _find_free_port()
    # extremely small chance of collision if a different process binds between these calls
    try:
        yield (p1, p2)
    finally:
        _kill_processes_on_port(p1)
        _kill_processes_on_port(p2)


@pytest.fixture(params=["io", "general_from_file"])
def loader_kind(request) -> str:
    return request.param


@pytest.fixture(params=[False, True])
def with_root(request) -> bool:
    return request.param


@pytest.fixture()
def env(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    root_override = tmp_path / "root_override"

    cfg_dir.mkdir(parents=True, exist_ok=True)
    root_override.mkdir(parents=True, exist_ok=True)

    # Create files in both locations so FilePath validation succeeds
    filenames = {
        "x": "X.npy",
        "y": "Y.npy",
        "split": "indices.txt",
        "ckpt": "model.ckpt",
    }

    paths = {
        "cfg_dir": cfg_dir,
        "root_override": root_override,
        "cfg": {name: cfg_dir / fname for name, fname in filenames.items()},
        "root": {name: root_override / fname for name, fname in filenames.items()},
    }

    for d in (paths["cfg"], paths["root"]):
        for p in d.values():
            p.write_text("ok")

    # Do not create default_root_dir; loader should ensure it exists

    return paths


def _write_config(config_path: Path, *, with_root: bool, env_paths: dict) -> None:
    # All references are simple relative filenames; resolution chooses the base.
    paths_block = (
        f'[PATHS]\nroot = "{env_paths["root_override"].as_posix()}"\n' if with_root else "[PATHS]\n"
    ) + (
        'output_dir = "outputs"\n'
        'checkpoints_dir = "checkpoints"\n'
        'figures_dir = "figures"\n'
        'predictions_dir = "preds"\n'
        'mlruns_dir = "mlruns"\n'
    )

    dataset_block = (
        "[DATASET]\n"
        'name = "FlexibleDataset"\n'
        'root_dir = "."\n\n'
        "[[DATASET.features]]\n"
        'name = "X"\n'
        'path = "X.npy"\n\n'
        "[[DATASET.targets]]\n"
        'name = "Y"\n'
        'path = "Y.npy"\n\n'
        "[DATASET.split]\n"
        'filepath = "indices.txt"\n'
    )

    model_block = (
        "[MODEL]\n"
        'name = "ConstantWidthFFNN"\n'
        'module_path = "dlkit.core.models.nn.ffnn.simple"\n'
        'checkpoint = "model.ckpt"\n'
    )

    session_block = '[SESSION]\nname = "test_session"\ninference = false\nseed = 42\n'

    mlflow_block = '[MLFLOW]\nenabled = true\nexperiment_name = "test_experiment"\n'

    trainer_block = '[TRAINING.trainer]\ndefault_root_dir = "work"\n'

    config_path.write_text(
        paths_block + dataset_block + model_block + session_block + mlflow_block + trainer_block
    )


@pytest.fixture()
def config_file(env: dict, with_root: bool) -> Path:
    cfg_path = env["cfg_dir"] / ("config_root.toml" if with_root else "config.toml")
    _write_config(cfg_path, with_root=with_root, env_paths=env)
    return cfg_path


@pytest.fixture()
def settings(env: dict, config_file: Path, loader_kind: str):
    if loader_kind == "io":
        return load_config(config_file)
    return GeneralSettings.from_file(config_file)


@pytest.fixture()
def expected_root(env: dict, config_file: Path, with_root: bool) -> Path:
    return (env["root_override"] if with_root else config_file.parent).resolve()


# --- Environment hardening and sandbox helpers (migrated from root conftest) ---


def _network_restricted() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.close()
        except Exception:
            pass
        return False
    except Exception:
        return True


def pytest_configure(config):
    """Configure pytest with test-specific settings.

    This hook runs before test collection and ensures that:
    1. Test environment variables are properly set
    2. Test artifacts directory is prepared
    """
    global _TEST_SESSION_ROOT, _TEST_HOME_DIR, _TEST_ARTIFACTS_DIR

    os.environ["DLKIT_TEST_MODE"] = "1"

    tmp_factory = getattr(config, "_tmp_path_factory", None)
    if tmp_factory is None:
        tmp_factory = TempPathFactory.from_config(config)
        config._tmp_path_factory = tmp_factory

    session_root = Path(str(tmp_factory.getbasetemp())) / "dlkit"
    home_dir = session_root / "home"
    artifacts_dir = session_root / "tests" / "artifacts"
    root_dir = session_root / "root"

    for directory in (home_dir, artifacts_dir, root_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(home_dir)
    os.environ["DLKIT_TEST_ARTIFACT_ROOT"] = str(artifacts_dir)
    os.environ.setdefault("DLKIT_ROOT_DIR", str(root_dir))

    pathlib.Path.home = classmethod(lambda cls, _home=home_dir: _home)  # type: ignore[assignment]
    global_environment.root_dir = str(root_dir)

    _TEST_SESSION_ROOT = session_root
    _TEST_HOME_DIR = home_dir
    _TEST_ARTIFACTS_DIR = artifacts_dir


def pytest_unconfigure(config):
    """Clean up after all tests complete.

    This hook runs after all tests and performs final cleanup
    to ensure no test artifacts remain.
    """
    # Clean up test environment variable
    os.environ.pop("DLKIT_TEST_MODE", None)
    os.environ.pop("DLKIT_TEST_ARTIFACT_ROOT", None)

    if _ORIGINAL_HOME_ENV is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _ORIGINAL_HOME_ENV

    os.environ.pop("DLKIT_ROOT_DIR", None)
    if _ORIGINAL_DLKIT_ROOT_DIR is not None:
        os.environ["DLKIT_ROOT_DIR"] = _ORIGINAL_DLKIT_ROOT_DIR

    pathlib.Path.home = _ORIGINAL_PATH_HOME

    global _TEST_SESSION_ROOT, _TEST_HOME_DIR, _TEST_ARTIFACTS_DIR
    _TEST_SESSION_ROOT = None
    _TEST_HOME_DIR = None
    _TEST_ARTIFACTS_DIR = None
    global_environment.root_dir = _ORIGINAL_ENV_ROOT


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Skip network-dependent tests in restricted sandboxes.

    We conservatively skip entire modules that require real sockets/HTTP:
    - tests/integration/test_mlflow_server_integration.py
    - tests/integration/test_server_lifecycle.py
    Additionally, any test marked `slow` is skipped by default when network is
    restricted.
    """
    if not _network_restricted():
        return

    skip_net = pytest.mark.skip(reason="Network/sockets not permitted in sandbox")
    for item in items:
        path = str(item.fspath)
        if path.endswith("test_mlflow_server_integration.py") or path.endswith(
            "test_server_lifecycle.py"
        ):
            item.add_marker(skip_net)
            continue
        # Also skip tests explicitly marked slow in restricted environments
        if any(mark.name == "slow" for mark in getattr(item, "iter_markers", lambda: [])()):
            item.add_marker(skip_net)


@pytest.fixture(autouse=True, scope="session")
def _fake_socket_when_restricted():  # noqa: D401
    """Install a minimal fake socket if sandbox denies real sockets."""
    if not _network_restricted():
        return
    _real = socket.socket

    class _FakeSocket:
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
            self._host = "127.0.0.1"
            self._port = 0

        def setsockopt(self, *args, **kwargs):
            return None

        def bind(self, address):
            host, port = address
            self._host = host or "127.0.0.1"
            self._port = int(port) if int(port) != 0 else 50000
            return None

        def listen(self, backlog):
            # No real socket operations in restricted environments.
            return None

        def getsockname(self):
            return (self._host, self._port)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _factory(family=-1, type=-1, proto=-1, fileno=None):
        return _FakeSocket(family, type, proto)

    socket.socket = _factory  # type: ignore[assignment]


@pytest.fixture(autouse=True, scope="session")
def _writable_home_dir():  # noqa: D401
    """Point HOME to a writable directory to allow file-based tracking.

    Some sandboxes disallow writing to the real home. The path is configured
    during pytest startup to ensure isolation under the session temp directory.
    """
    if _TEST_HOME_DIR is None:
        raise RuntimeError("Test home directory was not initialized")


@pytest.fixture(autouse=True, scope="session")
def _setup_test_artifacts_cleanup():  # noqa: D401
    """Ensure test artifacts directory exists and clean up after session.

    This fixture automatically:
    1. Creates tests/artifacts directory if it doesn't exist
    2. Cleans up all artifacts after the test session completes
    3. Ensures test isolation by providing a clean slate
    """
    if _TEST_ARTIFACTS_DIR is None:
        raise RuntimeError("Test artifacts directory was not initialized")

    tests_artifacts_dir = _TEST_ARTIFACTS_DIR
    tests_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Yield control to run tests
    yield

    # Cleanup: Remove all test artifacts after session
    if tests_artifacts_dir.exists():
        try:
            shutil.rmtree(tests_artifacts_dir)
            print(f"\n✓ Cleaned up test artifacts from {tests_artifacts_dir}")
        except OSError as e:
            # Log warning but don't fail tests due to cleanup issues
            print(f"\n⚠ Warning: Could not clean up test artifacts: {e}")


def _get_test_artifacts_dir() -> Path:
    """Helper to get the test artifacts directory path.

    Returns:
        Path: Path to tests/artifacts directory
    """
    if _TEST_ARTIFACTS_DIR is not None:
        return _TEST_ARTIFACTS_DIR

    explicit_root = os.environ.get("DLKIT_TEST_ARTIFACT_ROOT")
    if explicit_root:
        return Path(explicit_root)

    # Find project root (where tests/ directory exists)
    current = Path.cwd()

    # Try to find tests directory going up the directory tree
    for parent in [current] + list(current.parents):
        tests_dir = parent / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            return tests_dir / "artifacts"

    # Fallback: use current working directory + tests/artifacts
    return current / "tests" / "artifacts"


@pytest.fixture()
def test_artifacts_dir() -> Path:
    """Provide access to the test artifacts directory.

    This fixture provides tests with direct access to the artifacts
    directory for any custom artifact management needs.

    Returns:
        Path: Path to tests/artifacts directory (created if needed)
    """
    artifacts_dir = _get_test_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


# Import MLflow test fixtures
from tests.fixtures.mlflow_fixtures import (
    mlflow_global_state_isolation,
    mock_mlflow_client,
    mlflow_test_settings,
    mlflow_resource_manager,
    mock_mlflow_resource_manager,
    isolated_mlflow_tracker,
    process_leak_detector,
    thread_leak_detector,
    resource_leak_detection,
)
