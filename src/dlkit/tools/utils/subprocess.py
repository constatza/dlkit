import time
from typing import Final

import psutil
from loguru import logger

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
_DEFAULT_INITIAL_INTERVAL: Final[float] = 0.5  # seconds


# -------------------------------------------------------------------
# Pure helpers
# -------------------------------------------------------------------
def gather_process_tree(pid: int) -> list[psutil.Process]:
    """Return a list of psutil.Process for the given pid and its descendants.

    Args:
        pid: Process ID of the root process.

    Returns:
        List of psutil.Process objects including the root and its children.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess, psutil.Error, ValueError:
        return []
    return [parent] + parent.children(recursive=True)


def is_proc_alive(proc: psutil.Process) -> bool:
    """Determine if a process is still running and not a zombie.

    Args:
        proc: A psutil.Process instance.

    Returns:
        True if the process is alive and not a zombie, False otherwise.
    """
    return proc.is_running()


def get_alive_procs(procs: list[psutil.Process]) -> list[psutil.Process]:
    """Filter the given process list and return only those still alive.

    Args:
        procs: List of psutil.Process instances to check.

    Returns:
        List of processes that are still running.
    """
    return list(filter(is_proc_alive, procs))


def terminate_procs(procs: list[psutil.Process]) -> list[psutil.Process]:
    """Attempt to gracefully terminate each process in the list.

    Args:
        procs: List of psutil.Process instances to terminate.
    """
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
    gone, alive = psutil.wait_procs(procs, timeout=0.1)
    if alive:
        logger.warning(f"Failed to terminate processes: {alive}")
    return alive


def kill_procs(procs: list[psutil.Process]) -> list[psutil.Process]:
    """Forcefully kill each process in the list.

    Args:
        procs: List of psutil.Process instances to kill.
    """
    for proc in procs:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    gone, alive = psutil.wait_procs(procs, timeout=0.1)
    if alive:
        logger.error(f"Failed to kill processes: {alive}")
    return alive


# -------------------------------------------------------------------
# Action function
# -------------------------------------------------------------------
def stop_process_tree(
    pid: int,
    timeout: float,
    initial_interval: float = _DEFAULT_INITIAL_INTERVAL,
) -> None:
    """Terminate then kill a process tree with exponential backoff retries.

    Attempts to gracefully terminate the root pid and its descendants until
    `timeout` seconds elapse, doubling the wait interval after each round.
    If any processes remain, performs a hard kill, and finally reports any
    that could not be stopped.

    Args:
        pid: Process ID of the root process.
        timeout: Total time in seconds to keep retrying terminate().
        initial_interval: Starting interval in seconds between terminate() retries.
        log: Function to handle log messages (e.g., logger.info).
    """
    deadline = time.time() + timeout
    interval = initial_interval

    # Phase 1: graceful terminate() loop
    while time.time() < deadline:
        procs = gather_process_tree(pid)
        alive = get_alive_procs(procs)
        if not alive:
            logger.info(f"All processes in tree {pid} have exited gracefully.")
            return

        terminate_procs(alive)
        time.sleep(interval)
        interval *= 2  # exponential backoff

    # Phase 2: hard kill()
    procs = gather_process_tree(pid)
    alive = get_alive_procs(procs)
    if alive:
        logger.warning(
            f"Timeout reached ({timeout}s). "
            f"Attempting hard kill on remaining PIDs: {[p.pid for p in alive]}"
        )
        kill_procs(alive)
        time.sleep(0.1)

    # Phase 3: final check
    procs = gather_process_tree(pid)
    alive = get_alive_procs(procs)
    if alive:
        logger.error(f"Could not terminate the following processes: {[p.pid for p in alive]}")
    else:
        logger.info(f"Process tree {pid} stopped.")
