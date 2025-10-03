"""Process management implementation for server processes."""

import os
import threading
import time
from subprocess import PIPE, Popen
from typing import cast, Any
from dataclasses import dataclass, field

from loguru import logger

from dlkit.tools.utils.subprocess import stop_process_tree

from .protocols import ProcessManager


@dataclass
class _ProcessContext:
    """Context for managing process and associated threads."""

    process: Popen[bytes]
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    def signal_shutdown(self) -> None:
        """Signal all threads to shutdown."""
        self.shutdown_event.set()

    def wait_for_threads(self, timeout: float = 5.0) -> bool:
        """Wait for all threads to finish.

        Args:
            timeout: Maximum time to wait for threads

        Returns:
            True if all threads finished within timeout
        """
        threads = [t for t in [self.stdout_thread, self.stderr_thread] if t and t.is_alive()]

        for thread in threads:
            thread.join(timeout / len(threads) if threads else timeout)

        # Check if any threads are still alive
        alive_threads = [t for t in threads if t.is_alive()]
        if alive_threads:
            logger.warning(f"Some log draining threads did not finish: {len(alive_threads)} still alive")
            return False

        return True


def _drain_pipe_with_shutdown(pipe: Any, prefix: str, shutdown_event: threading.Event) -> None:
    """Continuously read from a pipe with coordinated shutdown support.

    This function replaces the daemon thread approach with a proper shutdown
    coordination mechanism that prevents deadlocks during test teardown.

    Args:
        pipe: Pipe to drain
        prefix: Prefix for log messages
        shutdown_event: Event to signal shutdown
    """
    if pipe is None:
        return

    import select

    try:
        # Use select with timeout to make readline() interruptible
        # This allows checking shutdown_event periodically
        while not shutdown_event.is_set():
            try:
                # On Unix, use select to check if data is available with a timeout
                # On Windows, select doesn't work with pipes, so we use a different approach
                if os.name != 'nt' and hasattr(select, 'select'):
                    # Wait up to 0.1 seconds for data to be available
                    ready, _, _ = select.select([pipe], [], [], 0.1)
                    if not ready:
                        # No data available, check shutdown_event again
                        continue

                    line = pipe.readline()
                else:
                    # Windows or no select support: use readline with periodic checks
                    # We can't make this truly non-blocking, but at least check shutdown frequently
                    line = pipe.readline()

                if not line:  # EOF reached
                    break

                try:
                    decoded = line.decode(errors="replace").rstrip()
                except Exception:
                    decoded = str(line).rstrip()

                if decoded:  # Don't log empty lines
                    msg = f"{prefix}: {decoded}"
                    # Log server output at DEBUG level (verbose subprocess output)
                    logger.debug(msg)

            except Exception as e:
                if not shutdown_event.is_set():
                    logger.debug(f"Error reading from pipe {prefix}: {e}")
                break

    except Exception as e:
        logger.debug(f"Error in pipe draining {prefix}: {e}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        logger.debug(f"Pipe draining thread {prefix} finished")


class SubprocessManager(ProcessManager):
    """Implementation of ProcessManager using Python subprocess with coordinated shutdown.

    This implementation fixes the daemon thread issues that cause test hanging
    by using proper thread coordination and shutdown signaling.
    """

    def __init__(self, shutdown_timeout: float = 10.0) -> None:
        """Initialize subprocess manager.

        Args:
            shutdown_timeout: Timeout for process shutdown in seconds
        """
        self._shutdown_timeout = shutdown_timeout
        self._process_contexts: dict[int, _ProcessContext] = {}
        self._lock = threading.Lock()

    def start_process(self, config: Any) -> Popen[bytes]:
        """Start a server process using subprocess with coordinated log draining.

        Args:
            config: Server configuration with 'command' attribute

        Returns:
            Popen process handle

        Raises:
            RuntimeError: If process startup fails
        """
        popen_kwargs = {
            "stdout": PIPE,
            "stderr": PIPE,
            "shell": False,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        try:
            logger.debug(f"SubprocessManager: Starting process with command: {' '.join(config.command)}")
            process = Popen(config.command, **popen_kwargs)

            # Create process context for coordinated management
            context = _ProcessContext(process=process)

            # Start NON-DAEMON threads with shutdown coordination
            prefix = f"{config.__class__.__name__.lower()}"

            context.stdout_thread = threading.Thread(
                target=_drain_pipe_with_shutdown,
                args=(process.stdout, f"{prefix}-stdout", context.shutdown_event),
                daemon=False,  # NOT daemon - allows proper cleanup
                name=f"{prefix}-stdout-{process.pid}"
            )
            context.stdout_thread.start()

            context.stderr_thread = threading.Thread(
                target=_drain_pipe_with_shutdown,
                args=(process.stderr, f"{prefix}-stderr", context.shutdown_event),
                daemon=False,  # NOT daemon - allows proper cleanup
                name=f"{prefix}-stderr-{process.pid}"
            )
            context.stderr_thread.start()

            # Register context for cleanup
            with self._lock:
                self._process_contexts[process.pid] = context

            logger.debug(f"Started process with PID={process.pid} and coordinated log draining")
            return cast(Popen[bytes], process)

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            raise RuntimeError(f"Process startup failed: {e}") from e

    def stop_process(self, process: Popen[bytes]) -> bool:
        """Stop a server process and coordinate cleanup of associated threads.

        Args:
            process: Process handle to stop

        Returns:
            True if stopped successfully
        """
        if process is None:
            return True

        try:
            pid = process.pid
            logger.debug(f"Stopping process PID={pid}")

            # Get and remove process context
            context = None
            with self._lock:
                context = self._process_contexts.pop(pid, None)

            # Signal threads to shutdown before stopping process
            if context:
                logger.debug(f"Signaling log draining threads to shutdown for PID={pid}")
                context.signal_shutdown()

            # Stop the process tree
            stop_process_tree(pid, timeout=self._shutdown_timeout)

            # Wait for log draining threads to finish
            if context:
                logger.debug(f"Waiting for log draining threads to finish for PID={pid}")
                if not context.wait_for_threads(timeout=min(5.0, self._shutdown_timeout)):
                    logger.warning(f"Some log draining threads did not finish cleanly for PID={pid}")

            logger.debug(f"Stopped process tree for PID={pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop process: {e}")
            return False

    def is_process_running(self, process: Popen[bytes]) -> bool:
        """Check if a process is still running.

        Args:
            process: Process handle to check

        Returns:
            True if process is running
        """
        if process is None:
            return False

        try:
            # poll() returns None if process is still running
            return process.poll() is None
        except Exception:
            return False

    def cleanup_all_processes(self) -> None:
        """Clean up all managed processes and their threads.

        This method is useful for test teardown and emergency cleanup.
        """
        logger.debug("Cleaning up all managed processes")

        with self._lock:
            contexts = list(self._process_contexts.values())
            pids = list(self._process_contexts.keys())
            self._process_contexts.clear()

        # Signal all threads to shutdown
        for context in contexts:
            context.signal_shutdown()

        # Stop all processes
        for context in contexts:
            try:
                if context.process.poll() is None:  # Still running
                    stop_process_tree(context.process.pid, timeout=self._shutdown_timeout / len(contexts))
            except Exception as e:
                logger.warning(f"Error stopping process {context.process.pid}: {e}")

        # Wait for all threads to finish
        for context in contexts:
            try:
                context.wait_for_threads(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error waiting for threads: {e}")

        logger.debug(f"Cleanup completed for {len(pids)} processes")

    def __del__(self) -> None:
        """Cleanup on deletion - ensure no resources leak."""
        try:
            self.cleanup_all_processes()
        except Exception:
            pass
