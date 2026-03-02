"""Health checking implementation for server services."""

import time
from abc import ABC, abstractmethod

import requests
from loguru import logger

from .protocols import HealthChecker, ServerStatus


class BaseHealthChecker(ABC, HealthChecker):
    """Abstract base class for health checkers using the Template Method pattern.

    This class implements the common wait/poll/backoff algorithm in `wait_for_health()`,
    while delegating the actual health check logic to subclasses via `check_health()`.

    The Template Method pattern is used here to eliminate ~60 lines of duplication
    between HTTPHealthChecker and MLflowAPIHealthChecker.

    Template Method: `wait_for_health()` - defines the algorithm skeleton
    Hook Methods: `check_health()` - implemented by subclasses for specific health checks

    Attributes:
        _request_timeout: Default timeout for individual health check requests (seconds)
        _wait_timeout: Default timeout for waiting for server to become healthy (seconds)
        _poll_interval: Default interval between health check polls (seconds)
    """

    def __init__(
        self,
        request_timeout: float = 1.0,
        wait_timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Initialize base health checker with timeout parameters.

        Args:
            request_timeout: Default timeout for individual health check requests (seconds)
            wait_timeout: Default timeout for waiting for server to become healthy (seconds)
            poll_interval: Default interval between health check polls (seconds)
        """
        self._request_timeout = request_timeout
        self._wait_timeout = wait_timeout
        self._poll_interval = poll_interval

    @abstractmethod
    def check_health(self, url: str, timeout: float | None = None) -> ServerStatus:
        """Check server health at given URL.

        This is the hook method that subclasses must implement with their
        specific health check logic.

        Args:
            url: Server URL to check
            timeout: Request timeout in seconds

        Returns:
            ServerStatus with health information
        """

    def wait_for_health(
        self,
        url: str,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> bool:
        """Wait for server to become healthy using exponential backoff.

        This is the template method that defines the algorithm skeleton.
        It uses the hook method `check_health()` for actual health checking.

        Algorithm:
        1. Initialize timeout and poll interval (with defaults)
        2. Calculate deadline for timeout
        3. Loop until deadline:
           a. Call check_health() (hook method)
           b. If healthy, return True
           c. Sleep with exponential backoff
           d. Update backoff (capped at 10.0s)
        4. If deadline reached, return False

        Args:
            url: Server URL to check
            timeout: Maximum time to wait in seconds (uses instance default if None)
            poll_interval: Time between checks in seconds (uses instance default if None)

        Returns:
            True if server became healthy within timeout, False otherwise
        """
        if timeout is None:
            timeout = self._wait_timeout
        if poll_interval is None:
            poll_interval = self._poll_interval

        timeout = float(timeout)
        backoff = float(poll_interval)
        deadline = time.monotonic() + timeout

        logger.debug(
            f"⏱️  HEALTH CHECK START: Waiting for server at {url} "
            f"(timeout: {timeout}s, poll_interval: {poll_interval}s)"
        )

        attempt = 0
        current = time.monotonic()
        while current < deadline:
            attempt += 1
            remaining = deadline - current

            # Call the hook method implemented by subclasses
            status = self.check_health(url, timeout=min(backoff, 10.0))
            check_duration = status.response_time if status.response_time is not None else 0.0

            logger.debug(
                f"🔍 Health check attempt #{attempt}: status={status.is_running}, "
                f"check_took={check_duration:.2f}s, backoff={backoff:.2f}s, remaining={remaining:.1f}s"
            )

            if status.is_running:
                elapsed = timeout - remaining
                logger.debug(
                    f"✅ Server became healthy after {max(elapsed, 0.0):.1f}s ({attempt} attempts)"
                )
                return True

            time.sleep(backoff)
            logger.debug(f"💤 Slept for {backoff:.2f}s (scheduled)")
            backoff = min(backoff * 1.5, 10.0)

            current = time.monotonic()

        logger.warning(
            f"❌ Server did not become healthy within {timeout}s at {url} (tried {attempt} times)"
        )
        return False


class HTTPHealthChecker(BaseHealthChecker):
    """Implementation of HealthChecker using HTTP requests.

    Default endpoint is '/'. MLflow's built-in server does not expose '/health'
    with a 200 status, but it returns 200 for the root path when running.

    This class inherits the wait/poll/backoff algorithm from BaseHealthChecker
    and implements only the HTTP-specific health check logic.

    Attributes:
        _health_endpoint: Health check endpoint path (e.g., '/', '/health')
    """

    def __init__(
        self,
        health_endpoint: str = "/",
        request_timeout: float = 1.0,
        wait_timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Initialize HTTP health checker.

        Args:
            health_endpoint: Health check endpoint path
            request_timeout: Default timeout for individual health check requests (seconds)
            wait_timeout: Default timeout for waiting for server to become healthy (seconds)
            poll_interval: Default interval between health check polls (seconds)
        """
        super().__init__(
            request_timeout=request_timeout,
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
        )
        self._health_endpoint = health_endpoint

    def check_health(self, url: str, timeout: float | None = None) -> ServerStatus:
        """Check server health using HTTP request.

        This is the hook method implementation for HTTP-based health checking.

        Args:
            url: Base server URL to check
            timeout: Request timeout in seconds

        Returns:
            ServerStatus with health information
        """
        health_url = f"{url.rstrip('/')}{self._health_endpoint}"

        if timeout is None:
            timeout = self._request_timeout

        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=timeout)
            response_time = time.time() - start_time

            is_running = response.status_code == 200
            error_message = None if is_running else f"HTTP {response.status_code}"

            return ServerStatus(
                is_running=is_running,
                url=url,
                response_time=response_time,
                error_message=error_message,
            )

        except requests.RequestException as e:
            logger.debug(f"Health check failed for {health_url}: {e}")
            return ServerStatus(
                is_running=False,
                url=url,
                response_time=None,
                error_message=str(e),
            )


class MLflowAPIHealthChecker(BaseHealthChecker):
    """Health checker that validates MLflow API endpoints are ready.

    This checker validates that MLflow's REST API endpoints are actually ready
    to accept requests, not just that the HTTP server is responding. This is
    important because MLflow's database migrations can take several seconds
    after the HTTP server starts.

    This class inherits the wait/poll/backoff algorithm from BaseHealthChecker
    and implements only the MLflow API-specific health check logic.

    Attributes:
        _api_endpoint: MLflow API endpoint to validate (e.g., '/api/2.0/mlflow/experiments/search')
    """

    def __init__(
        self,
        api_endpoint: str = "/api/2.0/mlflow/experiments/search",
        request_timeout: float = 2.0,
        wait_timeout: float = 25.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Initialize MLflow API health checker.

        Args:
            api_endpoint: API endpoint to validate (default: experiments search)
            request_timeout: Timeout for individual health check requests (seconds)
            wait_timeout: Timeout for waiting for API to become ready (seconds)
            poll_interval: Interval between health check polls (seconds)
        """
        super().__init__(
            request_timeout=request_timeout,
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
        )
        self._api_endpoint = api_endpoint

    def check_health(self, url: str, timeout: float | None = None) -> ServerStatus:
        """Check MLflow API health by making an API request.

        This is the hook method implementation for MLflow API-based health checking.

        Args:
            url: Base server URL to check
            timeout: Request timeout in seconds

        Returns:
            ServerStatus with API health information
        """
        api_url = f"{url.rstrip('/')}{self._api_endpoint}"

        if timeout is None:
            timeout = self._request_timeout

        try:
            start_time = time.time()
            # Use the search endpoint with max_results=1 to minimize response size
            # This is a reliable endpoint that exists in all MLflow versions
            response = requests.get(api_url, params={"max_results": 1}, timeout=timeout)
            response_time = time.time() - start_time

            # MLflow API returns 200 when ready (even if no experiments exist)
            # We accept 200 (success) as the only valid "healthy" status
            is_running = response.status_code == 200
            error_message = None if is_running else f"API returned HTTP {response.status_code}"

            return ServerStatus(
                is_running=is_running,
                url=url,
                response_time=response_time,
                error_message=error_message,
            )

        except requests.RequestException as e:
            logger.debug(f"MLflow API health check failed for {api_url}: {e}")
            return ServerStatus(
                is_running=False,
                url=url,
                response_time=None,
                error_message=str(e),
            )


class CompositeHealthChecker(HealthChecker):
    """Health checker that can combine multiple health check strategies."""

    def __init__(self, *checkers: HealthChecker) -> None:
        """Initialize with multiple health checkers.

        Args:
            *checkers: Health checker instances to use
        """
        if not checkers:
            raise ValueError("At least one health checker is required")
        self._checkers = checkers

    def check_health(self, url: str, timeout: float = 5.0) -> ServerStatus:
        """Check health using all checkers in sequence (all must pass).

        Args:
            url: Server URL to check
            timeout: Request timeout in seconds

        Returns:
            ServerStatus from the last checker if all pass, or first failure
        """
        last_status = None

        for checker in self._checkers:
            try:
                status = checker.check_health(url, timeout)
                if not status.is_running:
                    # If any checker fails, return immediately (short-circuit)
                    logger.debug(f"Health checker {checker.__class__.__name__} failed")
                    return status
                last_status = status
            except Exception as e:
                logger.debug(f"Health checker {checker.__class__.__name__} raised exception: {e}")
                return ServerStatus(
                    is_running=False,
                    url=url,
                    response_time=None,
                    error_message=str(e),
                )

        # All checkers passed - return last status
        return last_status or ServerStatus(
            is_running=False,
            url=url,
            response_time=None,
            error_message="No health checkers available",
        )

    def wait_for_health(self, url: str, timeout: float = 10.0, poll_interval: float = 0.5) -> bool:
        """Wait for server health using any available checker.

        Args:
            url: Server URL to check
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds

        Returns:
            True if any checker reports server as healthy within timeout
        """
        logger.debug(
            f"Waiting for server health at {url} using {len(self._checkers)} checkers (timeout: {timeout}s)"
        )

        deadline = time.monotonic() + float(timeout)

        for checker in self._checkers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"Composite checker timeout exhausted before {checker.__class__.__name__} could validate {url}"
                )
                return False

            if hasattr(checker, "wait_for_health"):
                if not checker.wait_for_health(
                    url,
                    timeout=remaining,
                    poll_interval=poll_interval,
                ):
                    logger.debug(
                        f"{checker.__class__.__name__}.wait_for_health reported unhealthy for {url}"
                    )
                    return False
            else:
                if not self._wait_with_check(checker, url, remaining, poll_interval):
                    return False

        logger.debug(f"All composite health checkers report healthy for {url}")
        return True

    def _wait_with_check(
        self,
        checker: HealthChecker,
        url: str,
        timeout: float,
        poll_interval: float,
    ) -> bool:
        """Fallback wait loop for checkers without wait_for_health."""
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            now = time.monotonic()
            if now >= deadline:
                logger.debug(f"{checker.__class__.__name__} wait loop timed out for {url}")
                return False

            attempt += 1
            remaining = deadline - now
            status = checker.check_health(url, timeout=min(poll_interval, 10.0))
            logger.debug(
                f"{checker.__class__.__name__} wait loop attempt #{attempt}: "
                f"status={status.is_running}, remaining={remaining:.1f}s"
            )
            if status.is_running:
                return True

            time.sleep(poll_interval)
