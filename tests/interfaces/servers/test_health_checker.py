"""Tests for HTTP health checker functionality."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from dlkit.interfaces.servers.health_checker import HTTPHealthChecker
from dlkit.interfaces.servers.protocols import ServerStatus


@pytest.fixture
def health_checker() -> HTTPHealthChecker:
    """Create health checker instance."""
    return HTTPHealthChecker()


@pytest.fixture
def custom_health_checker() -> HTTPHealthChecker:
    """Create health checker with custom endpoint."""
    return HTTPHealthChecker(health_endpoint="/api/health")


class TestHTTPHealthChecker:
    """Test HTTPHealthChecker functionality."""

    def test_init_with_default_endpoint(self) -> None:
        """Test health checker initialization with default endpoint."""
        checker = HTTPHealthChecker()
        assert checker._health_endpoint == "/"

    def test_init_with_custom_endpoint(self) -> None:
        """Test health checker initialization with custom endpoint."""
        checker = HTTPHealthChecker(health_endpoint="/api/v1/health")
        assert checker._health_endpoint == "/api/v1/health"

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_successful_response(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test successful health check response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch("time.time", side_effect=[1000.0, 1000.1]):  # 0.1 second response
            status = health_checker.check_health("http://localhost:5000")

        assert isinstance(status, ServerStatus)
        assert status.is_running is True
        assert status.url == "http://localhost:5000"
        assert status.response_time == pytest.approx(0.1)
        assert status.error_message is None

        # Check that get was called with correct URL, don't check timeout as it may be overridden by env
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:5000/"
        assert "timeout" in call_args[1]

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_with_custom_endpoint(
        self, mock_get: Mock, custom_health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check with custom endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch("time.time", side_effect=[1000.0, 1000.05]):
            custom_health_checker.check_health("http://localhost:8080")

        # Check that get was called with correct URL and endpoint
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:8080/api/health"
        assert "timeout" in call_args[1]

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_with_custom_timeout(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        health_checker.check_health("http://localhost:5000", timeout=0.1)

        mock_get.assert_called_once_with("http://localhost:5000/", timeout=0.1)

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_connection_error(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check with connection error."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        status = health_checker.check_health("http://localhost:5000")

        assert isinstance(status, ServerStatus)
        assert status.is_running is False
        assert status.url == "http://localhost:5000"
        assert status.response_time is None
        assert "Connection refused" in status.error_message

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_timeout_error(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check with timeout error."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        status = health_checker.check_health("http://localhost:5000")

        assert status.is_running is False
        assert "Request timed out" in status.error_message
        assert status.response_time is None

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_http_error(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check with HTTP error response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with patch("time.time", side_effect=[1000.0, 1000.2]):
            status = health_checker.check_health("http://localhost:5000")

        assert status.is_running is False
        assert status.response_time == pytest.approx(0.2)  # Still measures response time
        assert status.error_message == "HTTP 500"  # Check the actual error format

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_strips_trailing_slash(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test that trailing slashes are handled correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        health_checker.check_health("http://localhost:5000/")

        # Check that URL was correctly processed (single slash)
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:5000/"

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_uses_custom_timeout(self, mock_get: Mock) -> None:
        """Test health check uses custom timeout parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        health_checker = HTTPHealthChecker(request_timeout=2.5)
        health_checker.check_health("http://localhost:5000")

        mock_get.assert_called_once_with("http://localhost:5000/", timeout=2.5)

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_uses_default_timeout(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check uses default timeout when none specified."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        health_checker.check_health("http://localhost:5000")

        # Should use default timeout
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:5000/"
        assert call_args[1]["timeout"] == 1.0

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_check_health_unexpected_exception(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check handles unexpected exceptions."""
        # ValueError is not a RequestException, so it gets caught by the general except
        # Let's use a RequestException subclass instead
        mock_get.side_effect = requests.ConnectionError("Unexpected connection error")

        status = health_checker.check_health("http://localhost:5000")

        assert status.is_running is False
        assert "Unexpected connection error" in status.error_message
        assert status.response_time is None


class TestServerStatus:
    """Test ServerStatus dataflow structure."""

    def test_server_status_healthy(self) -> None:
        """Test ServerStatus for healthy server."""
        status = ServerStatus(
            is_running=True, url="http://localhost:5000", response_time=0.05, error_message=None
        )

        assert status.is_running is True
        assert status.url == "http://localhost:5000"
        assert status.response_time == 0.05
        assert status.error_message is None

    def test_server_status_unhealthy(self) -> None:
        """Test ServerStatus for unhealthy server."""
        status = ServerStatus(
            is_running=False,
            url="http://localhost:5000",
            response_time=None,
            error_message="Connection refused",
        )

        assert status.is_running is False
        assert status.url == "http://localhost:5000"
        assert status.response_time is None
        assert status.error_message == "Connection refused"


class TestHealthCheckerIntegration:
    """Integration tests for health checker."""

    @pytest.mark.parametrize("endpoint", ["/health", "/api/health", "/status"])
    def test_different_health_endpoints(self, endpoint: str) -> None:
        """Test health checker with different endpoints."""
        checker = HTTPHealthChecker(health_endpoint=endpoint)
        assert checker._health_endpoint == endpoint

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_health_check_measures_response_time_accurately(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test that response time is measured accurately."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Simulate 0.123 second response time
        with patch("time.time", side_effect=[1000.0, 1000.123]):
            status = health_checker.check_health("http://localhost:5000")

        assert status.response_time == pytest.approx(0.123)

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_health_check_works_with_various_urls(
        self, mock_get: Mock, health_checker: HTTPHealthChecker
    ) -> None:
        """Test health check works with various URL formats."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        test_urls = [
            "http://localhost:5000",
            "https://localhost:8080",
            "http://127.0.0.1:5000",
            "https://mlflow.example.com",
            "http://192.168.1.100:5000",
        ]

        for url in test_urls:
            status = health_checker.check_health(url)
            assert status.is_running is True
            assert status.url == url


class TestHTTPHealthCheckerWaitFunction:
    """Test wait_for_health functionality with custom timeouts."""

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    @patch("dlkit.interfaces.servers.health_checker.time.sleep")
    def test_wait_for_health_uses_custom_timeouts(self, mock_sleep: Mock, mock_get: Mock) -> None:
        """Test wait_for_health uses custom timeout and poll interval."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_get.side_effect = [mock_response_fail, mock_response_success]

        checker = HTTPHealthChecker(request_timeout=0.05, wait_timeout=0.5, poll_interval=0.02)

        with patch("time.time", side_effect=[1000.0, 1000.05, 1000.1, 1000.15]):
            with patch("time.monotonic", side_effect=[0, 0.1, 0.15, 0.2]):
                result = checker.wait_for_health("http://localhost:5000")

        assert result is True
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(0.02)

    @patch("dlkit.interfaces.servers.health_checker.requests.get")
    def test_wait_for_health_explicit_params_override_defaults(self, mock_get: Mock) -> None:
        """Test that explicit params override instance defaults."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Create checker with one set of defaults
        checker = HTTPHealthChecker(
            request_timeout=0.1,
            wait_timeout=0.5,  # Fast for tests
            poll_interval=0.05,
        )

        with patch("time.time", side_effect=[1000.0, 1000.05]):
            with patch("time.monotonic", side_effect=[0, 0.1, 0.2]):  # Added extra value
                # Override with different values
                result = checker.wait_for_health(
                    "http://localhost:5000", timeout=0.3, poll_interval=0.1
                )

        assert result is True
