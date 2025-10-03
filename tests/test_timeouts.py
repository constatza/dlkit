"""Centralized timeout configuration for integration tests.

These values can be adjusted based on hardware performance.

Usage:
    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_something_fast(self):
        ...

For slower machines, multiply the base values:
    SLOW_MACHINE_MULTIPLIER = 2.0
    FAST_TEST_TIMEOUT = int(30 * SLOW_MACHINE_MULTIPLIER)  # 60 seconds
"""

import os

# Optional environment variable to adjust all timeouts for slower hardware
# Set DLKIT_TEST_TIMEOUT_MULTIPLIER=2.0 for slower machines
TIMEOUT_MULTIPLIER = float(os.getenv("DLKIT_TEST_TIMEOUT_MULTIPLIER", "1.0"))

# Base timeout values (will be multiplied by TIMEOUT_MULTIPLIER)
_FAST_BASE = 30  # Fast tests using file:// tracking (no server startup)
_MEDIUM_BASE = 60  # Medium tests with multiple operations
_SLOW_BASE = 120  # Slow tests that start real HTTP servers (~7-10s startup)
_VERY_SLOW_BASE = 180  # Very slow tests (multiple servers or extensive training)

# Exported timeout constants (apply multiplier)
FAST_TEST_TIMEOUT = int(_FAST_BASE * TIMEOUT_MULTIPLIER)
MEDIUM_TEST_TIMEOUT = int(_MEDIUM_BASE * TIMEOUT_MULTIPLIER)
SLOW_TEST_TIMEOUT = int(_SLOW_BASE * TIMEOUT_MULTIPLIER)
VERY_SLOW_TEST_TIMEOUT = int(_VERY_SLOW_BASE * TIMEOUT_MULTIPLIER)
