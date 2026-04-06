from __future__ import annotations

import os


def recommended_uvicorn_workers() -> int:
    """Compute worker count using the standard (2 * cores) + 1 heuristic.

    Returns:
        Recommended number of Uvicorn worker processes.
    """
    num_cores = os.cpu_count() or 8
    # Gunicorn's recommended formula: (2 * n) + 1
    return (2 * num_cores) + 1
