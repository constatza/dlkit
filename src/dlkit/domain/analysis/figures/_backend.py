"""Matplotlib backend initialisation — import before any other matplotlib code."""

import matplotlib

matplotlib.use("Agg")  # non-interactive; safe on headless servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plt", "np"]
