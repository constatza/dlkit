import os
from pathlib import Path
from pydantic.networks import AnyUrl
from urllib.parse import urlparse


def mkdir_for_local(uri: AnyUrl | str) -> None:
    """Ensure the local directory for the given URI or path exists.

    Accepts either:
    - AnyUrl with scheme 'file'
    - A plain string path (no scheme)
    - A URL-like string with scheme 'file' or 'sqlite' (treated as local file)
    """
    parsed = None
    if isinstance(uri, str):
        parsed = urlparse(uri)
        scheme = parsed.scheme
        if scheme in ("", None):
            path_str = uri
        elif scheme in ("file", "sqlite"):
            path_str = parsed.path
        else:
            # Non-local schemes: nothing to do
            return
    else:
        # AnyUrl-like
        scheme = getattr(uri, "scheme", None)
        if scheme != "file":
            return
        path_str = uri.path

    if os.name == "nt":
        path_str = path_str.lstrip(r"/")
    else:
        # On Unix, remove leading double slashes that come from sqlite:////path URIs
        while path_str.startswith("//"):
            path_str = path_str[1:]
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def recommended_uvicorn_workers() -> int:
    """Compute worker count using the standard (2 * cores) + 1 heuristic."""
    num_cores = os.cpu_count() or 8
    # Gunicorn’s recommended formula: (2 * n) + 1
    return (2 * num_cores) + 1
