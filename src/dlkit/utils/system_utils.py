import os
from pathlib import Path

from pydantic.networks import FileUrl


def is_local_host(hostname: str | None) -> bool:
    """Check if a hostname is considered local.

    A URI is considered local if:
      - It has no hostname (which usually means a file path or similar local resource),
      - Its hostname is 'localhost' (case-insensitive),
      - Its hostname is the loopback IPv4 address ('127.0.0.1'),
      - Its hostname is a common representation of the IPv6 loopback address ('::1').

    Args:
        hostname (str | None): The hostname to check. Defaults to None.

    Returns:
        bool: True if the URI is considered local, otherwise False.
    """
    if hostname is None:
        return True
    hostname = hostname.lower()
    loopbacks = {"", "0.0.0.0", "127.0.0.0.1", "::1"}
    # Check known local identifiers
    if hostname in loopbacks:
        return True
    return False


def mkdir_for_local(uri: FileUrl) -> None:
    """Ensures that the local directory for the given URI exists if the URI starts with `file:`.

    Args:
        uri (str): A URI that might be local (file scheme).
    """
    path = uri.path
    if os.name == "nt" and path.startswith("/"):
        path = path.lstrip("/")

    local_path = Path(path)
    # check if it has extension
    if not local_path.suffix:
        local_path = local_path.parent
    local_path.mkdir(parents=True)
