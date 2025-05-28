import os
from pathlib import Path
from pydantic.networks import AnyUrl


def mkdir_for_local(uri: AnyUrl) -> None:
    """Ensures that the local directory for the given URI exists if the URI starts with `file:`.

    Args:
        uri (str): A URI that might be local (file scheme).
    """
    path = uri.path if os.name != "nt" else uri.path.lstrip(r"/")  # Handle Windows paths correctly
    Path(path).parent.mkdir(parents=True, exist_ok=True)
