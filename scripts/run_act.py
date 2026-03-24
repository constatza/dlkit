"""Run act for local CI checks; skips gracefully if act or docker is unavailable."""

import shutil
import subprocess
import sys


def main() -> None:
    for tool in ("act", "docker"):
        if shutil.which(tool) is None:
            print(f"{tool} not found — skipping local CI check (install {tool} to enable)")
            sys.exit(0)
    sys.exit(subprocess.run(["act", *sys.argv[1:]], check=False).returncode)


if __name__ == "__main__":
    main()
