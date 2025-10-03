#!/usr/bin/env python
"""Sync example and template TOML files from a single source of truth.

This script regenerates example/template configs using the same functions the
CLI uses, ensuring they stay in sync with the code.

Usage:
  - Write updated files:    python scripts/sync_templates.py --write
  - Check for drift only:   python scripts/sync_templates.py --check

Currently maintained targets:
  - example_config.toml (training template)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src_to_path() -> None:
    # Ensure local 'src' is importable when running as a script or pre-commit hook
    src = _project_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_template_text(kind: str) -> str:
    from dlkit.interfaces.cli import templates as tmpl

    return tmpl.render_template(kind)  # type: ignore[arg-type]


def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text() if path.exists() else None
    if old != content:
        path.write_text(content)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    _add_src_to_path()

    parser = argparse.ArgumentParser(description="Sync DLKit template/example TOML files")
    parser.add_argument("--write", action="store_true", help="Write updates to files")
    parser.add_argument(
        "--check", action="store_true", help="Check for drift; non-zero exit if different"
    )
    args = parser.parse_args(argv)

    root = _project_root()

    # Targets to maintain: (generator, destination)
    targets: list[tuple[callable[[], str], Path]] = [
        (lambda: build_template_text("training"), root / "examples" / "example_config.toml"),
        (lambda: build_template_text("training"), root / "config" / "templates" / "training.toml"),
        (
            lambda: build_template_text("inference"),
            root / "config" / "templates" / "inference.toml",
        ),
        (lambda: build_template_text("mlflow"), root / "config" / "templates" / "mlflow.toml"),
        (lambda: build_template_text("optuna"), root / "config" / "templates" / "optuna.toml"),
    ]

    had_changes = False
    has_drift = False

    for gen, path in targets:
        content = gen()
        current = path.read_text() if path.exists() else None
        if current != content:
            has_drift = True
            if args.write:
                changed = write_if_changed(path, content)
                had_changes = had_changes or changed

    if args.check:
        return 1 if has_drift else 0

    if args.write:
        return 0 if had_changes or has_drift else 0

    # Default: do nothing but indicate if drift exists
    return 1 if has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
