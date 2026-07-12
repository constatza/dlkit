"""Find (and optionally delete) MLflow runs polluted by the pre-fix `val_epoch/`/`test_epoch/` bug.

Before commit 5ddc2cc, `CoreLightningWrapper._on_eval_epoch_end` passed
"val_epoch"/"test_epoch" as a stage name into metric-key formatting, producing
literal MLflow metric keys like "val_epoch/Accuracy" and "test_epoch/Accuracy"
that clutter the MLflow UI as empty-looking extra groups. The producer bug is
fixed in code, but MLflow has no API to delete individual metric keys from a
run's history — old runs keep those keys until the run itself is deleted.

This script only lists affected runs by default. Pass --delete to actually
remove them, after reviewing the printed list.

Usage:
    uv run python scripts/purge_legacy_epoch_metric_runs.py [--experiment-id ID ...] [--delete]
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient

from dlkit.engine.tracking.discovery import default_sqlite_backend_uri
from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory

_LEGACY_PREFIXES = ("val_epoch/", "test_epoch/")


def _is_polluted(client: MlflowClient, run_id: str) -> bool:
    run = client.get_run(run_id)
    return any(key.startswith(_LEGACY_PREFIXES) for key in run.data.metrics)


def _find_polluted_runs(client: MlflowClient, experiment_ids: list[str]) -> list[str]:
    return [
        run.info.run_id
        for run in client.search_runs(experiment_ids=experiment_ids, max_results=50_000)
        if _is_polluted(client, run.info.run_id)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        help="Restrict the search to this experiment ID (repeatable). Defaults to all experiments.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the affected runs. Without this flag, only lists them.",
    )
    args = parser.parse_args()

    client = MLflowClientFactory.create_client(tracking_uri=default_sqlite_backend_uri())
    experiment_ids = args.experiment_ids or [e.experiment_id for e in client.search_experiments()]

    polluted = _find_polluted_runs(client, experiment_ids)
    if not polluted:
        print("No runs with legacy val_epoch/test_epoch metric keys found.")
        return

    print(f"Found {len(polluted)} run(s) with legacy val_epoch/test_epoch metric keys:")
    for run_id in polluted:
        run = client.get_run(run_id)
        print(f"  {run_id}  experiment={run.info.experiment_id}  start_time={run.info.start_time}")

    if not args.delete:
        print("\nRe-run with --delete to remove these runs.")
        return

    for run_id in polluted:
        client.delete_run(run_id)
    print(f"\nDeleted {len(polluted)} run(s).")


if __name__ == "__main__":
    main()
