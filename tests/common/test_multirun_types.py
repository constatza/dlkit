"""ChildOutcome discrimination via match/case, and MultiRunResult.tracking_uri default."""

from __future__ import annotations

from dlkit.common.results import ChildFailure, ChildOutcome, ChildSuccess, MultiRunResult


def _describe(outcome: ChildOutcome[int]) -> str:
    match outcome:
        case ChildSuccess(status="success", result=result):
            return f"ok:{result}"
        case ChildFailure(status="failure", exception_type=exception_type):
            return f"error:{exception_type}"


def test_match_discriminates_child_success_by_status() -> None:
    outcome: ChildOutcome[int] = ChildSuccess(child_id="child-0", run_id="run-1", result=42)

    assert _describe(outcome) == "ok:42"


def test_match_discriminates_child_failure_by_status() -> None:
    outcome: ChildOutcome[int] = ChildFailure(
        child_id="child-1",
        exception_type="ValueError",
        message="bad hyperparameter",
        run_id="run-2",
        stage="training",
    )

    assert _describe(outcome) == "error:ValueError"


def test_multirun_result_tracking_uri_defaults_to_none() -> None:
    result = MultiRunResult(parent_run_id="parent-run", children=(1, 2, 3))

    assert result.tracking_uri is None
