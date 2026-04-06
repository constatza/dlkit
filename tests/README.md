# Test Suite Rules

## OS-Agnostic Paths Are Mandatory

Every test under `tests/` must be valid on Linux, macOS, and Windows unless the test is explicitly marked as platform-specific with `pytest.mark.skipif(...)`.

This is a hard rule for all filesystem assertions:

- Never hard-code POSIX-only absolute paths like `"/tmp/..."` in shared tests.
- Never assume Windows drive-less path formatting in URIs.
- Always build expected paths from `pathlib.Path`, `tmp_path`, and the shared URI helpers in `dlkit.infrastructure.io.url_resolver`.
- When asserting on `file://` or `sqlite://` values, compare normalized URIs produced by shared helpers instead of hand-written strings.
- If the behavior is intentionally platform-specific, mark it loudly and explicitly in the test.

## Required Patterns

- Use `tmp_path` for temporary files and directories.
- Use `Path` joins for expected filesystem locations.
- Use `url_resolver.build_uri(...)`, `url_resolver.normalize_uri(...)`, or `url_resolver.resolve_local_uri(...)` for URI expectations.
- Use `Path.is_relative_to(base)` to assert a path is under a directory.
- When a string containment check on paths is unavoidable, normalize both sides with `.as_posix()` first: `Path(x).as_posix() in Path(y).as_posix()`.

## Anti-Patterns

- `assert uri == "file:///tmp/..."`
- `assert path == Path("/tmp/...")` in a cross-platform test
- Manual slicing/parsing of `file://` or `sqlite://` strings inside tests
- `assert str(some_path) in other_string` — `str(Path)` on Windows returns backslashes; use `is_relative_to()` or `.as_posix()` on both sides.
- `assert "foo/bar" in s or "foo\\bar" in s` — normalize instead of branching on separators.

## Scope Boundaries Are Mandatory

Tests must be self-contained within the `tests/` scope and pytest-managed temporary directories.

This is also a hard rule:

- Never write test artifacts into repository roots like `mlruns/`, `output/`, `checkpoints/`, or any other non-test directory.
- Never read seed data, configs, checkpoints, or artifacts from outside `tests/` unless the test is explicitly designed around a checked-in fixture under `tests/fixtures/`.
- Never depend on developer-machine state such as home-directory files, local databases, running services, or leftover artifacts from previous runs.
- Always create runtime artifacts under `tmp_path` or another pytest-managed temporary location.
- Always keep test input data under `tests/fixtures/` when it must be checked into the repository.
- If a test needs an external system, mark and isolate it explicitly instead of silently reaching outside the test boundary.

## Out-of-Scope Anti-Patterns

- Writing to `./mlruns`, `./output`, or `./checkpoints` from a test
- Reading `/tmp/...`, `~/...`, or arbitrary workspace files as test dependencies
- Reusing artifacts produced by an earlier manual run
- Assuming a local MLflow server, database, or dataset already exists

If a test touches paths or URIs, review it for Windows drive handling before merging.
If a test creates or reads artifacts, review it for scope isolation before merging.
