# Pydantic‑Only URL/Path Validation Plan

## Goals
- Eliminate urllib/httpx from validation logic; rely solely on Pydantic v2 primitives.
- Provide clean, composable Annotated types for MLflow backends, tracking URLs, artifact destinations, and secure local paths.
- Centralize tilde expansion and local path security checks with conservative behavior.
- Keep SOLID and KISS by composing small validators and scheme‑specific constraints.

## Design Principles
- Single Responsibility: each type validates one semantic (e.g., SQLite URL, local path).
- Open/Closed: add new schemes/types by composing constraints and small validators.
- KISS: prefer `Url` + `UrlConstraints` + simple regex; avoid parsing libraries.
- Consistency: same tilde/security rules across all consumers.

## Core Validators
- `tilde_expand_strict(value: str) -> str`
  - BeforeValidator for strings: expands leading `~`/`/~` in plain paths and in the first URL path segment only.
  - Regex‑based, no splitting/parsing libs; does not expand mid‑path `.../~/...`.
- `local_path_security_check(value: str) -> str`
  - AfterValidator: rejects NUL/control chars, directory traversal (`..` at boundaries), and suspicious `//` collapsing.

## Scheme Types (Pydantic Url)
- `HttpUrl`: `Annotated[Url, UrlConstraints(allowed_schemes=["http","https"]) ]`.
- `FileUrl`: `Annotated[Url, UrlConstraints(allowed_schemes=["file"], host_required=False)]`.
- `SQLiteUrl`: `Annotated[Url, UrlConstraints(allowed_schemes=["sqlite"], host_required=False)]` + AfterValidator enforcing triple‑slash form and non‑empty path.
- `CloudStorageUrl`: `Annotated[Url, UrlConstraints(allowed_schemes=["s3","gs","wasbs","hdfs"], host_required=False)]` + AfterValidator per‑scheme rules (e.g., S3 bucket regex, hdfs host optional).

## Custom Pydantic Type
- `DatabricksUrl`
  - Implements `__get_pydantic_core_schema__` returning a `no_info_plain_validator_function`.
  - Accepts `databricks://profile:workspace` (strict regex). Optional literal `databricks` shorthand only if explicitly required.
  - Applies `tilde_expand_strict` before regex.

## Composite Types
- `MLflowBackendUrl`
  - `Annotated[str, BeforeValidator(tilde_expand_strict), AfterValidator(validate_mlflow_backend)]`.
  - `validate_mlflow_backend` attempts validation as one of: `SQLiteUrl`, `FileUrl`, `HttpUrl`, `CloudStorageUrl`, `DatabricksUrl`; returns on first success.
- `MLflowTrackingUrl`
  - If restricting to server endpoints: alias to `HttpUrl`.
  - If broader support is desired: alias to `MLflowBackendUrl`.
- `ArtifactDestination`
  - `Annotated[str, BeforeValidator(tilde_expand_strict), AfterValidator(validate_artifact_destination)]`.
  - If `://` present → validate as `FileUrl|CloudStorageUrl`; else → validate as secure local path.
- `LocalPath`
  - `Annotated[str, BeforeValidator(tilde_expand_strict), AfterValidator(local_path_security_check)]`.
- `LocalOrRemote`
  - For general fields that may take either a path or URL: `Annotated[str, BeforeValidator(tilde_expand_strict), AfterValidator(dispatch_url_or_local)]`.

## Module Layout
- `src/dlkit/datatypes/urls.py`
  - Contains all types above (Annoted Url types, custom `DatabricksUrl`, composite types) and small validator helpers.
- `src/dlkit/datatypes/secure_uris.py`
  - Re‑exports `SecureMLflowBackendStoreUri = MLflowBackendUrl`, `SecureArtifactStoreUri = ArtifactDestination`, `SecureMLflowTrackingUri = HttpUrl` (or `MLflowBackendUrl`), `SecurePath = LocalPath`.
- `src/dlkit/settings/*`
  - Update to use the secure types; remove imports from legacy validation.
- Remove `src/dlkit/io/validation/*` and all tests relying on it.

## Replacement of Existing Uses
- `mlflow_settings.py`:
  - `server.backend_store_uri: SecureMLflowBackendStoreUri` (composite MLflow backend)
  - `server.artifacts_destination: SecureArtifactStoreUri` (URL or local path)
  - `client.tracking_uri: HttpUrl` (or `MLflowBackendUrl` if non‑HTTP should be allowed)
- `paths_settings.py`:
  - `SecurePath` uses `LocalPath` semantic; no I/O, no side effects.
- `io/config.py`:
  - Remove imports from legacy `io.validation`; only minimal pre‑resolution remains (tilde expansion + make relative paths absolute to computed root).

## Tests
- Remove legacy validation tests (`tests/io/*`, `tests/integration/test_mlflow_validation_integration.py`).
- Add targeted tests for new types:
  - Tilde expansion only at allowed positions (plain path and URL first segment).
  - SQLite triple‑slash + non‑empty path; FileUrl acceptance; HttpUrl normalization.
  - S3/GS/WASBS/HDFS acceptance + S3 bucket regex failures.
  - DatabricksUrl acceptance/rejection.
  - ArtifactDestination dispatch (URL vs local) and LocalPath security checks.

## Migration Notes
- Pydantic handles normalization (e.g., trailing slash for HTTP) consistently.
- Databricks shorthand literal is optional; if not required, do not support it.
- All urllib/httpx usage removed from validation path; only Pydantic validators remain.

## Acceptance Criteria
- No urllib/httpx imports in validation code.
- All settings validate with Pydantic-only types.
- Old `src/dlkit/io/validation/*` removed; the codebase builds and core tests unrelated to legacy validation still pass.
