# CLI Entry Point

`dlkit.interfaces.cli.app` owns the top-level Typer application: global
options, logging bootstrap, and command-group registration. Individual
command groups (`train`, `predict`, `evaluate`, `evaluate-multirun`,
`optimize`, `convert`, `config`, `converge`, `multirun`) are documented in
[`commands/commands.md`](commands/commands.md).

## Global Options

The `main()` callback (`app.py`) runs before any subcommand and configures
logging for the whole process via `configure_logging()`
(`dlkit.infrastructure.utils.logging_config`, see
[`../../infrastructure/utils/utils.md`](../../infrastructure/utils/utils.md)):

| Option | Effect |
|---|---|
| `--verbose` | Enables debug-level logging |
| `--debug` | Enables debug-level logging |
| `--log-level LEVEL` | Explicit level (DEBUG, INFO, WARNING, ERROR); overridden by `--debug`/`--verbose` |
| `--log-file` | Opts into file logging at `.dlkit/logs/dlkit_<timestamp>.log` (or `DLKIT_LOG_FILE` if set) |

File logging is opt-in only. Without `--log-file` (and without
`DLKIT_LOG_FILE` set), no file is written and no `.dlkit/` directory is
created — logs go to stderr only. `--log-file` without an explicit path
generates a fresh timestamped filename each run via `_resolve_log_file_path()`
(`app.py`), which defers to `EnvironmentSettings.get_internal_dir_path()`
(`dlkit.infrastructure.config.environment`) for the `.dlkit/` location.

```bash
dlkit train config.toml                    # stderr only
dlkit --log-file train config.toml         # also .dlkit/logs/dlkit_<timestamp>.log
DLKIT_LOG_FILE=run.log dlkit train config.toml  # explicit path, any command
```

## Commands

- `dlkit info` — prints installed dependency versions (PyTorch, Lightning,
  MLflow, Optuna).
- All other commands are registered as sub-`Typer` apps from
  `interfaces/cli/commands/` — see `commands/commands.md`.

## Related Modules

- `dlkit.infrastructure.utils.logging_config`: logging configuration this
  callback drives.
- `dlkit.infrastructure.config.environment`: `EnvironmentSettings` /
  `.dlkit/` resolution.
- `dlkit.interfaces.cli.commands`: the actual workflow commands.
