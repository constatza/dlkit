# CLI Middleware

`dlkit.interfaces.cli.middleware` owns CLI-side error presentation.

## Responsibilities
- translate DLKit exceptions into user-facing CLI output
- format validation and unexpected errors
- provide contextual follow-up suggestions
- handle keyboard interrupts cleanly

## Notes
- Error suggestions are registered by exception type instead of a long `isinstance` chain.
- Middleware stays presentation-focused; it does not own workflow behavior.
