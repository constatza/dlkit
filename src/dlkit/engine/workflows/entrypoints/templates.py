"""Runtime-owned template generation helpers."""

from __future__ import annotations

from typing import Any

from dlkit.infrastructure.config._template_helpers import TemplateKind

__all__ = ["TemplateKind", "generate_template", "validate_template"]


def generate_template(
    template_type: TemplateKind = "training",
    *,
    model: str | None = None,
    module_path: str | None = None,
) -> str:
    """Generate a TOML configuration template."""
    from dlkit.infrastructure.config._template_helpers import render_template

    return render_template(template_type, model=model, module_path=module_path)


def validate_template(
    template_content: str,
    template_type: TemplateKind | None = None,
) -> dict[str, Any]:
    """Validate TOML template content against JobConfig."""
    from dlkit.infrastructure.config.job_config import JobConfig

    errors: list[str] = []
    try:
        import tomlkit

        parsed = tomlkit.loads(template_content)
        try:
            JobConfig.model_validate(dict(parsed))
        except Exception as exc:
            errors.append(f"Settings validation failed: {exc}")
    except Exception as exc:
        errors.append(f"TOML parsing failed: {exc}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "template_type": template_type,
    }
