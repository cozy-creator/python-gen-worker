from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ProjectValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_project(root: str | Path, *, require_uv_lock: bool = False) -> ProjectValidationResult:
    """
    Validate a tenant project directory.

    Requirements:
    - `pyproject.toml` must exist
    - `[tool.cozy]` must exist in `pyproject.toml`
    - `requirements.txt` must not exist
    - `cozy.toml` must not exist (config is standardized in `[tool.cozy]`)

    Optionally:
    - require `uv.lock` if `require_uv_lock=True`
    """
    root_path = Path(root).expanduser().resolve()

    errors: list[str] = []
    warnings: list[str] = []

    pyproject = root_path / "pyproject.toml"
    if not pyproject.exists():
        errors.append("missing pyproject.toml")
        return ProjectValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    if (root_path / "requirements.txt").exists():
        errors.append("requirements.txt is not supported; use pyproject.toml/uv.lock")

    if (root_path / "cozy.toml").exists():
        errors.append("cozy.toml is not supported; use [tool.cozy] in pyproject.toml")

    if require_uv_lock and not (root_path / "uv.lock").exists():
        errors.append("missing uv.lock (required)")
    elif not (root_path / "uv.lock").exists():
        warnings.append("uv.lock not found (recommended for reproducible builds)")

    if tomllib is None:
        warnings.append("tomllib is unavailable; cannot validate [tool.cozy] presence")
        return ProjectValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse pyproject.toml: {exc}")
        return ProjectValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    tool = data.get("tool")
    cozy = tool.get("cozy") if isinstance(tool, dict) else None
    if not isinstance(cozy, dict):
        errors.append("missing [tool.cozy] in pyproject.toml")

    return ProjectValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

