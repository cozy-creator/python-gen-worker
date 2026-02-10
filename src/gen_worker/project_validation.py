from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ProjectValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_NON_SLUG_CHARS = re.compile(r"[^a-z0-9.]+")
_DUP_SLUG_SEPARATORS = re.compile(r"-{2,}")


def _normalize_project_name(raw: str) -> str:
    name = raw.strip().lower().replace("_", "-")
    if not name:
        return ""
    name = _NON_SLUG_CHARS.sub("-", name)
    name = _DUP_SLUG_SEPARATORS.sub("-", name)
    name = name.strip("-.")
    if len(name) > 128:
        name = name[:128].strip("-.")
    return name


def validate_project(root: str | Path, *, require_uv_lock: bool = False) -> ProjectValidationResult:
    """
    Validate a tenant project directory.

    Requirements:
    - `Dockerfile` must exist
    - `cozy.toml` must exist (flat schema)
      - schema_version = 1
      - name = "..."
      - main = "pkg.module"
      - gen_worker = ">=x,<y"
    - `pyproject.toml` must exist (Python packaging metadata)
    - `[project].name` must exist in `pyproject.toml` (normalized for URL-safe project paths)
    - `requirements.txt` must not exist

    Optionally:
    - require `uv.lock` if `require_uv_lock=True`
    """
    root_path = Path(root).expanduser().resolve()

    errors: list[str] = []
    warnings: list[str] = []

    if not (root_path / "Dockerfile").exists():
        errors.append("missing Dockerfile")

    cozy_toml = root_path / "cozy.toml"
    if not cozy_toml.exists():
        errors.append("missing cozy.toml")

    pyproject = root_path / "pyproject.toml"
    if not pyproject.exists():
        errors.append("missing pyproject.toml")

    if (root_path / "requirements.txt").exists():
        errors.append("requirements.txt is not supported; use pyproject.toml/uv.lock")

    if require_uv_lock and not (root_path / "uv.lock").exists():
        errors.append("missing uv.lock (required)")
    elif not (root_path / "uv.lock").exists():
        warnings.append("uv.lock not found (recommended for reproducible builds)")

    if tomllib is None:
        warnings.append("tomllib is unavailable; cannot validate cozy.toml/pyproject.toml")
        return ProjectValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    # Validate cozy.toml (flat schema).
    if cozy_toml.exists():
        try:
            cozy: dict[str, Any] = tomllib.loads(cozy_toml.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"failed to parse cozy.toml: {exc}")
            cozy = {}

        if cozy.get("schema_version") != 1:
            errors.append("cozy.toml missing or invalid schema_version (expected schema_version = 1)")

        cozy_name = cozy.get("name")
        if not isinstance(cozy_name, str) or cozy_name.strip() == "":
            errors.append("cozy.toml missing name")
        elif _normalize_project_name(cozy_name) == "":
            errors.append("cozy.toml invalid name")

        main = cozy.get("main")
        if not isinstance(main, str) or main.strip() == "":
            errors.append("cozy.toml missing main")

        gen_worker = cozy.get("gen_worker")
        if not isinstance(gen_worker, str) or gen_worker.strip() == "":
            errors.append("cozy.toml missing gen_worker constraint")

    # Validate pyproject.toml.
    if not pyproject.exists():
        return ProjectValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse pyproject.toml: {exc}")
        return ProjectValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    project = data.get("project")
    project_name = project.get("name") if isinstance(project, dict) else None
    if not isinstance(project_name, str) or project_name.strip() == "":
        errors.append("missing [project].name in pyproject.toml")
    elif _normalize_project_name(project_name) == "":
        errors.append("invalid [project].name in pyproject.toml")

    return ProjectValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
