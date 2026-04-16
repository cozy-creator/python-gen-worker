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
class EndpointValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_NON_SLUG_CHARS = re.compile(r"[^a-z0-9.]+")
_DUP_SLUG_SEPARATORS = re.compile(r"-{2,}")


def _normalize_endpoint_name(raw: str) -> str:
    name = raw.strip().lower().replace("_", "-")
    if not name:
        return ""
    name = _NON_SLUG_CHARS.sub("-", name)
    name = _DUP_SLUG_SEPARATORS.sub("-", name)
    name = name.strip("-.")
    if len(name) > 128:
        name = name[:128].strip("-.")
    return name


def validate_endpoint(root: str | Path, *, require_uv_lock: bool = False) -> EndpointValidationResult:
    """
    Validate a published endpoint directory.

    Requirements:
    - `Dockerfile` must exist
    - `endpoint.toml` must exist (flat schema)
      - schema_version = 1
      - name = "..."
      - main = "pkg.module"
    - `pyproject.toml` must exist (Python packaging metadata)
    - `[project].name` must exist in `pyproject.toml` (normalized for URL-safe endpoint paths)
    - `requirements.txt` must not exist

    Optionally:
    - require `uv.lock` if `require_uv_lock=True`
    """
    root_path = Path(root).expanduser().resolve()

    errors: list[str] = []
    warnings: list[str] = []

    if not (root_path / "Dockerfile").exists():
        errors.append("missing Dockerfile")

    endpoint_toml = root_path / "endpoint.toml"
    if not endpoint_toml.exists():
        errors.append("missing endpoint.toml")

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
        warnings.append("tomllib is unavailable; cannot validate endpoint.toml/pyproject.toml")
        return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    # Validate endpoint.toml (flat schema).
    if endpoint_toml.exists():
        try:
            tensorhub_cfg: dict[str, Any] = tomllib.loads(endpoint_toml.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"failed to parse endpoint.toml: {exc}")
            tensorhub_cfg = {}

        if tensorhub_cfg.get("schema_version") != 1:
            errors.append("endpoint.toml missing or invalid schema_version (expected schema_version = 1)")

        endpoint_name = tensorhub_cfg.get("name")
        if not isinstance(endpoint_name, str) or endpoint_name.strip() == "":
            errors.append("endpoint.toml missing name")
        elif _normalize_endpoint_name(endpoint_name) == "":
            errors.append("endpoint.toml invalid name")

        main = tensorhub_cfg.get("main")
        if not isinstance(main, str) or main.strip() == "":
            errors.append("endpoint.toml missing main")

    # Validate pyproject.toml.
    if not pyproject.exists():
        return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse pyproject.toml: {exc}")
        return EndpointValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    project = data.get("project")
    pyproject_name = project.get("name") if isinstance(project, dict) else None
    if not isinstance(pyproject_name, str) or pyproject_name.strip() == "":
        errors.append("missing [project].name in pyproject.toml")
    elif _normalize_endpoint_name(pyproject_name) == "":
        errors.append("invalid [project].name in pyproject.toml")

    return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
