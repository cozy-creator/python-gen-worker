from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import re

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from .names import slugify_name


@dataclass(frozen=True)
class EndpointValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndpointLockValidationResult:
    """Result of validating a discovered endpoint-lock ``functions`` list (#328).

    Constructed by ``validate_endpoint_lock``. ``ok`` is True iff
    ``errors`` is empty. Warnings are advisory (legacy `runtime` mismatch
    on a SerialWorker class, etc.).
    """

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_KNOWN_KINDS = frozenset(("inference", "training", "dataset", "conversion"))


def validate_endpoint_lock(lock_dict: Dict[str, Any]) -> EndpointLockValidationResult:
    """Validate a discovered endpoint.lock dict at bake time (#322/#328).

    Confirms every entry in ``lock_dict["functions"]`` is a class-shape
    (post-#322) declaration:

      1. ``class_name`` is present and non-empty — proves the entry came
         from a ``@inference`` / ``@training`` / ``@dataset`` / ``@conversion``
         decorated class, not a bare ``@inference``.
      2. ``archetype`` is ``"SerialWorker"`` or ``"BatchedWorker"``.
      3. ``kind`` is one of the four supported kinds.
      4. No two ``@inference.function`` methods on the SAME class slugify
         to the same wire route — that would silently shadow one of them
         at dispatch time.

    Returns an ``EndpointLockValidationResult`` whose ``errors`` lists every
    violation found, so a build can surface them all at once instead of one
    at a time. ``ok`` is True iff no errors.

    The intended caller is ``python -m gen_worker.discovery`` (bake time) and
    any CI lint that wants to gate-keep a pull request that drops a class
    declaration. Bake fails loudly when an endpoint still ships an old
    function-shape entry.
    """
    errors: List[str] = []
    warnings: List[str] = []

    functions = lock_dict.get("functions") if isinstance(lock_dict, dict) else None
    if not isinstance(functions, list):
        return EndpointLockValidationResult(
            ok=False,
            errors=("endpoint lock missing 'functions' list",),
        )
    if len(functions) == 0:
        warnings.append("no functions discovered (endpoint will advertise nothing)")

    # Per-class accumulator for the "two methods slugify to the same route"
    # check. Keyed by class_name → {function_slug: python_name}. A second
    # python_name on an existing slug under the same class is the violation.
    per_class_slugs: Dict[str, Dict[str, str]] = {}

    for idx, fn in enumerate(functions):
        if not isinstance(fn, dict):
            errors.append(f"functions[{idx}]: expected dict, got {type(fn).__name__}")
            continue
        fn_label = str(fn.get("name") or fn.get("python_name") or f"functions[{idx}]")

        kind = str(fn.get("kind") or "").strip()
        if kind not in _KNOWN_KINDS:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): kind must be one of "
                f"{sorted(_KNOWN_KINDS)}, got {kind!r}"
            )

        # Cross-method slug uniqueness within an endpoint group. The
        # orchestrator routes by ``slugify_name(function_name)``; two handlers
        # producing the same slug means one silently shadows the other.
        fn_name = str(fn.get("name") or "").strip()
        slug = slugify_name(fn_name)
        if not slug:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): function name "
                f"{fn_name!r} produces empty slug"
            )
            continue
        group = str(fn.get("class_name") or fn.get("module") or "<module>")
        py_name = str(fn.get("python_name") or "").strip()
        slugs = per_class_slugs.setdefault(group, {})
        prior_py = slugs.get(slug)
        if prior_py is not None and prior_py != py_name:
            errors.append(
                f"{group!r}: two handlers slugify to the same wire route "
                f"{slug!r}: {prior_py!r} and {py_name!r}. Rename one."
            )
        slugs[slug] = py_name

    return EndpointLockValidationResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


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
    - `pyproject.toml` must exist with `[tool.gen_worker].main = "pkg.module"`
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
        warnings.append("tomllib is unavailable; cannot validate pyproject.toml")
        return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    # Validate pyproject.toml.
    if not pyproject.exists():
        return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse pyproject.toml: {exc}")
        return EndpointValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    tool = data.get("tool")
    gw = tool.get("gen_worker") if isinstance(tool, dict) else None
    main = gw.get("main") if isinstance(gw, dict) else None
    if not isinstance(main, str) or main.strip() == "":
        errors.append("missing [tool.gen_worker].main in pyproject.toml")

    project = data.get("project")
    pyproject_name = project.get("name") if isinstance(project, dict) else None
    if not isinstance(pyproject_name, str) or pyproject_name.strip() == "":
        errors.append("missing [project].name in pyproject.toml")
    elif _normalize_endpoint_name(pyproject_name) == "":
        errors.append("invalid [project].name in pyproject.toml")

    return EndpointValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
