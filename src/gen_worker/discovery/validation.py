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


# #328: archetype-shape requirements. Every class-shape entry must have these
# keys populated; the migration sweep guarantees this — old function-shape
# entries (pre-#322) lack class_name, so they're the trip-wire.
_REQUIRED_CLASS_SHAPE_FIELDS = ("class_name", "archetype", "kind")
_KNOWN_ARCHETYPES = frozenset(("SerialWorker", "BatchedWorker"))
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

        # Old-shape entries (pre-#322 @inference) lack class_name.
        # That's the migration trip-wire — fail loud with a pointer to #328.
        missing = [f for f in _REQUIRED_CLASS_SHAPE_FIELDS if not fn.get(f)]
        if missing:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): missing required class-shape "
                f"field(s) {missing}. This is an old function-shape entry from "
                "before the #322 SDK refactor. Migration steps in progress.json "
                "#328 — rewrite the endpoint as a class with @inference / "
                "@training / @dataset / @conversion decorator. Bake-time hard "
                "fail to prevent shipping a stale endpoint that the worker "
                "can't dispatch."
            )
            continue

        # Type / value checks for the three required fields.
        cls_name = str(fn.get("class_name") or "").strip()
        archetype = str(fn.get("archetype") or "").strip()
        kind = str(fn.get("kind") or "").strip()
        if not cls_name:
            errors.append(f"functions[{idx}] ({fn_label!r}): class_name empty")
            continue
        if archetype not in _KNOWN_ARCHETYPES:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): archetype must be one of "
                f"{sorted(_KNOWN_ARCHETYPES)}, got {archetype!r}"
            )
        if kind not in _KNOWN_KINDS:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): kind must be one of "
                f"{sorted(_KNOWN_KINDS)}, got {kind!r}"
            )

        # Cross-method slug uniqueness within a class. The orchestrator
        # routes by ``slugify_name(function_name)``; two methods producing
        # the same slug means one silently shadows the other at dispatch.
        # (Discovery's outer check catches GLOBAL collisions across all
        # classes; this catches the more-likely SAME-class collision.)
        fn_name = str(fn.get("name") or "").strip()
        slug = slugify_name(fn_name)
        if not slug:
            errors.append(
                f"functions[{idx}] ({fn_label!r}): function name "
                f"{fn_name!r} produces empty slug"
            )
            continue
        py_name = str(fn.get("python_name") or "").strip()
        slugs = per_class_slugs.setdefault(cls_name, {})
        prior_py = slugs.get(slug)
        if prior_py is not None and prior_py != py_name:
            errors.append(
                f"class {cls_name!r}: two @inference.function methods slugify "
                f"to the same wire route {slug!r}: {prior_py!r} and {py_name!r}. "
                "Rename one of the methods (or set @inference.function(name=...))."
            )
        slugs[slug] = py_name

        # Cross-cutting hook sanity (#322): runtime= only valid on BatchedWorker.
        runtime = fn.get("runtime")
        if runtime is not None and archetype != "BatchedWorker":
            warnings.append(
                f"functions[{idx}] ({fn_label!r}): runtime={runtime!r} declared "
                f"on archetype={archetype} — runtime= is only valid on "
                "BatchedWorker (async class). Field will be ignored at dispatch."
            )

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
