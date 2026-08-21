from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from .names import slugify_name



@dataclass(frozen=True)
class EndpointLockValidationResult:
    """Result of validating a discovered endpoint-lock ``functions`` list (#328)."""

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_KNOWN_KINDS = frozenset(("inference", "training", "dataset", "conversion", "eval"))


def validate_endpoint_lock(lock_dict: Dict[str, Any]) -> EndpointLockValidationResult:
    """Validate a discovered endpoint.lock dict at bake time (#322/#328)."""
    errors: List[str] = []
    warnings: List[str] = []

    raw = lock_dict if isinstance(lock_dict, dict) else {}
    entrypoints = raw.get("entrypoints")
    if not isinstance(entrypoints, list):
        return EndpointLockValidationResult(
            ok=False,
            errors=("endpoint lock missing 'entrypoints' list",),
        )
    if len(entrypoints) == 0:
        errors.append(
            "this endpoint advertises NOTHING: no @entrypoint declarations "
            "were discovered, and hub admission refuses a manifest that "
            "declares no entrypoints[]"
        )
    functions = entrypoints

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


