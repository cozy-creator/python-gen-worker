from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from .names import slugify_name



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


_KNOWN_KINDS = frozenset(("inference", "training", "dataset", "conversion", "eval"))


def validate_endpoint_lock(lock_dict: Dict[str, Any]) -> EndpointLockValidationResult:
    """Validate a discovered endpoint.lock dict at bake time (#322/#328).

    Confirms every entry in ``lock_dict["entrypoints"]`` is a well-formed
    ``@entrypoint`` declaration:

      1. ``class_name`` is present and non-empty — proves the entry came
         from a ``@inference`` / ``@training`` / ``@dataset`` / ``@conversion``
         decorated class, not a bare ``@inference``.
      2. ``archetype`` is ``"SerialWorker"`` or ``"BatchedWorker"``.
      3. ``kind`` is one of the supported kinds.
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

    raw = lock_dict if isinstance(lock_dict, dict) else {}
    # pgw#1373: ONE declaration block. `functions[]`/`jobs[]` are deleted, so
    # there is nothing to fold and no both-keys ambiguity left to refuse.
    entrypoints = raw.get("entrypoints")
    if not isinstance(entrypoints, list):
        return EndpointLockValidationResult(
            ok=False,
            errors=("endpoint lock missing 'entrypoints' list",),
        )
    if len(entrypoints) == 0:
        # An ERROR, not a warning (pgw#1387): a release advertising nothing is
        # refused at hub admission, so a warning here just relocates the
        # failure to nine minutes after the image bake.
        errors.append(
            "this endpoint advertises NOTHING: no @entrypoint declarations "
            "were discovered, and hub admission refuses a manifest that "
            "declares no entrypoints[]"
        )
    functions = entrypoints

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


# pgw#1373: the A19 slot-layout gate (`_undeclared_slot_layouts` /
# `refuse_undeclared_slot_layouts` / `_check_slot_layout_declarations`) and the
# `aot_preconditions` gate are DELETED with the vocabularies that fed them.
# A19 demanded `Slot(layouts=...)` on every model slot; pgw#1394 established
# that a SERVING LANE is not an artifact-layout handle and removed `layouts`
# from v2 entrypoint slots entirely, so the gate could only ever fire falsely
# here — a v2 slot has no layouts to declare, by ruling. The lane travels on
# the release-derive document's `lane_contracts` and is gated there (th#2160).


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


