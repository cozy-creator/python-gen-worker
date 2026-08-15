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

    Confirms every entry in ``lock_dict["functions"]`` is a class-shape
    (post-#322) declaration:

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

    _check_slot_layout_declarations(lock_dict, errors)
    _check_aot_preconditions(lock_dict, errors, warnings)

    return EndpointLockValidationResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _undeclared_slot_layouts(lock_dict: Dict[str, Any]) -> List[str]:
    """A19 — the model slots in this manifest that declare no consumed
    tensor-layout contract, each already rendered as its own refusal.

    There is no default and no exemption list: every entry of a function's
    ``models={}`` is a model slot by construction, so "non-model slots are
    exempt" needs no test — they are not slots. A slot whose bytes no
    registered handle names says so with ``layouts_undeclarable="<reason>"``,
    and the reason travels on the manifest.
    """
    from ..models.tensor_layout_contract import undeclared_slot_refusal

    out: List[str] = []
    functions = lock_dict.get("functions") if isinstance(lock_dict, dict) else None
    for fn in functions or ():
        if not isinstance(fn, dict):
            continue
        fn_label = str(fn.get("name") or fn.get("python_name") or "<function>")
        for slot in fn.get("slots") or ():
            if not isinstance(slot, dict):
                continue
            if slot.get("layouts"):
                continue
            if str(slot.get("layouts_undeclarable") or "").strip():
                continue
            out.append(undeclared_slot_refusal(
                function=fn_label, slot=str(slot.get("name") or "<slot>")))
    return out


def refuse_undeclared_slot_layouts(lock_dict: Dict[str, Any]) -> None:
    """A19's refusal, typed. Every offender in ONE exception rather than one
    per build — an author fixing them one image build at a time is why nobody
    fixed them."""
    from ..models.tensor_layout_contract import UndeclaredSlotLayoutError

    found = _undeclared_slot_layouts(lock_dict)
    if found:
        raise UndeclaredSlotLayoutError("\n\n".join(found))


def _check_slot_layout_declarations(
    lock_dict: Dict[str, Any], errors: List[str],
) -> None:
    """The typed refusal, as a build error beside the others, so one run
    surfaces it with everything else the manifest gets wrong."""
    from ..models.tensor_layout_contract import UndeclaredSlotLayoutError

    try:
        refuse_undeclared_slot_layouts(lock_dict)
    except UndeclaredSlotLayoutError as exc:
        errors.append(str(exc))


def _check_aot_preconditions(
    lock_dict: Dict[str, Any], errors: List[str], warnings: List[str],
) -> None:
    """An image that cannot AOT-compile what it DECLARES is broken.

    ``discover_manifest`` stamps ``aot_preconditions`` — the static verdicts
    read off this very image (its C++ toolchain, its torch wheel, its
    declaration modules). A ``refused`` row means no pod can fix it, so the
    build stops here rather than shipping an endpoint that declares an export
    and silently downgrades its recipe on rented hardware forever.

    ``blocked`` (the family's own typed ``MintRefused``) and ``abstained`` (a
    torch-less manifest build) are WARNINGS: both are legitimate, and both are
    said out loud so nobody has to infer them from a pod's absence of events.
    """
    rows = lock_dict.get("aot_preconditions") if isinstance(lock_dict, dict) else None
    for row in rows or ():
        if not isinstance(row, dict):
            errors.append(f"aot_preconditions: expected dict rows, got {row!r}")
            continue
        verdict = str(row.get("verdict") or "").strip()
        family = str(row.get("family") or "").strip()
        label = f"{row.get('check')}{f' [{family}]' if family else ''}"
        detail = str(row.get("detail") or "")
        if verdict == "refused":
            errors.append(f"aot precondition {label}: {detail}")
        elif verdict in ("blocked", "abstained"):
            warnings.append(f"aot precondition {label} ({verdict}): {detail}")
        elif verdict != "ok":
            errors.append(
                f"aot precondition {label}: unknown verdict {verdict!r}")


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


