"""A dtype cast is TREE-wide; precision pins are PER-COMPONENT.

``families.facts`` states which component CLASSES must come off disk wider
than the composition's compute dtype (``AutoencoderKLWan -> fp32``), and
``models.loading`` honours it on every materialize. Without a producer half, a
plain ``dtype: "bf16"`` clone casts every weight group in the tree, so a
published "bf16" wan flavor carries a bf16 VAE the load side then upcasts back
into fp32 — truncated, and invisible at every gate.

This module is that producer half. It reads the TREE'S OWN
``model_index.json`` (authoritative: a fine-tune may substitute a class), asks
``families.facts`` — never a second table — and answers two questions:

* :func:`cast_exempt_components` — which components a narrowing cast must
  leave at source dtype. The tree-wide cast SKIPS them and says so; it does
  not refuse, because "publish a bf16 flavor of this model" is a tree-level
  intent with no per-component spelling that could satisfy both it and the
  pin, and the loader would widen the component right back anyway. Refusing
  would leave the caller with a workaround (a third pod leg, plus an operator
  who already knows the fact) as the only path to the same tree.
* :func:`check_explicit_pin_conflict` — the caller NAMING a pinned component
  as a quant target is a per-component instruction that contradicts the pin.
  That is refused, typed and by name: the caller can repair it (drop the
  component), and silently ignoring an explicit instruction is its own
  invisibility bug.

:func:`verify_produced_tree` is the publish gate. It compares the tree about
to be published against the pins and refuses when a pinned component is
narrower than its pin AND the source was not — i.e. when WE narrowed it. An
upstream whose own VAE ships bf16 is still mirrorable; the fact is recorded,
not enforced onto someone else's bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from ..families.facts import ComponentDtype, component_dtypes_for_classes

# Storage width per dtype spelling, for the ONE comparison this module makes:
# is the requested/produced precision NARROWER than the pin? Quant dtypes are
# all narrower than every pin, which is the answer that matters.
#
# pgw#1291: KEYS ARE SEPARATOR- AND CASE-FOLDED before lookup (see
# :func:`dtype_bits`), so `fp8:e4m3`, `fp8_e4m3`, `fp8-e4m3` and `F8_E4M3` are
# ONE entry. That is not tidiness — an unknown spelling scores 0 bits, and 0
# bits makes :func:`is_narrowing` answer False, which turns this whole gate
# SILENTLY OFF for the artifact it was pointed at. Four live spellings did
# exactly that: `int8` and `uint8` (which the module docstring claims are
# caught: "quant dtypes are all narrower than every pin"), and tensorhub's
# canonical `fp8_e4m3`/`fp8_e5m2`, which is the vocabulary `tensorlayout`
# derives and the catalog is moving to. Measured before the fix:
# `dtype_bits("int8") == 0`, `is_narrowing("int8", "fp32") is False`.
#
# `_dtype_vocabulary_is_complete` (tests/test_dtype_pins_vocabulary_pgw1291.py)
# is the fence: every token any producer in this repo can EMIT must have a
# width here, so the next vocabulary addition fails a test instead of
# disarming the gate 45 minutes into a conversion.
DTYPE_BITS: Dict[str, int] = {
    "fp64": 64, "f64": 64, "float64": 64,
    "fp32": 32, "f32": 32, "float32": 32,
    "bf16": 16, "bfloat16": 16, "fp16": 16, "f16": 16, "float16": 16,
    "fp8": 8, "fp8:e4m3": 8, "fp8:e5m2": 8, "f8:e4m3": 8, "f8:e5m2": 8,
    "int8": 8, "uint8": 8, "i8": 8, "u8": 8, "q8_0": 8,
    "q6_k": 6, "q5_k_m": 5, "q5_k_s": 5,
    "nvfp4": 4, "int4": 4, "int4:nf4": 4, "int4:fp4": 4, "nf4": 4, "fp4": 4,
    "q4_k_m": 4, "q4_k_s": 4, "q4_0": 4, "q4_1": 4,
    "q3_k_m": 3, "q3_k_s": 3, "q2_k": 2,
}

# The separators a dtype flavor is spelled with across the four vocabularies
# that meet here: `fp8:e4m3` (this module + clone), `fp8_e4m3` (tensorlayout /
# the catalog), `fp8-e4m3` (producer labels), `F8_E4M3` (the safetensors
# header). One fact, four spellings; fold them rather than enumerate them.
_DTYPE_SEPARATORS = ("_", "-")


def _fold_dtype(dtype: str) -> str:
    """The lookup key: lowercased, with every flavor separator folded to ':'.

    Applied to BOTH sides — the table is written with ':' and folded on load —
    so no entry can be reachable by one spelling and not another.
    """
    key = str(dtype or "").strip().lower()
    for sep in _DTYPE_SEPARATORS:
        key = key.replace(sep, ":")
    return key


# Fold the table itself, once, so lookups compare like with like. Written with
# ':' above for readability; `q4_k_m` and friends fold to `q4:k:m` and stay
# addressable only through the same fold.
_DTYPE_BITS_FOLDED: Dict[str, int] = {
    _fold_dtype(k): v for k, v in DTYPE_BITS.items()
}


class ComponentDtypePinError(ValueError):
    """A requested conversion contradicts a component's load-dtype pin.

    Carries the component, its class, the pin and the reason the pin exists,
    so the refusal names what to change instead of describing a rule.
    """

    def __init__(self, component: str, class_name: str, fact: ComponentDtype,
                 requested: str) -> None:
        self.component = str(component)
        self.class_name = str(class_name)
        self.pin = fact.dtype
        self.reason = fact.reason
        self.requested = str(requested)
        super().__init__(
            f"component {self.component!r} ({self.class_name}) is pinned to "
            f"{self.pin} and cannot be converted to {self.requested}: "
            f"{self.reason}. Drop {self.component!r} from the request "
            f"(the tree-wide cast leaves it at source dtype on its own)."
        )


def dtype_bits(dtype: str) -> int:
    """Storage width of a dtype spelling; 0 when unknown (never compared).

    Spelling-insensitive across the separator vocabularies (pgw#1291): a
    flavored fp8 answers 8 however it is written, because answering 0 disarms
    the caller's gate instead of failing it.
    """
    return _DTYPE_BITS_FOLDED.get(_fold_dtype(dtype), 0)


def is_narrowing(requested: str, pin: str) -> bool:
    """True when ``requested`` stores fewer bits than the pin requires."""
    req, pinned = dtype_bits(requested), dtype_bits(pin)
    return bool(req and pinned and req < pinned)


def model_index_classes(tree: Path | str) -> Dict[str, str]:
    """``{component: class name}`` a tree's ``model_index.json`` declares.

    Empty for single-file / transformers layouts, which carry no per-component
    class vocabulary and therefore no pins.
    """
    out: Dict[str, str] = {}
    try:
        with open(Path(tree) / "model_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return out
    if not isinstance(index, dict):
        return out
    for key, entry in index.items():
        if str(key).startswith("_"):
            continue
        if (isinstance(entry, (list, tuple)) and len(entry) == 2
                and all(isinstance(e, str) and e for e in entry)):
            out[str(key)] = entry[1]
    return out


def component_pins(tree: Path | str) -> Dict[str, ComponentDtype]:
    """``{component: ComponentDtype}`` the tree's own classes pin. One source
    of truth: :mod:`gen_worker.families.facts`, the table the loader reads."""
    return component_dtypes_for_classes(model_index_classes(tree))


def cast_exempt_components(tree: Path | str, requested_dtype: str) -> Dict[str, ComponentDtype]:
    """Components a cast to ``requested_dtype`` must NOT touch.

    Only NARROWING is exempted: a cast to fp32 of an fp32-pinned component is
    a no-op, and widening never contradicts a pin.
    """
    return {
        comp: fact for comp, fact in component_pins(tree).items()
        if is_narrowing(requested_dtype, fact.dtype)
    }


def check_explicit_pin_conflict(
    tree: Path | str, requested_dtype: str, named_components: Iterable[str] | None,
) -> None:
    """Refuse when the caller NAMED a pinned component as a conversion target.

    A tree-wide cast skips a pinned component (see module docstring); an
    explicit per-component instruction is refused instead, because it has a
    repair and ignoring it would be a second silent behaviour.
    """
    named = {str(c).strip() for c in (named_components or []) if str(c).strip()}
    if not named:
        return
    classes = model_index_classes(tree)
    for comp, fact in component_pins(tree).items():
        if comp in named and is_narrowing(requested_dtype, fact.dtype):
            raise ComponentDtypePinError(comp, classes.get(comp, ""), fact, requested_dtype)


def component_dtype(component_dir: Path | str) -> str:
    """Majority on-disk dtype of one component's safetensors, '' if unreadable."""
    from .ingest import detect_snapshot_dtype

    return detect_snapshot_dtype(Path(component_dir))


def component_dtypes_on_disk(tree: Path | str) -> Dict[str, str]:
    """``{component: dtype}`` read from the produced tree's own safetensors
    headers — the per-component precision report a publish carries so a
    downcast can never again be invisible. A tree with no component subdirs
    (single-file / transformers layout) reports its one weight set as
    ``"model"``, matching :func:`size_walk.compute_size_facts`."""
    root = Path(tree)
    out: Dict[str, str] = {}
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and any(entry.rglob("*.safetensors")):
            dt = component_dtype(entry)
            if dt:
                out[entry.name] = dt
    if not out and any(p.is_file() for p in root.glob("*.safetensors")):
        dt = component_dtype(root)
        if dt:
            out["model"] = dt
    return out


class ComponentDtypePinViolation(RuntimeError):
    """A tree about to be published carries a component NARROWER than its pin,
    and the source was not — i.e. this producer truncated it. Publishing would
    ship silent quality loss the load side cannot recover."""

    def __init__(self, component: str, class_name: str, pin: str, produced: str,
                 source: str, reason: str) -> None:
        self.component = component
        self.class_name = class_name
        self.pin = pin
        self.produced = produced
        self.source = source
        self.reason = reason
        super().__init__(
            f"refusing to publish: component {component!r} ({class_name}) is "
            f"pinned to {pin} but this tree carries {produced} "
            f"(source was {source or 'unknown'}). {reason}"
        )


def verify_produced_tree(
    tree: Path | str, *, source_dir: Optional[Path | str] = None,
    source_dtypes: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Publish gate. Returns ``{component: dtype}`` for the produced tree.

    Raises :class:`ComponentDtypePinViolation` when a pinned component is
    narrower than its pin AND the source component was not — the producer
    truncated it. A source that ALREADY ships the component narrow is
    mirrorable: the pin is a fact about the architecture, not a licence to
    refuse someone else's bytes, and the load side widens it either way.
    """
    produced = component_dtypes_on_disk(tree)
    pins = component_pins(tree)
    if source_dtypes is None:
        source_dtypes = (component_dtypes_on_disk(source_dir)
                         if source_dir is not None else {})
    classes = model_index_classes(tree)
    for comp, fact in pins.items():
        got = produced.get(comp, "")
        if not got or not is_narrowing(got, fact.dtype):
            continue
        src = str(source_dtypes.get(comp, "") or "")
        if src and is_narrowing(src, fact.dtype):
            continue  # upstream's own precision — mirrored, not truncated here
        raise ComponentDtypePinViolation(
            comp, classes.get(comp, ""), fact.dtype, got, src, fact.reason)
    return produced


__all__ = [
    "ComponentDtypePinError",
    "ComponentDtypePinViolation",
    "DTYPE_BITS",
    "cast_exempt_components",
    "check_explicit_pin_conflict",
    "component_dtype",
    "component_dtypes_on_disk",
    "component_pins",
    "dtype_bits",
    "is_narrowing",
    "model_index_classes",
    "verify_produced_tree",
]
