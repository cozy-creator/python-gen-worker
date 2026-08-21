"""A dtype cast is TREE-wide; precision pins are PER-COMPONENT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from ..families.facts import ComponentDtype, component_dtypes_for_classes

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

_DTYPE_SEPARATORS = ("_", "-")


def _fold_dtype(dtype: str) -> str:
    key = str(dtype or "").strip().lower()
    for sep in _DTYPE_SEPARATORS:
        key = key.replace(sep, ":")
    return key


_DTYPE_BITS_FOLDED: Dict[str, int] = {
    _fold_dtype(k): v for k, v in DTYPE_BITS.items()
}


class ComponentDtypePinError(ValueError):
    """A requested conversion contradicts a component's load-dtype pin."""

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
    """Storage width of a dtype spelling; 0 when unknown (never compared)."""
    return _DTYPE_BITS_FOLDED.get(_fold_dtype(dtype), 0)


def is_narrowing(requested: str, pin: str) -> bool:
    """True when ``requested`` stores fewer bits than the pin requires."""
    req, pinned = dtype_bits(requested), dtype_bits(pin)
    return bool(req and pinned and req < pinned)


def model_index_classes(tree: Path | str) -> Dict[str, str]:
    """``{component: class name}`` a tree's ``model_index.json`` declares."""
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
    """``{component: ComponentDtype}`` the tree's own classes pin."""
    return component_dtypes_for_classes(model_index_classes(tree))


def cast_exempt_components(tree: Path | str, requested_dtype: str) -> Dict[str, ComponentDtype]:
    """Components a cast to ``requested_dtype`` must NOT touch."""
    return {
        comp: fact for comp, fact in component_pins(tree).items()
        if is_narrowing(requested_dtype, fact.dtype)
    }


def check_explicit_pin_conflict(
    tree: Path | str, requested_dtype: str, named_components: Iterable[str] | None,
) -> None:
    """Refuse when the caller NAMED a pinned component as a conversion target."""
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
    """``{component: dtype}`` read from the produced tree's own safetensors headers — the per-component precision report a publish carries so a downcast can never again be invisible."""
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
    """A tree about to be published carries a component NARROWER than its pin, and the source was not — i.e."""

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
    """Publish gate."""
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
            continue
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
