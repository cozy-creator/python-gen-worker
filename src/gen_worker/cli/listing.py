"""Machine-readable endpoint introspection (``gen-worker run --list``).

Builds a stable JSON document describing the endpoint WITHOUT loading any
model: every routable function with its input JSON Schema, output type,
generator-ness, and model bindings, plus ``protocol_version`` +
``capabilities``. ``serve --list-functions --json`` emits the same
``functions`` array (one shared builder, one shape).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import msgspec

from .protocol import CAPABILITIES, PROTOCOL_VERSION, gen_worker_version

if TYPE_CHECKING:
    from .run import _SelectedFunction


def describe_binding(binding: Any) -> Dict[str, Any]:
    """Introspect one model binding into a JSON-able dict (no model load)."""
    from ..api.binding import BINDING_TYPES

    if isinstance(binding, BINDING_TYPES):
        out: Dict[str, Any] = {
            "type": type(binding).__name__,
            "provider": binding.provider,
            "ref": binding.ref,
        }
        for key in ("tag", "flavor", "revision", "dtype", "subfolder", "version"):
            v = getattr(binding, key, "")
            if v:
                out[key] = v
        files = tuple(getattr(binding, "files", ()) or ())
        if files:
            out["files"] = list(files)
        return out
    return {"type": type(binding).__name__}


def _function_input_schema(payload_type: Optional[type]) -> Dict[str, Any]:
    """JSON Schema for a function's payload Struct, top-level $ref inlined."""
    if payload_type is None:
        return {}
    try:
        s = msgspec.json.schema(payload_type)
    except Exception:
        return {}
    ref = s.get("$ref")
    defs = s.get("$defs")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and isinstance(defs, dict):
        key = ref.split("/")[-1]
        target = defs.get(key)
        if isinstance(target, dict):
            merged = dict(target)
            rest = {k: v for k, v in defs.items() if k != key}
            if rest:
                merged["$defs"] = rest
            return merged
    return s


def function_entries(
    candidates: List["_SelectedFunction"],
) -> List[Dict[str, Any]]:
    """The stable ``functions`` array, shared by ``run --list`` and
    ``serve --list-functions --json``."""
    out: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda c: c.fn_name):
        output_type = getattr(c, "output_type", None)
        out.append({
            "name": c.fn_name,
            "class": getattr(c.cls, "__name__", None) if c.cls is not None else None,
            "method": getattr(c, "attr_name", None) or None,
            "kind": getattr(c, "kind", ""),
            "is_generator": bool(getattr(c, "is_generator", False)),
            "input_schema": _function_input_schema(getattr(c, "payload_type", None)),
            "output": getattr(output_type, "__name__", None) if output_type else None,
            "models": {
                param: describe_binding(binding)
                for param, binding in (getattr(c, "bindings", {}) or {}).items()
            },
        })
    return out


def build_description(
    *,
    main_module: str,
    candidates: List["_SelectedFunction"],
) -> Dict[str, Any]:
    """Assemble the full description document (no model load)."""
    classes = sorted({
        c.cls.__name__ for c in candidates if c.cls is not None
    })
    kinds = sorted({str(getattr(c, "kind", "") or "") for c in candidates if getattr(c, "kind", "")})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "gen_worker_version": gen_worker_version(),
        "capabilities": list(CAPABILITIES),
        "endpoint": {
            "main_module": main_module,
            "kind": kinds[0] if len(kinds) == 1 else kinds,
            "classes": classes,
        },
        "functions": function_entries(candidates),
    }
