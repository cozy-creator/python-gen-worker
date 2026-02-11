from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, get_args, get_origin, get_type_hints, Annotated

import msgspec


@dataclass(frozen=True)
class Clamp:
    """
    Declarative clamp for numeric payload fields.

    Intended usage:

        from typing import Annotated
        from gen_worker.payload_constraints import Clamp

        class Input(msgspec.Struct):
            steps: Annotated[int | float, Clamp(20, 50, cast="int")] = 25

    Notes:
    - This does not affect msgspec decoding; it runs post-decode in gen-worker.
    - Only fields explicitly annotated with Clamp are modified.
    """

    min: float
    max: float
    cast: str = ""  # "", "int", "float"
    round: str = "half_up"  # only used when cast="int"


def _is_annotated(t: Any) -> bool:
    # typing.Annotated becomes origin=typing.Annotated
    return get_origin(t) is Annotated


def _extract_clamp(t: Any) -> tuple[Any, Clamp | None]:
    """
    Return (base_type, clamp) for Annotated types; otherwise (t, None).
    """
    if not _is_annotated(t):
        return t, None
    args = get_args(t)
    if not args:
        return t, None
    base = args[0]
    clamp = None
    for meta in args[1:]:
        if isinstance(meta, Clamp):
            clamp = meta
    return base, clamp


def _half_up_to_int(x: float) -> int:
    # Avoid python's bankers rounding; this matches what we used in examples.
    return int(math.floor(x + 0.5))


def _apply_one(value: Any, clamp: Clamp) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)):
        return value

    x = float(value)
    if not math.isfinite(x):
        x = clamp.max

    if x < clamp.min:
        x = clamp.min
    if x > clamp.max:
        x = clamp.max

    cast = (clamp.cast or "").strip().lower()
    if cast == "float":
        return float(x)
    if cast == "int":
        if (clamp.round or "").strip().lower() == "half_up":
            return _half_up_to_int(x)
        # fallback: truncate
        return int(x)
    return x if isinstance(value, float) else int(x)


def apply_payload_constraints(obj: Any) -> Any:
    """
    Apply Clamp annotations to a decoded msgspec.Struct payload.

    This mutates the object in-place when possible. If the object is immutable,
    it returns the original object (best-effort).
    """
    if obj is None:
        return obj

    if isinstance(obj, msgspec.Struct):
        try:
            hints = get_type_hints(type(obj), include_extras=True)
        except Exception:
            hints = {}

        for name in getattr(obj, "__struct_fields__", ()) or ():
            ann = hints.get(name)
            base, clamp = _extract_clamp(ann) if ann is not None else (None, None)

            try:
                cur = getattr(obj, name)
            except Exception:
                continue

            # Recurse into nested structs/lists/dicts even if this field isn't clamped.
            new_cur = apply_payload_constraints(cur)
            if new_cur is not cur:
                try:
                    setattr(obj, name, new_cur)
                except Exception:
                    pass
                cur = new_cur

            if clamp is not None:
                new_v = _apply_one(cur, clamp)
                if new_v is not cur:
                    try:
                        setattr(obj, name, new_v)
                    except Exception:
                        pass
        return obj

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = apply_payload_constraints(v)
        return obj

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = apply_payload_constraints(v)
        return obj

    return obj


def apply_schema_constraints(schema: Any, t: Any) -> Any:
    """
    Best-effort: enrich msgspec.json.schema(...) output with Clamp metadata.

    This helps clients see min/max bounds in the published function schema,
    even though clamping is enforced server-side post-decode.
    """
    if not isinstance(schema, dict):
        return schema
    if not (isinstance(t, type) and issubclass(t, msgspec.Struct)):
        return schema

    props = schema.get("properties")
    if not isinstance(props, dict):
        return schema

    try:
        hints = get_type_hints(t, include_extras=True)
    except Exception:
        hints = {}

    for field in getattr(t, "__struct_fields__", ()) or ():
        ann = hints.get(field)
        _, clamp = _extract_clamp(ann) if ann is not None else (None, None)
        if clamp is None:
            continue
        ent = props.get(field)
        if not isinstance(ent, dict):
            continue
        # Apply to the property schema (works for type=number/integer, and also when ent uses anyOf).
        ent["minimum"] = float(clamp.min)
        ent["maximum"] = float(clamp.max)
    return schema
