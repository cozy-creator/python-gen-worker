"""Ergonomic CLI payload args — httpie-style ``field=value`` instead of JSON.

Instead of ``--payload '{"prompt":"a cat","seed":42}'`` a user can type::

    gen-worker run "a cat" seed=42 guidance=3.5 hires=true

The grammar (decided in ``progress.json`` #350):

* ``field=value``  — set ``field``; value COERCED by the payload Struct's field
  type (``seed=42`` -> int, ``prompt=hi`` -> str, ``hires=true`` -> bool).
* ``field:=json``  — raw JSON value for lists/objects/explicit types:
  ``tags:='["a","b"]'``, ``size:=1024``.
* ``field@path``   — load the field's value from a file (long prompts, etc.).
* bare positional  — the payload's PRIMARY field (first required ``str`` field),
  so ``run "a cat"`` works without naming the prompt field.
* ``a.b=value``    — dotted key sets a nested object (best-effort coercion).

``--payload '<json>'`` stays the escape hatch; ``field=`` tokens merge over it.
Coercion uses the function's ``msgspec.Struct`` so errors and bounds match the
real decode path.
"""

from __future__ import annotations

import json
import re
import types as _types
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

_KEY = r"[A-Za-z_][\w.]*"
_RE_JSON = re.compile(rf"^(?P<key>{_KEY}):=(?P<val>.*)$", re.DOTALL)
_RE_FILE = re.compile(rf"^(?P<key>{_KEY})@(?P<val>.*)$", re.DOTALL)
_RE_SET = re.compile(rf"^(?P<key>{_KEY})=(?P<val>.*)$", re.DOTALL)


class ArgError(Exception):
    """A bad ergonomic-arg token. The caller maps this to a usage error."""


def looks_like_field_token(tok: str) -> bool:
    """True if ``tok`` is a ``field=`` / ``field:=`` / ``field@`` token.

    A bare positional (primary value) is NOT a field token. Used to decide
    whether an argument vector is ergonomic args vs. a single JSON/@file blob.
    """
    return bool(_RE_JSON.match(tok) or _RE_FILE.match(tok) or _RE_SET.match(tok))


# --------------------------------------------------------------------------
# Type-directed coercion against the payload Struct
# --------------------------------------------------------------------------

_NODEFAULT = getattr(msgspec, "NODEFAULT", object())


def _struct_field_types(struct_type: Any) -> Dict[str, Any]:
    try:
        return {f.name: f.type for f in msgspec.structs.fields(struct_type)}
    except Exception:
        return {}


def _is_required(field: Any) -> bool:
    return (
        getattr(field, "default", _NODEFAULT) is _NODEFAULT
        and getattr(field, "default_factory", _NODEFAULT) is _NODEFAULT
    )


def primary_field(struct_type: Any) -> Optional[str]:
    """The field a bare positional fills: first required ``str``, else first
    required, else first declared field. ``None`` if the struct has no fields."""
    try:
        fields = list(msgspec.structs.fields(struct_type))
    except Exception:
        return None
    for f in fields:
        if _is_required(f) and f.type is str:
            return f.name
    for f in fields:
        if _is_required(f):
            return f.name
    return fields[0].name if fields else None


def _to_bool(value: str) -> bool:
    low = value.strip().lower()
    if low in ("true", "1", "yes", "y", "on"):
        return True
    if low in ("false", "0", "no", "n", "off"):
        return False
    raise ArgError(f"expected a boolean (true/false), got {value!r}")


def _guess(value: str) -> Any:
    """Schema-less coercion for nested/unknown fields (httpie-ish)."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except (ValueError, TypeError):
            pass
    return value


def coerce(value: str, typ: Any) -> Any:
    """Coerce a string token to the declared field type (best-effort)."""
    if typ is None or typ is Any:
        return _guess(value)
    origin = typing.get_origin(typ)
    args = typing.get_args(typ)
    if origin is typing.Union or origin is getattr(_types, "UnionType", ()):
        if value == "" and type(None) in args:
            return None
        for a in (a for a in args if a is not type(None)):
            try:
                return coerce(value, a)
            except (ArgError, ValueError, TypeError):
                continue
        raise ArgError(f"value {value!r} does not match {typ}")
    if typ is str:
        return value
    if typ is bool:
        return _to_bool(value)
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    if origin in (list, tuple, set, dict):
        raise ArgError(
            f"field expects {getattr(origin,'__name__',origin)}; pass it as raw "
            f"JSON with ':=' (e.g. key:='[1,2]'), not '='"
        )
    # Enums, Literal, nested Struct, bytes, etc.: fall back to a best-effort guess.
    return _guess(value)


# --------------------------------------------------------------------------
# Token -> payload assembly
# --------------------------------------------------------------------------

def _set_path(obj: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def build_payload(
    tokens: List[str],
    struct_type: Any,
    *,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a payload dict from ergonomic tokens, coerced via ``struct_type``.

    ``base`` (e.g. a ``--payload`` JSON object) is the starting point; tokens
    merge over it. Raises :class:`ArgError` on a malformed token.
    """
    out: Dict[str, Any] = dict(base or {})
    field_types = _struct_field_types(struct_type)
    primary = primary_field(struct_type)
    primary_set = False

    for tok in tokens:
        m = _RE_JSON.match(tok)
        if m:
            key, raw = m.group("key"), m.group("val")
            try:
                val = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ArgError(f"{key}:= is not valid JSON: {e}") from e
            _set_path(out, key, val)
            continue
        m = _RE_FILE.match(tok)
        if m:
            key, path = m.group("key"), m.group("val")
            p = Path(path)
            if not p.exists():
                raise ArgError(f"{key}@: file not found: {path}")
            ftyp = field_types.get(key)
            if ftyp is bytes or ftyp is bytearray:
                import base64
                _set_path(out, key, base64.b64encode(p.read_bytes()).decode("ascii"))
            else:
                _set_path(out, key, coerce(p.read_text(encoding="utf-8"), ftyp))
            continue
        m = _RE_SET.match(tok)
        if m:
            key, raw = m.group("key"), m.group("val")
            if "." in key:
                _set_path(out, key, _guess(raw))
            elif not field_types:
                # No schema available (e.g. invoke without an importable module):
                # accept any field, guess its type (httpie-ish).
                out[key] = _guess(raw)
            elif key not in field_types:
                raise ArgError(
                    f"unknown field {key!r}; valid fields: {sorted(field_types)}"
                )
            else:
                out[key] = coerce(raw, field_types[key])
            continue
        # Bare positional -> primary field.
        if primary is None:
            raise ArgError(
                f"bare value {tok!r} but the payload has no field to fill; "
                "use field=value"
            )
        if primary_set:
            raise ArgError(
                f"more than one bare positional value (second was {tok!r}); "
                f"name the field explicitly, e.g. {primary}=..."
            )
        out[primary] = coerce(tok, field_types.get(primary))
        primary_set = True

    return out
