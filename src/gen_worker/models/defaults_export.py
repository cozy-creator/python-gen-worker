"""The mechanical sync artifact (pgw#1377 point 5): ``{names, schemas}``.

Generated from the ``Defaults`` structs — never hand-maintained. The hub's
core consumes the NAMES only (the recognized-set guard on the ``model``
column); the schema blobs become load-bearing when th#2141 adds write-time
validation. Rides the release derive document as ``defaults_schema``
(th#2133) on the same emitter run as the graphs[] derive (pgw#1370), and the
``gen-worker models export`` CLI emits it standalone.

Every field in every struct has a default, so the emitted schemas naturally
validate PARTIAL row objects — the hub's JSONB is exactly that. Unknown
fields are deliberately NOT forbidden (no ``additionalProperties: false``):
the evolution rule says an old reader ignores newer rows' fields, and the
validator must not be stricter than the decoder.
"""

from __future__ import annotations

from typing import cast

import msgspec

from .model_types import defaults_vocabularies

_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def defaults_json_schema(name: str) -> dict[str, object]:
    """Standalone JSON Schema (draft 2020-12) for one recognized name."""
    vocab = defaults_vocabularies()
    cls = vocab.get(name)
    if cls is None:
        raise KeyError(
            f"no model type named {name!r}; recognized: {sorted(vocab)}"
        )
    raw = msgspec.json.schema(cls)
    defs = dict(raw.get("$defs") or {})
    body = defs.pop(cls.__name__, None)
    if body is None:
        body = {k: v for k, v in raw.items() if k not in ("$ref", "$defs")}
    schema: dict[str, object] = {
        "$schema": _SCHEMA_DIALECT,
        "$id": f"https://schemas.cozy.art/gen-worker/models/{name}.defaults.schema.json",
        **body,
    }
    schema["title"] = name
    if defs:
        schema["$defs"] = defs
    # Round-trip through JSON so Python-native default VALUES (tuples, Knob
    # instances) come out JSON-safe — same canonicalization the families
    # exporter established.
    return cast(dict[str, object], msgspec.json.decode(msgspec.json.encode(schema)))


def export_document() -> dict[str, object]:
    """``{names: [...], schemas: {name: json-schema}}`` — base types first,
    then the LoRA overlays, in declaration order."""
    names = list(defaults_vocabularies())
    return {
        "names": names,
        "schemas": {name: defaults_json_schema(name) for name in names},
    }


__all__ = ["defaults_json_schema", "export_document"]
