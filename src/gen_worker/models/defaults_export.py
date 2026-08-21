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
    return cast(dict[str, object], msgspec.json.decode(msgspec.json.encode(schema)))


def export_document() -> dict[str, object]:
    """``{names: [...], schemas: {name: json-schema}}`` — base types first, then the LoRA overlays, in declaration order."""
    names = list(defaults_vocabularies())
    return {
        "names": names,
        "schemas": {name: defaults_json_schema(name) for name in names},
    }


__all__ = ["defaults_json_schema", "export_document"]
