"""ProducedFlavor — what a tenant transform returns per output.

A tenant's ``@conversion`` returns ``list[ProducedFlavor]`` — one
entry per flavor the job produces into the destination checkpoint. The
library uploads each flavor's ``path`` (file OR directory) and attaches
the declared ``attributes`` to the final checkpoint flavor publish payload.

Attribute-bag ownership (issue #22 — server-authoritative metadata):
  - Tenant declares ONLY tenant-specific attributes (technique config,
    quant_library + family-required keys, human-readable labels).
  - dtype / file_layout / file_type / kind / library are SERVER-INFERRED
    from the uploaded files — tenant SHOULD NOT emit them. The server
    reads the bytes and writes canonical values regardless of what the
    tenant supplies. Tenant-supplied values are logged as divergence but
    not used.
  - Attributes with keys starting with ``_`` are REJECTED by the server
    (reserved for server-computed reserved fields like
    ``_tensor_key_fingerprint``).
  - Do NOT duplicate inputs-to-the-job on the variant — that's drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import msgspec


# JSON-schema bridge for pathlib.Path: discovery emits a per-class JSON schema
# for every msgspec.Struct in the function signature, and msgspec's schema
# generator rejects custom types unless they're annotated with extra_json_schema
# or a schema_hook. ProducedFlavor's `path` / `extra_files` are filesystem
# pointers used by the library to know what to upload — on the wire they're
# absolute paths represented as strings. The annotation keeps the field typed
# as Path for tenant ergonomics while making it discoverable.
_PathField = Annotated[Path, msgspec.Meta(extra_json_schema={"type": "string"})]


class ProducedFlavor(msgspec.Struct):
    """One checkpoint flavor emitted by a transform tenant function.

    Fields:
      - path: file (e.g. ``model.safetensors``, ``model.q4_k_m.gguf``) OR
        directory (e.g. a ``save_pretrained`` output tree).
      - attributes: per-flavor attribute bag. See module docstring for
        what belongs here vs what belongs in the orchestrator job record.
      - extra_files: rare escape hatch — sibling artifacts attached to the
        same flavor (e.g. a tokenizer.json next to a non-tree output).
      - flavor: optional owner-facing row label such as ``bf16``, ``fp8``,
        or ``int4``. When empty, the library falls back to attributes such
        as ``flavor``.
      - flavors: optional full flavor-label set such as
        ``["fp8", "flashpack", "aio"]``. ``flavor`` is kept as the primary
        compatibility label and is included automatically when present.
    """

    path: _PathField
    attributes: dict = msgspec.field(default_factory=dict)
    extra_files: list[_PathField] = msgspec.field(default_factory=list)
    flavor: str = ""
    flavors: list[str] = msgspec.field(default_factory=list)


__all__ = ["ProducedFlavor"]
