"""ProducedFlavor — what a tenant transform hands to ``publish_flavors``.

A producer endpoint builds ``list[ProducedFlavor]`` — one entry per flavor
the job produces into the destination checkpoint — and calls
``gen_worker.convert.publish_flavors(ctx, flavors)``. The library uploads each
flavor's ``path`` (file OR directory) as one Tensorhub commit and attaches
the declared ``attributes`` to the commit payload.

Attribute-bag ownership (server-authoritative metadata):
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

    **A18 / §1.32(d), pgw#1319: there is no ``flavor`` field.** It was the last
    of the axis — a producer-local label that named no catalog row but still
    decided the ``precision_class`` stamp through
    ``classify_flavor_token``, which is deleted with it. What the bytes ARE is
    stated by the ``dtype`` attribute and, when the producer knows it, an
    ``artifact_contract`` attribute (``ns.name@N``, PROVEN hub-side against the
    safetensors header — §1.33); what LANE they are on is stated by a
    ``precision_class`` attribute, DECLARED from a structural fact and checked
    against :data:`gen_worker.models.ladder.PRECISION_CLASSES`. A tree of
    sub-16-bit bytes that declares none is a refusal, not an unstamped publish.

    A job that emits several artifacts hands over several ``ProducedFlavor``
    entries: N publishes joining ONE tag group. There is no flavor-label set
    on a single publish (``flavors`` was deleted with the wire field).
    """

    path: _PathField
    attributes: dict = msgspec.field(default_factory=dict)
    extra_files: list[_PathField] = msgspec.field(default_factory=list)


__all__ = ["ProducedFlavor"]
