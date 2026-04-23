"""ProducedVariant — what a tenant transform function returns per output.

A tenant's ``@training_function`` returns ``list[ProducedVariant]`` — one
entry per variant the job produces into the destination checkpoint. The
library uploads each variant's ``path`` (file OR directory) and attaches
the declared ``attributes`` to the upload-commit.

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

import msgspec


class ProducedVariant(msgspec.Struct):
    """One variant emitted by a transform tenant function.

    Fields:
      - path: file (e.g. ``model.safetensors``, ``model.q4_k_m.gguf``) OR
        directory (e.g. a ``save_pretrained`` output tree).
      - attributes: per-variant attribute bag. See module docstring for
        what belongs here vs what belongs in the orchestrator job record.
      - extra_files: rare escape hatch — sibling artifacts attached to the
        same variant (e.g. a tokenizer.json next to a non-tree output).
    """

    path: Path
    attributes: dict = msgspec.field(default_factory=dict)
    extra_files: list[Path] = msgspec.field(default_factory=list)


__all__ = ["ProducedVariant"]
