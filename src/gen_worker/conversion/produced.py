"""ProducedVariant — what a tenant transform function returns per output.

A tenant's ``@conversion_function`` returns ``list[ProducedVariant]`` — one
entry per variant the job produces into the destination checkpoint. The
library uploads each variant's ``path`` (file OR directory) and attaches the
declared ``attributes`` to the upload-commit.

Attribute-bag ownership:
  - Tenant declares only keys that identify the variant or are read at
    inference/load time (dtype, file_layout, file_type, technique config,
    quant_library + family-required keys, etc.).
  - Library appends a single provenance key (``produced_by_job_id``) that
    joins to everything else (source ref, datasets, specs, training
    hyperparameters, timestamps) in the orchestrator job record.
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
