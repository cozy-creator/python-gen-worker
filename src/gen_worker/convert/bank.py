"""th#592 provider-hash download-skip — bank keys + manifest payloads.

A *bank key* is a sha256 over everything that determines a clone flavor's
published bytes, computed from provider METADATA alone (HF lfs.oid sha256s /
civitai API sha256s + the output spec + toolchain epoch). After a successful
publish, the flavor's file manifest (path/blake3/size + commit attrs) is
recorded hub-side under that key; a later clone of identical inputs looks
the key up first and, on a hit whose blobs all still exist in CAS, publishes
by reference — zero provider bytes downloaded, identical checkpoint id
(the streaming conversion is deterministic; proven live on ltx-2.3-dev,
e2e#118 runs 6/7).

Fail-open by design: every consumer treats bank errors as a miss.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol

# Bump when conversion output bytes change for identical inputs (cast/quant
# serialization, sharding thresholds, gguf toolchain): a bump = bank miss =
# one full re-clone per source, never corruption (a stale hit would still
# publish the internally-consistent PRIOR bytes).
TOOLCHAIN_EPOCH = "1"

BANK_KEY_PREFIX = "clone-flavor/v1:"


class SourcePlan(Protocol):
    """What bank key derivation needs from a provider plan
    (:class:`gen_worker.convert.ingest.HFSourcePlan` / ``CivitaiSourcePlan``)."""

    @property
    def provider(self) -> str: ...

    def bank_files(self) -> list[tuple[str, int, str]]: ...

    def bank_extra(self) -> dict[str, str]: ...


def flavor_bank_key(
    plan: SourcePlan,
    spec_label: str,
    *,
    layout_hint: str = "",
    quantize_components: list[str] | None = None,
    gguf_quant: str | None = None,
) -> str:
    """Derive the bank key for one output spec, or "" when the source has no
    complete per-file content identity (then there is no safe key)."""
    files = plan.bank_files()
    if not files:
        return ""
    canonical = json.dumps(
        {
            "epoch": TOOLCHAIN_EPOCH,
            "provider": str(plan.provider),
            "files": [[p, s, c] for p, s, c in files],
            "extra": dict(sorted(plan.bank_extra().items())),
            "spec": str(spec_label),
            "layout_hint": str(layout_hint or ""),
            "quantize_components": sorted(quantize_components or []),
            "gguf_quant": str(gguf_quant or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return BANK_KEY_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_bank_payload(
    *,
    files: list[dict[str, Any]],
    flavor: str,
    dtype: str,
    file_layout: str,
    file_type: str,
    metadata: Mapping[str, Any],
    repo_spec: Mapping[str, str],
    source_revision: str,
) -> dict[str, Any]:
    """The manifest recorded hub-side after a successful publish: everything
    a later bank-hit commit needs, with no local bytes."""
    return {
        "files": [
            {
                "path": str(f["path"]),
                "blake3": str(f["blake3"]),
                "size_bytes": int(f["size_bytes"]),
            }
            for f in files
        ],
        "flavor": str(flavor),
        "dtype": str(dtype),
        "file_layout": str(file_layout),
        "file_type": str(file_type),
        "metadata": dict(metadata),
        "repo_spec": {k: str(v) for k, v in dict(repo_spec or {}).items()},
        "source_revision": str(source_revision or ""),
    }


__all__ = [
    "BANK_KEY_PREFIX",
    "TOOLCHAIN_EPOCH",
    "SourcePlan",
    "build_bank_payload",
    "flavor_bank_key",
]
