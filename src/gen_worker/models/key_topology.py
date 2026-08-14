"""Which tensor-KEY convention an artifact on disk is written in.

Header reads only — no tensor data, no torch, no model construction. That is
the whole point: the answer must be available BEFORE a 71 GB fetch turns into
a rented pod that discovers the mismatch as `Cannot detect the model type`
from an md5-over-key:shape lookup five libraries down.

Classification is by the keys that DIFFER, not by counting: the minimax-h3
diffusers repackaging and the minimax-native tree share exactly one key out of
638/535, and the ATTENTION PROJECTIONS are where they part — fused
`…attn.qkv_proj` versus split `…attn.to_q` / `to_k` / `to_v`. That
discriminator is used directly rather than through a block-prefix pattern,
because the block prefix varies across families (`transformer_blocks`,
`single_transformer_blocks`, `blocks`, `down_blocks.N.attentions.N.…`) and the
projection split does not.

**Unclassified is not silently OK.** The caller is told three things — the
token, whether the tree is in the DENOISER position, and whether any tensors
were read — and it fails closed on the one combination that is dangerous: a
denoiser whose key convention matches nothing registered. The axis does not
apply to a VAE, a text encoder or a scheduler, and reporting "unknown" for
those is a fact, not a hedge; `gen_worker.discovery.decode_set` is where that
distinction becomes a refusal or a pass.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Iterable

import msgspec

from .safetensors_header import header_len_ok
from .tensor_layout_contract import (
    KEYS_DIFFUSERS_SPLIT_QKV,
    KEYS_NATIVE_FUSED_QKV,
    KEYS_TRANSFORMERS_NATIVE,
)

# Ordered: the first rule that matches wins. FUSED is checked first because a
# tree carrying fused projections is the one no diffusers class can ingest, and
# a tree carrying both spellings is a repackaging in progress, not a diffusers
# tree.
_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (KEYS_NATIVE_FUSED_QKV, re.compile(r"\.(qkv_proj|to_qkv|qkv)\.")),
    (KEYS_DIFFUSERS_SPLIT_QKV, re.compile(r"\.(to_q|to_k|to_v)\.")),
    (KEYS_TRANSFORMERS_NATIVE,
     re.compile(r"\.(q_proj|k_proj|v_proj|query|key|value)\.")),
)

#: Header bytes are bounded by `header_len_ok`; this caps how many FILES we
#: open. It is high, and the scan stops at the first file that classifies,
#: because the cap now sits on a FAIL-CLOSED path: a 30-shard denoiser whose
#: leading shards hold only embeddings would otherwise read as unclassified
#: and refuse. Attention projections appear early in practice, so the walk
#: costs one or two headers on every real tree and the cap only bounds the
#: pathological one.
_MAX_FILES = 64

#: How many keys a refusal quotes. Enough to recognize the convention, few
#: enough that the message stays readable.
_SAMPLE = 6


class SnapshotKeys(msgspec.Struct, frozen=True, kw_only=True):
    """What the header scan found, and where it looked."""

    topology: str          # a registered token, or "" for unclassified
    denoiser: bool         # the scanned tree is in the DENOISER position
    saw_tensors: bool      # any safetensors header was actually read
    sample: tuple[str, ...] = ()

    @property
    def unclassified_denoiser(self) -> bool:
        """The one combination that must fail closed."""
        return self.denoiser and self.saw_tensors and not self.topology


def tensor_keys(files: Iterable[Path]) -> tuple[str, ...]:
    """Every tensor name in the given safetensors files' headers."""
    keys: list[str] = []
    for count, path in enumerate(files):
        if count >= _MAX_FILES:
            break
        try:
            with open(path, "rb") as handle:
                raw = handle.read(8)
                if len(raw) < 8:
                    continue
                (length,) = struct.unpack("<Q", raw)
                if not header_len_ok(length):
                    continue
                header = json.loads(handle.read(length))
        except (OSError, ValueError, struct.error):
            continue
        if isinstance(header, dict):
            keys.extend(k for k in header if k != "__metadata__")
    return tuple(keys)


def identify_keys(keys: Iterable[str]) -> str:
    """The key convention a tensor-name set is written in, or ``""``."""
    names = list(keys)
    for token, pattern in _RULES:
        if any(pattern.search(name) for name in names):
            return token
    return ""


def _scan(directory: Path) -> tuple[str, tuple[str, ...]]:
    """Classify a directory's shards, stopping at the first file that answers.

    Early exit matters on the fail-closed path: reading every shard of a
    30-way tree to reach the same answer the first one gave is wasted IO, and
    reading only the first few and giving up would REFUSE a tree whose leading
    shards happen to hold embeddings.
    """
    files = sorted(p for p in directory.glob("*.safetensors") if p.is_file())
    if not files:
        return "", ()
    seen: list[str] = []
    for path in files[:_MAX_FILES]:
        keys = tensor_keys([path])
        seen.extend(keys)
        topology = identify_keys(keys)
        if topology:
            return topology, tuple(sorted(seen)[:_SAMPLE])
    return "", tuple(sorted(seen)[:_SAMPLE])


def classify_snapshot(root: Path, component: str = "") -> SnapshotKeys:
    """Classify a snapshot's tree, reporting whether it is the DENOISER.

    ``component`` names the subdirectory to read; empty scans the denoiser
    component dirs and then the root. A ROOT-layout tree — one with no
    ``model_index.json`` — is itself the denoiser (the singlefile and
    sharded-transformers artifacts the quantized loaders detect), which is why
    the position is derived here rather than guessed by the caller.
    """
    from ..component_vocab import denoiser_components

    base = Path(root)
    denoisers = set(denoiser_components())
    if base.is_file():
        keys = tensor_keys([base])
        return SnapshotKeys(topology=identify_keys(keys), denoiser=True,
                            saw_tensors=bool(keys),
                            sample=tuple(sorted(keys)[:_SAMPLE]))
    if not base.is_dir():
        return SnapshotKeys(topology="", denoiser=False, saw_tensors=False)

    if component:
        topology, sample = _scan(base / component)
        return SnapshotKeys(
            topology=topology, denoiser=component in denoisers,
            saw_tensors=bool(sample), sample=sample)

    for name in denoiser_components():
        directory = base / name
        if not directory.is_dir():
            continue
        topology, sample = _scan(directory)
        if sample:
            return SnapshotKeys(topology=topology, denoiser=True,
                                saw_tensors=True, sample=sample)
    topology, sample = _scan(base)
    # A root-layout tree IS the denoiser; a pipeline root that merely happens
    # to hold loose tensors beside a `model_index.json` is not.
    return SnapshotKeys(
        topology=topology,
        denoiser=bool(sample) and not (base / "model_index.json").exists(),
        saw_tensors=bool(sample), sample=sample)
