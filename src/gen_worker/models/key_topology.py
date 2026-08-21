"""Which tensor-KEY convention an artifact on disk is written in — a HEURISTIC,
and the fallback for a tree that carries no v2 topology stamp of its own.

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
denoiser whose key convention matches nothing here. The axis does not
apply to a VAE, a text encoder or a scheduler, and reporting "unknown" for
those is a fact, not a hedge; `gen_worker.discovery.decode_set` is where that
distinction becomes a refusal or a pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import msgspec

from .safetensors_header import read_header

# ── THIS VOCABULARY IS LOCAL, AND IT IS A HEURISTIC (pgw#1621) ───────────────
#
# These three tokens used to be `KEYS_*` constants in `tensor_layout_contract`,
# one of the five v1 DECODE AXES. They are gone from there because the axis
# they enumerated is now the TOPOLOGY half of a v2 lane stamp — a finite
# `{key -> logical shape}` map EXTRACTED MECHANICALLY from a reference
# checkpoint's headers, ratified in tensorfs `spec/v2/topologies/`, and
# compared exactly. `minimax-h3.diffusers@1` and `minimax-h3.native@1` are two
# such topologies related by a ratified morphism.
#
# THE STAMP IS THE AUTHORITATIVE ANSWER. What follows is a REGEX GUESS over
# attention-projection spellings, kept for the one thing the stamp cannot do
# here: a tree arriving at this worker carries no stamp of its own, and the
# question "is this denoiser addressed in a way anything in this image knows
# how to read" has to be answerable from its bytes, before a 71 GB fetch turns
# into a rented pod. So these tokens name CONVENTIONS, never handles: nothing
# intersects on them, no manifest carries them, and the only thing they feed is
# the fail-closed refusal in `discovery.decode_set`.
KEYS_NATIVE_FUSED_QKV = "native.fused-qkv"
KEYS_DIFFUSERS_SPLIT_QKV = "diffusers.split-qkv"
KEYS_TRANSFORMERS_SPLIT_QKV = "transformers.split-qkv"

_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (KEYS_NATIVE_FUSED_QKV, re.compile(r"\.(qkv_proj|to_qkv|qkv)\.")),
    (KEYS_DIFFUSERS_SPLIT_QKV, re.compile(r"\.(to_q|to_k|to_v)\.")),
    # th#1937's ruled keying for the transformers convention:
    # `*layers.N.self_attn.q_proj` OR `*layers.N.attention.self.query`, so
    # encoder-style text encoders answer alongside decoder-style ones.
    (KEYS_TRANSFORMERS_SPLIT_QKV,
     re.compile(r"layers?\.\d+\.(self_attn\.q_proj|attention\.self\.query)\.")),
)


def known_key_conventions() -> tuple[str, ...]:
    """Every convention this heuristic can name. A refusal quotes it so the
    reader sees what was tried, not just that nothing matched."""
    return tuple(token for token, _ in _RULES)

_MAX_FILES = 64

_SAMPLE = 6

_ATTENTION_SHAPED = re.compile(r"(^|\.)(attn|attention|self_attn)[\._]")


class SnapshotKeys(msgspec.Struct, frozen=True, kw_only=True):
    """What the header scan found, and where it looked."""

    topology: str
    denoiser: bool
    saw_tensors: bool
    attention_shaped: bool = False
    sample: tuple[str, ...] = ()

    @property
    def unclassified_denoiser(self) -> bool:
        """The one combination that must fail closed: a DENOISER carrying attention substructure spelled in no way this image recognizes."""
        return (
            self.denoiser
            and self.saw_tensors
            and self.attention_shaped
            and not self.topology
        )


def tensor_keys(files: Iterable[Path]) -> tuple[str, ...]:
    """Every tensor name in the given safetensors files' headers."""
    keys: list[str] = []
    for count, path in enumerate(files):
        if count >= _MAX_FILES:
            break
        header = read_header(
            path,
            why="the tensor key names classify this checkpoint's attention "
                "topology; an empty set silently classifies it as neither",
        )
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


def attention_shaped(keys: Iterable[str]) -> bool:
    """Whether a tensor-name set carries attention substructure at all."""
    return any(_ATTENTION_SHAPED.search(name) for name in keys)


def _scan(directory: Path) -> tuple[str, bool, tuple[str, ...]]:
    files = sorted(p for p in directory.glob("*.safetensors") if p.is_file())
    if not files:
        return "", False, ()
    seen: list[str] = []
    for path in files[:_MAX_FILES]:
        keys = tensor_keys([path])
        seen.extend(keys)
        topology = identify_keys(keys)
        if topology:
            return topology, True, tuple(sorted(seen)[:_SAMPLE])
    return "", attention_shaped(seen), tuple(sorted(seen)[:_SAMPLE])


def classify_snapshot(root: Path, component: str = "") -> SnapshotKeys:
    """Classify a snapshot's tree, reporting whether it is the DENOISER."""
    from ..component_vocab import denoiser_components

    base = Path(root)
    denoisers = set(denoiser_components())
    if base.is_file():
        keys = tensor_keys([base])
        return SnapshotKeys(topology=identify_keys(keys), denoiser=True,
                            saw_tensors=bool(keys),
                            attention_shaped=attention_shaped(keys),
                            sample=tuple(sorted(keys)[:_SAMPLE]))
    if not base.is_dir():
        return SnapshotKeys(topology="", denoiser=False, saw_tensors=False)

    if component:
        topology, attention, sample = _scan(base / component)
        return SnapshotKeys(
            topology=topology, denoiser=component in denoisers,
            saw_tensors=bool(sample), attention_shaped=attention,
            sample=sample)

    for name in denoiser_components():
        directory = base / name
        if not directory.is_dir():
            continue
        topology, attention, sample = _scan(directory)
        if sample:
            return SnapshotKeys(topology=topology, denoiser=True,
                                saw_tensors=True, attention_shaped=attention,
                                sample=sample)
    topology, attention, sample = _scan(base)
    return SnapshotKeys(
        topology=topology,
        denoiser=bool(sample) and not (base / "model_index.json").exists(),
        saw_tensors=bool(sample), attention_shaped=attention, sample=sample)
