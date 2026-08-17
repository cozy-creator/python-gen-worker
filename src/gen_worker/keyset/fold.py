"""Fold a shipped/cached/traced graph axis with THIS pod's runtime axes.

The one arithmetic step every route shares. A key set arrives from one of three
places (:class:`KeySource`) and all three converge here, so a shipped key and a
traced key are the same value by construction rather than by care — the property
``boot_key`` already relied on when its memo path skipped the traces.
"""

from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Tuple

from .. import boot_phases, compile_cache as cc
from .identifiers import (
    ClassHash, ClosureDigest, CompiledGraphKey, GraphClassName, KeySetError,
    parse_compiled_graph_key)

__all__ = [
    "DerivedKeySet",
    "KeySource",
    "MemoVerdict",
    "fold_entry_keys",
]


class KeySource(enum.Enum):
    """WHERE this boot's graph axis came from. A typed state, not a tag.

    ``SHIPPED`` is pgw#1327's whole point: the mint lane derived these hashes and
    this pod read them as data. ``DURABLE`` is the same document off the
    platform-placed store root (pgw#1353) — derived by a trace some pod of THIS
    endpoint ran, on storage that outlived the pod that paid for it. ``MEMO`` is
    the same document out of this pod's own cache, written by a trace this
    machine ran on an earlier boot. ``TRACED`` means this process ran
    ``torch.export`` children, which on a serve pod is the thing #1327 exists to
    delete.

    ``DURABLE`` is a member rather than a flavour of ``MEMO`` because the two
    answer different questions and the fleet reads the answer. ``keys_from=memo``
    on a fleet pod is very nearly a contradiction — the pod-local cache is
    ``/tmp`` and a fresh pod's is empty — so a durable hit reported as ``memo``
    would be indistinguishable from the impossible case, and the ONE number this
    issue exists to move (how many pods pay the 805 s) would be unreadable off
    the boot events.

    ``HUB`` is th#2123's store, and it is a DISTINCT value for the same reason
    ``DURABLE`` is. The four tiers answer four different operational questions:
    ``shipped`` says the mint lane baked it, ``durable`` says this endpoint has
    storage that outlives a pod, ``hub`` says the platform holds it for an
    endpoint shape that has neither, and ``traced`` says somebody paid 805 s.
    Collapsing any pair would make "which fix is actually reaching the fleet"
    unanswerable off the boot events — which is the exact blindness pgw#1353
    was filed to end.
    """

    SHIPPED = "shipped"
    DURABLE = "durable"
    HUB = "hub"
    MEMO = "memo"
    TRACED = "traced"


class MemoVerdict(enum.Enum):
    """What the pod-local cache had to say, when anything asked it."""

    ABSENT = "absent"
    HIT = "hit"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DerivedKeySet:
    """The exact ``cg-key-v1`` set this boot will ask for, and where it came from.

    Neutral by design: ``boot_adopt`` consumes THIS type, so the serve path's
    result type does not live in the tracer's module and the serve path does not
    import a tracer to name its own answer (pgw#1327 / pgw#1328).
    """

    #: graph class -> that class's ``cg-key-v1``. THE thing resolve asks for.
    entry_keys: Mapping[GraphClassName, CompiledGraphKey]
    source: KeySource
    closure: ClosureDigest
    wall_ms: int = 0
    memo: MemoVerdict = MemoVerdict.ABSENT
    #: Trace-lane measurements. Zero/empty on every non-``TRACED`` source, which
    #: is itself the readable statement that no child process ran.
    workers: int = 0
    width_reason: str = ""
    traced: int = 0
    trace_ms: Mapping[GraphClassName, int] = field(default_factory=dict)
    nodes: Mapping[GraphClassName, int] = field(default_factory=dict)

    @property
    def keys(self) -> Tuple[CompiledGraphKey, ...]:
        """Every derived entry key, sorted — the batch a resolve carries."""
        return tuple(sorted(set(self.entry_keys.values())))


@contextlib.contextmanager
def _fold_span(family: str) -> Iterator[Any]:
    if not boot_phases.in_boot():
        yield None
        return
    with boot_phases.span(
        boot_phases.PHASE_KEY_FOLD, function=str(family or ""),
    ) as span:
        yield span


def fold_entry_keys(
    class_hashes: Mapping[GraphClassName, ClassHash],
    *,
    family: str,
) -> Dict[GraphClassName, CompiledGraphKey]:
    """Fold TCG graph axes with freshly stated runtime axes.

    ``sm`` and the toolchain digest are restated HERE, every time, from this
    process — never carried in a document. That is what makes a shipped key set
    machine-independent: the same class hashes on an sm_86 pod and an sm_100 pod
    fold to different keys, and a toolchain upgrade moves every key on the very
    next boot without anyone invalidating anything.
    """
    from gen_worker._vendor.torchcg.identity import from_axes, toolchain_axis_digest

    if not class_hashes:
        raise KeySetError("keyset_invalid", "an empty class set folds to no key")
    sm = str(cc.runtime_key().get("sm") or "")
    if not sm:
        raise KeySetError(
            "no_runtime_sm",
            "this process cannot read a CUDA compute capability, so it cannot "
            "fold a compiled-graph key; a key folded without an sm would name "
            "graphs no card produced")
    toolchain = toolchain_axis_digest(dict(cc.toolchain_digest()))
    with _fold_span(family) as span:
        entry_keys = {
            name: parse_compiled_graph_key(str(from_axes({
                "graph": str(class_hash),
                "sm": sm,
                "toolchain": toolchain,
            })))
            for name, class_hash in class_hashes.items()
        }
        if span is not None:
            span.note(f"classes={len(class_hashes)}")
    return entry_keys
