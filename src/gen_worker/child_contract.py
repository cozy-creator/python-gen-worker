"""The vocabulary a serving parent and any trace/compile child share.

pgw#1215 (th#1834 Phase 3). These types lived in ``mint_process``, whose OTHER
half is the out-of-process mint DRIVER — and that driver is being replaced by a
serving parent that supervises compile children directly, one graph class each.
The vocabulary is not:
five modules that survive the rewrite already import it (``executor``,
``boot_key``, ``boot_adopt``, ``measure_child``, ``local_serve``), plus both
operator rigs.

So it moves out FIRST, on its own, while nothing else changes. That ordering is
the point: with the substrate gone the driver's removal is a CUT, and with it
still inside it would be an untangle — a deletion that has to preserve two
thirds of what it deletes is how a rewrite acquires a compatibility shim nobody
asked for.

What belongs here is anything a parent must SAY to a child about the work:
which checkpoints (``MintSlot``), what the declaration asked for
(``CompileSpec``), what identity those resolve to (``slot_subjects``), and
how the child reports back while it runs (``MintFrame``). What does not belong
here is any statement about the child's PROCESS — spawn, supervision, exit
codes and reports stay with the driver that owns them.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import msgspec

from . import graph_facts
from .api.binding import ModelRef, wire_ref

#: Every child progress frame is one stdout line with this prefix. Anything
#: else the child (or a library it imports) prints is diagnostic tail, never
#: parsed — a mint must not be steerable by a stray print.
FRAME_PREFIX = "MINT_FRAME "


class CompileSpec(msgspec.Struct, frozen=True, kw_only=True):
    """The declared compile contract, flattened — exactly the facts the CHILD
    reads.

    The PARENT sends this rather than letting the child re-derive it from the
    decorator, because the class-scoped ``guidance_scales``/``text_lens``
    unions live on the spec and not on the decl: a child rebuilding from
    ``@endpoint`` alone would export a different declaration than the parent
    asked for. It is NOT a key-derivation wire — since pgw#758/#1010 the child
    computes no key at all; the parent stamps it from the returned envelope.

    pgw#1034 therefore dropped ``regional``/``text_len``/``dynamic``: they
    crossed the wire and no child consumer read them
    (``fleet_cells.aot_export_spec`` and ``compile_cache.resolve_targets`` read
    family/targets/shapes/text_lens/guidance/bucket, and nothing else does).
    Any field added back must name the child code that reads it.
    """

    shapes: Tuple[Tuple[int, ...], ...] = ()
    targets: Tuple[str, ...] = ()
    family: str = ""
    lora_bucket: int = 0
    guidance_scales: Tuple[float, ...] = ()
    text_lens: Tuple[int, ...] = ()


class MintSlot(msgspec.Struct, frozen=True, kw_only=True):
    """One setup slot, as the parent resolved it. Present and complete, or absent.

    pgw#974. This used to be parallel slot-keyed dicts on ``MintRequest`` —
    ``snapshots`` (bytes) and ``slot_bindings`` (identity, pgw#969) — written
    by separate statements, each independently allowed to be empty. Some of
    the combinations that describes are incoherent, and one of them cost two
    L40S pods: ``{"pipeline": "/tmp/x"}`` with no binding decoded, type-checked
    and looked complete, and the child died 0.0 s into ``warmup_forward`` at
    ``ctx.slots["pipeline"]``. ``ref`` and ``path`` therefore carry no
    defaults: a slot with bytes and no identity cannot be constructed and
    cannot be decoded.

    A slot the parent did not resolve is ABSENT from the map — never a present
    one with a hole in it. ``child_preflight.assert_slots_resolvable`` still
    refuses one that the endpoint declares and does not mark optional.

    * ``ref`` — WHICH checkpoint. ``ctx.slots`` is built from bindings, and a
      child re-runs discovery, so a hub-catalog slot (``Slot(selected_by=...)``
      with no ``default_checkpoint=``, which is sdxl's shape) rediscovers
      nothing at all. A slot WITH a code default is the quieter half of the
      same defect: the child resolves the DECLARED checkpoint while the parent
      serves the hub's pick, and traces graphs for a model this pod never runs.
    * ``path`` — WHERE its bytes already are, materialized by the parent, so
      the child never touches the network: a mint is compute, and a mint
      process that could download is one that can stall on a lemon host.

    th#1941: nothing rides beside them. The hub composes the manifest, so
    ``path`` names a COHERENT tree by construction — there is no second tree
    for the child to be handed and no narrowing for it to detect.
    """

    ref: ModelRef
    path: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError(
                f"a resolved slot must name the tree its bytes are in; got an "
                f"empty path for {wire_ref(self.ref)!r}")


def slot_subjects(
    slots: Mapping[str, MintSlot],
    digests: Optional[Mapping[str, str]] = None,
) -> Tuple[graph_facts.SlotSubject, ...]:
    """The resolved SUBJECT of one arm or one boot trace (pgw#1113).

    THE single derivation, so the arm token, the local-store memo and the
    boot-key memo cannot disagree about which checkpoint a pipeline is bound
    to. ``path`` is deliberately excluded — where the bytes were materialized
    is a location on this machine, never an identity.
    """
    have = dict(digests or {})
    return tuple(
        graph_facts.SlotSubject(
            slot=str(name),
            refs=(wire_ref(slot.ref),),
            snapshot_digest=str(have.get(str(name), "") or ""),
        )
        for name, slot in sorted(slots.items())
    )


class MintFrame(msgspec.Struct, frozen=True, kw_only=True):
    """One progress frame. Reporting only — never a liveness claim."""

    phase: str = ""
    step: int = 0
    total: int = 0
    note: str = ""


def frame_line(
    phase: str = "", step: int = 0, total: int = 0, note: str = "",
) -> str:
    """One wire line for a progress frame — the child's half of the protocol,
    kept here so both sides share one encoder."""
    body = msgspec.json.encode(
        MintFrame(phase=phase, step=step, total=total, note=note[:400]))
    return FRAME_PREFIX + body.decode() + "\n"


__all__ = [
    "FRAME_PREFIX",
    "CompileSpec",
    "MintFrame",
    "MintSlot",
    "frame_line",
    "slot_subjects",
]
