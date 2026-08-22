"""The layout rung a worker is ON, and the one it is EARNING (pgw#1645).

`research/layout-morphisms/0-DESIGN.md` §3, the worker half. A mint compiles
against a byte layout and DECLARES it (tcg#83); while compiling it also records
the layout inductor actually asked for, as a wishlist. Those two facts, read off
the artifact that is serving RIGHT NOW, are the whole of this module:

* the artifact declares `torch.contiguous@1` and wished for nothing — this IS
  the ideal layout and there is no rung above it;
* it declares one layout and wished for another — the worker is serving the
  stored layout and EARNING the ideal one. That is a rung it has not reached
  yet, not a fault, and it must read that way in the status or an operator
  chasing a slow pod will chase this instead;
* the wish is outside the ratified catalog — it is a CANDIDATE, emitted for a
  human to ratify. Machines derive along ratified morphisms and never invent
  one;
* the wish is ratified and the platform will not deliver it yet — a DECLINE,
  typed and named, never a silently dropped wish.

**Nothing here transforms anything.** The rung is a reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Mapping, Sequence, Tuple

__all__ = [
    "LayoutRung",
    "LayoutState",
    "LayoutWish",
    "PER_ELEMENT_DECLINE",
    "fill_path",
    "read_rung",
    "rungs_of",
]

#: The reason a ratified, compiler-wished arrangement is not delivered TODAY,
#: stated once and dated so it can be retired rather than accumulated.
#:
#: tensorfs#157, MEASURED on a 4070 (release, best of 10): an identity fill runs
#: at 5.15 GiB/s and a `torch.channels_last-2d@1` fill at 0.12 GiB/s -- ~31 ns
#: per element, because that arrangement's innermost storage axis is the channel
#: and its source stride is H*W, so the fold in `Plan::for_each_run` folds
#: NOTHING and the copy becomes one run per element. Extrapolated over SDXL's
#: 635 MiB conv-weight set that is ~5 s of host time per load, against a step
#: cost the same arrangement saves single-digit percent of. Paying it by default
#: would be a regression wearing an optimization's name.
#:
#: WHAT REPLACES THIS, and it is not a guess: `FillStats::runs_per_element` is
#: the number, computed by the one implementation that folds the runs. The
#: moment the fill path is reachable from Python (tensorfs#159 / pgw#1648) this
#: constant dies and the decision is read from the stats of the fill that would
#: be performed.
#: gen-worker deliberately does NOT compute a run count of its own -- that would
#: be a second implementation of the fold, which is the defect this whole
#: program keeps removing.
PER_ELEMENT_DECLINE = (
    "tensorfs#157: this arrangement's fill is a per-element gather "
    "(0.12 GiB/s measured, ~5 s host-side for SDXL's conv set), so the "
    "platform declines to deliver it. The wish stands and the candidate is "
    "recorded; it is the DELIVERY that is withheld, not the ratification"
)


class LayoutState(StrEnum):
    """Where a serving artifact sits against the layout it wished for.

    The values are INTENDED as a wire contract in the same way `EagerPhase`'s
    are -- a member may be added, and renaming one would orphan history the day
    something joins to them. **Nothing consumes them yet**, and saying otherwise
    would be a claim about a contract that does not exist: today the rungs reach
    `ServeAdoption.facts()` -> `worker.mint_facts()` -> a `logger.debug` call,
    and no hub query groups on them. Freeze the spellings anyway -- they are
    cheap to fix now and unfixable once a query does.
    """

    #: The mint asked for no layout change. This artifact IS at its ideal.
    NO_WISH = "no_wish"
    #: Served layout == wished layout. The delivery already happened.
    AT_IDEAL = "at_ideal"
    #: A ratified wish is outstanding and deliverable: serving the stored
    #: layout while the ideal-layout mint is pending. EARNING, not broken.
    EARNING = "earning"
    #: Ratified, and the platform will not deliver it yet. See
    #: :data:`PER_ELEMENT_DECLINE`.
    DECLINED = "declined"
    #: The wish names no ratified morphism. It rides out as a candidate for
    #: ratification and the mint keeps the stored layout -- the permanent
    #: fallback the design forbids removing.
    CANDIDATE = "candidate"
    #: Two constants want two different ratified arrangements. There is no
    #: single ideal layout for this graph, so there is no rung to earn: a
    #: re-mint would move the copies rather than delete them.
    NO_SINGLE_IDEAL = "no_single_ideal"


@dataclass(frozen=True)
class LayoutWish:
    """One constant whose layout the mint asked to change.

    `morphism` is a ratified handle or `""`. An empty handle carries only the
    stride order inductor asked for: a name is never invented for it.
    """

    fqn: str
    morphism: str
    stride_order: Tuple[int, ...]

    @property
    def ratified(self) -> bool:
        return bool(self.morphism)


@dataclass(frozen=True)
class LayoutRung:
    """One graph's layout position: what it serves, what it wants, and why."""

    graph: str
    served: str
    ideal: str
    state: LayoutState
    detail: str
    wishes: Tuple[LayoutWish, ...] = ()

    @property
    def earning(self) -> bool:
        """Whether a re-mint against a better layout is owed to this graph."""

        return self.state is LayoutState.EARNING

    @property
    def settled(self) -> bool:
        """Whether no RE-MINT is owed -- for any reason, including a decline
        and including an unratified candidate.

        Settled is not the same as ideal. A decline is a settled position with
        a stated cause and a candidate is settled pending a HUMAN, not a
        machine; conflating either with EARNING makes a permanent state look
        like a stuck mint queue, which is the reading this whole confession
        exists to prevent."""

        return self.state is not LayoutState.EARNING

    def line(self) -> str:
        """The confession, one line, in the shape `rung.transition_line` uses."""

        head = f"LAYOUT_RUNG={self.state} graph={self.graph} served={self.served}"
        if self.ideal:
            head = f"{head} ideal={self.ideal}"
        if self.wishes:
            head = f"{head} wishes={len(self.wishes)}"
        return f"{head}: {self.detail}" if self.detail else head

    def facts(self) -> Dict[str, Any]:
        return {
            "graph": self.graph,
            "served": self.served,
            "ideal": self.ideal,
            "state": str(self.state),
            "detail": self.detail,
            "wishes": [
                {
                    "fqn": wish.fqn,
                    "morphism": wish.morphism,
                    "stride_order": list(wish.stride_order),
                }
                for wish in self.wishes
            ],
        }


def _wishes(raw: object) -> Tuple[LayoutWish, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    out = []
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        out.append(
            LayoutWish(
                fqn=str(row.get("fqn", "")),
                morphism=str(row.get("morphism", "")),
                stride_order=tuple(int(v) for v in (row.get("stride_order") or ())),
            )
        )
    return tuple(out)


def fill_path() -> Any:
    """The layout-applying fill this process can reach, or `None`.

    A PROBE of what is installed, not a flag: the transform has exactly one
    implementation (`tensorfs_core::fill`, one plan, two backends behind the
    `FillSink` trait), and a worker either has it in reach or it does not.
    Today it does not, so every non-identity arrangement is declined for the
    plainest possible reason: nothing in this process can apply one.

    **THIS PROBE IS AIMED AT THE WRONG SURFACE AND IS OWED A REPOINT** (pgw#1648).
    It asks the vendored tensorfs for a `fill` attribute. The real client is
    `serving.streaming.fill_client.client_for` over tensorfs' native
    `CudaFillClient` / `HostFillClient` (tensorfs#159) -- and this probe could
    never go true anyway, because those live in the COMPILED extension and the
    vendored tensorfs deliberately carries none (pgw#1310, which is why
    `tensors.py` has a pure-Python `_MappedObject`). It is left aimed here, and
    said out loud, rather than pointed at a module that does not exist on this
    branch: the VERDICT it produces is correct today either way -- there is no
    fill in reach -- and a probe that lies about which absence it found is
    worse than one that names its own aim.

    Once `fill_client` lands, this returns that client and the decline moves
    from "there is no fill" to the per-arrangement judgement
    :data:`PER_ELEMENT_DECLINE` describes, priced by the fill's own
    `runs_per_element` rather than by anything computed here.
    """

    try:
        from .._vendor import tensorfs as _tensorfs
    except ImportError:  # pragma: no cover - the vendored package is always present
        return None
    return getattr(_tensorfs, "fill", None)


def _delivery(handle: str) -> str:
    """`""` when the platform will deliver this arrangement, else the reason."""

    if fill_path() is None:
        return (
            f"no layout-applying fill is reachable from this process, so "
            f"{handle!r} cannot be delivered to VRAM at all (pgw#1648 + "
            f"tensorfs#159: tensorfs owns the plan and the transform and is "
            f"being bound to Python as a fill client; varena hands it the "
            f"destination address). {PER_ELEMENT_DECLINE}"
        )
    return ""


def read_rung(graph: str, metadata: Mapping[str, Any]) -> LayoutRung:
    """Read one serving artifact's layout position off its OWN metadata.

    Both facts come from the artifact -- never from a caller's belief about it,
    and never from this repository's idea of what the fleet declares. An
    artifact that states no ratified `declared_input_layout` cannot be read
    here at all; `torchcg.artifact.validate_metadata` already refuses it before
    the bytes are ever loaded, which is why this function has no arm for it.
    """

    served = str(metadata.get("declared_input_layout", ""))
    wishes = _wishes(metadata.get("layout_wishlist"))
    if not wishes:
        return LayoutRung(
            graph=graph,
            served=served,
            ideal="",
            state=LayoutState.NO_WISH,
            detail="the mint asked for no layout change",
        )

    ratified = sorted({wish.morphism for wish in wishes if wish.ratified})
    if not ratified:
        return LayoutRung(
            graph=graph,
            served=served,
            ideal="",
            state=LayoutState.CANDIDATE,
            detail=(
                f"{len(wishes)} constant(s) want an arrangement no ratified "
                f"morphism names; the orders ride out as candidates and the "
                f"mint keeps {served!r}"
            ),
            wishes=wishes,
        )
    if len(ratified) > 1:
        return LayoutRung(
            graph=graph,
            served=served,
            ideal="",
            state=LayoutState.NO_SINGLE_IDEAL,
            detail=(
                f"constants want {ratified!r}: there is no single ideal layout "
                f"for this graph, so a re-mint would move the copies rather "
                f"than delete them"
            ),
            wishes=wishes,
        )

    ideal = ratified[0]
    if ideal == served:
        return LayoutRung(
            graph=graph,
            served=served,
            ideal=ideal,
            state=LayoutState.AT_IDEAL,
            detail="the delivered layout is the wished one",
            wishes=wishes,
        )
    refusal = _delivery(ideal)
    if refusal:
        return LayoutRung(
            graph=graph,
            served=served,
            ideal=ideal,
            state=LayoutState.DECLINED,
            detail=refusal,
            wishes=wishes,
        )
    return LayoutRung(
        graph=graph,
        served=served,
        ideal=ideal,
        state=LayoutState.EARNING,
        detail=(
            f"serving {served!r} while the {ideal!r} mint is pending; this pod "
            f"is EARNING that rung, not failing to reach it"
        ),
        wishes=wishes,
    )


def rungs_of(session: Any) -> Tuple[LayoutRung, ...]:
    """Every armed graph's layout position, read from the artifact SERVING it.

    Walks the adopt session's own dispatcher registry and reads each armed
    runner's verified metadata -- the block the loader already validated on the
    way in. Nothing is re-materialized and nothing is re-derived: the artifact
    that is answering requests right now is the artifact that answers here.
    """

    home = getattr(session, "_home", None)
    if not isinstance(home, Mapping):
        return ()
    out = []
    for graph, dispatcher in sorted(home.items()):
        for record, compiled in tuple(getattr(dispatcher, "_entries", ())):
            if record.graph != graph:
                continue
            metadata = _metadata_of(compiled)
            if metadata is None:
                continue
            out.append(read_rung(graph, metadata))
    return tuple(out)


def _metadata_of(compiled: Any) -> Mapping[str, Any] | None:
    runner = getattr(compiled, "runner", None)
    graph = getattr(runner, "_graph", None)
    metadata = getattr(graph, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else None
