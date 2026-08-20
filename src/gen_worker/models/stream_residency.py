"""Per-module budgeted partial residency with cast-per-forward streaming.

The finest rung on the degrade ladder (pgw#1497), ported from ComfyUI's
lowvram mechanism (`comfy/model_patcher.py:982-1111`, `comfy/ops.py:337-459`,
`comfy/model_management.py:1340-1494`) — THE reason a 20 GB model is usable on
a 6 GB card over there.

Three parts:

* **The budget split** (:func:`plan_residency`) — walk the leaf modules sorted
  by offload cost, fill a VRAM budget largest-first, and reserve the
  ``streams``-deep IN-FLIGHT WINDOW so the streaming tail can never overshoot
  the budget it was fitted to. Pure arithmetic, no torch, deterministic.
* **The streaming tail** (:class:`StreamedResidency`) — every leaf that did
  not fit rests in PINNED host RAM and is cast onto the device inside its own
  forward, on round-robin offload streams writing into persistent reused cast
  buffers, so leaf N+1's H2D overlaps leaf N's compute. Both fences matter:
  the stream waits on compute before its buffer is refilled, and compute waits
  on the stream before it reads.
* **Partial unload/load** (:meth:`StreamedResidency.rebudget`) — the eviction
  and demotion primitive. Freeing N bytes trims a model's cold tail instead of
  dropping the model, which is what lets several warm endpoints share one card
  and IS the implementation of the host tier's ``demote_to_host`` /
  ``promote_to_device``.

**The budget is handed down, never estimated.** ComfyUI derives it from
hand-fitted per-architecture activation lambdas, and their own code is fleeing
that (report A, weaknesses 1-3). Ours comes from the residency lease:
admission already reserved ``weights + headroom`` before a byte moved, and the
number this module is given is what that admission decided the resident
weights may occupy. Nothing here measures, guesses or samples VRAM.

**One walk, not two.** ComfyUI has a load walk and a separate inverse
``partially_unload`` walk which can disagree about the same model. Here there
is exactly one planner: unload is a re-plan at a smaller budget, load a re-plan
at a larger one, and the transition is the set difference. Demotion to the host
tier is the same call with a budget of zero.

**Hooks, not op subclasses.** ComfyUI owns its layer classes; we serve
arbitrary diffusers trees, so the mechanism is a forward pre/post hook pair
over true leaf modules (no children, own tensors). A module that owns
parameters AND children stays resident: its post-hook would fire after its
children's forwards, holding a cast-buffer view alive across their casts.

**LoRA.** Ours is attach-based (:mod:`gen_worker.utils.lora` →
``load_lora_weights``/``set_adapters``), so an adapter is a pair of tiny
``lora_A``/``lora_B`` leaves NEXT TO the base layer, never a patch fused into
the base weight. That makes ComfyUI's LowVramPatch machinery unnecessary: peft
computes ``base_layer(x) + lora_B(lora_A(x)) * scale``, the base layer streams
through its own hook exactly as an unpatched one does, and the adapter leaves
are forced resident (they are megabytes, and their cost is dwarfed by the
launch overhead of streaming them). A MERGED adapter is just a different
weight and streams like any other.

**Cells and the schedule** (pgw#1515). Two things above the split, both
PLANNER-side and therefore shared by every mechanism the rung can sit on:

* **Cells** (:class:`CellPolicy`, :class:`ResidencyCell`) — the unit of
  back/unback/stream is a CELL, a packed group of leaves, not a leaf. A
  mechanism that maps memory at a page granularity pays that granularity ONCE
  PER REGION, so per-leaf regions charge the alignment remainder once per leaf:
  pgw#1507 measured **248 MiB = 13.2 %** of sd1.5's weights lost that way
  (1.830 GiB of weight in 2.072 GiB of span over 239 regions, 2 MiB VMM
  granularity), and ~98 µs of page-table work per streamed leaf per forward on
  top. Packing the small leaves first collapses both: the remainder is paid
  once per CELL, and the plan REPORTS the residual so the elimination is a
  measured claim and not an assertion (:attr:`ResidencyPlan.granularity_tax_bytes`
  against :attr:`ResidencyPlan.leaf_granular_tax_bytes`).
* **The schedule** (:class:`ResidencySchedule`) — when the assigned {VRAM, RAM}
  pair admits holding the whole ACTIVE component for a call, the cells rotate at
  CALL boundaries (:meth:`StreamedResidency.hold_component`) instead of
  per-forward. That is ``model_offload``'s per-CALL amortization — measured
  1.57× against this rung's per-STEP 1.91× at the same peak VRAM (pgw#1497) —
  on this rung's mechanism and at this rung's finer granularity. The choice is
  DETERMINISTIC ARITHMETIC over the budget and the cell layout, decided in the
  plan, never a runtime heuristic reacting to pressure (Paul's boundary ruling).

Admission-first is untouched: the budget is still handed down, the fill is still
the greedy largest-first walk run to a fixed point, and under the default
leaf-granular policy every number this module produces is bit-for-bit what it
produced before cells existed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import staging

logger = logging.getLogger(__name__)

#: Round-robin offload streams. Two is ComfyUI's NVIDIA/AMD default and the
#: point where a second in-flight cast stops buying overlap on one copy engine.
DEFAULT_STREAMS = 2

#: Leaves smaller than this are always resident. Streaming a 4 KiB norm weight
#: costs two hook calls, a carve and a fence to save four kilobytes — the
#: launch overhead dominates the transfer. (ComfyUI force-loads ≤16 KiB modules
#: on its AIMDO path for the same reason; ours is coarser on purpose.)
DEFAULT_MIN_STREAM_BYTES = 1 << 20

#: Cast-buffer carve alignment. A uint8 slice must be re-viewable as the
#: weight's dtype, so every sub-buffer starts on a 512-byte boundary.
_ALIGN = 512

#: varena's VMM chunk granularity — the page size the driver maps in. The
#: number behind pgw#1507's measured 13.2 % tax, and the reason cells exist.
VMM_GRANULARITY_BYTES = 2 << 20

#: Set on a module whose leaves this rung has hooked.
ENGAGED_ATTR = "_cozy_stream_residency"

#: Substrings marking adapter leaves that are never streamed (see the module
#: docstring's LoRA note).
_ADAPTER_MARKERS = ("lora_a", "lora_b", "lora_embedding", "lora_magnitude")


def align_to(n: int, alignment: int) -> int:
    """``n`` rounded up to a multiple of ``alignment``. One rounding rule for
    both alignments a mechanism has: the cast-buffer carve and the page."""
    a = max(1, int(alignment))
    return (int(n) + a - 1) // a * a


def aligned(n: int) -> int:
    return align_to(n, _ALIGN)


# ---------------------------------------------------------------------------
# The budget split — pure arithmetic, no torch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryBudget:
    """The {VRAM, RAM} PAIR a model is assigned (Paul, 2026-08-19).

    A memory profile is assigned top-down — worker, then execution group, then
    model — and BOTH halves determine what this rung may do: the resident set
    spends the VRAM half, and the streaming tail and every demotion to the host
    tier spend the RAM half. Pinned staging is a slice of the same RAM budget,
    not free space beside it.

    ``ram_bytes == 0`` means UNSTATED, not unlimited. The plan then reports its
    host cost (:attr:`ResidencyPlan.host_bytes`) without a verdict on it.

    **ENFORCEMENT IS A NAMED DEFERRAL** (pgw#1497 follow-up): partial unload
    must REFUSE, or degrade further down the ladder, when the RAM half cannot
    absorb what VRAM evicts. Today it plans against the VRAM half and reports
    the RAM cost. The signature is the pair now so that landing enforcement is
    a change of behaviour and not a change of shape.
    """

    vram_bytes: int
    ram_bytes: int = 0

    @classmethod
    def of(cls, value: "int | MemoryBudget") -> "MemoryBudget":
        """Accept either half of the migration: a bare VRAM int, or the pair."""
        if isinstance(value, MemoryBudget):
            return value
        return cls(vram_bytes=max(0, int(value)))


@dataclass(frozen=True)
class LeafCost:
    """One leaf module's two prices.

    ``resident_bytes`` is what keeping it on the card costs. ``cast_bytes`` is
    what streaming it costs *while it is in flight* — the cast-buffer space it
    occupies. They differ only where storage and compute dtype differ (an fp8
    leaf casting to bf16 needs twice its stored size in the buffer); ComfyUI
    also folds an on-the-fly LoRA cost in here, which we do not need because
    attached adapters are separate resident leaves.
    """

    name: str
    resident_bytes: int
    cast_bytes: int = 0
    #: Which COMPONENT this leaf belongs to — the first path segment of its
    #: qualified name (``unet.down_blocks.0.attn.to_q`` -> ``unet``), which is
    #: exactly the granularity ``model_offload`` swaps at and therefore the unit
    #: the call-boundary schedule holds hot. A dotless name has no component and
    #: joins the unnamed one, so a flat tree is one component rather than N.
    component: str = ""

    def __post_init__(self) -> None:
        if not self.cast_bytes:
            object.__setattr__(self, "cast_bytes", int(self.resident_bytes))
        if not self.component and "." in self.name:
            object.__setattr__(self, "component", self.name.split(".", 1)[0])


@dataclass(frozen=True)
class CellPolicy:
    """A MECHANISM's paging geometry — the only thing the planner needs to know
    about how a rung actually holds bytes (pgw#1515).

    Three numbers, each a property of the mechanism and not of the model:

    * ``cell_bytes`` — the target size of one back/unback/stream unit. ``0``
      means LEAF-GRANULAR: every leaf is its own cell, which is what the rung
      did before cells existed and what :data:`LEAF_CELLS` states.
    * ``granularity_bytes`` — the page size a REGION is rounded up to. The
      software cast ring has no region concept and states ``1``; varena maps in
      2 MiB VMM chunks and states :data:`VMM_GRANULARITY_BYTES`, which is the
      whole source of pgw#1507's 13.2 % tax.
    * ``intra_align_bytes`` — the alignment BETWEEN members inside one cell. A
      uint8 slice must be re-viewable as its weight's dtype, so a mechanism that
      carves views out of a shared region states :data:`_ALIGN`; one that does
      not states ``1``.

    Stating a geometry is how a mechanism gets priced honestly: with
    ``granularity_bytes=1`` the tax arithmetic collapses to zero and every
    number the planner reports is the pre-cell number, exactly.
    """

    cell_bytes: int = 0
    granularity_bytes: int = 1
    intra_align_bytes: int = 1

    @classmethod
    def of(cls, value: "int | CellPolicy | None") -> "CellPolicy":
        """Accept a bare target-cell-bytes int, a policy, or nothing."""
        if value is None:
            return LEAF_CELLS
        if isinstance(value, CellPolicy):
            return value
        return cls(cell_bytes=max(0, int(value)))

    @classmethod
    def vmm(cls, cell_bytes: int) -> "CellPolicy":
        """The varena geometry: 2 MiB pages, 512-byte views inside a cell."""
        return cls(
            cell_bytes=max(0, int(cell_bytes)),
            granularity_bytes=VMM_GRANULARITY_BYTES,
            intra_align_bytes=_ALIGN,
        )

    def span_of(self, payload_bytes: int) -> int:
        """What a region holding ``payload_bytes`` actually MAPS."""
        return align_to(payload_bytes, self.granularity_bytes)

    def member_bytes(self, leaf_bytes: int) -> int:
        """What one member costs inside a cell, aligned for its neighbours."""
        return align_to(leaf_bytes, self.intra_align_bytes)


#: The default: one cell per leaf, no page granularity, no intra-cell padding.
#: Under this policy the planner's arithmetic is identical to pgw#1497's.
LEAF_CELLS = CellPolicy()


@dataclass(frozen=True)
class ResidencyCell:
    """One back/unback/stream unit: a packed group of leaves that share a fate.

    Members of a cell are always in the same place. That is the point — a
    mechanism that can map, stream or release half a cell is back to per-leaf
    granularity and back to paying its page remainder per leaf.
    """

    index: int
    component: str
    members: Tuple[str, ...]
    #: The weight bytes the members really hold (no alignment of any kind).
    weight_bytes: int
    #: What the mechanism MAPS for this cell: the members packed at the
    #: intra-cell alignment, rounded up to one page granularity.
    span_bytes: int
    #: The in-flight cast cost when this cell streams — the whole cell, since
    #: the cell is the stream unit.
    cast_bytes: int
    #: True when every member is always-resident (below the streaming floor, or
    #: excluded by the caller). Forced and streamable leaves never share a cell:
    #: a cell's fate has to be one thing.
    forced: bool
    #: The same members priced ONE REGION PER LEAF — the baseline this packing
    #: exists to beat, carried so the elimination is reported and not asserted.
    leaf_span_bytes: int

    @property
    def granularity_tax_bytes(self) -> int:
        return max(0, self.span_bytes - self.weight_bytes)

    @property
    def leaf_granular_tax_bytes(self) -> int:
        return max(0, self.leaf_span_bytes - self.weight_bytes)


def pack_cells(
    costs: Sequence[LeafCost],
    *,
    policy: "int | CellPolicy | None" = None,
    min_stream_bytes: int = DEFAULT_MIN_STREAM_BYTES,
    exclude: Iterable[str] = (),
) -> Tuple[ResidencyCell, ...]:
    """Group ``costs`` into cells. Deterministic, order-independent, pure.

    **Small leaves are packed FIRST**, which is not a tie-break but the whole
    mechanism: a leaf's page remainder is bounded by the granularity, so a 4 KiB
    norm weight in its own 2 MiB region wastes 99.8 % of it while a 300 MiB
    attention block wastes at most 0.7 %. Filling ascending puts the small ones
    together until a cell reaches its target and leaves the large ones as
    singletons, where they were already efficient. pgw#1507 measured the cost of
    not doing this: 248 MiB over 239 regions on a 1.830 GiB model.

    Two things a cell may never mix, because a cell has ONE fate and ONE home:

    * forced and streamable leaves — a forced member would pin the whole cell;
    * two components — the call-boundary schedule holds a COMPONENT hot, so a
      cell straddling two of them could not be swapped at a call boundary.

    Order-independence is a hard requirement inherited from
    :func:`plan_residency`: groups are emitted in sorted key order and filled in
    sorted member order, so the cells are a function of the SET of costs, never
    of the sequence they arrived in.
    """
    geometry = CellPolicy.of(policy)
    skip = {str(n) for n in exclude}
    floor = int(min_stream_bytes)

    groups: Dict[Tuple[str, bool], List[LeafCost]] = {}
    for cost in costs:
        forced = cost.name in skip or cost.cast_bytes < floor
        groups.setdefault((cost.component, forced), []).append(cost)

    cells: List[ResidencyCell] = []
    for key in sorted(groups):
        component, forced = key
        members = sorted(
            groups[key], key=lambda c: (c.cast_bytes, c.resident_bytes, c.name)
        )
        batch: List[LeafCost] = []
        payload = 0
        for cost in members:
            share = geometry.member_bytes(cost.resident_bytes)
            if batch and payload + share > max(0, geometry.cell_bytes):
                cells.append(_cell(len(cells), component, forced, batch, payload, geometry))
                batch, payload = [], 0
            batch.append(cost)
            payload += share
        if batch:
            cells.append(_cell(len(cells), component, forced, batch, payload, geometry))
    return tuple(cells)


def _cell(
    index: int,
    component: str,
    forced: bool,
    members: Sequence[LeafCost],
    payload_bytes: int,
    geometry: CellPolicy,
) -> ResidencyCell:
    return ResidencyCell(
        index=index,
        component=component,
        members=tuple(c.name for c in members),
        weight_bytes=sum(c.resident_bytes for c in members),
        span_bytes=geometry.span_of(payload_bytes),
        cast_bytes=sum(c.cast_bytes for c in members),
        forced=forced,
        leaf_span_bytes=sum(
            geometry.span_of(geometry.member_bytes(c.resident_bytes)) for c in members
        ),
    )


class ResidencySchedule(str, Enum):
    """WHEN cells rotate — decided in the plan, never under fire.

    ``CALL_BOUNDARY`` is the ``model_offload``-equivalent mode on this rung's
    mechanism: the assigned {VRAM, RAM} pair admits holding the whole active
    component's cells for the duration of a call, so the swap happens once per
    call and its cost amortises over every step (pgw#1497 measured that split
    directly — ``model_offload`` 1.57× per-CALL against this rung's 1.91×
    per-STEP at the SAME 1.9 GB peak).

    ``PER_STEP`` is the fallback the budget forces: no component fits whole, so
    cells rotate inside the forward and the tax is paid every step. It is also
    the mode that reaches below ``model_offload``'s floor, which is the band
    this rung exists for.
    """

    CALL_BOUNDARY = "call_boundary"
    PER_STEP = "per_step"


@dataclass(frozen=True)
class ResidencyPlan:
    """Which leaves stay on the card, which stream, and what it reserves."""

    budget_bytes: int
    streams: int
    #: Forced resident (below ``min_stream_bytes``, or excluded by the caller).
    forced: Tuple[str, ...]
    #: Chose to stay resident, in fill order (largest first).
    resident: Tuple[str, ...]
    #: Streams from pinned host RAM, in the same walk order.
    streamed: Tuple[str, ...]
    resident_bytes: int
    streamed_bytes: int
    #: The in-flight cast window this split reserved out of the budget. The
    #: subtle half of the port: without it a plan fits at rest and overshoots
    #: the instant two casts are in flight.
    window_bytes: int
    #: The RAM half of the assigned pair, 0 when unstated. Reported, not yet
    #: enforced — see :class:`MemoryBudget`.
    ram_budget_bytes: int = 0

    # -- cells (pgw#1515) ---------------------------------------------------
    #: The paging geometry this plan was drawn for.
    policy: CellPolicy = LEAF_CELLS
    #: Every cell, forced and streamable, in packing order.
    cells: Tuple[ResidencyCell, ...] = ()
    #: The cells the card holds — indices into :attr:`cells`, in fill order.
    resident_cells: Tuple[int, ...] = ()
    #: The cells that stream, in the same walk order.
    streamed_cells: Tuple[int, ...] = ()
    #: What the mechanism actually MAPS for the resident set: the cell spans,
    #: page-remainder included. Equals :attr:`resident_bytes` exactly when the
    #: geometry has no page granularity, which is why the default policy leaves
    #: every pre-cell number unchanged.
    resident_span_bytes: int = 0
    #: The SAME resident leaves priced one region per leaf — pgw#1507's measured
    #: baseline, carried so that :attr:`tax_eliminated_bytes` is a subtraction
    #: of two numbers this plan computed rather than a claim about a past run.
    leaf_granular_span_bytes: int = 0

    # -- the schedule (pgw#1515) --------------------------------------------
    schedule: ResidencySchedule = ResidencySchedule.PER_STEP
    #: (component, mapped span) for every component, largest first. The
    #: arithmetic the schedule was decided on, published so the decision can be
    #: audited without re-deriving it.
    component_spans: Tuple[Tuple[str, int], ...] = ()
    #: The largest component's mapped span — what CALL_BOUNDARY must hold.
    hot_component_bytes: int = 0
    #: What rests off-card while one component is hot. This is the RAM half's
    #: real bill under CALL_BOUNDARY, and the reason a stated RAM budget can
    #: force PER_STEP on a card that would otherwise admit the swap.
    cold_component_bytes: int = 0
    #: The components that may be held whole for a call, largest first. Empty
    #: under PER_STEP — an empty tuple IS the refusal.
    hold_while_hot: Tuple[str, ...] = ()
    #: Which component this plan is currently holding hot, "" while the plan is
    #: the budget-greedy split rather than a call-boundary arrangement.
    hot_component: str = ""

    @property
    def host_bytes(self) -> int:
        """What this plan costs in HOST RAM: the pinned streaming tail."""
        return self.streamed_bytes

    @property
    def host_fits(self) -> bool:
        """False when the tail is over the stated RAM half. Always True while
        the RAM half is unstated — an unstated budget is not a passed one, and
        callers that need the distinction read ``ram_budget_bytes``."""
        return not self.ram_budget_bytes or self.host_bytes <= self.ram_budget_bytes

    @property
    def device_bytes(self) -> int:
        """What the card actually holds under this plan, at peak.

        The SPAN, not the weight bytes: a region is charged at what it maps, so
        ``fits`` stays exact under a mechanism that pages (pgw#1507 charged its
        regions the same way, which is what made its ``resident_bytes`` equal
        the driver's mapped bytes to the byte)."""
        return self.resident_span_bytes + self.window_bytes

    # -- the granularity tax, reported (pgw#1515) ---------------------------

    @property
    def granularity_tax_bytes(self) -> int:
        """What this plan's page remainders cost. Zero is the goal, and the
        residual is published rather than assumed away."""
        return max(0, self.resident_span_bytes - self.resident_bytes)

    @property
    def leaf_granular_tax_bytes(self) -> int:
        """What the same resident leaves would have cost one region each —
        pgw#1507 measured this at 248 MiB / 13.2 % on sd1.5."""
        return max(0, self.leaf_granular_span_bytes - self.resident_bytes)

    @property
    def tax_eliminated_bytes(self) -> int:
        """Weight bytes the packing bought back at the same lease."""
        return max(0, self.leaf_granular_span_bytes - self.resident_span_bytes)

    @property
    def granularity_tax_ratio(self) -> float:
        return self.granularity_tax_bytes / self.resident_bytes if self.resident_bytes else 0.0

    @property
    def leaf_granular_tax_ratio(self) -> float:
        return (
            self.leaf_granular_tax_bytes / self.resident_bytes
            if self.resident_bytes
            else 0.0
        )

    @property
    def fits(self) -> bool:
        """False when even the forced set plus one window is over budget — the
        split ran out of room to give back, and the caller is about to serve
        over its lease. A confession, never a silent clamp."""
        return self.device_bytes <= self.budget_bytes

    @property
    def all_resident(self) -> Tuple[str, ...]:
        return self.forced + self.resident


def plan_residency(
    costs: Sequence[LeafCost],
    *,
    budget_bytes: "int | MemoryBudget",
    streams: int = DEFAULT_STREAMS,
    min_stream_bytes: int = DEFAULT_MIN_STREAM_BYTES,
    exclude: Iterable[str] = (),
    cells: "int | CellPolicy | None" = None,
) -> ResidencyPlan:
    """Split ``costs`` into a resident set and a streaming tail under a budget.

    ``budget_bytes`` is the lease's, not an estimate (see the module
    docstring). The walk is ComfyUI's ``ModelPatcher.load`` (`:1000-1001`):
    sorted by offload cost descending, a leaf is admitted only when what is
    already resident, plus the leaf, plus the in-flight cast window still
    fits.

    The window arithmetic is TIGHTER than theirs, and deliberately. They
    reserve ``this leaf + the next NUM_STREAMS leaves`` — a lookahead over
    leaves that may well end up RESIDENT, so the reservation is counted twice
    and a budget that can genuinely stream the model reports that it cannot
    (measured on a 6-leaf pyramid: 15 MB reserved where 12 MB is the real
    peak, and the whole model then streams with a resident set of zero). What
    the ring actually holds is ``streams`` buffers, each grown to the largest
    leaf that ever rode it, so the honest reservation is ``streams x largest
    STREAMED cast`` — and it matches :attr:`_CastRing.buffer_peak_bytes` at
    run time.

    That reservation depends on the split, and the split depends on the
    reservation, so the walk is run to a FIXED POINT: start at a zero window
    (fill by weight alone), compute what the tail that did not fit really
    needs, refill under that reservation, repeat. The window only ever grows,
    so it converges in at most one pass per leaf, and it terminates at zero
    when nothing streams — which is what makes a budget equal to the model's
    own size mean full residency instead of an empty resident set holding a
    window for a tail that isn't there.

    Deterministic: ties break on name, so the same tree and the same budget
    always produce the same split. That is a hard requirement, not a
    nicety — a mint's traced graph specializations and a residency reservation
    both depend on the answer.

    **The unit of the walk is a CELL** (pgw#1515), not a leaf: cells are packed
    first (:func:`pack_cells`), the fill admits whole cells charged at their
    MAPPED SPAN, and the in-flight window reserves ``streams`` × the largest
    streamed CELL, because a cell is what moves. Under the default
    leaf-granular geometry every cell is one leaf with no page remainder, so
    this is the identical arithmetic to the pre-cell planner — verified as a
    standing test, not as an argument.

    The plan also decides its own SCHEDULE from the same two inputs, so that
    "hold the hot component and swap at call boundaries" is a property of the
    plan and never a runtime reaction to pressure.
    """
    streams = max(1, int(streams))
    pair = MemoryBudget.of(budget_bytes)
    budget = max(0, int(pair.vram_bytes))
    geometry = CellPolicy.of(cells)
    packed = pack_cells(
        costs, policy=geometry, min_stream_bytes=min_stream_bytes, exclude=exclude
    )

    forced_cells = [c for c in packed if c.forced]
    order = sorted(
        (c for c in packed if not c.forced),
        key=lambda c: (-c.cast_bytes, -c.weight_bytes, c.index),
    )
    floor = sum(c.span_bytes for c in forced_cells)

    window = 0
    span = floor
    resident: List[ResidencyCell] = []
    streamed: List[ResidencyCell] = []
    for _ in range(len(order) + 1):
        span = floor
        resident = []
        streamed = []
        for cell in order:
            if span + cell.span_bytes + window <= budget:
                resident.append(cell)
                span += cell.span_bytes
            else:
                streamed.append(cell)
        needed = streams * max((c.cast_bytes for c in streamed), default=0)
        if needed <= window:
            break
        window = needed

    held = forced_cells + resident
    component_span: Dict[str, int] = {}
    for cell in packed:
        component_span[cell.component] = component_span.get(cell.component, 0) + cell.span_bytes
    ranked = tuple(sorted(component_span.items(), key=lambda kv: (-kv[1], kv[0])))
    hot = ranked[0][1] if ranked else 0
    cold = sum(v for _, v in ranked) - hot
    # The swap moves a cell at a time and is pipelined exactly like a per-step
    # cast, so the same ring reservation applies at a call boundary.
    swap_window = streams * max((c.cast_bytes for c in packed), default=0)
    ram = max(0, int(pair.ram_bytes))
    call_boundary = bool(ranked) and hot + swap_window <= budget and (not ram or cold <= ram)

    return ResidencyPlan(
        budget_bytes=budget,
        streams=streams,
        forced=tuple(n for c in forced_cells for n in c.members),
        resident=tuple(n for c in resident for n in c.members),
        streamed=tuple(n for c in streamed for n in c.members),
        resident_bytes=sum(c.weight_bytes for c in held),
        streamed_bytes=sum(c.weight_bytes for c in streamed),
        window_bytes=window,
        ram_budget_bytes=ram,
        policy=geometry,
        cells=packed,
        resident_cells=tuple(c.index for c in held),
        streamed_cells=tuple(c.index for c in streamed),
        resident_span_bytes=span,
        leaf_granular_span_bytes=sum(c.leaf_span_bytes for c in held),
        schedule=(
            ResidencySchedule.CALL_BOUNDARY if call_boundary else ResidencySchedule.PER_STEP
        ),
        component_spans=ranked,
        hot_component_bytes=hot,
        cold_component_bytes=cold,
        hold_while_hot=tuple(name for name, _ in ranked) if call_boundary else (),
    )


def _hot_plan(plan: ResidencyPlan, component: str) -> ResidencyPlan:
    """``plan`` rearranged so ``component`` is held whole and the rest parks.

    Pure, and derived from the cells the plan already carries — the schedule
    decides WHETHER this arrangement is affordable, and this decides what it
    looks like. The forced core (adapters, sub-floor leaves, caller exclusions)
    stays resident in every arrangement: an exclusion is a statement about
    residency, and a call boundary does not repeal it.
    """
    held = [c for c in plan.cells if c.forced or c.component == component]
    parked = [c for c in plan.cells if not c.forced and c.component != component]
    window = plan.streams * max((c.cast_bytes for c in parked), default=0)
    return ResidencyPlan(
        budget_bytes=plan.budget_bytes,
        streams=plan.streams,
        forced=tuple(n for c in held if c.forced for n in c.members),
        resident=tuple(n for c in held if not c.forced for n in c.members),
        streamed=tuple(n for c in parked for n in c.members),
        resident_bytes=sum(c.weight_bytes for c in held),
        streamed_bytes=sum(c.weight_bytes for c in parked),
        window_bytes=window,
        ram_budget_bytes=plan.ram_budget_bytes,
        policy=plan.policy,
        cells=plan.cells,
        resident_cells=tuple(c.index for c in held),
        streamed_cells=tuple(c.index for c in parked),
        resident_span_bytes=sum(c.span_bytes for c in held),
        leaf_granular_span_bytes=sum(c.leaf_span_bytes for c in held),
        schedule=plan.schedule,
        component_spans=plan.component_spans,
        hot_component_bytes=plan.hot_component_bytes,
        cold_component_bytes=plan.cold_component_bytes,
        hold_while_hot=plan.hold_while_hot,
        hot_component=component,
    )


@dataclass(frozen=True)
class PlanTransition:
    """The moves from one plan to another. Empty tuples = nothing to do."""

    demote: Tuple[str, ...]
    promote: Tuple[str, ...]
    freed_bytes: int
    claimed_bytes: int


def plan_transition(
    old: ResidencyPlan,
    new: ResidencyPlan,
    costs: Sequence[LeafCost],
    *,
    allow_promote: bool = True,
) -> PlanTransition:
    """What moving from ``old`` to ``new`` costs and frees.

    ``allow_promote=False`` is what makes a partial UNLOAD an unload: the
    greedy fill is not perfectly monotone in the budget (dropping a large leaf
    frees room a smaller one can take), and a call asked to free bytes must
    never answer by moving bytes onto the card.
    """
    by_name = {c.name: c for c in costs}
    was = set(old.all_resident)
    now = set(new.all_resident)
    demote = tuple(n for n in old.all_resident if n not in now)
    promote = tuple(n for n in new.all_resident if n not in was) if allow_promote else ()
    return PlanTransition(
        demote=demote,
        promote=promote,
        freed_bytes=sum(by_name[n].resident_bytes for n in demote if n in by_name),
        claimed_bytes=sum(by_name[n].resident_bytes for n in promote if n in by_name),
    )


# ---------------------------------------------------------------------------
# The streaming tail — cast per forward on round-robin offload streams
# ---------------------------------------------------------------------------


class _CastRing:
    """``streams`` offload streams, each owning one persistent cast buffer.

    The buffer is reused for every leaf that rides that stream, which is the
    whole memory argument: the tail costs ``streams`` × the largest streamed
    leaf, not the tail's size. Off CUDA the ring degrades to synchronous
    copies into ordinary host buffers — the SAME driver loop, so every
    ordering decision below is exercised by the CPU-side tests and only the
    overlap is left for the card to prove.
    """

    def __init__(self, torch: Any, device: Any, streams: int) -> None:
        self._torch = torch
        self.device = device
        self.cuda = getattr(device, "type", str(device)) == "cuda"
        self._n = max(1, int(streams))
        self._streams: List[Any] = (
            [torch.cuda.Stream(device=device) for _ in range(self._n)]
            if self.cuda
            else [None] * self._n
        )
        self._buffers: List[Optional[Any]] = [None] * self._n
        self._index = 0
        #: Peak bytes any one buffer grew to — telemetry, and the number the
        #: plan's ``window_bytes`` reservation is checked against.
        self.buffer_peak_bytes = 0

    def acquire(self, nbytes: int) -> Tuple[Any, Any]:
        """The next (stream, buffer), fenced so the buffer is free to refill.

        The fence LAGS by one acquisition, exactly as ``get_offload_stream``
        does. Fencing the stream we are about to hand out against the compute
        stream *now* would make its H2D wait for the forward that was just
        enqueued — which is the serialization this rung exists to avoid.
        """
        torch = self._torch
        if self.cuda:
            current = torch.cuda.current_stream(self.device)
            self._streams[self._index].wait_stream(current)
        self._index = (self._index + 1) % self._n
        stream = self._streams[self._index]
        buffer = self._buffers[self._index]
        if buffer is None or int(buffer.numel()) < int(nbytes):
            if stream is not None:
                stream.synchronize()
            buffer = torch.empty(int(nbytes), dtype=torch.uint8, device=self.device)
            self._buffers[self._index] = buffer
            self.buffer_peak_bytes = max(self.buffer_peak_bytes, int(nbytes))
        return stream, buffer

    def release(self) -> None:
        for stream in self._streams:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception:  # noqa: BLE001
                    pass
        self._buffers = [None] * self._n


@dataclass
class _TensorSlot:
    """One streamed tensor: where it rests and how to put it back."""

    attr: str
    is_param: bool
    host: Any
    nbytes: int
    offset: int


class _StreamedLeaf:
    """The forward pre/post hook pair over one leaf module.

    Runs under the residency manager's single-flight guarantee (one request
    per model instance), so the in-flight stream handle needs no lock.
    """

    __slots__ = ("name", "module", "slots", "total", "ring", "_torch", "_stream", "_handles")

    def __init__(
        self,
        name: str,
        module: Any,
        slots: List[_TensorSlot],
        total: int,
        ring: _CastRing,
        torch: Any,
    ) -> None:
        self.name = name
        self.module = module
        self.slots = slots
        self.total = total
        self.ring = ring
        self._torch = torch
        self._stream: Any = None
        self._handles: List[Any] = []

    def install(self) -> None:
        # PREPENDED, so the H2D lands before any other pre-hook on this leaf
        # (an fp8 upcast window, say) reads the weight. `always_call` on the
        # post-hook keeps the restore on an exception: a mid-forward CUDA OOM
        # must not leave a cast-buffer view bound as the module's weight,
        # because the next acquisition of that stream overwrites it.
        self._handles.append(
            self.module.register_forward_pre_hook(self._pre, prepend=True)
        )
        self._handles.append(
            self.module.register_forward_hook(self._post, always_call=True)
        )

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()

    def _pre(self, module: Any, args: Any) -> None:
        torch = self._torch
        stream, buffer = self.ring.acquire(self.total)
        context = (
            torch.cuda.stream(stream) if stream is not None else _nullcontext()
        )
        with context:
            for slot in self.slots:
                host = slot.host
                view = (
                    buffer[slot.offset : slot.offset + slot.nbytes]
                    .view(host.dtype)
                    .view(host.shape)
                )
                view.copy_(host, non_blocking=stream is not None)
                bind_tensor(module, slot.attr, view, slot.is_param)
        if stream is not None:
            torch.cuda.current_stream(self.ring.device).wait_stream(stream)
        self._stream = stream

    def _post(self, module: Any, args: Any, output: Any = None) -> None:
        for slot in self.slots:
            bind_tensor(module, slot.attr, slot.host, slot.is_param)
        stream, self._stream = self._stream, None
        if stream is not None:
            # The other half of the fence (ComfyUI's ``uncast_bias_weight``):
            # the buffer must not be refilled while compute is still reading
            # the views this forward bound.
            stream.wait_stream(self._torch.cuda.current_stream(self.ring.device))


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def bind_tensor(module: Any, attr: str, value: Any, is_param: bool) -> None:
    if is_param:
        module._parameters[attr].data = value
    else:
        module._buffers[attr] = value


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------


def own_tensors(module: Any) -> List[Tuple[str, bool, Any]]:
    """(attr, is_param, tensor) for the module's OWN parameters and buffers."""
    out: List[Tuple[str, bool, Any]] = []
    for name, param in getattr(module, "_parameters", {}).items():
        if param is not None:
            out.append((name, True, param))
    for name, buf in getattr(module, "_buffers", {}).items():
        if buf is not None:
            out.append((name, False, buf))
    return out


def tensor_bytes(t: Any) -> int:
    return int(t.numel()) * int(t.element_size())


def is_adapter_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _ADAPTER_MARKERS)


def discover_leaves(
    roots: Sequence[Tuple[str, Any]],
) -> Tuple[Dict[str, Any], List[LeafCost], Set[str]]:
    """``({name: module}, [LeafCost], adapter_names)`` for one module forest.

    ONE walk, shared by every mechanism that holds this rung's contract — the
    software tail here and the arena facade in
    :mod:`gen_worker.models.arena_residency`. The two must agree leaf-for-leaf
    or their plans are not comparable, and a second copy of this walk is the
    way they would silently stop agreeing.
    """
    leaves: Dict[str, Any] = {}
    costs: List[LeafCost] = []
    adapters: Set[str] = set()
    for root_name, root in roots:
        for name, module in root.named_modules():
            if not is_streamable_leaf(module):
                continue
            qualified = f"{root_name}.{name}" if name else root_name
            if is_adapter_name(qualified):
                adapters.add(qualified)
            own = own_tensors(module)
            total = sum(tensor_bytes(t) for _, _, t in own)
            if total <= 0:
                continue
            leaves[qualified] = module
            # `cast_bytes` is the ALIGNED carve, so the plan's window
            # reservation is the number the ring really allocates rather than
            # one that ignores the 512-byte sub-buffer boundaries.
            costs.append(
                LeafCost(
                    qualified,
                    total,
                    sum(aligned(tensor_bytes(t)) for _, _, t in own),
                    # STATED, not derived from the name: a root that is itself a
                    # leaf has a dotless qualified name and still belongs to its
                    # own component for the call-boundary schedule (pgw#1515).
                    component=root_name,
                )
            )
    return leaves, costs, adapters


def is_streamable_leaf(module: Any) -> bool:
    """True leaves only: no children, and tensors of its own.

    A module that owns parameters AND children cannot be hooked safely — its
    post-hook fires after its children's forwards, so the cast-buffer views it
    bound would still be live while the ring hands the same buffer to a child.
    """
    for _ in module.children():
        return False
    return bool(own_tensors(module))


class StreamedResidency:
    """A budgeted resident set plus a streaming tail over one module tree.

    Construct over an author model, a pipeline or a bare ``nn.Module``
    (:meth:`over` finds the roots), :meth:`engage` to split at the lease's
    budget, :meth:`rebudget` to trim or extend it, :meth:`release` to put the
    tree back the way it was found.
    """

    def __init__(
        self,
        roots: Sequence[Tuple[str, Any]],
        *,
        device: Any = None,
        budget_bytes: "int | MemoryBudget" = 0,
        streams: int = DEFAULT_STREAMS,
        min_stream_bytes: int = DEFAULT_MIN_STREAM_BYTES,
        exclude: Iterable[str] = (),
        cells: "int | CellPolicy | None" = None,
    ) -> None:
        import torch

        self._torch = torch
        # `device=None` DERIVES the execution device from where this tree's
        # weights already are. A global probe would answer "cuda" for a
        # pipeline the CPU rung deliberately put on the host, and the first
        # promote would then move it somewhere nobody asked for.
        self.device = torch.device(
            device if device is not None else tree_device(roots) or "cpu"
        )
        self.budget = MemoryBudget.of(budget_bytes)
        self.streams = max(1, int(streams))
        self.min_stream_bytes = int(min_stream_bytes)
        self.cells = CellPolicy.of(cells)
        self._exclude = {str(n) for n in exclude}
        self._roots = list(roots)
        self._leaves: Dict[str, Any] = {}
        self._costs: List[LeafCost] = []
        self._streamed: Dict[str, _StreamedLeaf] = {}
        self._ring: Optional[_CastRing] = None
        self._lock = threading.Lock()
        self.plan: Optional[ResidencyPlan] = None
        self._discover()

    # -- construction -------------------------------------------------------

    @classmethod
    def over(
        cls,
        obj: Any,
        *,
        device: Any = None,
        budget_bytes: "int | MemoryBudget" = 0,
        **kwargs: Any,
    ) -> "StreamedResidency":
        """Build over whatever the caller has: a module, a diffusers pipeline
        (its ``components``), or an author model object holding either."""
        return cls(module_roots(obj), device=device, budget_bytes=budget_bytes, **kwargs)

    def _discover(self) -> None:
        self._leaves, self._costs, adapters = discover_leaves(self._roots)
        self._exclude |= adapters

    # -- read side ----------------------------------------------------------

    @property
    def costs(self) -> Tuple[LeafCost, ...]:
        return tuple(self._costs)

    @property
    def total_bytes(self) -> int:
        return sum(c.resident_bytes for c in self._costs)

    @property
    def buffer_peak_bytes(self) -> int:
        return self._ring.buffer_peak_bytes if self._ring is not None else 0

    # -- the one operation --------------------------------------------------

    def engage(self) -> ResidencyPlan:
        """Split at the construction budget and put the tree where it says."""
        return self._apply(
            plan_residency(
                self._costs,
                budget_bytes=self.budget,
                streams=self.streams,
                min_stream_bytes=self.min_stream_bytes,
                exclude=self._exclude,
                cells=self.cells,
            ),
            allow_promote=True,
        )

    def rebudget(self, budget_bytes: "int | MemoryBudget") -> ResidencyPlan:
        """Re-split at a new budget — partial unload down, partial load up.

        Downward is the eviction primitive: the cold tail is trimmed and
        nothing is promoted. ``rebudget(0)`` is a full demotion to the host
        tier; ``rebudget(self.total_bytes)`` promotes everything back.
        """
        pair = MemoryBudget.of(budget_bytes)
        # The RAM half survives a re-plan that only names a VRAM number: a
        # demotion does not revoke the profile the model was assigned.
        if not pair.ram_bytes and self.budget.ram_bytes:
            pair = MemoryBudget(pair.vram_bytes, self.budget.ram_bytes)
        allow_promote = (
            self.plan is None or pair.vram_bytes >= self.plan.budget_bytes
        )
        self.budget = pair
        return self._apply(
            plan_residency(
                self._costs,
                budget_bytes=pair,
                streams=self.streams,
                min_stream_bytes=self.min_stream_bytes,
                exclude=self._exclude,
                cells=self.cells,
            ),
            allow_promote=allow_promote,
        )

    def partial_unload(self, need_bytes: int) -> int:
        """Free at least ``need_bytes`` of device weights; returns bytes freed.

        Trims this model's cold tail rather than dropping the model — the
        primitive that lets several warm endpoints degrade smoothly on one
        card instead of evicting each other whole.
        """
        before = self.plan.resident_bytes if self.plan is not None else self.total_bytes
        self.rebudget(max(0, before - int(need_bytes)))
        after = self.plan.resident_bytes if self.plan is not None else 0
        return max(0, before - after)

    def partial_load(self, extra_bytes: int) -> int:
        """Promote up to ``extra_bytes`` more of the tail; bytes claimed."""
        before = self.plan.resident_bytes if self.plan is not None else 0
        self.rebudget(before + max(0, int(extra_bytes)))
        after = self.plan.resident_bytes if self.plan is not None else 0
        return max(0, after - before)

    # -- the call-boundary schedule (pgw#1515) ------------------------------

    def hold_component(self, component: str) -> ResidencyPlan:
        """Hold ``component``'s cells whole and park every other component.

        THE call-boundary swap, and the ``model_offload``-equivalent mode on
        this rung's mechanism: the hot component runs fully resident for the
        whole call, so its move cost is paid once per CALL and amortises over
        every step, instead of the per-STEP cast this rung otherwise pays. The
        difference is measured, not argued — pgw#1497 clocked ``model_offload``
        at 1.57× against this rung's 1.91× at the SAME 1.9 GB peak, purely
        because of where the tax lands.

        REFUSES when the plan's schedule is not
        :attr:`ResidencySchedule.CALL_BOUNDARY`. A budget that cannot hold the
        biggest component whole would answer this call by silently overshooting
        its lease, and the whole point of deciding the schedule in the plan is
        that the answer is known before anything moves.
        """
        plan = self.plan if self.plan is not None else self.engage()
        wanted = str(component)
        if plan.schedule is not ResidencySchedule.CALL_BOUNDARY:
            raise ValueError(
                "hold_component(%r) refused: the assigned budget admits no whole "
                "component (hot %d B + window %d B > VRAM %d B, cold %d B vs RAM %d B) "
                "— this plan is scheduled %s"
                % (
                    wanted,
                    plan.hot_component_bytes,
                    plan.window_bytes,
                    plan.budget_bytes,
                    plan.cold_component_bytes,
                    plan.ram_budget_bytes,
                    plan.schedule.value,
                )
            )
        if wanted not in plan.hold_while_hot:
            raise ValueError(
                "hold_component(%r) refused: no such component — this tree has %r"
                % (wanted, plan.hold_while_hot)
            )
        return self._apply(_hot_plan(plan, wanted), allow_promote=True)

    def demote_to_host(self) -> int:
        """Every weight to pinned host RAM. The residency HOST tier."""
        return self.partial_unload(self.total_bytes)

    def promote_to_device(self) -> int:
        """Every weight back onto the card. An H2D, never a disk walk."""
        return self.partial_load(self.total_bytes)

    # -- execution ----------------------------------------------------------

    def _apply(self, plan: ResidencyPlan, *, allow_promote: bool) -> ResidencyPlan:
        with self._lock:
            torch = self._torch
            old = self.plan
            if old is None:
                demote = tuple(plan.streamed)
                promote = tuple(plan.all_resident)
            else:
                transition = plan_transition(
                    old, plan, self._costs, allow_promote=allow_promote
                )
                demote = transition.demote
                promote = transition.promote
                if not allow_promote:
                    # An unload keeps everything the trim did not name where it
                    # already is, so the plan we RECORD must say that rather
                    # than the one the planner would have drawn.
                    plan = self._recorded(plan, old, demote)

            with torch.no_grad():
                for name in demote:
                    self._to_host(name)
                for name in promote:
                    self._to_device(name)
                self._place_residue()
            self.plan = plan
            for _, root in self._roots:
                try:
                    setattr(root, ENGAGED_ATTR, bool(plan.streamed))
                except Exception:  # noqa: BLE001
                    pass
            return plan

    def _recorded(
        self, planned: ResidencyPlan, old: ResidencyPlan, demote: Tuple[str, ...]
    ) -> ResidencyPlan:
        """The plan an unload actually produced: ``old`` minus the trim."""
        dropped = set(demote)
        by_name = {c.name: c for c in self._costs}
        resident = tuple(n for n in old.resident if n not in dropped)
        forced = tuple(n for n in old.forced if n not in dropped)
        streamed = tuple(old.streamed) + tuple(
            n for n in old.all_resident if n in dropped
        )
        # The cells that survived the trim, so the span and tax numbers keep
        # describing what is really mapped rather than what the planner drew.
        kept = set(forced) | set(resident)
        held = [c for c in planned.cells if kept.issuperset(c.members)]
        parked = [c for c in planned.cells if not kept.issuperset(c.members)]
        return ResidencyPlan(
            budget_bytes=planned.budget_bytes,
            streams=planned.streams,
            forced=forced,
            resident=resident,
            streamed=streamed,
            resident_bytes=sum(
                by_name[n].resident_bytes for n in forced + resident if n in by_name
            ),
            streamed_bytes=sum(
                by_name[n].resident_bytes for n in streamed if n in by_name
            ),
            window_bytes=planned.window_bytes,
            ram_budget_bytes=planned.ram_budget_bytes,
            policy=planned.policy,
            cells=planned.cells,
            resident_cells=tuple(c.index for c in held),
            streamed_cells=tuple(c.index for c in parked),
            resident_span_bytes=sum(c.span_bytes for c in held),
            leaf_granular_span_bytes=sum(c.leaf_span_bytes for c in held),
            schedule=planned.schedule,
            component_spans=planned.component_spans,
            hot_component_bytes=planned.hot_component_bytes,
            cold_component_bytes=planned.cold_component_bytes,
            hold_while_hot=planned.hold_while_hot,
        )

    def _place_residue(self) -> None:
        """Everything OUTSIDE the streaming tail must be ON the device.

        Not an optimization — a correctness clause, and dropping it is a defect
        MEASURED on the card: a tensor owned by a module that has children
        (CLIP's ``position_ids``, an embedding table's siblings, a final norm
        hanging off a container) is never a leaf, so nothing else here ever
        touches it. Left on the host it takes down the first kernel that reads
        it with ``index is on cpu, different from other tensors on cuda:0`` —
        which is exactly how sd1.5 died at a 5% budget.
        """
        for root_name, root in self._roots:
            for name, module in root.named_modules():
                qualified = f"{root_name}.{name}" if name else root_name
                if qualified in self._streamed:
                    continue  # its tensors rest on the host BY DESIGN
                for attr, is_param, tensor in own_tensors(module):
                    if tensor.is_meta or tensor.device == self.device:
                        continue
                    bind_tensor(module, attr, tensor.to(self.device), is_param)

    def _ring_for(self) -> _CastRing:
        if self._ring is None:
            self._ring = _CastRing(self._torch, self.device, self.streams)
        return self._ring

    def _to_host(self, name: str) -> None:
        """Park a leaf in pinned host RAM and hook its forward."""
        module = self._leaves.get(name)
        if module is None or name in self._streamed:
            return
        torch = self._torch
        slots: List[_TensorSlot] = []
        offset = 0
        for attr, is_param, tensor in own_tensors(module):
            if tensor.is_meta or tensor.storage_offset() != 0:
                # Meta or an alias into a larger storage: this mover cannot
                # represent it, so the leaf stays where it is rather than
                # being half-moved. Loud, because it silently costs budget.
                logger.warning(
                    "stream residency: %s.%s is meta or a partial-view alias; "
                    "the leaf stays resident and its bytes are over budget",
                    name, attr,
                )
                return
            host = staging.alloc_pinned_like(torch, tensor)
            if host is None:
                host = torch.empty_like(tensor, device="cpu")
            host.copy_(tensor)
            nbytes = tensor_bytes(host)
            slots.append(
                _TensorSlot(
                    attr=attr, is_param=is_param, host=host, nbytes=nbytes, offset=offset
                )
            )
            offset += aligned(nbytes)
            bind_tensor(module, attr, host, is_param)
        leaf = _StreamedLeaf(name, module, slots, offset, self._ring_for(), torch)
        leaf.install()
        self._streamed[name] = leaf

    def _to_device(self, name: str) -> None:
        """Put a leaf back on the card and drop its hooks."""
        module = self._leaves.get(name)
        if module is None:
            return
        leaf = self._streamed.pop(name, None)
        if leaf is not None:
            leaf.remove()
            for slot in leaf.slots:
                try:
                    pinned = bool(slot.host.is_pinned())
                except Exception:  # noqa: BLE001
                    pinned = False
                bind_tensor(
                    module,
                    slot.attr,
                    slot.host.to(self.device, non_blocking=pinned),
                    slot.is_param,
                )
            return
        for attr, is_param, tensor in own_tensors(module):
            if tensor.is_meta or tensor.device == self.device:
                continue
            bind_tensor(module, attr, tensor.to(self.device), is_param)

    def release(self) -> None:
        """Un-hook everything and leave every weight on the device."""
        with self._lock:
            with self._torch.no_grad():
                for name in list(self._streamed):
                    self._to_device(name)
            if self._ring is not None:
                self._ring.release()
                self._ring = None
            for _, root in self._roots:
                try:
                    setattr(root, ENGAGED_ATTR, False)
                except Exception:  # noqa: BLE001
                    pass
            self.plan = None


# ---------------------------------------------------------------------------
# Finding the tree
# ---------------------------------------------------------------------------


def module_roots(obj: Any) -> List[Tuple[str, Any]]:
    """(name, module) for every ``nn.Module`` tree under ``obj``.

    A bare module is itself; a diffusers pipeline is its ``components``; an
    author model object is whichever of its attributes are either. One walk
    for all three, because the serve path holds all three shapes.
    """
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        return []
    if isinstance(obj, nn.Module):
        return [(type(obj).__name__, obj)]
    components = getattr(obj, "components", None)
    if isinstance(components, dict) and components:
        return [(n, m) for n, m in components.items() if isinstance(m, nn.Module)]
    out: List[Tuple[str, Any]] = []
    for name, value in vars(obj).items():
        if name.startswith("_"):
            continue
        if isinstance(value, nn.Module):
            out.append((name, value))
            continue
        nested = getattr(value, "components", None)
        if isinstance(nested, dict) and nested:
            out.extend(
                (f"{name}.{n}", m) for n, m in nested.items() if isinstance(m, nn.Module)
            )
    return out


def tree_device(roots: Sequence[Tuple[str, Any]]) -> Optional[Any]:
    """Where this tree's weights ARE — the device of its first real tensor.

    Measured, never assumed: the execution device of an already-loaded object
    is a property of the object, and a probe of the machine answers a
    different question.
    """
    for _, root in roots:
        for module in root.modules():
            for _attr, _is_param, tensor in own_tensors(module):
                if not tensor.is_meta:
                    return tensor.device
    return None


def stream_residency_active(obj: Any) -> bool:
    """True when this rung has a streaming tail armed anywhere under ``obj``."""
    if getattr(obj, ENGAGED_ATTR, False):
        return True
    return any(getattr(root, ENGAGED_ATTR, False) for _, root in module_roots(obj))


__all__ = [
    "DEFAULT_MIN_STREAM_BYTES",
    "DEFAULT_STREAMS",
    "ENGAGED_ATTR",
    "LEAF_CELLS",
    "VMM_GRANULARITY_BYTES",
    "CellPolicy",
    "LeafCost",
    "MemoryBudget",
    "PlanTransition",
    "ResidencyCell",
    "ResidencyPlan",
    "ResidencySchedule",
    "StreamedResidency",
    "align_to",
    "aligned",
    "bind_tensor",
    "discover_leaves",
    "is_adapter_name",
    "is_streamable_leaf",
    "module_roots",
    "own_tensors",
    "pack_cells",
    "tensor_bytes",
    "plan_residency",
    "plan_transition",
    "stream_residency_active",
    "tree_device",
]
