"""The varena facade: the same residency contract, the arena as the mechanism.

pgw#1507. :mod:`gen_worker.models.stream_residency` (pgw#1497) shipped the
CONTRACT — :func:`~gen_worker.models.stream_residency.plan_residency`, a
``{VRAM, RAM}`` :class:`~gen_worker.models.stream_residency.MemoryBudget`, and
``engage``/``rebudget``/``partial_unload``/``demote_to_host``. It shipped one
MECHANISM behind it: a pinned-CPU tail cast per forward into a shared ring
buffer. This module is the SECOND mechanism behind the SAME contract, so the
rung swaps without an endpoint noticing.

The mechanism, in one paragraph. One :class:`varena.Arena` per model owns the
lease's VRAM half; one ``Reservation`` owns a virtual range big enough for
every managed weight and **that range never moves**. Each leaf gets a
granularity-aligned region inside it. A RESIDENT leaf's region is ``back()``-ed
— physical chunks mapped under the stable address — and the module's parameter
is a zero-copy DLPack view of it. A STREAMED leaf's region is unbacked, costing
zero VRAM, and its forward pre-hook backs the region, refills it (from the
pinned host mirror, or straight off disk through varena's ``RefillEngine``) and
binds the view; a bounded ring of DEFERRED unbacks releases it again once the
compute that read it has actually finished.

**Why the address is the point.** The software rung's tail lands in a shared
cast buffer, so a leaf's weight sits at a different address on every forward —
one of the two reasons compiled graphs and offload are mutually exclusive
(``serving/context.py``). Here the address is a property of the LAYOUT, not of
residency: promote, demote and rebudget move physical pages under a fixed
pointer. **That exclusivity is NOT lifted in this lane** — it is roadmap phase
3 — but the mechanism that lifts it is this one.

**The granularity tax, stated up front.** CUDA VMM maps physical memory in
chunks (2 MiB on this fleet's cards). Independent per-leaf ``back``/``unback``
therefore needs per-leaf granularity-aligned regions, so a leaf pays for the
remainder of its last chunk. That cost is hidden nowhere: the layout reports
each region's ALIGNED SPAN as its
:class:`~gen_worker.models.stream_residency.LeafCost`, so the planner spends
the real number, ``plan.resident_bytes`` equals the bytes actually mapped, and
``plan.fits`` is exact instead of optimistic. :attr:`ArenaLayout.tax_bytes`
names the total.

**Nothing here is a policy.** No NVML, no probe, no prefetch, no eviction
heuristic: the budget is the lease's, the split is pgw#1497's planner, and the
only decision this module makes is arithmetic. varena's minimalism ruling
applies to its caller too — the dtype→DLPack table, the layout, the triple
resolution and the fencing live HERE because they are caller policy, and
nothing in this lane asked varena to grow.

**Two contracts every caller of this facade inherits** (va#12; varena's
README carries the same words — this paragraph exists for readers who never
open it):

* **The arena governs WEIGHTS only.** Activations, VAE workspaces and
  text-encoder transients live outside it, in torch's allocator, and are the
  caller's problem — both arms of the ComfyUI floor measurement died at
  ``VAEDecode``, on memory varena would never have been asked about.
  "Demand-paged VRAM residency" is a promise about a model's parameters,
  never about its peak. Weights are immutable and CAS-backed, so demote is
  unmap-only — parking MUTABLE state would need a write-back leg varena
  deliberately lacks (refill is one-way disk→pin→VRAM).
* **The launch window: nothing is unbacked between pre-resident and
  launch-complete.** The caller is the only party that ever calls
  ``unback``, so this is a contract to KEEP, not a mechanism varena can
  enforce or violate on its own. Pre-resident the whole working set, launch,
  and only then re-budget — the :class:`UnbackRing`'s event gating is this
  module's implementation of exactly that. It is what makes a stable pointer
  safe to bake into a compiled artifact or a CUDA graph, and it is universal
  practice in the field, not a peculiarity of ours: no comparable system
  pages during a compiled launch.

Residency signatures (scalar ``signature()`` and va#12's per-2 MiB-chunk
``page_signatures``) are BACKING signatures, never CONTENT signatures: bytes
changed under an unchanged mapping are invisible to them, and content
integrity is a caller-side D2H digest (the checkpoint juggle, pgw#1607,
banks per-region digests at ingest for that reason).
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import staging
from .safetensors_header import header_len_ok
from .stream_residency import (
    DEFAULT_MIN_STREAM_BYTES,
    DEFAULT_STREAMS,
    ENGAGED_ATTR,
    LeafCost,
    MemoryBudget,
    ResidencyPlan,
    aligned,
    bind_tensor,
    discover_leaves,
    module_roots,
    own_tensors,
    plan_residency,
    plan_transition,
    tensor_bytes,
    tree_device,
)

logger = logging.getLogger(__name__)

#: DLPack type codes. varena hands the caller this mapping on purpose (its
#: minimalism ruling), so it is stated once, here.
DL_INT, DL_UINT, DL_FLOAT, DL_BFLOAT, DL_BOOL = 0, 1, 2, 4, 6

#: Fallback CUDA VMM allocation granularity. The real number comes from
#: ``Arena.granularity``; this exists so the pure-arithmetic layout is testable
#: with no driver present.
DEFAULT_GRANULARITY = 2 << 20

#: The layout's name for the packed always-resident region.
CORE_REGION = "__core__"


def dlpack_dtype(torch: Any, dtype: Any) -> Tuple[int, int]:
    """``(code, bits)`` for a torch dtype, or a refusal.

    Deliberately a closed list. An unmapped dtype must be a loud refusal and
    never a guess: a wrong code hands torch a correctly-SIZED view of
    misinterpreted bytes, which is a silent numerical fault rather than a
    crash.
    """
    table = {
        torch.float16: (DL_FLOAT, 16),
        torch.float32: (DL_FLOAT, 32),
        torch.float64: (DL_FLOAT, 64),
        torch.bfloat16: (DL_BFLOAT, 16),
        torch.int8: (DL_INT, 8),
        torch.int16: (DL_INT, 16),
        torch.int32: (DL_INT, 32),
        torch.int64: (DL_INT, 64),
        torch.uint8: (DL_UINT, 8),
        torch.bool: (DL_BOOL, 8),
    }
    try:
        return table[dtype]
    except KeyError:
        raise TypeError(
            f"arena residency has no DLPack code for {dtype}; refusing rather "
            f"than guessing one (a wrong code is a silent numerical fault)"
        ) from None


# ---------------------------------------------------------------------------
# The layout — pure arithmetic, no torch, no driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """One parameter or buffer at a fixed offset in the reservation."""

    leaf: str
    attr: str
    is_param: bool
    offset: int
    nbytes: int
    shape: Tuple[int, ...]
    dtype_code: int
    dtype_bits: int


@dataclass(frozen=True)
class RegionSpec:
    """One independently mappable unit: a leaf, or the packed forced core.

    ``span`` is granularity-aligned and is what ``back``/``unback`` are called
    with, so two regions can never share a chunk — which is what makes a
    per-forward unback unable to pull the ground out from under a neighbour
    that is still in flight.
    """

    name: str
    offset: int
    span: int
    weight_bytes: int
    slots: Tuple[SlotSpec, ...]

    @property
    def tax_bytes(self) -> int:
        return self.span - self.weight_bytes


@dataclass(frozen=True)
class ArenaLayout:
    """Where every managed weight lives inside the reservation."""

    granularity: int
    regions: Tuple[RegionSpec, ...]
    #: Leaf names packed into the always-resident core region.
    core_names: Tuple[str, ...]
    virtual_bytes: int
    weight_bytes: int

    @property
    def tax_bytes(self) -> int:
        """Chunk-remainder bytes the alignment costs when everything is backed."""
        return sum(r.tax_bytes for r in self.regions)

    def by_name(self) -> Dict[str, RegionSpec]:
        return {r.name: r for r in self.regions}

    def costs(self) -> Tuple[LeafCost, ...]:
        """The planner's costs, priced in ALIGNED SPANS.

        The tax is charged to the planner rather than discovered afterwards. A
        plan whose ``resident_bytes`` was the raw weight bytes would
        under-report what the card holds by exactly the remainder of every
        region's last chunk — a number wrong in the direction that OOMs.
        """
        return tuple(LeafCost(r.name, r.span, r.span) for r in self.regions)


#: One leaf as the layout needs it: ``(attr, is_param, nbytes, shape, code, bits)``.
TensorSpec = Tuple[str, bool, int, Tuple[int, ...], int, int]


def plan_layout(
    leaf_tensors: Sequence[Tuple[str, Sequence[TensorSpec]]],
    *,
    granularity: int = DEFAULT_GRANULARITY,
    min_stream_bytes: int = DEFAULT_MIN_STREAM_BYTES,
    exclude: Iterable[str] = (),
) -> ArenaLayout:
    """Lay every managed weight out in one reservation. Pure arithmetic.

    Two kinds of region:

    * **The core**, at offset 0: every leaf that can never stream (smaller than
      ``min_stream_bytes``, or excluded by the caller — adapters, the
      dtype-fragile components). Packed together at 512-byte alignment, so the
      set pays ONE chunk remainder instead of one each. On sd1.5's unet that is
      226 leaves totalling 67 MB paying a single sub-chunk remainder.
    * **One region per streaming candidate**, granularity-aligned. This is what
      buys independent residency: ``unback`` releases the chunks WHOLLY inside
      its request, so two leaves sharing a chunk could never release it — and
      two leaves sharing a chunk while one is in flight would race. Alignment
      is the price of per-leaf residency and it is charged to the planner.

    Region order is the planner's fill order (descending size, ties on name),
    so the resident set is very nearly a prefix of the reservation and the
    layout is deterministic for a given tree.
    """
    gran = max(1, int(granularity))
    skip = {str(n) for n in exclude}
    floor = int(min_stream_bytes)

    sized = [(name, list(slots), sum(int(s[2]) for s in slots)) for name, slots in leaf_tensors]
    core = [e for e in sized if e[0] in skip or e[2] < floor]
    core_names = {e[0] for e in core}
    cand = [e for e in sized if e[0] not in core_names]
    cand.sort(key=lambda e: (-e[2], e[0]))
    core.sort(key=lambda e: e[0])

    def _slots(entries: Sequence[Any], base: int) -> Tuple[Tuple[SlotSpec, ...], int]:
        out: List[SlotSpec] = []
        off = base
        for leaf_name, slots, _total in entries:
            for attr, is_param, nbytes, shape, code, bits in slots:
                out.append(
                    SlotSpec(
                        leaf=leaf_name,
                        attr=attr,
                        is_param=bool(is_param),
                        offset=off,
                        nbytes=int(nbytes),
                        shape=tuple(int(d) for d in shape),
                        dtype_code=int(code),
                        dtype_bits=int(bits),
                    )
                )
                off += aligned(int(nbytes))
        return tuple(out), off - base

    regions: List[RegionSpec] = []
    cursor = 0
    if core:
        slots, used = _slots(core, 0)
        span = -(-used // gran) * gran
        regions.append(
            RegionSpec(CORE_REGION, 0, span, sum(e[2] for e in core), slots)
        )
        cursor = span
    for entry in cand:
        slots, used = _slots([entry], cursor)
        span = -(-used // gran) * gran
        regions.append(RegionSpec(entry[0], cursor, span, entry[2], slots))
        cursor += span

    return ArenaLayout(
        granularity=gran,
        regions=tuple(regions),
        core_names=tuple(e[0] for e in core),
        virtual_bytes=cursor,
        weight_bytes=sum(e[2] for e in sized),
    )


# ---------------------------------------------------------------------------
# (path, offset, len) triples from the local weight files
# ---------------------------------------------------------------------------


def safetensors_triples(
    directory: "str | Path", *, variant: Optional[str] = None
) -> Dict[str, Tuple[Path, int, int]]:
    """``{tensor key: (path, byte offset, byte length)}`` for one component dir.

    The triples ``RefillEngine`` eats. The safetensors header already states
    each tensor's byte range within the data section, and the data section
    starts at ``8 + header_len`` — so the file offset is that plus the header's
    own ``data_offsets[0]``. No tensor is materialised to learn this.

    This function is THE SEAM. The local snapshot is one producer of triples;
    pgw#1498's CAS-native ingest is another, and it arrives as a different
    producer of this same dict — one constructor argument
    (:class:`ArenaResidency` ``triples=``), not a rewrite.
    """
    root = Path(directory)
    tail = f".{variant}.safetensors" if variant else ".safetensors"
    files = sorted(p for p in root.glob("*.safetensors") if p.name.endswith(tail))
    if not variant:
        files = [p for p in files if p.name.count(".") == 1]
    if not files:
        raise FileNotFoundError(f"no *{tail} weight files under {root}")

    out: Dict[str, Tuple[Path, int, int]] = {}
    for path in files:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (hlen,) = struct.unpack("<Q", raw)
            if not header_len_ok(hlen):
                raise ValueError(f"{path}: implausible safetensors header length {hlen}")
            header = json.loads(fh.read(hlen))
        base = 8 + hlen
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            out[key] = (path, base + int(start), int(end) - int(start))
    return out


# ---------------------------------------------------------------------------
# The deferred-unback ring
# ---------------------------------------------------------------------------


class UnbackRing:
    """Deferred unbacks, ``depth`` deep, each gated on a real CUDA event.

    THE correctness point of the module. A forward post-hook fires when the
    leaf's kernels are ENQUEUED, not when they have run, so unbacking there
    would unmap memory a live kernel is still reading — a fault the software
    rung cannot have, because it only rebinds a buffer it keeps alive. Instead
    the post-hook records an event on the compute stream and the region joins
    this ring; the unback happens when a LATER leaf needs the room, after that
    event has actually completed.

    ``depth`` is the planner's ``streams``, and that is not a coincidence: the
    ring holds at most ``depth`` backed streamed regions at once, which is
    exactly the in-flight window ``plan_residency`` reserved out of the budget.
    """

    def __init__(self, reservation: Any, depth: int) -> None:
        self._res = reservation
        self.depth = max(1, int(depth))
        self._pending: List[Tuple[RegionSpec, Any]] = []
        self.unbacked_bytes = 0
        self.unbacks = 0

    def retire(self, keep: int = 0) -> None:
        while len(self._pending) > max(0, keep):
            region, event = self._pending.pop(0)
            if event is not None:
                event.synchronize()
            self.unbacked_bytes += int(self._res.unback(region.offset, region.span))
            self.unbacks += 1

    def make_room(self) -> None:
        self.retire(keep=self.depth - 1)

    def defer(self, region: RegionSpec, event: Any) -> None:
        self._pending.append((region, event))

    def drain(self) -> None:
        self.retire(keep=0)

    def __len__(self) -> int:
        return len(self._pending)


class _ArenaLeaf:
    """The forward pre/post hook pair over one streamed leaf."""

    __slots__ = ("region", "module", "owner", "_handles")

    def __init__(self, region: RegionSpec, module: Any, owner: "ArenaResidency") -> None:
        self.region = region
        self.module = module
        self.owner = owner
        self._handles: List[Any] = []

    def install(self) -> None:
        # Prepended and ``always_call``-ed for pgw#1497's reasons: the bytes
        # must land before any other pre-hook reads the weight, and a
        # mid-forward exception must not leave a view of a region this ring is
        # about to unmap bound as the module's parameter.
        self._handles.append(self.module.register_forward_pre_hook(self._pre, prepend=True))
        self._handles.append(self.module.register_forward_hook(self._post, always_call=True))

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()

    def _pre(self, module: Any, args: Any) -> None:
        self.owner._page_in(self.region)

    def _post(self, module: Any, args: Any, output: Any = None) -> None:
        self.owner._page_out(self.region)


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------


class ArenaResidency:
    """pgw#1497's residency contract, served out of a varena arena.

    Surface-identical to
    :class:`~gen_worker.models.stream_residency.StreamedResidency`: construct
    over a pipeline or module tree, :meth:`engage`, :meth:`rebudget`,
    :meth:`partial_unload`, :meth:`demote_to_host`, :meth:`promote_to_device`,
    :meth:`release`. Each returns the same
    :class:`~gen_worker.models.stream_residency.ResidencyPlan`.

    ``triples`` is the disk source (see :func:`safetensors_triples`).
    ``host_mirror=False`` states a RAM half of zero: the tail then has no
    pinned mirror and every page-in comes off disk through the RefillEngine.
    That is pgw#1497's owed RAM-half enforcement arriving as a DEGRADE rather
    than a refusal — a model whose RAM allowance cannot hold the tail still
    serves, one rung slower.
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
        triples: Optional[Dict[str, Tuple[Path, int, int]]] = None,
        host_mirror: bool = True,
    ) -> None:
        import torch
        import varena

        self._torch = torch
        self._varena = varena
        self.device = torch.device(device if device is not None else tree_device(roots) or "cuda")
        if self.device.type != "cuda":
            raise ValueError(
                f"arena residency needs a CUDA device; this tree's weights are on {self.device}"
            )
        self.budget = MemoryBudget.of(budget_bytes)
        self.streams = max(1, int(streams))
        self.min_stream_bytes = int(min_stream_bytes)
        self._triples = dict(triples or {})
        self._host_mirror = bool(host_mirror)
        self._roots = list(roots)

        self._leaves, _discovered, adapters = discover_leaves(self._roots)
        self._exclude = {str(n) for n in exclude} | adapters

        self.arena = varena.Arena(
            device=int(self.device.index or 0), budget_bytes=max(0, self.budget.vram_bytes)
        )
        self.layout = plan_layout(
            self._tensor_specs(),
            granularity=self.arena.granularity,
            min_stream_bytes=self.min_stream_bytes,
            exclude=self._exclude,
        )
        self.reservation = self.arena.reserve(
            max(self.layout.virtual_bytes, self.arena.granularity)
        )
        self._regions = self.layout.by_name()
        self._costs = list(self.layout.costs())
        self._backed: Dict[str, bool] = {}
        self._hooks: Dict[str, _ArenaLeaf] = {}
        self._host: Dict[str, List[Any]] = {}
        self.ring = UnbackRing(self.reservation, self.streams)
        self._offload = [
            torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
            for _ in range(self.streams)
        ]
        self._offload_index = 0
        self._engine: Any = None
        self._slab_pool: Any = None
        self._staging_slab: Any = None
        self.plan: Optional[ResidencyPlan] = None
        self.adopted = False
        self.page_ins = 0
        self.unpinned_slots = 0
        #: Every residency mutation's reservation signature, newest last. The
        #: audit trail the "signature-verified" done-criterion reads.
        self.signatures: List[int] = [int(self.reservation.signature())]

    @classmethod
    def over(cls, obj: Any, **kwargs: Any) -> "ArenaResidency":
        return cls(module_roots(obj), **kwargs)

    @classmethod
    def arm(cls, pipeline: Any, **kwargs: Any) -> "ArenaResidency":
        """Arm the facade on a diffusers pipeline, the way the rung arms.

        The same three obligations ``_apply_partial_stream`` discharges, and
        each of them is a defect pgw#1497 MEASURED on this card rather than a
        precaution:

        1. **Components no rung may hook are excluded AND placed.** An
           exclusion is a statement about hooks, never about residency — the
           dtype-fragile VAE excluded and then left to nobody killed the first
           decode with ``weight type torch.HalfTensor``.
        2. **The component vocabulary is not all modules.** A pipeline's
           ``components`` holds tokenizers, schedulers and feature extractors;
           handing one to ``named_modules`` raises, and the rung then silently
           served on a coarser one.
        3. **The handle is stamped on the pipeline and the execution-device
           repair is installed.** Parking leaves on the host makes
           ``pipeline.device`` — a PUBLIC answer — report the host, and the
           pipeline then builds ``input_ids`` there and dies in the first
           embedding with ``index is on cpu``. The repair already exists and is
           mechanism-agnostic: it asks the stamped handle where the model
           executes, so the arena gets it by being stamped.
        """
        from .memory import (
            STREAM_RESIDENCY_ATTR,
            _named_components,
            install_execution_device_fallback,
            unhookable_components,
        )

        excluded = set(unhookable_components(pipeline))
        roots = [
            (name, module)
            for name, module in _named_components(pipeline)
            if name not in excluded and hasattr(module, "named_modules")
        ]
        if not roots and hasattr(pipeline, "named_modules"):
            roots = [(type(pipeline).__name__, pipeline)]
        if not roots:
            raise ValueError(
                f"arena residency: no hookable nn.Module tree on "
                f"{type(pipeline).__name__}"
            )
        device = kwargs.get("device") or "cuda"
        components = getattr(pipeline, "components", None)
        for name in sorted(excluded):
            module = components.get(name) if isinstance(components, dict) else None
            if module is None or not hasattr(module, "to"):
                continue
            try:
                module.to(device)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "arena residency: could not keep excluded component %r on "
                    "%s (%s: %s); it will not serve", name, device, type(exc).__name__, exc,
                )
        residency = cls(roots, **kwargs)
        try:
            setattr(pipeline, STREAM_RESIDENCY_ATTR, residency)
        except Exception:  # noqa: BLE001
            logger.warning(
                "arena residency: could not stamp the handle on %s; "
                "`pipeline.device` will keep answering with the host while the "
                "tail is parked there", type(pipeline).__name__,
            )
        install_execution_device_fallback()
        return residency

    # -- read side ----------------------------------------------------------

    @property
    def costs(self) -> Tuple[LeafCost, ...]:
        return tuple(self._costs)

    @property
    def total_bytes(self) -> int:
        """What full residency costs ON THE CARD — spans, not raw weights."""
        return sum(r.span for r in self.layout.regions)

    @property
    def weight_bytes(self) -> int:
        return self.layout.weight_bytes

    def stats(self) -> Dict[str, Any]:
        out = dict(self.arena.stats())
        out["signature"] = int(self.reservation.signature())
        out["backed_regions"] = sum(1 for v in self._backed.values() if v)
        out["layout_tax_bytes"] = self.layout.tax_bytes
        out["page_ins"] = self.page_ins
        out["unbacks"] = self.ring.unbacks
        out["unpinned_slots"] = self.unpinned_slots
        return out

    def is_resident(self, name: str) -> bool:
        return bool(self._backed.get(name))

    # -- the one operation --------------------------------------------------

    def engage(self) -> ResidencyPlan:
        plan = self._plan(self.budget)
        self._set_ceiling(plan)
        return self._apply(plan, allow_promote=True)

    def rebudget(self, budget_bytes: "int | MemoryBudget") -> ResidencyPlan:
        pair = MemoryBudget.of(budget_bytes)
        # A re-plan naming only a VRAM number does not revoke the assigned RAM
        # half (pgw#1497's pair rule).
        if not pair.ram_bytes and self.budget.ram_bytes:
            pair = MemoryBudget(pair.vram_bytes, self.budget.ram_bytes)
        allow_promote = self.plan is None or pair.vram_bytes >= self.plan.budget_bytes
        self.budget = pair
        # The arena's ceiling IS the lease: raise it before a promote so the
        # `back()` has room, lower it after a demote so a later `back()` past
        # the assigned VRAM is REFUSED by the allocator rather than discovered
        # as an OOM three kernels later.
        plan = self._plan(pair)
        if pair.vram_bytes > int(self.arena.stats()["budget_bytes"]):
            self._set_ceiling(plan)
        applied = self._apply(plan, allow_promote=allow_promote)
        self._set_ceiling(applied)
        return applied

    def _set_ceiling(self, plan: ResidencyPlan) -> None:
        """The arena ceiling: the lease, or what the plan holds if that is more.

        A plan whose forced core plus one in-flight window is already over the
        lease is over it — ``plan.fits`` says so and pgw#1497's confession
        machinery reports it. Setting the ARENA's ceiling below what that plan
        will map would turn a reported over-lease into a ``BudgetExceeded``
        four kernels into the next forward, which is a worse failure of the
        same fact.
        """
        ceiling = max(int(self.budget.vram_bytes), int(plan.device_bytes))
        if ceiling > int(self.budget.vram_bytes):
            logger.warning(
                "arena residency: the plan maps %.2f MiB against a %.2f MiB "
                "lease (forced core %.2f MiB + %.2f MiB in-flight window); the "
                "arena ceiling is raised to what the plan holds and plan.fits "
                "is False",
                ceiling / 1048576, self.budget.vram_bytes / 1048576,
                plan.resident_bytes / 1048576, plan.window_bytes / 1048576,
            )
        self.arena.set_budget(max(0, ceiling))

    def partial_unload(self, need_bytes: int) -> int:
        before = self.plan.resident_bytes if self.plan is not None else self.total_bytes
        self.rebudget(max(0, before - int(need_bytes)))
        after = self.plan.resident_bytes if self.plan is not None else 0
        return max(0, before - after)

    def partial_load(self, extra_bytes: int) -> int:
        before = self.plan.resident_bytes if self.plan is not None else 0
        self.rebudget(before + max(0, int(extra_bytes)))
        after = self.plan.resident_bytes if self.plan is not None else 0
        return max(0, after - before)

    def demote_to_host(self) -> int:
        """Every managed weight off the card — ``unback``, not a copy."""
        return self.partial_unload(self.total_bytes)

    def promote_to_device(self) -> int:
        """Every managed weight back, at the SAME addresses it left."""
        return self.partial_load(self.total_bytes)

    # -- planning -----------------------------------------------------------

    def _plan(self, budget: MemoryBudget) -> ResidencyPlan:
        # `min_stream_bytes=0` and the core exclusion, because the LAYOUT
        # already decided what can stream; re-deciding it here would let the
        # two disagree.
        return plan_residency(
            self._costs,
            budget_bytes=budget,
            streams=self.streams,
            min_stream_bytes=0,
            exclude=(CORE_REGION,),
        )

    def _tensor_specs(self) -> List[Tuple[str, List[TensorSpec]]]:
        torch = self._torch
        out: List[Tuple[str, List[TensorSpec]]] = []
        for name, module in self._leaves.items():
            slots: List[TensorSpec] = []
            for attr, is_param, tensor in own_tensors(module):
                if tensor.storage_offset() != 0 or not tensor.is_contiguous():
                    logger.warning(
                        "arena residency: %s.%s is a partial-view alias or "
                        "non-contiguous; the whole leaf stays outside the arena",
                        name, attr,
                    )
                    slots = []
                    break
                code, bits = dlpack_dtype(torch, tensor.dtype)
                slots.append(
                    (attr, is_param, tensor_bytes(tensor), tuple(tensor.shape), code, bits)
                )
            if slots:
                out.append((name, slots))
        return out

    # -- execution ----------------------------------------------------------

    def _apply(self, plan: ResidencyPlan, *, allow_promote: bool) -> ResidencyPlan:
        torch = self._torch
        with torch.no_grad():
            self.ring.drain()
            if not self.adopted:
                self._adopt(plan)
                self.adopted = True
            else:
                old = self.plan
                assert old is not None
                transition = plan_transition(old, plan, self._costs, allow_promote=allow_promote)
                if not allow_promote:
                    plan = _recorded(plan, old, transition.demote, self._costs)
                for name in transition.demote:
                    self._demote(name)
                for name in transition.promote:
                    self._promote(name)
            self._place_residue()
        self.signatures.append(int(self.reservation.signature()))
        self.plan = plan
        for _, root in self._roots:
            try:
                setattr(root, ENGAGED_ATTR, bool(plan.streamed))
            except Exception:  # noqa: BLE001
                pass
        return plan

    def _adopt(self, plan: ResidencyPlan) -> None:
        """First engage: every managed weight leaves the tree's own tensors.

        Region by region, in LAYOUT order, so the arena's chunk pool is filled
        front-to-back and the resident prefix is contiguous. Torch's caching
        allocator is emptied afterwards: the weights it held are unreferenced
        now, but they stay in ITS pool and are invisible to the driver — which
        is precisely the coexistence bite that makes an arena `back()` fail
        with memory that is technically free.
        """
        torch = self._torch
        resident = set(plan.all_resident)
        if all(self._is_cold(region) for region in self.layout.regions):
            self._adopt_cold(plan)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            return
        for region in self.layout.regions:
            cold = self._is_cold(region)
            if region.name in resident:
                self.reservation.back(region.offset, region.span)
                if cold:
                    # COLD: the tree was built on `meta` and the bytes have
                    # never been in this process. Disk to arena in one pass —
                    # no torch tensor is allocated to hold them on the way.
                    self._fill_from_disk(region)
                    for slot in region.slots:
                        self._bind(slot, self._view(slot))
                else:
                    for slot in region.slots:
                        view = self._view(slot)
                        view.copy_(self._live(slot))
                        self._bind(slot, view)
                self._backed[region.name] = True
            else:
                self._capture_host(region, from_live=not cold)
                self._backed[region.name] = False
                self._install_hook(region)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _adopt_cold(self, plan: ResidencyPlan) -> None:
        """A wholly cold tree: ONE batch, both destinations, one engine pass.

        MEASURED, and it is the whole cold-load story: adopting region by
        region costs one ``submit``/``wait`` round trip EACH — 239 of them on
        sd1.5 — and each round trip drains the engine's queue to empty, so the
        disk never has more than one region's reads outstanding and the io_uring
        queue depth buys nothing. Handing the engine every triple at once is
        what lets it keep the queue full, and it is also the shape pgw#1498's
        CAS ingest will want.
        """
        torch = self._torch
        resident = set(plan.all_resident)
        base = int(self.reservation.base_ptr)
        requests: List[Tuple[str, int, int, int, int]] = []
        pending_host: Dict[str, List[Any]] = {}
        for region in self.layout.regions:
            if region.name in resident:
                self.reservation.back(region.offset, region.span)
                for slot in region.slots:
                    path, offset, length = self._triple(slot)
                    requests.append((str(path), offset, length, 0, base + slot.offset))
            else:
                mirrors: List[Any] = []
                for slot in region.slots:
                    dtype = _dtype_of(torch, slot)
                    template = torch.empty(slot.shape, dtype=dtype, device="meta")
                    pinned = staging.alloc_pinned_like(torch, template)
                    if pinned is None:
                        self.unpinned_slots += 1
                        pinned = torch.empty(slot.shape, dtype=dtype, device="cpu")
                    path, offset, length = self._triple(slot)
                    requests.append((str(path), offset, length, int(pinned.data_ptr()), 0))
                    mirrors.append(pinned)
                pending_host[region.name] = mirrors
        self._engine_for().submit(requests, 0).wait()
        torch.cuda.synchronize(self.device)
        for region in self.layout.regions:
            if region.name in resident:
                for slot in region.slots:
                    self._bind(slot, self._view(slot))
                self._backed[region.name] = True
            else:
                self._host[region.name] = pending_host[region.name]
                self._rebind_off_device(region)
                self._backed[region.name] = False
                self._install_hook(region)

    def _is_cold(self, region: RegionSpec) -> bool:
        """True when this region's weights are not in the process yet.

        A meta-initialised tree states its shapes and dtypes and holds no
        bytes; that is not an error to route around but the CHEAP path — the
        weights go disk → arena and never occupy a torch allocation at all.
        """
        return any(self._live(slot).is_meta for slot in region.slots)

    def _promote(self, name: str) -> None:
        """``back`` the region, refill it, bind views. The address is unchanged."""
        region = self._regions.get(name)
        if region is None or self._backed.get(name):
            return
        self.reservation.back(region.offset, region.span)
        self._fill(region)
        for slot in region.slots:
            self._bind(slot, self._view(slot))
        self._backed[name] = True
        hook = self._hooks.pop(name, None)
        if hook is not None:
            hook.remove()
        self._host.pop(name, None)
        self._torch.cuda.synchronize(self.device)

    def _demote(self, name: str) -> None:
        """Mirror the region to host RAM (or to nothing), then ``unback`` it."""
        region = self._regions.get(name)
        if region is None or not self._backed.get(name):
            return
        if name == CORE_REGION:
            # The forced core has no hook, so nothing would ever page it back
            # in. `plan_residency` keeps forced leaves resident at every
            # budget, so reaching here means the two disagree.
            raise AssertionError(
                "arena residency: the forced core was scheduled for demotion; "
                "the layout and the planner disagree about what can stream"
            )
        self._capture_host(region, from_live=False)
        self.reservation.unback(region.offset, region.span)
        self._backed[name] = False
        self._install_hook(region)

    def _install_hook(self, region: RegionSpec) -> None:
        if region.name == CORE_REGION or region.name in self._hooks:
            return
        module = self._leaves.get(region.name)
        if module is None:
            return
        hook = _ArenaLeaf(region, module, self)
        hook.install()
        self._hooks[region.name] = hook

    # -- the streamed path --------------------------------------------------

    def _page_in(self, region: RegionSpec) -> None:
        torch = self._torch
        self.ring.make_room()
        self.reservation.back(region.offset, region.span)
        stream = self._offload[self._offload_index]
        self._offload_index = (self._offload_index + 1) % self.streams
        # The offload stream must not refill a region the previous forward is
        # still reading; the compute stream must not read before the refill.
        # Both fences, exactly as pgw#1497's ring does them.
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._fill(region, stream=stream)
            for slot in region.slots:
                self._bind(slot, self._view(slot))
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.page_ins += 1

    def _page_out(self, region: RegionSpec) -> None:
        torch = self._torch
        # Rebind FIRST: nothing may still point into a range the ring will
        # unmap.
        self._rebind_off_device(region)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.ring.defer(region, event)

    # -- byte movement ------------------------------------------------------

    def _fill(self, region: RegionSpec, *, stream: Any = None) -> None:
        """Get a backed region's bytes in, from RAM if there is RAM else disk."""
        host = self._host.get(region.name)
        if host is not None:
            for slot, pinned in zip(region.slots, host):
                self._view(slot).copy_(pinned, non_blocking=True)
            return
        self._fill_from_disk(region, stream=stream)

    def _fill_from_disk(self, region: RegionSpec, *, stream: Any = None) -> None:
        base = int(self.reservation.base_ptr)
        requests = []
        for slot in region.slots:
            path, offset, length = self._triple(slot)
            requests.append((str(path), offset, length, 0, base + slot.offset))
        handle = self._engine_for().submit(
            requests, int(stream.cuda_stream) if stream is not None else 0
        )
        handle.wait()

    def _read_into(self, slot: SlotSpec, pinned: Any) -> None:
        """One slot's bytes from disk into a pinned host page. Host destination.

        The same engine, the same triple; only the destination differs. A host
        mirror built this way never allocates a device byte, which is what
        makes a cold load of the STREAMED tail cost nothing on the card.
        """
        path, offset, length = self._triple(slot)
        handle = self._engine_for().submit(
            [(str(path), int(offset), int(length), int(pinned.data_ptr()), 0)], 0
        )
        handle.wait()

    def _triple(self, slot: SlotSpec) -> Tuple[Path, int, int]:
        key = self._triple_key(slot)
        triple = self._triples.get(key)
        if triple is None:
            raise KeyError(
                f"arena residency: no (path, offset, len) triple for {key!r}; "
                f"this region has neither a host mirror nor a disk source"
            )
        path, offset, length = triple
        if int(length) != slot.nbytes:
            raise ValueError(
                f"arena residency: {key!r} is {length} bytes on disk and "
                f"{slot.nbytes} in the tree — refusing a partial fill"
            )
        return Path(path), int(offset), int(length)

    def _engine_for(self) -> Any:
        if self._engine is None:
            # Device-only destinations bounce through the engine's own pinned
            # staging ring. One slab, one engine, for the life of the model.
            self._slab_pool = self._varena.SlabPool(32 << 20, pin="require")
            self._staging_slab = self._slab_pool.alloc(16 << 20)
            self._engine = self._varena.RefillEngine(
                staging=self._staging_slab, chunk_bytes=4 << 20
            )
        return self._engine

    def _triple_key(self, slot: SlotSpec) -> str:
        """``unet.down_blocks.0.x`` + ``weight`` → ``down_blocks.0.x.weight``.

        The root component name is the first segment of the qualified leaf name
        (:func:`~gen_worker.models.stream_residency.discover_leaves` builds it
        that way), and a component's own weight file keys its tensors relative
        to itself.
        """
        _root, _, rest = slot.leaf.partition(".")
        return f"{rest}.{slot.attr}" if rest else slot.attr

    def _capture_host(self, region: RegionSpec, *, from_live: bool) -> None:
        """Mirror a region into pinned host RAM, unless the RAM half says no."""
        torch = self._torch
        if self._host.get(region.name) is not None:
            # A mirror already exists (a prior capture, or the checkpoint
            # juggle's normalized image, pgw#1607). Weights are immutable so
            # it cannot be stale: demote is unmap-only, never copy-out (the
            # va#11 write-back invariant).
            self._rebind_off_device(region)
            return
        if not self._host_mirror:
            missing = [
                self._triple_key(s)
                for s in region.slots
                if self._triple_key(s) not in self._triples
            ]
            if missing:
                raise ValueError(
                    f"arena residency: {region.name} has no host mirror and no "
                    f"disk triple for {missing[:3]}; there would be nowhere to "
                    f"page it back from"
                )
            self._rebind_off_device(region)
            return
        mirrors: List[Any] = []
        for slot in region.slots:
            dtype = _dtype_of(torch, slot)
            template = torch.empty(slot.shape, dtype=dtype, device="meta")
            pinned = staging.alloc_pinned_like(torch, template)
            if pinned is None:
                # Pageable RAM: correct, and MUCH slower, because `non_blocking`
                # H2D silently becomes synchronous off a pageable page. Counted
                # so the pricing run can say whether it was measuring the rung
                # or the pinned pool's ceiling.
                self.unpinned_slots += 1
                pinned = torch.empty(slot.shape, dtype=dtype, device="cpu")
            if from_live:
                pinned.copy_(self._live(slot))
            elif self._backed.get(region.name):
                pinned.copy_(self._view(slot))
            else:
                # COLD mirror: the bytes have never been in this process, so
                # they come off disk straight into the pinned page.
                self._read_into(slot, pinned)
            mirrors.append(pinned)
        torch.cuda.synchronize(self.device)
        self._host[region.name] = mirrors
        self._rebind_off_device(region)

    # -- binding ------------------------------------------------------------

    def _view(self, slot: SlotSpec) -> Any:
        return self._torch.from_dlpack(  # type: ignore[attr-defined]
            self.reservation.tensor(
                slot.offset, list(slot.shape), slot.dtype_code, slot.dtype_bits
            )
        )

    def _live(self, slot: SlotSpec) -> Any:
        module = self._leaves[slot.leaf]
        return getattr(module, "_parameters" if slot.is_param else "_buffers")[slot.attr]

    def _bind(self, slot: SlotSpec, value: Any) -> None:
        """Point a module's parameter or buffer at ``value``.

        ``Parameter.data = ...`` is the normal move and it preserves the
        Parameter's identity, which matters: hooks, LoRA wrappers and anything
        holding the object keep working across every promote and demote.

        A META parameter is the one case where it cannot be used — torch
        REFUSES ``set_data`` across the meta boundary ("incompatible tensor
        type"), measured on the cold-load leg. There is no identity worth
        preserving there either: a meta parameter is a declared shape that has
        never held a byte, so the object is replaced rather than filled.
        """
        module = self._leaves.get(slot.leaf)
        if module is None:
            return
        if slot.is_param:
            current = module._parameters.get(slot.attr)
            if current is None or bool(current.is_meta) or bool(value.is_meta):
                import torch.nn as nn

                module._parameters[slot.attr] = nn.Parameter(value, requires_grad=False)
                return
        bind_tensor(module, slot.attr, value, slot.is_param)

    def _rebind_off_device(self, region: RegionSpec) -> None:
        """Point every slot at its host mirror, or at meta when there is none.

        A meta tensor is the honest stand-in for "this weight exists, is this
        shape, and is nowhere right now": any kernel that reads it fails
        loudly instead of reading an unmapped address.
        """
        torch = self._torch
        host = self._host.get(region.name)
        for index, slot in enumerate(region.slots):
            if host is not None:
                self._bind(slot, host[index])
            else:
                self._bind(
                    slot,
                    torch.empty(slot.shape, dtype=_dtype_of(torch, slot), device="meta"),
                )

    # -- everything outside the managed set ---------------------------------

    def _place_residue(self) -> None:
        """Tensors no region owns must be ON the device. pgw#1497's clause.

        A tensor hanging off a module that HAS children (CLIP's
        ``position_ids``) is never a leaf, so no region covers it. Left on the
        host it takes down the first kernel that reads it, which is exactly how
        sd1.5 died at a 5% budget over there.
        """
        managed = set(self._regions) | set(self.layout.core_names)
        for root_name, root in self._roots:
            for name, module in root.named_modules():
                qualified = f"{root_name}.{name}" if name else root_name
                if qualified in managed:
                    continue
                for attr, is_param, tensor in own_tensors(module):
                    if tensor.is_meta or tensor.device == self.device:
                        continue
                    bind_tensor(module, attr, tensor.to(self.device), is_param)

    # -- teardown -----------------------------------------------------------

    def release(self) -> None:
        """Un-hook, copy every managed weight OUT of the arena, drop residency.

        The tree must be able to outlive the arena: after this every weight is
        an ordinary torch tensor on the device, exactly as it was found.
        """
        torch = self._torch
        with torch.no_grad():
            self.ring.drain()
            for name, hook in list(self._hooks.items()):
                hook.remove()
                self._hooks.pop(name, None)
            for region in self.layout.regions:
                if self._backed.get(region.name):
                    for slot in region.slots:
                        self._bind(slot, self._view(slot).clone())
                else:
                    host = self._host.get(region.name)
                    if host is None:
                        self.reservation.back(region.offset, region.span)
                        self._fill(region)
                        for slot in region.slots:
                            self._bind(slot, self._view(slot).clone())
                        torch.cuda.synchronize(self.device)
                        self.reservation.unback(region.offset, region.span)
                        continue
                    for index, slot in enumerate(region.slots):
                        self._bind(slot, host[index].to(self.device))
            torch.cuda.synchronize(self.device)
            for region in self.layout.regions:
                if self._backed.get(region.name):
                    self.reservation.unback(region.offset, region.span)
                    self._backed[region.name] = False
        self._host.clear()
        self._engine = None
        self._staging_slab = None
        self._slab_pool = None
        for _, root in self._roots:
            try:
                setattr(root, ENGAGED_ATTR, False)
            except Exception:  # noqa: BLE001
                pass
        self.plan = None


def _dtype_of(torch: Any, slot: SlotSpec) -> Any:
    for dtype in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool,
    ):
        if dlpack_dtype(torch, dtype) == (slot.dtype_code, slot.dtype_bits):
            return dtype
    raise TypeError(f"no torch dtype for DLPack ({slot.dtype_code}, {slot.dtype_bits})")


def _recorded(
    planned: ResidencyPlan,
    old: ResidencyPlan,
    demote: Tuple[str, ...],
    costs: Sequence[LeafCost],
) -> ResidencyPlan:
    """The plan an UNLOAD actually produced: ``old`` minus the trim.

    Same clause as the software rung's: the greedy fill is not monotone in the
    budget, so a call asked to free bytes records what it did, not what a fresh
    plan at the new budget would have drawn.
    """
    dropped = set(demote)
    by_name = {c.name: c for c in costs}
    resident = tuple(n for n in old.resident if n not in dropped)
    forced = tuple(n for n in old.forced if n not in dropped)
    streamed = tuple(old.streamed) + tuple(n for n in old.all_resident if n in dropped)
    return ResidencyPlan(
        budget_bytes=planned.budget_bytes,
        streams=planned.streams,
        forced=forced,
        resident=resident,
        streamed=streamed,
        resident_bytes=sum(by_name[n].resident_bytes for n in forced + resident if n in by_name),
        streamed_bytes=sum(by_name[n].resident_bytes for n in streamed if n in by_name),
        window_bytes=planned.window_bytes,
        ram_budget_bytes=planned.ram_budget_bytes,
    )


__all__ = [
    "CORE_REGION",
    "DEFAULT_GRANULARITY",
    "ArenaLayout",
    "ArenaResidency",
    "RegionSpec",
    "SlotSpec",
    "TensorSpec",
    "UnbackRing",
    "dlpack_dtype",
    "plan_layout",
    "safetensors_triples",
]
