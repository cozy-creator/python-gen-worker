"""The grant seam: the endpoint DECLARES, the platform GRANTS, varena EXECUTES.

Paul, ratifying pgw#1598: *"model requests memory from varena. varena always says yes and
then makes it work. graph has only one specialization per lane, and compiles once per lane
and serves it normally."* And the standing direction behind this module: *"these memory
tricks move out of endpoint code into varena and python-gen-worker itself as much as
possible. Endpoint code declares, the platform executes."*

## Two arenas, and only one of them owns bytes

**WEIGHT arena** — real residency. Manifest arithmetic plus phase-boundary paging under
STABLE virtual addresses. varena maps and unmaps physical pages beneath a pointer that never
moves, which is what lets a compiled artifact survive a residency change (pgw#1607).

**REQUEST arena** — an ACCOUNTING-ONLY headroom guarantee. Activations stay in torch's
allocator and are never paged by us. The request arena is a NUMBER the weight arena must
leave unspent, not a place bytes live. varena is never told about it.

## The one admission rule

**COMPILED IFF FULLY RESIDENT, ELSE EAGER-STREAMED.** There is no compiled-below-residency
rung and no author-visible ladder. Two riders, both from measurement:

* Full residency is NECESSARY but not SUFFICIENT for compiled. The compiled regime also has
  to fit in ``driver_free`` ALONE — AOTI's first-call pool allocates outside torch's caching
  allocator, so cached blocks are eager-spendable money and cannot fund a compiled admit
  (pgw#1627).
* Compiled needs a MEASURED demand stamp, **and the stamp must come from a SUCCESSFUL mint
  run, never from a death trace** (pgw#1627's second re-open, on-card 2026-08-21). That rule
  was paid for: a "+1154 MiB, 4/4 runs, batch-invariant" figure circulated as sdxl sm_89's
  compiled first-call demand and was FALSIFIED — with 1326 MiB more freed, the first call
  consumed ~2474 of 2506 available and died identically. **A death only ever reports the free
  memory it consumed**, so a demand read off one measures the card, not the artifact. sdxl
  sm_89's demand is UNKNOWN, lower-bounded >2501 MiB, and 8 GiB is a measured NO for compiled
  SDXL UNet-only.

  Without a stamp this module returns eager, **structurally** — not by accident. A mid-graph
  OOM in a compiled artifact is process death (pgw#1255 leg 2), so "we did not measure it"
  can never be spent as "it probably fits". The falsification above is the argument for that
  rule rather than a footnote to it: the one number anybody had was an artifact of the
  failure that produced it.

## Fully resident when it fits, and why that is the DEFAULT rather than an upgrade

Measured, RTX 4070 Laptop 7.62 GiB (7803 MiB), 2026-08-21 anima head-to-head (varena#3):
our serve pinned peak at **6102 MiB in every leg** — identical at load 5, 14 and 91, because
it is budget-fed — while ComfyUI, fully resident, peaked **6778-7226 MiB on the same card**
across four legs without an OOM, and ran the same 1024x1024/20-step/CFG request **20-37%
faster**. The card had ~7284 MiB free. Full residency FIT, by a demonstrated margin, and our
decider refused it.

**And pgw#1604 measured the same defect independently, on SDXL, and stated its arithmetic.**
Finding 1 of the VRAM-limbo curve, verbatim: *"SDXL is NEVER fully resident on this card, and
cannot be.* ``select_auto_mode``'s *fit test is* ``needed <= avail - 2.0``; *the confessed*
``needed_gb`` *is 6.5, so residency wants **8.5 GiB free on a 7.62 GiB card**. There is no*
``off``/``native`` *row at any budget. The "ceiling" is already a degraded rung."*

Finding 5 of the same curve is why the answer is to DELETE the constant rather than raise it:
the same request peaks at **2603 MiB at a 6.0 GiB cap, 2218 at 2.5, 1962 at 2.0, and 494 once
tiling engages** — the caching allocator hands cached blocks back under pressure without being
asked. *"Any VRAM sizing rule derived from a high-water mark measured on a roomy card
systematically overstates the requirement."* A reserve fitted to a roomy-card high-water mark
is measuring the allocator's generosity, not the request's need, and every GiB it overstates
is a GiB the card had and the request did not get.

It refused it on arithmetic, not on a measurement: ``select_auto_mode`` demands
``weights <= free - _DEFAULT_SAFETY_MARGIN_GB`` and ``_plan_partial_resident`` then demands
``weights <= free - PARTIAL_RESIDENT_RESERVE_GB`` — two independent 2 GiB guesses at one
unknown (the per-request activation peak), with ``_TRANSIENT_RESERVE_BYTES`` a third number
for a fourth quantity inside the search. The constant carries its own confession:
*"TEMPORARY. This is a constant standing in for a measurement."*

So this module does not tune the guess. It **deletes** it:

1. Full residency is the default posture. It is refused only by a MEASUREMENT — a banked
   per-endpoint request peak that does not fit, or the on-card probe.
2. The probe is that measurement, and it already exists. ``partial_resident.probe_plan``
   does the worst onload once, reads free, and parks back; the fully-resident equivalent is
   free, because the placement happens anyway. The arithmetic estimate was never trustworthy
   on its own — on the campaign card it admitted a plan that then OOMed by 5 MiB — which is
   exactly why the probe was built.
3. A budget below card capacity with no co-tenant is self-harm. The deterministic 6102 MiB is
   a neutral fact about a configuration, not a virtue.

## What lives here and what does not

Everything in this module is ARITHMETIC over declared bytes: no torch, no driver, no
pipeline. That is deliberate — the decision is the part that has been wrong, and it should be
falsifiable on a cardless box. The byte movement is varena's; the pipeline-shaped half is
``memory.py``'s; the streamed-set SEARCH stays in
``partial_resident.plan_component_residency``, which is measured, tested and correct about
the thing it does (minimum BYTES moved, never minimum count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RESIDENT",
    "STREAMED",
    "EAGER",
    "COMPILED",
    "ComponentDecl",
    "RequestArena",
    "Spendable",
    "Grant",
    "plan_grant",
]

_GIB = 1 << 30
_MIB = 1 << 20

#: The grant vocabulary. Two words, per varena#10 / pgw#1598 §4 — ``floor`` died with the UVM
#: rung when Paul ruled compiled-below-residency out. A component is either on the card for
#: the process's life, or it is paged in at its phase boundary and out again.
RESIDENT = "resident"
STREAMED = "streamed"

#: The two regimes. This is ``residency_ledger.REGIMES`` and ``partial_resident``'s
#: ``regime=`` argument, named once so the three cannot drift.
EAGER = "eager"
COMPILED = "compiled"

#: What the on-card probe demands still be free once placement is done. The ONLY reserve
#: constant this module spends, and the only one with provenance that survives contact: the
#: planner's arithmetic admitted a plan that then OOMed by 5 MiB, and no constant fitted to
#: one card fixes that, because allocator fragmentation and the co-tenants' share are in none
#: of the numbers a planner can read. Lifted from ``partial_resident._PROBE_FLOOR_BYTES``
#: rather than re-derived; it means the same thing in both places.
PROBE_FLOOR_BYTES = 256 * _MIB


@dataclass(frozen=True)
class ComponentDecl:
    """One thing the endpoint author declares. The whole author-visible vocabulary.

    ``phase`` orders the paging schedule: components in the same phase are co-resident while
    that phase runs, and the schedule is STATIC, which is why varena needs no tracing
    machinery to prefetch (varena#10, ZeRO-Infinity steal list). ``pinned`` is the author
    saying *this one may never be paged* — dtype-fragile VAEs, content-shared encoders, and
    components diffusers drives by method rather than ``forward`` (a parked VAE never onloads
    because the call is ``self.vae.decode(...)``; pgw#1619 ruled that a REFUSAL, not a second
    hook, and this flag is where that refusal is now stated up front).

    There is no rung, no ``low_vram_mode``, no ``enable_*_offload`` here, and that is the
    second half of the acceptance bar: the programming model gets SIMPLER for the author.
    """

    name: str
    weight_bytes: int
    phase: int = 0
    pinned: bool = False


@dataclass(frozen=True)
class RequestArena:
    """The accounting-only headroom guarantee, and where its number came from.

    ``basis`` is load-bearing and is not decoration:

    * ``measured``  — a banked per-request peak for THIS endpoint. The only basis that may
      fund a compiled admit.
    * ``declared``  — ``Resources.peak_vram_per_request_gb``. Zero of the 26 shipped
      endpoints set it, which is the honest reason the default path has never had a real
      number to use.
    * ``probe``     — no prior number; the on-card probe decides, and until it runs the
      arena asks only for :data:`PROBE_FLOOR_BYTES`. This is the cold-start case and it
      means *try full residency*, not *reserve 2 GiB against a shape nobody measured*.

    ``compiled_extra_bytes`` is pgw#1601's mint-time demand stamp: what the compiled regime
    additionally spends OUTSIDE torch's allocator on its first call. ``None`` means no stamp
    exists, and no stamp means no compiled admit.

    **The stamp may only come from a SUCCESSFUL mint-child run** (pgw#1627's stamp-source
    rule, on-card 2026-08-21). A death trace reports the free memory the call consumed before
    dying, which is a fact about the card, not about the artifact — the figure that
    circulated as sdxl sm_89's demand was exactly that, and it was falsified by giving the
    same call more room and watching it consume that too. No consumer of this field can tell
    a good stamp from a bad one, so the producer carries the rule.
    """

    bytes: int
    basis: str
    compiled_extra_bytes: Optional[int] = None

    @classmethod
    def cold(cls) -> "RequestArena":
        """No prior measurement: ask for the probe floor and let the card answer."""
        return cls(bytes=PROBE_FLOOR_BYTES, basis="probe")

    def demand(self, regime: str) -> int:
        """Bytes the weight arena must leave unspent under ``regime``."""
        if regime == COMPILED:
            extra = self.compiled_extra_bytes or 0
            return int(self.bytes) + int(extra)
        return int(self.bytes)

    @property
    def funds_compiled(self) -> bool:
        """A compiled admit needs a measurement on both halves, or it is not an admit."""
        return self.basis in ("measured", "declared") and self.compiled_extra_bytes is not None


@dataclass(frozen=True)
class Spendable:
    """What the card will actually give, split the way pgw#1627 proved it must be.

    ``driver_free_bytes`` is what a NEW allocation would get. ``allocator_cache_bytes`` is
    reserved-but-unallocated inside torch's caching allocator: real money for eager
    activations, and no money at all for AOTI, which allocates its first-call pool outside
    the allocator entirely. Counting the cache on the compiled path killed every 8 GiB
    compiled-SDXL leg, and the confession string that should have caught it was a constant.
    """

    driver_free_bytes: int
    allocator_cache_bytes: int = 0

    def for_regime(self, regime: str) -> Tuple[int, str]:
        """(spendable bytes, the basis name that goes on the confession line)."""
        if regime == COMPILED:
            return int(self.driver_free_bytes), "driver_free"
        return int(self.driver_free_bytes) + int(self.allocator_cache_bytes), "free+cache"


@dataclass(frozen=True)
class Grant:
    """varena always says yes. So this has no ``fits`` and no ``refusal``.

    What it has instead is an honest split: which components are RESIDENT, which are
    STREAMED, and what the arena was measured against when that was decided. A caller that
    wants to know whether it got what it asked for reads :attr:`fully_resident`; a caller
    that wants to know whether it may compile reads :attr:`regime`. Nothing downstream ever
    branches on "is there enough memory" — that question is answered here, once.
    """

    residency: Mapping[str, str]
    regime: str
    weight_budget_bytes: int
    resident_bytes: int
    streamed_bytes: int
    request_bytes: int
    request_basis: str
    spendable_bytes: int
    headroom_basis: str
    #: Set when the grant is honoured only because the reactive net exists — the declared
    #: demand exceeds what the card can spend even with everything unpinned streamed. varena
    #: still says yes; this names the fact so the confession can carry it.
    over_card: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def fully_resident(self) -> bool:
        return not self.streamed_bytes

    @property
    def streamed(self) -> Tuple[str, ...]:
        return tuple(n for n, v in self.residency.items() if v == STREAMED)

    @property
    def resident(self) -> Tuple[str, ...]:
        return tuple(n for n, v in self.residency.items() if v == RESIDENT)

    def line(self) -> str:
        """One line, and it names every number the decision was made on.

        A confession that cannot go red is not a confession — ``headroom_basis`` shipped as a
        constant ``"free+cache"`` and could not report the bug it was added to expose
        (pgw#1627). Every field printed here is an input, not a restatement of the verdict.
        """
        head = (
            f"grant regime={self.regime} "
            f"{'FULLY RESIDENT' if self.fully_resident else f'streamed={len(self.streamed)}'} "
            f"weights={self.resident_bytes / _GIB:.2f}+{self.streamed_bytes / _GIB:.2f} GiB "
            f"request={self.request_bytes / _GIB:.2f} GiB ({self.request_basis}) "
            f"spendable={self.spendable_bytes / _GIB:.2f} GiB ({self.headroom_basis})"
        )
        if self.streamed:
            head += f" paged={','.join(self.streamed)}"
        if self.over_card:
            head += " OVER-CARD (reactive net is the backstop)"
        for n in self.notes:
            head += f" | {n}"
        return head


def _totals(components: Sequence[ComponentDecl]) -> int:
    return sum(int(c.weight_bytes) for c in components)


def _all_resident(components: Sequence[ComponentDecl]) -> Dict[str, str]:
    return {c.name: RESIDENT for c in components}


def plan_grant(
    components: Sequence[ComponentDecl],
    *,
    spendable: Spendable,
    request: RequestArena,
    compile_intent: bool = False,
    stream_selector: Optional[Callable[..., Sequence[str]]] = None,
) -> Grant:
    """The ONE admission decision. Pure arithmetic over declared bytes.

    ``stream_selector`` is how the streamed set gets chosen when full residency does not
    fit. It is injected rather than implemented here because the search that already exists
    (``partial_resident.plan_component_residency``) is measured, tested, and correct about
    the thing it does — minimum BYTES moved, never minimum count, with the denoiser never a
    candidate. This module owns the DECISION; that function owns the SEARCH. It is called as
    ``stream_selector(components, budget_bytes=...)`` and returns the names to stream; a
    ``None`` selector, or one that returns nothing, still yields a grant — varena always says
    yes — with :attr:`Grant.over_card` set so the caller can say so out loud.

    Order of questions, and it is the admission rule read top to bottom:

    1. Can everything be resident with the COMPILED demand met out of ``driver_free`` alone,
       and is there a measurement to fund it? Then compiled.
    2. Can everything be resident under the eager basis? Then eager, fully resident. This is
       the branch the old decider could not reach and the one the anima row says we want.
    3. Otherwise eager-streamed: page the cheapest set out at phase boundaries.
    """
    declared = _totals(components)

    if compile_intent and request.funds_compiled:
        budget, basis = spendable.for_regime(COMPILED)
        need = request.demand(COMPILED)
        if declared + need <= budget:
            return Grant(
                residency=_all_resident(components),
                regime=COMPILED,
                weight_budget_bytes=declared,
                resident_bytes=declared,
                streamed_bytes=0,
                request_bytes=need,
                request_basis=request.basis,
                spendable_bytes=budget,
                headroom_basis=basis,
            )

    notes: Tuple[str, ...] = ()
    if compile_intent and not request.funds_compiled:
        # Named, not silent. This is the difference between "compiled did not fit" and
        # "compiled was never asked" — pgw#1627's dead-code lesson is that a branch nobody
        # reaches reports as a branch that passed.
        missing = "no measured request peak" if request.basis == "probe" else "no mint demand stamp"
        notes += (f"compiled not admitted: {missing}",)

    budget, basis = spendable.for_regime(EAGER)
    need = request.demand(EAGER)

    if declared + need <= budget:
        return Grant(
            residency=_all_resident(components),
            regime=EAGER,
            weight_budget_bytes=declared,
            resident_bytes=declared,
            streamed_bytes=0,
            request_bytes=need,
            request_basis=request.basis,
            spendable_bytes=budget,
            headroom_basis=basis,
            notes=notes,
        )

    weight_budget = max(0, budget - need)
    streamed: Tuple[str, ...] = ()
    if stream_selector is not None:
        streamed = tuple(stream_selector(components, budget_bytes=weight_budget) or ())
    pinned = {c.name for c in components if c.pinned}
    # A selector that names a pinned component is a bug in the selector, not a licence.
    # Dropping it silently would page a dtype-fragile VAE and produce black images.
    refused = tuple(n for n in streamed if n in pinned)
    if refused:
        streamed = tuple(n for n in streamed if n not in pinned)
        notes += (f"selector named pinned components, refused: {','.join(refused)}",)

    residency = {c.name: (STREAMED if c.name in streamed else RESIDENT) for c in components}
    streamed_bytes = sum(int(c.weight_bytes) for c in components if c.name in streamed)
    resident_bytes = declared - streamed_bytes

    return Grant(
        residency=residency,
        regime=EAGER,
        weight_budget_bytes=weight_budget,
        resident_bytes=resident_bytes,
        streamed_bytes=streamed_bytes,
        request_bytes=need,
        request_basis=request.basis,
        spendable_bytes=budget,
        headroom_basis=basis,
        over_card=resident_bytes + need > budget,
        notes=notes,
    )
