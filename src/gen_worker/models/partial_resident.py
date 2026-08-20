"""Component-granular residency — offload the MINIMUM bytes, keep the rest.

``model_offload`` is all-or-nothing: every component leaves the card after every
request and comes back over PCIe before the next one. Measured on an RTX 4070
with SDXL (pgw#1577) that is **13 GiB of traffic per request** — 6.5 GiB in,
6.5 GiB out, ~4.8 s at the card's ~2.7 GB/s effective rate — to reclaim the
**1.2 GiB** the pipeline was actually over budget by. ComfyUI's
``free_memory(memory_required)`` frees only what it must; this rung is that
policy at component granularity.

The plan is computed ONCE, before any weight lands, from free VRAM and measured
component sizes. The resident set never changes at runtime, so there is no
eviction loop for pgw#1560's non-raising allocator thrash to live in, and no
except-OOM retry: an OOM inside a compiled graph is process death, so admission
is the only honest place to decide.

The mechanism is a PINNED HOST MIRROR per evicted component, not accelerate's
``CpuOffload``. Weights are read-only during inference, so the host copy never
goes stale and eviction is a pointer swap rather than a device-to-host copy —
which deletes half of ``model_offload``'s bill outright, and leaves the other
half as a pinned H2D instead of a pageable one. Three rules hold the arithmetic
the plan admitted:

* an evicted component's forward-pre-hook onloads it and parks every other one,
  so at most one is ever on the card;
* the first RESIDENT module after the chain parks them all, so the encoders are
  gone BEFORE the denoise loop rather than at the end of the call;
* ``maybe_free_model_hooks`` is replaced, because the stock one re-runs
  ``enable_model_cpu_offload``, whose first statement is ``self.to("cpu")`` —
  that would bounce the resident denoiser off the card and back on every single
  call, which is strictly worse than the rung it replaces.

It also honors ``unhookable_components`` for real. ``_exclude_from_cpu_offload``
does not: diffusers consults it only for components ABSENT from
``model_cpu_offload_seq``, and SDXL's ``vae`` — the ``force_upcast`` one
gw#441/gw#469 requires to stay resident — is in that sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

_GIB = 1 << 30

#: Set on a pipeline this rung armed: the :class:`ResidencyPlan` it armed, so a
#: later reader can state the resident set without re-deriving it.
COMPONENT_RESIDENCY_ATTR = "_cozy_component_residency"

#: Set on a pipeline this rung armed: ``{name: ParkedComponent}``, so a later
#: reader can state where each evicted component actually is.
PARKED_COMPONENTS_ATTR = "_cozy_parked_components"

#: Set on each MODULE this rung parks (pgw#1587). The mark rides the module
#: rather than the pipeline because the reader that needs it — ``ctx.compile``
#: — is handed a compile TARGET and nothing else, and a mark it has to find a
#: pipeline to interpret is a mark it will one day fail to find. A module
#: without it keeps a stable device pointer for the life of the load, which is
#: exactly the precondition a compiled graph's by-reference constants need.
PARKED_MODULE_ATTR = "_cozy_parked_module"

#: The device this rung's resident set actually executes on. ``DiffusionPipeline.device``
#: answers with the FIRST component's device, and an evicted encoder parked on
#: the host makes that ``cpu`` for a pipeline whose denoiser is on the card —
#: the same public-answer break pgw#1497 measured for ``partial_stream``, where
#: the endpoint then builds ``input_ids`` on the host and the first embedding
#: dies with *"index is on cpu, different from other tensors on cuda:0"*.
#: ``memory.install_execution_device_fallback`` reads this.
PARTIAL_RESIDENT_DEVICE_ATTR = "_cozy_partial_resident_device"

#: The typed phase an arming FAILURE confesses under, mirroring
#: ``PARTIAL_STREAM_UNARMED_PHASE``: a rung that could not arm and fell through
#: to a coarser one is a placement the operator asked for and did not get.
PARTIAL_RESIDENT_UNARMED_PHASE = "partial_resident_unarmed"

#: Headroom held back for the transient onload of an evicted component.
#:
#: MEASURED TWICE ON THE CARD, and both numbers are in the record because the
#: first one was mine and it was wrong:
#:
#: * At **512 MiB** the check refused the whole rung once free VRAM fell from
#:   7.3 to 7.1 GiB — a co-tenant taking 200 MiB reverting SDXL to
#:   `model_offload`. A performance cliff.
#: * At **256 MiB** the rung admitted the single-encoder plan (6.95 GiB peak of
#:   7.3 free, 96%) and the onload **OOMed in `ParkedComponent.onload`**:
#:   *"Tried to allocate 20.00 MiB … 14.94 MiB is free … this process has
#:   6.96 GiB in use."* A failed request.
#:
#: The 256 MiB reasoning — "the encode phase has no activations to reserve for"
#: — is wrong about the mechanism. The caching allocator does not return the
#: DENOISE phase's blocks between requests, so the next request's encode does
#: not start on a clean card: it starts on a pool that still holds them, next to
#: whatever co-tenants took. The reserve must cover both, and it is therefore an
#: activation-sized number after all.
#:
#: The two failures are not symmetric. 512 MiB's cliff falls back to today's
#: rung — slow, correct, loud. 256 MiB's failure is an OOM on a paid request.
#: That asymmetry picks the number, not the arithmetic.
#:
#: OWED: the cliff is real and should be closed by a ceiling that scales with
#: the card rather than a constant subtracted from it. Filed with the rung.
_TRANSIENT_RESERVE_BYTES = 512 * (1 << 20)

#: The activation headroom this rung reserves, and it is NOT
#: ``memory._DEFAULT_SAFETY_MARGIN_GB``'s twin by accident any more — see below.
#:
#: ⚠️ **RAISED 1.25 -> 2.00 GiB, 2026-08-20 (pgw#1595, coordinator ruling). THE 1.25 WAS
#: DERIVED FROM THE WRONG RESIDENCY CONFIGURATION AND WAS UNDER-SIZED FROM THE DAY IT
#: SHIPPED.** It came from pgw#1570's ``model_offload`` run — 5956 MiB peak over 5437 MiB of
#: weights, so **519 MiB** of activations. That number does not transfer to THIS rung, and
#: pgw#1586's own GREEN arm proves it: 7540 MiB peak over 5693 MiB of resident weights is
#: **1847 MiB = 1.80 GiB**, 3.6x the figure the constant was built on.
#:
#: The two configurations differ in the allocator, not in the maths. Under ``model_offload``
#: the denoiser is freed and re-allocated every request, so the pool churns and blocks are
#: reused; under ``partial_resident`` the weights are PINNED in the pool for the process's
#: life and activations must come out of what is left, so the high-water mark carries
#: fragmentation the offloaded rung never pays.
#:
#: 2.00 = the measured 1.80 plus margin. It is also what the safer plan needs: at 2.00 the
#: search takes BOTH encoders (peak 6.70 GiB, ~1.1 GiB of headroom) instead of the single
#: encoder the 1.25 admitted (peak 6.95, **268 MB** of headroom — the configuration that
#: survived pgw#1586's leg by 13.5 MB and thrashed pgw#1595's by 6.6 MB).
#:
#: **CONSEQUENCE, STATED RATHER THAN DISCOVERED LATER:** the rung needs
#: ``free - reserve >= 5.31 GiB`` (the forced unet+vae set), so at 2.00 it is only available
#: above **~7.31 GiB free**. On this 7.62 GiB card that means a nearly-empty card and the rung
#: will often REFUSE to ``model_offload`` — which is the honest answer, not a regression. On
#: fleet cards (12/16/24 GiB) SDXL never demotes this far and none of this applies.
#:
#: **TEMPORARY.** This is a constant standing in for a measurement, sized to the worst shape
#: we have actually run. It is replaced by the per-endpoint measured peak (pgw#1586 item (c),
#: ``reserve_source=measured``) as soon as that lands.
PARTIAL_RESIDENT_RESERVE_GB = 2.00

#: What the ON-CARD PROBE demands still be free with the largest evicted
#: component onloaded. The arithmetic above is an ESTIMATE — measured on the
#: campaign card it admitted a plan that then OOMed by 5 MiB — and no constant
#: fitted to one card fixes that, because the allocator's fragmentation and the
#: co-tenants' share are not in any of the numbers the planner can read. So the
#: plan is checked by DOING it once, at load, before a request can hit it.
_PROBE_FLOOR_BYTES = 256 * (1 << 20)

#: How many progressively more expensive plans the probe may reject before the
#: rung gives up and lets the load fall to `model_offload`.
_MAX_PROBE_ATTEMPTS = 4


@dataclass(frozen=True)
class ResidencyPlan:
    """Which components stay on the card, which commute, and the arithmetic.

    ``fits`` is the only verdict. ``refusal`` names why not, and is "" exactly
    when ``fits`` — a plan that does not fit must say what beat it, because the
    caller's next move (fall to ``model_offload``) erases the evidence.
    """

    resident: Tuple[str, ...]
    offloaded: Tuple[str, ...]
    resident_bytes: int
    offloaded_bytes: int
    budget_bytes: int
    transient_peak_bytes: int
    free_bytes: int
    fits: bool
    refusal: str = ""

    def summary(self) -> str:
        return (
            f"resident={','.join(self.resident) or '-'} "
            f"({self.resident_bytes / _GIB:.2f} GiB) "
            f"offloaded={','.join(self.offloaded) or '-'} "
            f"({self.offloaded_bytes / _GIB:.2f} GiB/request, H2D only) "
            f"budget={self.budget_bytes / _GIB:.2f} GiB "
            f"transient_peak={self.transient_peak_bytes / _GIB:.2f} GiB "
            f"free={self.free_bytes / _GIB:.2f} GiB"
        )


def plan_component_residency(
    *,
    sizes: Mapping[str, int],
    order: Sequence[str],
    denoiser: str,
    forced_resident: Collection[str] = (),
    budget_bytes: int,
    free_bytes: int,
    transient_reserve_bytes: int = _TRANSIENT_RESERVE_BYTES,
    min_moved_bytes: int = 0,
) -> ResidencyPlan:
    """The minimum-BYTES subset of components to evict, or a refusal.

    ``min_moved_bytes`` asks for the next plan STRICTLY more expensive than a
    given one. It exists because the arithmetic here is an estimate and the
    allocator is not: :func:`apply_component_residency` probes the plan it was
    given on the real card, and when the probe says no the caller comes back for
    the next-cheapest one rather than giving up on the rung.

    Minimum bytes, not minimum count: every evicted byte is paid twice per
    request (out and back), so the cheapest plan that fits is the one that moves
    the least — which is usually neither the largest component nor the smallest
    number of them. ``order`` is the pipeline's own execution order and is what
    makes the transient peak a single component rather than a sum: an offloaded
    component is released by the next one's pre-forward, so at most one of them
    is on the card at a time.

    The denoiser is never a candidate. It is the component the rung exists to
    keep — evicting it reproduces ``model_offload`` at a higher complexity.
    """
    known = [n for n in order if n in sizes]
    known += sorted(n for n in sizes if n not in known)
    total = sum(int(sizes[n]) for n in known)
    forced = {str(n) for n in forced_resident}
    forced.add(str(denoiser))

    if denoiser not in sizes:
        return ResidencyPlan(
            (), (), total, 0, budget_bytes, total, free_bytes, False,
            f"no denoiser named {denoiser!r} among {known}",
        )

    pinned_bytes = sum(int(sizes[n]) for n in known if n in forced)
    if pinned_bytes > budget_bytes:
        return ResidencyPlan(
            (), (), total, 0, budget_bytes, total, free_bytes, False,
            f"the forced-resident set alone is {pinned_bytes / _GIB:.2f} GiB "
            f"against a {budget_bytes / _GIB:.2f} GiB budget",
        )

    candidates = [n for n in known if n not in forced and int(sizes[n]) > 0]
    best: Optional[Tuple[int, int, Tuple[str, ...]]] = None
    # Few components (<= 8 in every shipped family), so the EXACT answer is
    # cheap. It is also the only correct one: fewest components and fewest bytes
    # are different plans — evicting one 1.5 GiB encoder to free 1.0 GiB moves
    # more per request than evicting two 0.6 GiB ones — and bytes is what PCIe
    # charges for. Every subset is enumerated for that reason.
    for k in range(0, len(candidates) + 1):
        for subset in combinations(candidates, k):
            moved = sum(int(sizes[n]) for n in subset)
            resident_bytes = total - moved
            if resident_bytes > budget_bytes:
                continue
            # One at a time: each offloaded component is released by the next
            # one's pre-forward (or by the release hook on the first resident
            # module after the chain), so the transient ceiling sees the largest
            # of them, never their sum. An empty subset has no transient at all.
            peak = resident_bytes + max(
                (int(sizes[n]) for n in subset), default=0
            )
            ceiling = max(0, free_bytes - transient_reserve_bytes)
            if subset and peak > ceiling:
                continue
            if moved <= min_moved_bytes:
                continue
            key = (moved, len(subset), subset)
            if best is None or key < best:
                best = key

    if best is None:
        return ResidencyPlan(
            (), (), total, 0, budget_bytes, total, free_bytes, False,
            f"no subset of {candidates or ['(nothing evictable)']} brings "
            f"{total / _GIB:.2f} GiB under the {budget_bytes / _GIB:.2f} GiB "
            f"budget without breaking the transient ceiling",
        )

    moved, _, subset = best
    offloaded = tuple(n for n in known if n in set(subset))
    resident = tuple(n for n in known if n not in set(subset))
    resident_bytes = total - moved
    peak = resident_bytes + max((int(sizes[n]) for n in offloaded), default=0)
    return ResidencyPlan(
        resident, offloaded, resident_bytes, moved, budget_bytes, peak,
        free_bytes, True, "",
    )


def _component_sizes(
    pipeline: Any, names: Sequence[str], sizer: Any
) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for name in names:
        comp = getattr(pipeline, name, None)
        if comp is None or not hasattr(comp, "parameters"):
            continue
        try:
            n = int(sizer(comp))
        except Exception:
            continue
        if n > 0:
            sizes[name] = n
    return sizes


def _pipeline_component_names(pipeline: Any) -> List[str]:
    names: List[str] = []
    seq = str(getattr(pipeline, "model_cpu_offload_seq", "") or "")
    for part in seq.split("->"):
        part = part.strip()
        if part and part not in names:
            names.append(part)
    try:
        components = dict(getattr(pipeline, "components", {}) or {})
    except Exception:
        components = {}
    for name in components:
        if name not in names:
            names.append(str(name))
    return names


def _denoiser_name(pipeline: Any, names: Sequence[str]) -> str:
    """The component whose residency is the point. Named by attribute, in the
    order diffusers itself uses across families."""
    for candidate in ("unet", "transformer", "prior", "decoder"):
        if candidate in names and getattr(pipeline, candidate, None) is not None:
            return candidate
    return ""


def plan_for_pipeline(
    pipeline: Any,
    *,
    budget_bytes: int,
    free_bytes: int,
    sizer: Any,
    forced_resident: Collection[str] = (),
    transient_reserve_bytes: int = _TRANSIENT_RESERVE_BYTES,
    min_moved_bytes: int = 0,
) -> ResidencyPlan:
    """:func:`plan_component_residency` against a live pipeline's own tree."""
    names = _pipeline_component_names(pipeline)
    sizes = _component_sizes(pipeline, names, sizer)
    denoiser = _denoiser_name(pipeline, list(sizes))
    return plan_component_residency(
        sizes=sizes,
        order=names,
        denoiser=denoiser,
        forced_resident=forced_resident,
        budget_bytes=budget_bytes,
        free_bytes=free_bytes,
        transient_reserve_bytes=transient_reserve_bytes,
        min_moved_bytes=min_moved_bytes,
    )


class ParkedComponent:
    """One offloaded component, mirrored in PINNED host RAM.

    **Weights are read-only during inference**, so the host mirror never goes
    stale — and that single fact is what lets ``park`` be a pointer swap rather
    than a device-to-host copy. Half of ``model_offload``'s PCIe bill is that
    copy, paid on every component on every request for data the host already
    had. The half that remains is a PINNED H2D, which the card takes at roughly
    4x the pageable rate accelerate's ``CpuOffload`` gets.

    The invariant has a cost: anything that MUTATES weights in place after
    arming (a fused LoRA, a dtype cast) invalidates the mirror. Arming happens
    once at load, before serving, like every other rung, and re-arming is how a
    mutation is absorbed.
    """

    __slots__ = ("name", "module", "_slots", "_torch", "_device", "on_device", "bytes")

    def __init__(self, name: str, module: Any, slots: List[Any], torch_mod: Any,
                 device: Any, nbytes: int) -> None:
        self.name = name
        self.module = module
        self._slots = slots
        self._torch = torch_mod
        self._device = device
        self.on_device = False
        self.bytes = nbytes

    @classmethod
    def mirror(
        cls, name: str, module: Any, *, device: Any, torch_mod: Any,
        log: logging.Logger,
    ) -> Optional["ParkedComponent"]:
        """Snapshot ``module``'s weights into pinned host RAM and park it.

        Returns None — having changed nothing — when any tensor cannot be
        represented (meta, or an alias into a larger storage). A half-moved
        component is the one outcome worse than not arming.
        """
        from .staging import alloc_pinned_like
        from .stream_residency import own_tensors, tensor_bytes

        pending: List[Any] = []
        total = 0
        for sub in module.modules():
            for attr, is_param, tensor in own_tensors(sub):
                if tensor.is_meta or tensor.storage_offset() != 0:
                    log.warning(
                        "partial_resident: %s.%s is meta or a partial-view "
                        "alias; the component cannot be mirrored", name, attr,
                    )
                    return None
                host = alloc_pinned_like(torch_mod, tensor)
                if host is None:
                    host = torch_mod.empty_like(tensor, device="cpu")
                host.copy_(tensor)
                total += tensor_bytes(host)
                pending.append((sub, attr, is_param, host))
        if not pending:
            return None
        parked = cls(name, module, pending, torch_mod, device, total)
        parked._bind_host()
        return parked

    def _bind_host(self) -> None:
        from .stream_residency import bind_tensor

        with self._torch.no_grad():
            for sub, attr, is_param, host in self._slots:
                bind_tensor(sub, attr, host, is_param)
        self.on_device = False

    def park(self) -> None:
        """Drop the device copy. No copy back — the mirror is authoritative."""
        if not self.on_device:
            return
        self._bind_host()

    def onload(self) -> None:
        """Pinned, non-blocking H2D."""
        if self.on_device:
            return
        from .stream_residency import bind_tensor

        with self._torch.no_grad():
            for sub, attr, is_param, host in self._slots:
                try:
                    pinned = bool(host.is_pinned())
                except Exception:  # noqa: BLE001
                    pinned = False
                bind_tensor(
                    sub, attr, host.to(self._device, non_blocking=pinned), is_param
                )
        self.on_device = True


def _install_residency_hooks(
    pipeline: Any, parked: Dict[str, ParkedComponent], order: Sequence[str],
    log: logging.Logger,
) -> None:
    """One rule, applied everywhere: **at most one parked component is on the
    card at a time.** Each parked component's pre-forward onloads itself and
    parks the others; the first RESIDENT module after the chain parks them all,
    which is what puts the encoders off the card BEFORE the denoise loop rather
    than at the end of the call. Both are the arithmetic
    :func:`plan_component_residency` admitted, enforced rather than assumed."""

    def _park_all_but(keep: str = "") -> None:
        for name, comp in parked.items():
            if name != keep:
                comp.park()

    for name, comp in parked.items():
        def _pre(_m: Any, _a: Any, _n: str = name, _c: ParkedComponent = comp) -> None:
            _park_all_but(_n)
            _c.onload()

        comp.module.register_forward_pre_hook(_pre)
        # pgw#1587: the mark goes on HERE and not at `mirror()`, because THIS
        # is the moment the component starts moving per forward. A mirror the
        # probe then refuses is rolled up by the caller's fall to the next
        # rung, and marking it would have told `ctx.compile` that a component
        # nothing is hooking relocates — a refusal on a load where no rung
        # armed at all. (Found by the pgw#1587 red arm, which is the shape a
        # gate keyed on the wrong moment always has.)
        try:
            setattr(comp.module, PARKED_MODULE_ATTR, True)
        except Exception:  # noqa: BLE001 — a module that refuses an attribute
            log.debug("partial_resident: %s could not carry the park mark", name)

    seen_parked = False
    for name in order:
        if name in parked:
            seen_parked = True
            continue
        if not seen_parked:
            continue
        module = getattr(pipeline, name, None)
        if module is None or not hasattr(module, "register_forward_pre_hook"):
            continue

        def _release(_m: Any, _a: Any) -> None:
            _park_all_but()

        module.register_forward_pre_hook(_release)
        break

    def maybe_free_model_hooks(_self: Any = pipeline) -> None:
        """THE TRAP THIS OVERRIDE EXISTS FOR (pgw#1577):
        ``DiffusionPipeline.maybe_free_model_hooks`` ends by calling
        ``enable_model_cpu_offload``, whose FIRST statement is
        ``self.to("cpu")``. Left in place it drags the resident denoiser to the
        host and back on every call — the exact cost this rung deletes,
        reintroduced at the end of the request instead of the start."""
        for component in dict(getattr(_self, "components", {}) or {}).values():
            reset = getattr(component, "_reset_stateful_cache", None)
            if callable(reset):
                reset()
        _park_all_but()

    pipeline.maybe_free_model_hooks = maybe_free_model_hooks
    # The stock method returns early on an empty `_all_hooks`, so anything that
    # still reaches the base implementation is a no-op rather than a re-arm.
    pipeline._all_hooks = []
    log.debug("partial_resident: hooks installed for %s", ",".join(parked))


def _placement_attribution(torch_mod: Any) -> Dict[str, int]:
    """Split what the PROCESS holds on the card into its four real parts.

    pgw#1586: the budget subtracts a reserve from DRIVER-level free
    (``mem_get_info``), but the resident SET is sized in WEIGHT BYTES — so
    everything the process holds that is not a weight is spent out of the
    reserve before the first activation, invisibly. Measured on the campaign
    card: arithmetic said ~2.1 GiB should remain for activations and the probe
    read 0.45. Attribution, not inference, is the only way to say where the
    difference went, so the four parts are read from the allocator itself:

    * ``alloc`` — live tensors. The weights, and what the plan actually sized.
    * ``cache`` — ``reserved - allocated``: blocks the allocator holds and WILL
      reuse for activations, but which driver-free counts as gone. A parked
      component's device blocks land here: ``park()`` drops the reference, the
      caching allocator keeps the block, and ``mem_get_info`` never sees it come
      back without ``empty_cache()``.
    * ``ctx`` — ``(total - driver_free) - reserved``: the CUDA context, cuDNN and
      cuBLAS workspaces, and anything else outside the caching allocator.
    * ``driver_free`` — what a fresh allocation can take from the driver.
    """
    out: Dict[str, int] = {}
    try:
        alloc = int(torch_mod.cuda.memory_allocated())
        reserved = int(torch_mod.cuda.memory_reserved())
        driver_free, total = torch_mod.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - attribution must never fail a placement
        return out
    out["attr_alloc_bytes"] = alloc
    out["attr_cache_bytes"] = max(0, reserved - alloc)
    out["attr_ctx_bytes"] = max(0, (int(total) - int(driver_free)) - reserved)
    out["attr_driver_free_bytes"] = int(driver_free)
    return out


def probe_plan(
    parked: Dict[str, ParkedComponent], *, free_bytes_now: Any,
    floor_bytes: int = _PROBE_FLOOR_BYTES,
) -> tuple[bool, int]:
    """DO the worst onload once and read what is left. Returns (ok, free_bytes).

    The planner's transient ceiling is arithmetic over numbers it can read, and
    on the campaign card that arithmetic admitted a plan whose onload then died
    5 MiB short — because allocator fragmentation and a co-tenant's share are in
    neither the component sizes nor the free-VRAM probe. This is the same
    question asked of the card instead of of a constant, and it costs one copy
    at load.
    """
    if not parked:
        return True, int(free_bytes_now())
    worst = max(parked.values(), key=lambda c: c.bytes)
    worst.onload()
    free = int(free_bytes_now())
    cache = 0
    try:
        import torch

        cache = int(_placement_attribution(torch).get("attr_cache_bytes", 0))
    except Exception:  # noqa: BLE001 - a probe that cannot attribute still probes
        cache = 0
    worst.park()
    # pgw#1586. THE FLOOR IS CHECKED AGAINST WHAT ACTIVATIONS CAN ACTUALLY HAVE,
    # WHICH IS NOT DRIVER-FREE. A parked component's blocks stay in the caching
    # allocator's pool: `park()` drops the reference, the allocator keeps the
    # block, and `mem_get_info` never sees it return. So the same bytes were
    # counted as freed by the plan and as used by this probe. Measured on the
    # card: driver_free 0.45 + reusable cache 1.56 = 2.01 GiB against a 2.00 GiB
    # reserve — the reserve was honoured all along and the probe was reading a
    # number 1.56 GiB too small.
    #
    # ⚠️ CACHE IS *SOFT* AVAILABILITY, NOT A HARD BUDGET TERM. 1.56 GiB of
    # cached blocks is not 1.56 GiB of CONTIGUOUS space, and the 214 allocator
    # retries this leg logged ARE that discount showing up at runtime — the
    # allocator freeing and remapping segments that do not fit the requested
    # activation. Counting it here is right because the bug being fixed is a
    # SPURIOUS REFUSAL and the failure mode past the probe is retries-and-serve,
    # not a fatal (degrade-never-OOM holds). Do NOT promote this term into a
    # hard budget: the measured-placement work should treat it as soft, with the
    # retry count as the natural signal for how soft.
    return (free + cache) >= floor_bytes, free


def parks_module(target: Any) -> bool:
    """Does this rung MOVE ``target``'s weights between host and device per
    forward? (pgw#1587)

    The question ``ctx.compile`` asks before arming a compiled graph on a
    target. A compiled graph binds its constants to device pointers and cannot
    page mid-graph, so a target this rung parks must serve eager — while every
    other component, denoiser included, keeps a stable pointer for the life of
    the load and compiles exactly as it would with no rung engaged. That is
    Paul's SDXL case: the text encoders come off the card, the compiled UNet
    stays on it, and the two do not conflict.

    Accepts a module OR a pipeline: for a pipeline the answer is "does it park
    ANY of my components", which is the honest answer for ``ctx.compile(pipe)``
    — that form marks the whole component tree.
    """
    if target is None:
        return False
    if bool(getattr(target, PARKED_MODULE_ATTR, False)):
        return True
    components = getattr(target, "components", None)
    if isinstance(components, Mapping):
        return any(
            bool(getattr(component, PARKED_MODULE_ATTR, False))
            for component in components.values()
        )
    return False


def apply_component_residency(
    pipeline: Any,
    plan: ResidencyPlan,
    *,
    device: Any,
    log: logging.Logger,
    free_bytes_now: Any = None,
    facts: Optional[Dict[str, Any]] = None,
) -> bool:
    """Arm ``plan`` on ``pipeline``, PROBING it on the card first. Returns
    whether it armed.

    A False return means nothing was armed and the caller either asks for a
    more expensive plan or falls to the next rung — never that a partial
    arrangement was left behind.
    """
    if not plan.fits:
        return False
    try:
        import torch
    except Exception as exc:
        log.warning("partial_resident: unavailable (%s: %s)", type(exc).__name__, exc)
        return False

    dev = torch.device(device)
    parked: Dict[str, ParkedComponent] = {}
    try:
        remove = getattr(pipeline, "remove_all_hooks", None)
        if callable(remove):
            remove()
        for name in plan.resident:
            comp = getattr(pipeline, name, None)
            if comp is not None and hasattr(comp, "to"):
                comp.to(dev)
        for name in plan.offloaded:
            comp = getattr(pipeline, name, None)
            if comp is None:
                continue
            mirrored = ParkedComponent.mirror(
                name, comp, device=dev, torch_mod=torch, log=log
            )
            if mirrored is None:
                log.warning(
                    "partial_resident: %s could not be mirrored; the caller "
                    "falls to the next rung", name,
                )
                return False
            parked[name] = mirrored
        if not parked:
            log.warning(
                "partial_resident: the plan named %d component(s) to offload "
                "and none could be parked", len(plan.offloaded),
            )
            return False
        if free_bytes_now is not None:
            ok, free = probe_plan(parked, free_bytes_now=free_bytes_now)
            # The probe's MEASUREMENT, handed back so the caller can put it on
            # the loud line. pgw#1595: success was `log.info` and inaudible at
            # the endpoint's WARNING level, which makes "probe passed" and
            # "probe never ran" the same picture (pgw#1559 class).
            if facts is not None:
                facts["probe_free_bytes"] = int(free)
                facts.update(_placement_attribution(torch))
            if not ok:
                log.warning(
                    "partial_resident: PROBE REFUSED %s — onloading the largest "
                    "evicted component left %.2f GiB free, under the %.2f GiB "
                    "floor. The plan's arithmetic said %.2f GiB peak of %.2f "
                    "free; the card disagrees, and the card is right.",
                    ",".join(plan.offloaded), free / _GIB,
                    _PROBE_FLOOR_BYTES / _GIB,
                    plan.transient_peak_bytes / _GIB, plan.free_bytes / _GIB,
                )
                return False
            log.info(
                "partial_resident: probe OK — %.2f GiB still free with %s "
                "onloaded", free / _GIB, ",".join(plan.offloaded),
            )
        _install_residency_hooks(
            pipeline, parked, _pipeline_component_names(pipeline), log
        )
    except Exception as exc:
        log.warning(
            "partial_resident: arming failed (%s: %s); the caller falls to the "
            "next rung", type(exc).__name__, exc,
        )
        return False

    setattr(pipeline, COMPONENT_RESIDENCY_ATTR, plan)
    setattr(pipeline, PARKED_COMPONENTS_ATTR, parked)
    setattr(pipeline, PARTIAL_RESIDENT_DEVICE_ATTR, dev)
    log.info("partial_resident: armed — %s", plan.summary())
    return True


__all__ = [
    "COMPONENT_RESIDENCY_ATTR",
    "PARKED_COMPONENTS_ATTR",
    "PARKED_MODULE_ATTR",
    "PARTIAL_RESIDENT_DEVICE_ATTR",
    "PARTIAL_RESIDENT_RESERVE_GB",
    "PARTIAL_RESIDENT_UNARMED_PHASE",
    "ParkedComponent",
    "ResidencyPlan",
    "apply_component_residency",
    "parks_module",
    "plan_component_residency",
    "plan_for_pipeline",
    "probe_plan",
]
