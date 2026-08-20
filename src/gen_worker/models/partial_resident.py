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

#: Allocator slack for the ONLOAD COPY itself — and nothing else. It is NOT an
#: activation estimate: the only phase in which an evicted component is on the
#: card is the one that runs it (text encoding, 77 tokens), whose activations are
#: noise next to the weights being moved. The activation estimate is the BUDGET's
#: `PARTIAL_RESIDENT_RESERVE_GB`, and it guards the denoise phase, where
#: activations actually exist.
#:
#: MEASURED CLIFF, found on the card and recorded because the number looks
#: arbitrary otherwise: at 512 MiB this check REFUSED the whole rung once free
#: VRAM dropped from 7.3 to 7.1 GiB — a 200 MiB shift by a co-tenant silently
#: reverting SDXL to the 13 GiB-per-request rung. Charging a denoise-sized
#: reserve against an encode-sized phase is what produced that cliff.
_TRANSIENT_RESERVE_BYTES = 256 * (1 << 20)

#: The activation headroom this rung reserves, and it is NOT
#: ``memory._DEFAULT_SAFETY_MARGIN_GB``. That 2.0 GiB is a PLACEMENT heuristic
#: for the resident-vs-offload decision, deliberately pessimistic because it
#: guards a rung with no offload at all. Here the measurement exists: pgw#1570
#: recorded SDXL 1024^2 batch-2 CFG at a 5956 MiB peak with 5437 MiB of weights
#: resident — **519 MiB of activations**. 1.25 GiB is 2.4x that, and holding
#: back 2.0 would refuse the plan that keeps the denoiser resident, which is
#: Paul's ruling inverted: *"if the space is available, and it helps us run
#: faster, why wouldn't varena take it?"*
PARTIAL_RESIDENT_RESERVE_GB = 1.25


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
) -> ResidencyPlan:
    """The minimum-BYTES subset of components to evict, or a refusal.

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


def apply_component_residency(
    pipeline: Any,
    plan: ResidencyPlan,
    *,
    device: Any,
    log: logging.Logger,
) -> bool:
    """Arm ``plan`` on ``pipeline``. Returns whether it armed.

    A False return means nothing was armed and the caller must fall to the next
    rung — never that a partial arrangement was left behind.
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
    "PARTIAL_RESIDENT_DEVICE_ATTR",
    "PARTIAL_RESIDENT_RESERVE_GB",
    "PARTIAL_RESIDENT_UNARMED_PHASE",
    "ParkedComponent",
    "ResidencyPlan",
    "apply_component_residency",
    "plan_component_residency",
    "plan_for_pipeline",
]
