"""pgw#737: the VRAM pre-budget every self-mint capture must pass.

gw#587's premise — "the serving worker's boot warmup IS a perfect mint by
construction" — holds only while the capture FITS. Nothing checked. On
wan-2.2 1.3.1 (27B MoE, ~54.2 GiB resident bf16 plus dual-expert LoRA branch
containers) the mint's seed passes OOMed three times on an 80 GiB H100 and
the TENANT request died with them at 78.07 GiB peak — 26 of 40 denoise steps
banked and lost, both tiers, both pods, five hub re-dispatches and a second
pod bought for a deterministic failure.

The budget is deliberately a MEASUREMENT, not a model of the graph: the CUDA
peak high-water minus the current resident set is the largest transient this
process has actually sustained on the card — the load, at the boot gate, and
this family's activation working set at serving shapes once a forward has
run (the driver re-checks after the boot warm). Below that floor sits the
pre-forward estimate, a quarter of the resident set: the ratio the wan-2.2
incident measured, and the reason the boot gate can refuse before any
capture exists. On top of the activation working set a mint needs:

  * its own seed forwards' working set — the seeds run the SAME warm-plan
    shapes the tenant runs, eager, in their own turns; and
  * whatever the capture RETAINS across turns (the per-signature dummy
    batches held in the warm queue, inductor's workspace, the compiled
    artifacts' own buffers), which the tenant's next peak has to fit
    alongside.

Both scale with the activation working set, so the need is stated as two of
them plus a flat inductor working-set floor. It is a floor, not a
prediction: when it is not met the mint is DECLINED and the worker serves
eager with the cell absent — a roomier config, or a smaller-resident flavor,
mints it later. An unprobeable device never blocks a mint (CPU rigs,
non-CUDA workers keep today's behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

_GIB = 1 << 30

#: Activation working set as a fraction of the resident set, used only when
#: no forward has run yet (no measured high-water to read).
_UNMEASURED_ACTIVATION_FRACTION = 0.25

#: Inductor's own compile-time working set — autotune candidate buffers,
#: codegen scratch, the cudagraph pool. Family-independent and flat; same
#: order as the pgw#677 warm-thread headroom floor.
_COMPILE_WORKSPACE_BYTES = 4 * _GIB


#: A second CUDA context on the same card (pgw#784). The mint child is its own
#: OS process, so it pays a context, cuBLAS/cuDNN handles and its own allocator
#: block — a real, unavoidable cost of the process boundary. This is a FLOOR
#: used only until a child reports its measured peak (``record_child_peak``);
#: it is not a prediction and never the number a decision rests on twice.
_CUDA_CONTEXT_FLOOR_BYTES = 1 * _GIB

#: Measured child peaks, keyed by (family, weight lane). One mint teaches the
#: next: the second ask on a pod is a fact, not this module's arithmetic.
_CHILD_PEAKS: Dict[Tuple[str, str], int] = {}


def _gib(value: int) -> str:
    return f"{value / _GIB:.2f}GiB"


@dataclass(frozen=True)
class MintBudget:
    """One capture-headroom verdict. ``probed=False`` = no CUDA to read;
    ``fits`` is then True by construction (never block on a blind probe)."""

    fits: bool
    probed: bool = False
    measured: bool = False
    free_bytes: int = 0
    need_bytes: int = 0
    resident_bytes: int = 0
    activation_bytes: int = 0

    def line(self, event: str, reason: str) -> str:
        """The one structured line a decline logs and puts on the wire."""
        if not self.probed:
            return f"{event} reason={reason} headroom=unprobeable"
        return (
            f"{event} reason={reason} headroom={_gib(self.free_bytes)} "
            f"needed~={_gib(self.need_bytes)} "
            f"resident={_gib(self.resident_bytes)} "
            f"activation={_gib(self.activation_bytes)}"
            f"({'measured' if self.measured else 'estimated'})"
        )


_UNPROBEABLE = MintBudget(fits=True, probed=False)


def device_of(pipeline: Any) -> Optional[int]:
    """The CUDA device a pipeline's weights live on (None = unknown, read
    the current device instead)."""
    try:
        import torch
    except Exception:
        return None
    candidates = [pipeline]
    try:
        candidates.extend(vars(pipeline).values())
    except TypeError:
        pass
    for obj in candidates:
        if not isinstance(obj, torch.nn.Module):
            continue
        for param in obj.parameters():
            if param.is_cuda:
                return int(param.device.index or 0)
            break
    return None


def probe(device: Optional[int] = None) -> MintBudget:
    """Can a self-mint capture run here without taking the tenant down?"""
    try:
        import torch

        if not torch.cuda.is_available():
            return _UNPROBEABLE
        dev = torch.cuda.current_device() if device is None else int(device)
        free, _total = torch.cuda.mem_get_info(dev)
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        peak = int(torch.cuda.max_memory_allocated(dev))
    except Exception:
        return _UNPROBEABLE
    # The allocator's cached-but-unallocated pool is reclaimable headroom
    # (ie#468's planner reads free VRAM the same way).
    free_bytes = int(free) + max(0, reserved - allocated)
    measured_activation = max(0, peak - allocated)
    activation = max(
        measured_activation,
        int(allocated * _UNMEASURED_ACTIVATION_FRACTION),
    )
    need = 2 * activation + _COMPILE_WORKSPACE_BYTES
    return MintBudget(
        fits=free_bytes >= need,
        probed=True,
        measured=measured_activation > 0,
        free_bytes=free_bytes,
        need_bytes=need,
        resident_bytes=allocated,
        activation_bytes=activation,
    )


def record_child_peak(family: str, weight_lane: str, peak_bytes: int) -> None:
    """Bank a mint child's MEASURED device peak for the next ask.

    Monotone by design: a mint that peaked higher once can peak that high
    again, and the ask must not drift down on a lucky run.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _CHILD_PEAKS[key] = max(_CHILD_PEAKS.get(key, 0), int(peak_bytes))


def child_peak(family: str, weight_lane: str) -> int:
    return _CHILD_PEAKS.get((str(family or ""), str(weight_lane or "")), 0)


def co_residency(
    device: Optional[int] = None,
    *,
    family: str = "",
    weight_lane: str = "",
) -> MintBudget:
    """pgw#784: can a MINT CHILD live on this card next to the eager server?

    The contract implies co-residency — "THE WORKER IS AVAILABLE THE ENTIRE
    TIME WHILE IT IS MINTING" — so the question is never whether to share the
    card, only whether the share is affordable and what BOUNDS each side.

    The model, stated
    -----------------
    The child is a separate OS process, so it holds its OWN copy of what it
    compiles. Its ask is:

        resident weights + ONE activation working set
                         + the inductor compile workspace
                         + one CUDA context

    ``resident`` and ``activation`` are read off THIS process, which is a
    legitimate proxy and not a guess: the child loads the same weights at the
    same lane and runs the same declared shapes.

    Note what LEAVES the serving process in exchange. The in-process capture
    needed ``2 * activation + workspace`` **on the serving card, inside the
    serving process** — the tenant's forward, the seed's forward, and whatever
    the capture retained across turns (per-signature dummy batches, inductor
    scratch, compiled buffers), all of which the tenant's next peak had to fit
    around. Delegating removes every byte of that from the server. So the real
    delta is::

        + one weight copy      (the child's)
        + one CUDA context     (the child's)
        - one activation set   (the seed forward is no longer co-resident
                                with the tenant's inside one allocator)

    which makes delegation CHEAPER for activation-heavy families (video at
    high resolution) and dearer for weight-heavy ones (a 54 GiB MoE). The
    weight-heavy case declines — and a decline is not a failure: the worker
    serves eager, the cell stays absent, and a roomier pod mints it. That is
    pgw#737's existing policy, unchanged.

    Enforcement, not hope
    ---------------------
    ``need_bytes`` is handed to the child as a hard
    ``set_per_process_memory_fraction`` cap. An under-estimate therefore
    becomes the CHILD's OOM — a typed failed mint reported by a live worker —
    instead of the tenant's, which is the failure the wan-2.2 incident was.
    And the child's measured peak is banked (``record_child_peak``), so the
    second ask on a pod is a fact.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return _UNPROBEABLE
        dev = torch.cuda.current_device() if device is None else int(device)
        free, _total = torch.cuda.mem_get_info(dev)
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        peak = int(torch.cuda.max_memory_allocated(dev))
    except Exception:
        return _UNPROBEABLE
    free_bytes = int(free) + max(0, reserved - allocated)
    measured_activation = max(0, peak - allocated)
    activation = max(
        measured_activation,
        int(allocated * _UNMEASURED_ACTIVATION_FRACTION),
    )
    banked = child_peak(family, weight_lane)
    need = max(
        banked + _CUDA_CONTEXT_FLOOR_BYTES if banked else 0,
        allocated + activation + _COMPILE_WORKSPACE_BYTES
        + _CUDA_CONTEXT_FLOOR_BYTES,
    )
    return MintBudget(
        fits=free_bytes >= need,
        probed=True,
        measured=measured_activation > 0 or banked > 0,
        free_bytes=free_bytes,
        need_bytes=need,
        resident_bytes=allocated,
        activation_bytes=activation,
    )


__all__ = [
    "MintBudget",
    "child_peak",
    "co_residency",
    "device_of",
    "probe",
    "record_child_peak",
]
