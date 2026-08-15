"""Does a SERVE still fit after this adopt? (pgw#1265)

The adopt raises this pod's residency floor permanently: the loaded runner
stays resident for the life of the arm, and the §4.32 parity gate runs two
forwards beside it. Nothing asked whether what it was about to allocate fit.
Measured 2026-08-14 on z-image, four times in one day: resident 32.01 GiB of
44.42 GiB before the adopt (three functions, two full pipelines), then
``ComputeProcessDied: cause=signal:SIGSEGV function=generate`` with 17.9 MiB
free. The exhaustion is UNCATCHABLE — this SDK imposes
``expandable_segments:True`` (``settings_authority.py``), under which running
out is a native mapping abort no ``except`` sees — so ``_bind_headroom``'s
attempt-and-catch, which is the honest gate for a CATCHABLE bind OOM, cannot
be the whole answer. A query has to run first.

**What this is NOT.** ``mint_budget.adopt_headroom`` (pgw#1164) was deleted for
cause in pgw#1175: it PREDICTED the arm's cost as ``2 * activation`` where
``activation`` was an admittedly unmeasured quarter of a resident set the free
figure already excluded, and it refused stickily. It could not refuse the
failure it was written for and did refuse cards that were fine. Nothing here
predicts the adopt's cost, no bank is read (pgw#1205's device-peak bank stays
measurement-only), and there is no margin, factor or floor constant to tune.

**The two terms, both measured, both at the decision point.**

``have`` — ``torch.cuda.mem_get_info`` free plus the allocator cache this
process can hand back (``reserved - allocated``), the identical construction
``hot_swap._headroom_ok`` already uses.

``need`` — the largest device transient ONE forward has actually taken in this
process on this card, banked nowhere and bracketed at the seams every forward
passes (the executor's warm forward and its request invoke). It is keyed by
DEVICE, not by pipeline, deliberately: a release serving three functions holds
several pipelines on one card, and the question "can a serve still run here" is
the card's.

``need == 0`` (nothing has been measured yet) and an unprobeable device both
FIT BY CONSTRUCTION, exactly like every other budget in this tree. An honest
under-refusal beats a number invented to look like a bound; the refusal only
ever fires on evidence this process produced itself.

**THE RESIDENT FLOOR IS NEVER COMPUTED HERE, AND MUST NOT BE.** ``have`` is the
DRIVER's free figure, so every byte anything holds on this card — this record's
weights, a sibling instance's, and any component a placement rung kept resident
rather than offloading (ie#721's exclusion set) — is already subtracted from it
before this module sees it. There is deliberately no floor term to keep in step
with `models/memory.place_pipeline`: a second derivation of a quantity the card
already reports is how two subsystems come to disagree about it. If a future
change wants the exclusion accounted for, it is accounted for — by observation,
at the instant of the decision, which is stronger than any published set could
be. The one thing this cannot see is a floor that rises AFTER the verdict (a
rung transition, a sibling's rotation); no pre-check can, and that residue is
exactly what the serve-time allocator-exhaustion ruling covers.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Dict, Iterator, Optional
from .hostfacts import cuda_ready
from . import hostfacts

logger = logging.getLogger(__name__)

#: The typed refusal reason. A CAPACITY verdict about this pod at this instant
#: — never a contract verdict about the artifact, which is what quarantines a
#: key (th#1819). Another pod, or this one with less resident, adopts it fine.
REASON = "insufficient_adopt_headroom"

_GIB = float(1 << 30)

_lock = threading.Lock()
#: device index -> the largest single-forward device transient measured in this
#: process. In-process only: it dies with the worker, is never persisted, never
#: emitted as a bank row, and never read by a width or placement decision.
_forward_peak: Dict[int, int] = {}


def _torch() -> Any:
    try:
        import torch

        if not cuda_ready():
            return None
        return torch
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return None


def _device_index(device: Optional[int]) -> Optional[int]:
    torch = _torch()
    if torch is None:
        return None
    try:
        return int(torch.cuda.current_device()) if device is None else int(device)
    except Exception:  # noqa: BLE001
        return None


def record_forward_peak(device: Optional[int], transient: int) -> None:
    """Bank ONE forward's measured device transient, monotonically."""
    index = _device_index(device)
    if index is None or transient <= 0:
        return
    with _lock:
        if transient > _forward_peak.get(index, 0):
            _forward_peak[index] = int(transient)


def forward_peak(device: Optional[int] = None) -> int:
    """The largest single-forward transient measured on this device, or 0."""
    index = _device_index(device)
    if index is None:
        return 0
    with _lock:
        return int(_forward_peak.get(index, 0))


def reset() -> None:
    """Forget every measurement (tests; a fresh process starts here anyway)."""
    with _lock:
        _forward_peak.clear()


@contextlib.contextmanager
def forward_watermark(device: Optional[int] = None) -> Iterator[None]:
    """Bracket ONE forward and record what it transiently cost the device.

    ``max_memory_allocated`` is process-monotone and shared with other readers,
    so it is never reset here (pgw#652's activation learning reads the same
    counter): the transient is ``peak_after - allocated_at_entry``, which is
    this span's own high-water above the resident set it started from. A span
    that allocates nothing new records nothing.
    """
    torch = _torch()
    index = _device_index(device)
    if torch is None or index is None:
        yield
        return
    try:
        before = int(torch.cuda.memory_allocated(index))
    except Exception:  # noqa: BLE001
        yield
        return
    try:
        yield
    finally:
        try:
            peak = int(torch.cuda.max_memory_allocated(index))
            record_forward_peak(index, peak - before)
        except Exception:  # noqa: BLE001 — telemetry never fails a forward
            logger.debug("adopt-fit: forward watermark failed", exc_info=True)


def headroom(device: Optional[int] = None) -> Optional[int]:
    """Device bytes a new allocation could actually get, or ``None`` when the
    device cannot be probed. Driver-free plus the allocator cache this process
    can return — cached blocks are headroom, and counting only ``free`` would
    refuse a card holding a large idle cache."""
    index = _device_index(device)
    return None if index is None else hostfacts.headroom_bytes(index)


def refusal(what: str, device: Optional[int] = None) -> str:
    """``""`` when ``what`` may proceed; else the typed refusal's detail.

    Fits by construction when nothing has been measured on this device yet, or
    when the device cannot be probed. Both are the deliberate under-refusal:
    this gate only ever speaks from evidence this process produced.
    """
    need = forward_peak(device)
    if need <= 0:
        return ""
    have = headroom(device)
    if have is None or have >= need:
        return ""
    return (
        f"{what}: this pod has {have / _GIB:.3f} GiB of usable device memory "
        f"(free + reclaimable cache) and ONE forward on this card has measured "
        f"a {need / _GIB:.3f} GiB transient in this process, so a serve would "
        f"not fit beside what the adopt is about to hold. Adoption is "
        f"abandoned; this pod serves EAGER and stays alive. Nothing is said "
        f"about the artifact — this is a fact about the card at this instant, "
        f"and another pod, or this one with less resident, adopts it fine "
        f"(pgw#1265)"
    )


__all__ = [
    "REASON",
    "forward_peak",
    "forward_watermark",
    "headroom",
    "record_forward_peak",
    "refusal",
    "reset",
]
