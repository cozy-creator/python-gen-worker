"""What a mint MEASURES. Nothing here predicts VRAM (§4.33, pgw#1175).

This module is what is left of ``mint_budget`` after the deletion. That module
was a prediction layer standing where an attempt belongs: five peak banks, a
sticky decline, and an arithmetic whose central term charged the compiling
child for ``allocated`` — the PARENT's resident weights, which ``free_bytes``
already excludes. The premise it rested on (*"the resident set the child
provably re-holds"*) died at ``fc77b923``, when every production mint became
weight-free (meta/FakeTensor instantiation, pgw#1080). Four families were told
they needed 49-113 GiB and recorded hardware-unsatisfiable; §4.33 retracted all
four.

What replaces it:

* **K = f(cores, one measured child RSS)** — the two facts below feed
  :func:`gen_worker.aot_compile_pool.entry_workers`, which now derives width
  from CPU and host RAM only. A compile child's device footprint is not
  estimated, banked, divided or compared against anything.
* **The attempt is the signal.** A compile child that runs out of device
  memory dies in its own process, is classified (``MintResourceExhausted``)
  and reported; an ADOPT that cannot bind an entry raises there and the pod
  serves eager (``aot_serve.load_and_wrap``'s per-entry OOM guard). Neither
  needed a number computed in advance to be correct.

Everything here is a READING taken after the fact. If a function in this module
ever grows a term it did not measure, it belongs in the deleted module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


#: One entry child's measured HOST high-water, keyed by (family, weight lane).
#: The one bank that survives, because it is the divisor in ``K``'s host-RAM
#: bound and nothing else. Monotone at the WRITE: a child that peaked higher
#: once can peak that high again, and a lucky run must not talk the ask down.
_ENTRY_RSS_PEAKS: Dict[Tuple[str, str], int] = {}


def record_entry_peak_rss(family: str, weight_lane: str, peak_bytes: int) -> None:
    """Bank one entry child's measured host high-water for the next mint.

    Written on EVERY outcome including failures: an aborted mint's entries
    still peaked where they peaked, and the attempt that follows is exactly
    the one that needs the fact.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _ENTRY_RSS_PEAKS[key] = max(_ENTRY_RSS_PEAKS.get(key, 0), int(peak_bytes))


def entry_peak_rss(family: str, weight_lane: str) -> int:
    """0 = never measured on this pod; ``entry_workers`` says so in its basis."""
    return _ENTRY_RSS_PEAKS.get((str(family or ""), str(weight_lane or "")), 0)


def device_of(pipeline: Any) -> Optional[int]:
    """The CUDA device a pipeline's weights live on (None = unknown).

    An address, not a budget — it names which card an adopt brackets itself
    against and which index a mint child is handed.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
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


def adopt_watermark(device: Optional[int] = None) -> Tuple[int, int]:
    """``(allocated_now, peak_so_far)`` on this device, or ``(0, 0)``.

    The pair an adopt brackets itself with, and the instrument behind the
    ``cell_adopt_budget`` row — the only answer anyone has to "where does a
    loaded cell's device memory go". ``max_memory_allocated`` is
    process-monotone, so the caller takes ``peak_after - allocated_before``
    and never resets the counter, which other readers on this process share.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, 0
        dev = torch.cuda.current_device() if device is None else int(device)
        return (int(torch.cuda.memory_allocated(dev)),
                int(torch.cuda.max_memory_allocated(dev)))
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return 0, 0


__all__ = [
    "adopt_watermark",
    "device_of",
    "entry_peak_rss",
    "record_entry_peak_rss",
]
