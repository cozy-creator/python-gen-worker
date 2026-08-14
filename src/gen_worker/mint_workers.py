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

from typing import Any, Dict, NamedTuple, Optional, Tuple


#: One entry child's measured HOST high-water, keyed by (family, weight lane).
#: The one bank that survives, because it is the divisor in ``K``'s host-RAM
#: bound and nothing else. Monotone at the WRITE: a child that peaked higher
#: once can peak that high again, and a lucky run must not talk the ask down.
_ENTRY_RSS_PEAKS: Dict[Tuple[str, str], int] = {}


def record_compiled_graph_peak_rss(family: str, weight_lane: str, peak_bytes: int) -> None:
    """Bank one entry child's measured host high-water for the next mint.

    Written on EVERY outcome including failures: an aborted mint's entries
    still peaked where they peaked, and the attempt that follows is exactly
    the one that needs the fact.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _ENTRY_RSS_PEAKS[key] = max(_ENTRY_RSS_PEAKS.get(key, 0), int(peak_bytes))


def compiled_graph_peak_rss(family: str, weight_lane: str) -> int:
    """0 = never measured on this pod; ``entry_workers`` says so in its basis."""
    return _ENTRY_RSS_PEAKS.get((str(family or ""), str(weight_lane or "")), 0)


# ---------------------------------------------------------------------------
# pgw#1205: the DEVICE reading, banked per GRAPH CLASS with its provenance
# ---------------------------------------------------------------------------
#
# §4.33 deleted a prediction layer and left one sentence standing: *"an estimate
# may never act as a floor a measurement is not allowed to correct, and an
# absent measurement means NO EVIDENCE."* pgw#1199 then deleted the last thing
# that made a mint weight-scale, so the honest requirement is what a compile
# actually costs a card — activation scale — and that is a MEASUREMENT nobody
# was keeping.
#
# WHY THE MACHINE, AND NOT THE CELL MANIFEST. The manifest is written when a
# cell SEALS, and the mint that most needs measuring is the one that OOMed and
# sealed nothing — a bank whose writer dies exactly when the interesting data
# exists is not a bank. And the consumer is local: K is decided in the mint
# child, on this card, so a fleet table reachable only through a hub is the
# wrong shape to read it from and cozy-local has no hub at all. So the reading
# goes to two sinks that answer different questions: HERE for "what did this
# cost on THIS machine", and onto `aot_mint_phases` for the fleet view. One
# measurement, taken once, in the child that ran it.
#
# WHAT THIS IS NOT. It sizes nothing. `entry_workers` is still f(cores,
# measured child RSS) and no width, placement or admission decision reads a row
# below. Wiring one in would re-create precisely what §4.33 deleted, and this
# module's own header says where such a term belongs: the deleted module.


class DevicePeak(NamedTuple):
    """One compile's device high-water, both readings.

    Both, deliberately: ``allocated`` is what the compile needed and
    ``reserved`` is what the caching allocator HELD and therefore what a
    concurrent sibling could not have. The GAP between them is allocator
    fragmentation, which is itself worth seeing and which a single number
    hides.
    """

    allocated_bytes: int
    reserved_bytes: int


class DevicePeakKey(NamedTuple):
    """WHAT was measured, and under WHICH conditions.

    Every axis is here because the number is meaningless without it: the same
    graph class costs a different amount on a different card, under a different
    toolchain, at a different weight lane, and in a different PHASE of the mint.
    A row that cannot say all five is not a measurement, it is a number.
    """

    graph_class: str
    #: The card, both ways it can be named: a human-legible SKU slug
    #: (``h100-80gb-hbm3``) and the arch the kernels were built for
    #: (``sm_90``). A cell minted at the wrong arch is unadoptable, so the
    #: reading must not be shared across arches.
    card: str
    sm: str
    #: The SAME digest the cell key's toolchain axis uses
    #: (``cell_key.toolchain_axis_digest``), so a banked row and the cell it
    #: was measured for agree about what "this toolchain" means.
    toolchain: str
    gen_worker: str
    weight_lane: str
    #: WHICH window this high-water covers — an export peak and an entry
    #: compile peak are different questions and must never be maxed together.
    phase: str


#: The bank. Monotone at the WRITE, exactly as the RSS bank is: a compile that
#: peaked high once can peak that high again, and a lucky run must not talk the
#: reading down.
_DEVICE_PEAKS: Dict[DevicePeakKey, DevicePeak] = {}


def record_entry_device_peak(
    key: DevicePeakKey, allocated_bytes: int, reserved_bytes: int,
) -> None:
    """Bank one compile's measured device high-water.

    Written on EVERY outcome including failures — pgw#848's rule for the host
    half, which applies here with more force: the attempt that ran out of
    device memory is the one whose reading the next attempt most needs, and it
    is exactly the attempt that seals no cell.

    Each reading is maxed INDEPENDENTLY. They come from one child and normally
    move together, but taking ``max`` per field can only ever widen a reading,
    and a bank that under-reports is the failure mode that matters.
    """
    allocated, reserved = max(0, int(allocated_bytes)), max(0, int(reserved_bytes))
    if allocated <= 0 and reserved <= 0:
        return
    if not str(key.graph_class or "").strip():
        # A row with no subject cannot be looked up and would silently
        # accumulate every class into one entry.
        return
    held = _DEVICE_PEAKS.get(key)
    if held is None:
        _DEVICE_PEAKS[key] = DevicePeak(allocated, reserved)
        return
    _DEVICE_PEAKS[key] = DevicePeak(
        max(held.allocated_bytes, allocated),
        max(held.reserved_bytes, reserved))


def entry_device_peak(key: DevicePeakKey) -> Optional[DevicePeak]:
    """The banked reading, or ``None`` — which means NO EVIDENCE.

    ``None`` rather than a zero-valued row on purpose (§4.33): "never measured
    here" and "measured at zero" are different facts, and a caller that cannot
    tell them apart will read the first as the second.
    """
    return _DEVICE_PEAKS.get(key)


def device_peak_rows() -> Dict[DevicePeakKey, DevicePeak]:
    """Every row this machine has banked. A copy — the bank is append-and-widen
    only, and a caller must not be able to lower a reading by holding it."""
    return dict(_DEVICE_PEAKS)


def _forget_device_peaks() -> None:
    """Drop the bank.

    Private, and meant to stay so: production never un-measures a card, so a
    public name here would be public surface with no production caller. Tests
    reach it deliberately.
    """
    _DEVICE_PEAKS.clear()


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
    "compiled_graph_peak_rss",
    "record_compiled_graph_peak_rss",
]
