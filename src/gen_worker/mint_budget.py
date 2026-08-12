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

#: pgw#848: the same loop for the entry-compile pool's HOST ask, keyed the
#: same way. Separate from ``_CHILD_PEAKS`` because they measure different
#: processes on different resources: that one is the mint child's DEVICE
#: peak, this one is ONE entry child's host high-water (interpreter + the
#: compiler it runs), which is what bounds K.
_ENTRY_RSS_PEAKS: Dict[Tuple[str, str], int] = {}

#: pgw#877: and the DEVICE half of the entry loop, which is the one that was
#: missing entirely. ``_CHILD_PEAKS`` above is the MINT CHILD's device peak —
#: one process holding a whole pipeline. This is ONE ENTRY CHILD's, measured
#: by ``aot_compile_child._peak_device`` and carried out on the pool's phase
#: table. They are different processes with different footprints and must not
#: share a figure: the entry ask used to be the mint child's whole
#: co-residency estimate, which is how K stayed at 1-2 on a host that could
#: run it 37-127 wide.
_ENTRY_DEVICE_PEAKS: Dict[Tuple[str, str], int] = {}

#: pgw#1164: and the ADOPT's, which is a different process AND a different
#: step from all three above. Those measure a compile; this measures loading
#: the finished cell onto the serving card and proving it against eager. It is
#: the only one of the four that has destroyed a paid mint (th#1825).
_ADOPT_PEAKS: Dict[Tuple[str, str], int] = {}

#: pgw#1169: (family, weight lane) this process has already refused to adopt.
#: The refusal is STICKY — see :func:`note_adopt_declined`.
_ADOPT_DECLINED: "set[Tuple[str, str]]" = set()


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
    #: pgw#877 #3: ``reserved - allocated``, the part of ``free_bytes`` that is
    #: free to THIS process only. Reported so the cross-process over-count can
    #: be measured on a real pod instead of reasoned about.
    cache_slack_bytes: int = 0
    #: pgw#848: the CEILING the child is actually given, which is NOT
    #: ``need_bytes``. See :func:`co_residency` — the estimate answers "should
    #: this start", the ceiling answers "how far may it go", and using one
    #: number for both is what capped two mints at 11.09 GiB on cards with
    #: 21.48 GiB free.
    cap_bytes: int = 0

    @property
    def card_bytes(self) -> int:
        """th#1800: the TOTAL device memory a card must carry to admit this
        mint at all — the server's own resident set plus the child's whole ask.

        A decline that says only "not here" cannot be acted on; §4.28 leaves
        exactly one way to get a cell for a family that does not fit
        (*"pre-warming a release/SKU = boot an ordinary serving pod there"*),
        and acting on that needs a card CLASS, not a shortfall. wan-2.2-t2v-a14b
        declined at ``resident=40.65 need~=72.54`` on an 80 GiB H100 — the H200
        that admits it is a fact of THIS number (113.19 GiB), and it had to be
        re-derived by hand from a log line before it could be stated.

        ``resident_bytes`` is added because the server's weights are already
        allocated and therefore already outside ``free_bytes``: the card must
        hold both processes, not just the child.
        """
        if not self.probed:
            return 0
        return self.resident_bytes + self.need_bytes

    def line(self, event: str, reason: str) -> str:
        """The one structured line a decline logs and puts on the wire."""
        if not self.probed:
            return f"{event} reason={reason} headroom=unprobeable"
        return (
            f"{event} reason={reason} headroom={_gib(self.free_bytes)} "
            f"needed~={_gib(self.need_bytes)} "
            f"resident={_gib(self.resident_bytes)} "
            f"activation={_gib(self.activation_bytes)}"
            f"({'measured' if self.measured else 'estimated'}) "
            f"cap={_gib(self.cap_bytes)} "
            f"card>={_gib(self.card_bytes)} "
            f"cache_slack={_gib(self.cache_slack_bytes)}"
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


@dataclass(frozen=True)
class _DeviceRead:
    """One CUDA reading, and the two derived quantities both budgets need.

    pgw#877: :func:`probe` and :func:`co_residency` each carried their own
    verbatim copy of this — same five ``torch.cuda`` calls, same ``free_bytes``
    sum, same ``activation`` floor. Two copies of one definition is how the two
    answers drift apart on a card neither of them can re-read.

    ``free_bytes`` counts THIS process's cached-but-unallocated allocator
    blocks as headroom. That is exact for :func:`probe`, whose capture runs in
    this process and this allocator. It is NOT exact for
    :func:`co_residency`, whose consumer is a different OS process: the
    caching allocator does not hand cached blocks back to the driver without
    ``empty_cache()``, so those bytes are free to nobody but us. See that
    function's note.
    """

    free_bytes: int
    allocated: int
    measured_activation: int
    activation: int
    #: pgw#877 #3: THE OVER-COUNT, named and carried rather than argued about.
    #: ``reserved - allocated`` — this process's cached-but-unallocated
    #: allocator blocks, which `free_bytes` counts as headroom. Exact for
    #: :func:`probe`; for :func:`co_residency` it is bytes a DIFFERENT process
    #: cannot have, so it inflates both `fits` and `cap_bytes` by this much,
    #: out of the tenant's reserve. Off-pod it cannot be measured (this box has
    #: no usable CUDA), so the instrument ships and the next real mint reports
    #: it: compare `cache_slack` in the decline line against `nvidia-smi`'s
    #: free. Nothing branches on it — measure first, then decide.
    cache_slack: int


def _read_device(device: Optional[int]) -> Optional[_DeviceRead]:
    """The reading, or ``None`` when there is no CUDA to read."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device() if device is None else int(device)
        free, _total = torch.cuda.mem_get_info(dev)
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        peak = int(torch.cuda.max_memory_allocated(dev))
    except Exception:
        return None
    measured_activation = max(0, peak - allocated)
    return _DeviceRead(
        free_bytes=int(free) + max(0, reserved - allocated),
        allocated=allocated,
        cache_slack=max(0, reserved - allocated),
        measured_activation=measured_activation,
        activation=max(
            measured_activation,
            int(allocated * _UNMEASURED_ACTIVATION_FRACTION),
        ),
    )


def probe(device: Optional[int] = None) -> MintBudget:
    """Can a self-mint capture run here without taking the tenant down?

    The IN-PROCESS capture's gate (pgw#737), still reached whenever
    delegation declines — ``executor._background_mint_run`` falls through to
    it when no pending is delegated.
    """
    read = _read_device(device)
    if read is None:
        return _UNPROBEABLE
    need = 2 * read.activation + _COMPILE_WORKSPACE_BYTES
    return MintBudget(
        fits=read.free_bytes >= need,
        probed=True,
        measured=read.measured_activation > 0,
        free_bytes=read.free_bytes,
        need_bytes=need,
        resident_bytes=read.allocated,
        activation_bytes=read.activation,
        cache_slack_bytes=read.cache_slack,
        cap_bytes=need,
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


def note_adopt_declined(family: str, weight_lane: str) -> None:
    """Make an adopt refusal STICKY for this process (pgw#1169, §4.31).

    §4.31's serve-first posture is that a cell-attributable failure de-arms the
    cell and serves eager IN-REQUEST. An adopt that cannot fit the card is that
    failure in its worst-behaved form — it takes the process rather than
    returning an error — so the refusal has to survive the decision that
    produced it. Without this, a pod re-asks a question whose answer cannot
    have improved and re-runs a load that killed the last attempt.

    Deliberately NOT in :func:`compile_cache.arming_block`, which documents
    that every reason it names *"is deterministic for the life of this
    process: none of them can differ on a retry."* Free VRAM is not: it moves
    with what is resident. So the STICKINESS lives here, next to the bank whose
    numbers it is made of, and `arming_block`'s invariant stays true.
    """
    _ADOPT_DECLINED.add((str(family or ""), str(weight_lane or "")))


def adopt_declined(family: str, weight_lane: str) -> bool:
    """Whether this process has already refused to adopt this (family, lane)."""
    return (str(family or ""), str(weight_lane or "")) in _ADOPT_DECLINED


def record_adopt_peak(family: str, weight_lane: str, peak_bytes: int) -> None:
    """Bank the ADOPT-ARM's measured device high-water for the next ask.

    pgw#1164. Every other bank in this module measures a COMPILE; this one
    measures the step after it, which is the one that has actually killed a
    mint. ``adopt_delegated_mint`` loads the packed cell onto the serving card
    and runs ``gate_cell_numerics``, whose probe holds, simultaneously: the
    retained eager callable, EVERY loaded AOTI entry runner, and two forwards'
    activation working set per axis (``numerics_probe.probe_cell`` iterates
    ``axes_from_meta`` and reads ``state["original"]`` and ``state["runner"]``
    for each). sdxl declares 36 entries. Nothing measured that.

    Monotone for the same reason the others are: an adopt that peaked higher
    once can peak that high again, and the ask must not drift down on a lucky
    run.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _ADOPT_PEAKS[key] = max(_ADOPT_PEAKS.get(key, 0), int(peak_bytes))


def adopt_peak(family: str, weight_lane: str) -> int:
    """0 = this pod has never completed an adopt for this (family, lane)."""
    return _ADOPT_PEAKS.get((str(family or ""), str(weight_lane or "")), 0)


def adopt_headroom(
    family: str = "", weight_lane: str = "", device: Optional[int] = None,
) -> MintBudget:
    """Can the ADOPT-ARM run here without taking the card down? (pgw#1164)

    The gap this closes, stated exactly: :func:`probe` and :func:`co_residency`
    budget the COMPILE. Nothing budgeted the adopt, so a pod could pass its
    pre-mint gate, compile 36/36 entries over 1 h 37 m, and then die loading
    what it had just built — which is th#1825, \\$0.81, one step from durable.
    The budget it passed never described the step that killed it.

    TWO BASES, and the difference is stated rather than blended:

    * **measured** — this pod has completed an adopt for this (family, lane),
      so :func:`adopt_peak` holds a real high-water and ``need`` IS that fact.
      This is the ask that can refuse.
    * **unmeasured** — no adopt has ever completed here. ``need`` is then a
      FLOOR built only of measured terms: ``2 * activation``, the verify's own
      two-forward working set, on the same construction :func:`probe` already
      uses and for the same reason (the probe runs eager AND compiled on one
      feed, so both allocate).

    **The unmeasured floor is deliberately INCOMPLETE and must not be read as
    a prediction.** It omits the loaded entry runners' own device footprint —
    the term nobody has ever measured, and the term this function exists to
    start banking. So an unmeasured verdict can refuse a card that cannot even
    hold the verify, and it CANNOT refuse a card that merely cannot hold 36
    runners. That is an honest under-refusal, not a guess dressed as a bound:
    inventing a per-entry constant here is exactly the magic number this
    codebase forbids, and the real figure arrives the moment one adopt
    completes anywhere on this family.

    ``probed=False`` (no CUDA) fits by construction, like every other budget
    here — an unprobeable device never blocks a mint.
    """
    read = _read_device(device)
    if read is None:
        return _UNPROBEABLE
    banked = adopt_peak(family, weight_lane)
    need = banked if banked > 0 else 2 * read.activation
    # pgw#1169: a refusal already taken stands. Re-asking would re-run a load
    # that this process has already decided it cannot survive, and a transient
    # dip in someone else's residency is not evidence that it can.
    fits = read.free_bytes >= need and not adopt_declined(family, weight_lane)
    return MintBudget(
        fits=fits,
        probed=True,
        measured=banked > 0,
        free_bytes=read.free_bytes,
        need_bytes=need,
        resident_bytes=read.allocated,
        activation_bytes=read.activation,
        cache_slack_bytes=read.cache_slack,
        cap_bytes=need,
    )


def adopt_watermark(device: Optional[int] = None) -> Tuple[int, int]:
    """``(allocated_now, peak_so_far)`` on this device, or ``(0, 0)``.

    The pair an adopt brackets itself with. ``max_memory_allocated`` is
    process-monotone, so the caller takes ``peak_after - allocated_before`` —
    the high-water the adopt added ABOVE the resident set it started from —
    and never resets the counter, which other readers on this process share.
    """
    read = _read_device(device)
    if read is None:
        return 0, 0
    try:
        import torch

        dev = torch.cuda.current_device() if device is None else int(device)
        return read.allocated, int(torch.cuda.max_memory_allocated(dev))
    except Exception:
        return read.allocated, 0


def record_entry_peak_rss(family: str, weight_lane: str, peak_bytes: int) -> None:
    """Bank one ENTRY child's measured HOST high-water for the next mint.

    pgw#848: the device side of the pool's width has had this loop since
    pgw#784 (``record_child_peak`` above); the host side never did.
    ``aot_compile_pool.entry_workers`` was called with ``device_bytes=`` and
    never with ``peak_rss_bytes=``, so ``mem_workers`` divided available RAM
    by a 3 GiB CONSTANT on every mint the fleet has ever run and
    ``per_entry_rss_basis`` read ``"default"`` forever — a field that exists
    to distinguish a measured K from a guessed one, permanently pinned to
    "guessed". The pool has measured the real figure the whole time
    (``peak_child_rss_bytes`` in its ledger); nothing read it.

    Monotone, for the same reason the device bank is: a mint that peaked
    higher once can peak that high again, and an ask must not drift down on
    a lucky run.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _ENTRY_RSS_PEAKS[key] = max(_ENTRY_RSS_PEAKS.get(key, 0), int(peak_bytes))


def entry_peak_rss(family: str, weight_lane: str) -> int:
    """0 = never measured on this pod; the width falls back to its constant."""
    return _ENTRY_RSS_PEAKS.get((str(family or ""), str(weight_lane or "")), 0)


def record_entry_device_peak(
    family: str, weight_lane: str, peak_bytes: int,
) -> None:
    """Bank ONE ENTRY child's measured DEVICE high-water (pgw#877 #1/#2).

    The third bank, and the last one that was missing. pgw#868 A4 taught the
    entry child to measure this (``EntryReport.peak_device_bytes`` /
    ``..._reserved_bytes``) and deliberately left it telemetry-only; nothing
    ever read it, so the per-entry device ask stayed
    ``co_residency().need_bytes`` — the MINT CHILD's whole footprint, ~56 % of
    which was never observed — reported as ``per_entry_device_basis:
    'measured'``.

    Written by the SERVING PARENT only, exactly like its two siblings, and
    read back onto ``MintRequest.entry_device_peak_bytes``. That wire hop is
    the fix for the defect this bank would otherwise reproduce: a
    module-global read inside the mint child is a read of an empty dict.

    Monotone, for the reason both siblings are: a mint that peaked higher once
    can peak that high again.
    """
    if peak_bytes <= 0:
        return
    key = (str(family or ""), str(weight_lane or ""))
    _ENTRY_DEVICE_PEAKS[key] = max(
        _ENTRY_DEVICE_PEAKS.get(key, 0), int(peak_bytes))


def entry_device_peak(family: str, weight_lane: str) -> int:
    """0 = no entry child has ever been watched on this pod for this lane."""
    return _ENTRY_DEVICE_PEAKS.get(
        (str(family or ""), str(weight_lane or "")), 0)


def entry_device_ask(peak_bytes: int) -> int:
    """One entry child's device ask from ITS OWN measured peak (0 = none).

    The peak is the caching allocator's high-water. A CUDA context, the
    cuBLAS/cuDNN handles and the driver's own per-process overhead live
    OUTSIDE the allocator and are invisible to it, so the context floor is
    added rather than assumed to be inside the measurement.
    """
    return int(peak_bytes) + _CUDA_CONTEXT_FLOOR_BYTES if peak_bytes > 0 else 0


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

    Enforcement, not hope — and the ESTIMATE IS NOT THE CEILING (pgw#848)
    --------------------------------------------------------------------
    The child gets a hard ``set_per_process_memory_fraction`` cap. For most of
    this module's life that cap was ``need_bytes``, i.e. the estimate above,
    and that conflation cost the program its whole-graph proof twice:

        pod   card total   free at OOM   cap imposed   entries exported
        4090   23.52 GiB      660 MiB     11.09 GiB     1 of 36
        L40S   44.39 GiB    21.48 GiB     11.08 GiB     5 of 36

    **21.48 GiB free, and the mint died for 30 MiB.** The cap did not move
    across a 2x card change or a ``vram_gb`` 12->20 change, because it was a
    property of neither: sdxl's UNet is ~4.87 GiB resident and
    4.87 x 1.25 + 5 = 11.09 GiB, exactly what both pods printed. The mint was
    not running out of GPU. It was enforcing a self-imposed ceiling derived
    from :data:`_UNMEASURED_ACTIVATION_FRACTION` — a fraction nobody measured —
    and then reporting the result as a deterministic refusal.

    So the two questions are separated, because they are different questions:

    * ``need_bytes`` — *should this start?* An estimate of what the child will
      use, deliberately conservative, and the thing ``fits`` compares.
    * ``cap_bytes`` — *how far may it go?* ``free_bytes - activation``: what
      the card actually has, less what the tenant needs for its next forward
      (its weights are already allocated and so already outside ``free``).
      A property of the CARD, not of a guess.

    This does NOT weaken pgw#784's premise. The tenant's next peak is still
    reserved by construction, and an under-estimate still becomes the CHILD's
    OOM rather than the tenant's — which is the failure the wan-2.2 incident
    was. What changes is that a roomy card now licenses a roomy child.

    KNOWN OVERSTATEMENT, unfixed (pgw#877)
    --------------------------------------
    ``free_bytes`` is ``mem_get_info().free + (reserved - allocated)`` — the
    driver's free plus THIS process's cached-but-unallocated allocator blocks.
    That sum is exact for :func:`probe`, whose capture runs in this allocator.
    It is not exact here: the consumer is a different OS process, and PyTorch's
    caching allocator does not return cached blocks to the driver without an
    ``empty_cache()`` nobody calls on this path. So both ``fits`` and
    ``cap_bytes`` are overstated by exactly ``reserved - allocated``, and the
    overstatement comes out of the tenant's reserve. ``aot_compile_pool
    ._probe_free_device_bytes`` reads the card the same way for the same
    cross-process consumer, and its own docstring already says the quiet part
    ("a cached block the tenant is not using is free to nobody but this
    process"). Observable: log ``reserved - allocated`` beside ``free_bytes``
    and diff ``free_bytes`` against ``nvidia-smi``'s free — the gap is the
    over-count.

    On :data:`_UNMEASURED_ACTIVATION_FRACTION`: it is still a guess, and it is
    deliberately NOT replaced with a different off-pod constant — substituting
    one unmeasured number for another is a move this program has already paid
    for. It now bounds only the admission estimate and the tenant reserve,
    never the child, and the child's measured peak is banked
    (``record_child_peak``) so the second ask on a pod is a fact.
    """
    read = _read_device(device)
    if read is None:
        return _UNPROBEABLE
    free_bytes = read.free_bytes
    allocated = read.allocated
    measured_activation = read.measured_activation
    activation = read.activation
    # pgw#877 #4 — A MEASUREMENT REPLACES THE GUESSES IT MEASURED.
    #
    # This was `max(banked + ctx, allocated + activation + workspace + ctx)`,
    # so a child that really peaked BELOW the estimate re-asked for the
    # estimate forever: an estimate acting as a floor a measurement isn't
    # allowed to correct. That is this subsystem's whole disease in one line.
    #
    # The monotone ratchet belongs at the WRITE (`record_child_peak` keeps the
    # high-water across attempts, so a lucky run cannot talk the ask down) and
    # NOT at the read, where it can only ever pin the answer to the guess.
    #
    # The floor that makes narrowing safe is `allocated + ctx`, and it is
    # chosen rather than assumed: `allocated` is a MEASUREMENT of the resident
    # set the child provably re-holds, while `_UNMEASURED_ACTIVATION_FRACTION`
    # and `_COMPILE_WORKSPACE_BYTES` are the two guesses. Only the guesses may
    # be corrected away. It matters because `record_child_peak` banks on EVERY
    # outcome including failures (pgw#848, deliberately) — a child that OOMed
    # during `load` banks a tiny peak, and a narrowing that trusted it blindly
    # would admit the next mint onto a card that cannot hold one weight copy.
    banked = child_peak(family, weight_lane)
    if banked:
        need = max(banked, allocated) + _CUDA_CONTEXT_FLOOR_BYTES
    else:
        need = (allocated + activation + _COMPILE_WORKSPACE_BYTES
                + _CUDA_CONTEXT_FLOOR_BYTES)
    # pgw#848: the CEILING, which is a property of the CARD and not of the
    # estimate. Everything the tenant will need for its next forward is
    # `activation` — its weights are already allocated and therefore already
    # out of `free_bytes` — so what is genuinely spare is `free - activation`,
    # and that is what the child may have. Never BELOW `need`: if the card is
    # so tight that the reserve eats the estimate, `fits` has already declined
    # and no cap is issued at all.
    cap = max(need, free_bytes - activation)
    return MintBudget(
        fits=free_bytes >= need,
        probed=True,
        measured=measured_activation > 0 or banked > 0,
        free_bytes=free_bytes,
        need_bytes=need,
        resident_bytes=allocated,
        activation_bytes=activation,
        cache_slack_bytes=read.cache_slack,
        cap_bytes=cap,
    )


__all__ = [
    "MintBudget",
    "adopt_declined",
    "adopt_headroom",
    "adopt_peak",
    "adopt_watermark",
    "child_peak",
    "co_residency",
    "device_of",
    "entry_device_ask",
    "entry_device_peak",
    "entry_peak_rss",
    "probe",
    "note_adopt_declined",
    "record_adopt_peak",
    "record_child_peak",
    "record_entry_device_peak",
    "record_entry_peak_rss",
]
