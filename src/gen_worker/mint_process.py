"""pgw#784: a mint runs in its OWN OS process — the worker serves eager and
reports throughout.

Paul's contract (``WORKER-CONTRACTS.md`` §2, verbatim)::

    "the process for mint is: 1. compute key 2. try to fetch from hub, if
    miss then begin compiling and serve eager until the local cell is
    available; then switch over to using that. 2b. if cache-hit, then load
    the AOT/JIT cell and serve compiled. that's it. THE WORKER IS AVAILABLE
    THE ENTIRE TIME WHILE IT IS MINTING!"

The defect this closes (th#1299)
--------------------------------
The mint's compile driver ran INSIDE the serving process. Inductor's
orchestration layer is long-running GIL-holding Python, so a mint starved the
one asyncio task that carries BOTH the 10s heartbeat and eager serving: a pod
declared hung resumed reporting 6 seconds later, an activity counter froze for
126s, and the hub killed it at ~60s — correctly, per §1. A worker whose
reporting can be muted by its own compute is a broken worker; the fix is
worker-side and structural, never hub-side patience.

Spawn, never fork
-----------------
The child is a FRESH interpreter (``python -m gen_worker.mint_child``), not a
fork and not a ``multiprocessing`` worker:

* a CUDA context cannot survive ``fork()`` — the parent has one by the time a
  mint is decided, so forking is not on the table at all;
* ``multiprocessing`` (even with the spawn context) would pickle live parent
  objects across the boundary. Nothing live crosses here. The whole boundary
  is one JSON request file in, one JSON report file + one artifact out — the
  child loads what it needs itself, from the same baked manifest and the same
  already-downloaded weights on disk.

That boundary is also why the child is diagnosable: the request is a file you
can re-run by hand (``python -m gen_worker.mint_child request.json``).

Nothing in the parent blocks
----------------------------
Every wait here is loop-native — ``await reader.readline()`` for the child's
frames, ``await asyncio.sleep`` for the evidence poll, ``await proc.wait()``
for the reap. The parent's 10s beat and its eager serving run untouched, so
"report mint progress" needs no new protocol: the child's phase frames land on
``Activity.phase`` and ride the ordinary beat, and the child's CPU is already
inside ``activity._process_cpu_evidence`` (which sums live children
recursively), so ``evidence:self_mint_compile`` advances by itself.

Observe the machine, never a wall clock
---------------------------------------
There is no total-runtime bound. A mint that keeps making progress is never
killed; a mint that stops making progress is killed quickly. Progress is
MEASURED — the child's process-tree CPU seconds plus the bytes it has written
into its work root — not asserted by a frame the child could emit while
wedged. That is the same rule ``stall.SilenceWindow`` / pgw#655 already apply
to downloads, and the same rule §1 states for supervisors.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import msgspec

from .api.binding import ModelRef, wire_ref
from .stall import SilenceWindow

logger = logging.getLogger(__name__)

#: The child entry point. Kept as a module so the boundary is argv+files.
MINT_CHILD_MODULE = "gen_worker.mint_child"

#: Every child progress frame is one stdout line with this prefix. Anything
#: else the child (or a library it imports) prints is diagnostic tail, never
#: parsed — a mint must not be steerable by a stray print.
FRAME_PREFIX = "MINT_FRAME "

REQUEST_NAME = "request.json"
REPORT_NAME = "report.json"
#: pgw#848: the mint's phase table AS IT RUNS, rewritten atomically on every
#: beat. `REPORT_NAME` is written once, at a terminus the child reaches under
#: its own power — a child that is KILLED never writes one, and until this
#: existed a 29-minute abandoned mint reported `total_s=1741.33 — no cell
#: produced` with zero entry rows and no pool row. The measurements were all
#: made; nothing carried them out of the process.
PHASES_SNAPSHOT_NAME = "mint_phases.json"

# Exit codes. The child's exit code is a CLASSIFICATION, not just a failure
# bit: the retry policy below reads it, so a deterministic refusal is never
# retried and a resource shortfall is.
EXIT_MINTED = 0
EXIT_REFUSED = 2      # typed, deterministic (gate/toolchain/decl) — terminal
EXIT_RESOURCE = 3     # VRAM / host RAM shortfall — retry once, re-budgeted
EXIT_BAD_REQUEST = 4  # malformed request: a wiring defect, never retried

#: Terminal statuses a parent sees.
MINTED = "minted"
REFUSED = "refused"
RESOURCE = "resource"
CRASHED = "crashed"
ABANDONED = "abandoned"

#: How often the parent re-reads the child's measured evidence.
_OBSERVE_INTERVAL_S = 5.0

#: How long the child's MEASURED evidence (process-tree CPU + work-root bytes)
#: may fail to advance before the parent calls it wedged. Generous because it
#: bounds nothing that is working: an inductor phase that burns CPU advances
#: this every poll. It is not a mint budget — there is no mint budget.
_EVIDENCE_WINDOW_S = 300.0

#: Evidence advance (CPU-seconds + MiB written) that counts as progress.
_EVIDENCE_EPS = 0.05

#: Bytes of the child's stderr kept for the failure report.
_STDERR_TAIL_BYTES = 8192

#: How long to wait for the CORPSE after `_terminate_group` has already signaled
#: the child's process group. pgw#973 (§4.24 / gw#666): an explicitly EXEMPT
#: fixed duration, on `executor._CANCEL_UNWIND_GRACE_S`'s model — the kill
#: decision was made upstream on `_EVIDENCE_WINDOW_S`, which IS a progress
#: signal, so nothing here can end work that was advancing. What it bounds is
#: the parent's own control loop: a child ignoring SIGTERM must not hold the
#: serving process's teardown open. Expiry is not a failure — the reap future
#: outlives it (`shield`), and the exit status is simply not waited for.
_REAP_GRACE_S = 15.0

#: One retry, and only for the classes that can plausibly differ next time
#: (a resource shortfall the tenant's peak caused, a crash). A deterministic
#: refusal is terminal on the first attempt: re-running it burns a pod.
MAX_ATTEMPTS = 2


class CompileCellSpec(msgspec.Struct, frozen=True, kw_only=True):
    """The declared compile contract, flattened — exactly the facts the CHILD
    reads.

    The PARENT sends this rather than letting the child re-derive it from the
    decorator, because the class-scoped ``guidance_scales``/``text_lens``
    unions live on the spec and not on the decl: a child rebuilding from
    ``@endpoint`` alone would export a different declaration than the parent
    asked for. It is NOT a key-derivation wire — since pgw#758/#1010 the child
    computes no key at all; the parent stamps it from the returned envelope.

    pgw#1034 therefore dropped ``regional``/``text_len``/``dynamic``: they
    crossed the wire and no child consumer read them
    (``fleet_cells.aot_export_spec`` and ``compile_cache.resolve_targets`` read
    family/targets/shapes/text_lens/guidance/bucket, and nothing else does).
    Any field added back must name the child code that reads it.
    """

    shapes: Tuple[Tuple[int, ...], ...] = ()
    targets: Tuple[str, ...] = ()
    family: str = ""
    lora_bucket: int = 0
    guidance_scales: Tuple[float, ...] = ()
    text_lens: Tuple[int, ...] = ()


class MintSlot(msgspec.Struct, frozen=True, kw_only=True):
    """One setup slot, as the parent resolved it. Present and complete, or absent.

    pgw#974. This used to be THREE parallel slot-keyed dicts on
    ``MintRequest`` — ``snapshots`` (bytes), ``slot_bindings`` (identity,
    pgw#969) and ``component_paths`` (composition, pgw#816) — written by three
    separate statements, each independently allowed to be empty. Two of the
    three combinations that describes are incoherent, and one of them cost two
    L40S pods: ``{"pipeline": "/tmp/x"}`` with no binding decoded, type-checked
    and looked complete, and the child died 0.0 s into ``warmup_forward`` at
    ``ctx.slots["pipeline"]``. ``ref`` and ``path`` therefore carry no
    defaults: a slot with bytes and no identity cannot be constructed and
    cannot be decoded.

    A slot the parent did not resolve is ABSENT from the map — never a present
    entry with a hole in it. ``mint_child.assert_slots_resolvable`` still
    refuses one that the endpoint declares and does not mark optional.

    * ``ref`` — WHICH checkpoint. ``ctx.slots`` is built from bindings, and a
      child re-runs discovery, so a hub-catalog slot (``Slot(selected_by=...)``
      with no ``default_checkpoint=``, which is sdxl's shape) rediscovers
      nothing at all. A slot WITH a code default is the quieter half of the
      same defect: the child resolves the DECLARED checkpoint while the parent
      serves the hub's pick, and traces graphs for a model this pod never runs.
    * ``path`` — WHERE its bytes already are, materialized by the parent, so
      the child never touches the network: a mint is compute, and a mint
      process that could download is one that can stall on a lemon host.
    * ``component_paths`` — pgw#617 per-component overrides, comp -> that
      component's own local tree. Empty for most slots and part of the
      composition when not: th#1330 B2 EXCLUDES an overridden component's
      files from the base fetch, so ``path`` alone then names a narrowed tree
      (``<digest>__x<fp>``) that no loader can open.
    """

    ref: ModelRef
    path: str
    component_paths: Dict[str, str] = {}

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError(
                f"a resolved slot must name the tree its bytes are in; got an "
                f"empty path for {wire_ref(self.ref)!r}")


class MintRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the child needs, and nothing live."""

    function: str
    modules: Tuple[str, ...]
    family: str
    #: The parent's ArmIdentity token (``arm1-…``) — the obligation this
    #: child discharges. NOT a cell key (pgw#1059): the cell's key is
    #: stamped by the mint itself and returned in ``MintReport.cell_key``.
    arm_token: str
    target: str          # artifact the child writes (.tar.gz), atomically
    #: The child's work root (target, export tree, snapshots). The parent
    #: watches its BYTE GROWTH as one of the two progress signals — pgw#1010
    #: re-aimed it from the retired inductor capture dir, which an AOT mint
    #: never wrote a byte into, so that signal read 0 for every real mint.
    work_root: str
    report: str          # typed terminal report the child writes
    cfg: CompileCellSpec
    #: The parent's resolution, whole — slot -> identity + bytes + composition.
    #: See ``MintSlot``: the three views this replaced could disagree.
    slots: Dict[str, MintSlot] = {}
    device: int = -1     # CUDA ordinal; -1 = leave the child's default
    #: The child's device ceiling in bytes. 0 = NO CAP, which is the honest
    #: value for an unprobeable card (`mint_budget.co_residency` reports
    #: `probed=False`) and is stated rather than inferred: `mint_child.cap_vram`
    #: returns a "vram cap NOT applied" note the child frames, so an uncapped
    #: mint is distinguishable from a capped one in the phase table (pgw#973
    #: §4.24 item 4 — it used to return "" and record nothing at all).
    vram_cap_bytes: int = 0
    #: pgw#848: where the child rewrites its live phase table, so a mint the
    #: parent KILLS still leaves its measurements behind. Empty = no snapshot.
    phases_snapshot: str = ""
    #: pgw#848 item 5: the CROSS-ATTEMPT resume bank root (``aot_resume``).
    #: Deliberately NOT under the per-attempt ``child-<n>`` workdir — that is
    #: precisely why a crash at entry 30 of 36 used to discard ~5.2 h of
    #: compile. Empty = this mint does not resume and behaves as it did before.
    resume: str = ""
    #: pgw#848: one ENTRY child's measured host high-water, banked by the
    #: parent from a previous mint on this pod (``mint_budget.entry_peak_rss``).
    #: 0 = never measured here, and the pool's width falls back to its
    #: constant. It has to travel on the request because the width is computed
    #: INSIDE the mint child, whose memory dies with it — the same reason
    #: ``vram_cap_bytes`` is computed parent-side and handed down.
    entry_peak_rss_bytes: int = 0
    #: pgw#877: the DEVICE twin of the field above, and the fix for the defect
    #: that made the per-entry device ask a permanent estimate. ONE ENTRY
    #: child's measured device high-water, banked by the parent from a
    #: previous mint of this (family, lane).
    #:
    #: It has to travel HERE for exactly the reason the RSS figure does, and
    #: the asymmetry between the two was the proof: `mint_budget._CHILD_PEAKS`
    #: is written only in the SERVING PARENT, while the width is computed
    #: INSIDE the mint child, where that dict is empty by construction. A bank
    #: read in a process that can never have written it is not a bank.
    #: 0 = never measured here, and the ask falls back to the estimate.
    entry_device_peak_bytes: int = 0
    #: The hub-resolved execution lane (``ctx.lane``) and the effective
    #: declared-parameter values per function (th#1087). Both STEER the warm
    #: forwards, so both must be the parent's values — a child warming at
    #: different config traces different graphs and the parent's proof misses.
    execution_lane: str = ""
    configs: Dict[str, Dict[str, Any]] = {}


class MintFrame(msgspec.Struct, frozen=True, kw_only=True):
    """One progress frame. Reporting only — never a liveness claim."""

    phase: str = ""
    step: int = 0
    total: int = 0
    note: str = ""


class MintReport(msgspec.Struct, frozen=True, kw_only=True):
    """The child's typed terminal word, written to ``MintRequest.report``.

    Written even on refusal/resource exits: the reason a mint did not happen
    is the load-bearing part of a failed mint, and a pod exposes no logs.
    """

    status: str
    artifact: str = ""
    digest: str = ""
    cell_key: str = ""
    detail: str = ""
    phase: str = ""
    #: Measured, so the NEXT mint's co-residency ask is a fact rather than
    #: an estimate (§1: no magic numbers). Read by
    #: ``mint_delegate.record_child_peak``.
    #:
    #: pgw#877: there was a `peak_rss_bytes` beside it, written from
    #: `getrusage(SELF)+getrusage(CHILDREN)` on both minted termini and read by
    #: NOTHING — the parent banks the host high-water from
    #: `mint_phases["pool"]["peak_child_rss_bytes"]`, which is a per-entry
    #: VmHWM tree sum and the number `entry_workers` actually divides by.
    peak_vram_bytes: int = 0
    elapsed_s: float = 0.0
    #: th#1322: per-phase seconds, measured by the CHILD (`load`,
    #: `warmup_forward`, `inductor_compile`, `seal_publish`, `finalize`). The
    #: parent turns these into `jit_compile` events so a JIT mint's cost is a
    #: numeric hub record with the same shape as an AOT mint's — the child owns
    #: the clock, so deriving spans from the parent's frame receipts would
    #: measure pipe latency instead of work. Empty from a child that died before
    #: writing its report, which is honest: no measurement, not zero.
    phases: Dict[str, float] = msgspec.field(default_factory=dict)
    #: pgw#1080 — the FAKE-EXPORT FOOTPRINT. The child builds its compile
    #: targets from code + config, so the process that traces holds no
    #: checkpoint values; these fields are how a reader knows that held, for
    #: WHICH components, and what it cost instead.
    #:
    #: ``export_peak_bytes`` is the number the pgw#992 census must size the
    #: next export child against: it is measured from a peak reset taken AFTER
    #: the warm proof, so weight-scale random values cannot leak into it.
    #: ``warm_proof_peak_bytes`` is the pgw#984 proof's own window, and
    #: ``warm_proof_values`` records what it ran on (``random``) — a
    #: numerics-adjacent surprise is then one lookup, not a re-derivation.
    structure_only_components: Tuple[str, ...] = ()
    #: Empty when the property held; the typed reason when a component could
    #: not be built from config and that slot loaded its weights instead.
    structure_refusal: str = ""
    #: Checkpoint bytes the child would have held and did not.
    virtual_param_bytes: int = 0
    #: Config-derived tables it legitimately did hold (literals live here).
    structure_real_bytes: int = 0
    warm_proof_peak_bytes: int = 0
    warm_proof_values: str = ""
    export_peak_bytes: int = 0
    #: pgw#805: the AOT recipe's per-entry phase TABLE
    #: (``aot_mint._mint_phase_table``). The child emits it too, but a mint
    #: child holds no orchestrator session, so the child's events go nowhere —
    #: the parent re-emits from this, exactly as it does for `jit_compile`.
    mint_phases: Dict[str, Any] = msgspec.field(default_factory=dict)


@dataclass(frozen=True)
class MintOutcome:
    """What the parent learned. A mint outcome is never a worker outcome."""

    status: str
    detail: str = ""
    artifact: Optional[Path] = None
    report: Optional[MintReport] = None
    exit_code: Optional[int] = None
    stderr_tail: str = ""
    last_phase: str = ""
    elapsed_s: float = 0.0
    #: pgw#848: the child's live phase table, recovered from disk when the
    #: child died without writing a report. Deliberately a SEPARATE field
    #: rather than a synthesized ``report``: ``retryable`` branches on
    #: ``report is None``, and inventing a report to carry telemetry would
    #: silently change a retry decision.
    partial_phases: Dict[str, Any] = field(default_factory=dict)

    @property
    def minted(self) -> bool:
        return self.status == MINTED

    @property
    def retryable(self) -> bool:
        """A retry may plausibly differ. A refusal never will.

        ``RESOURCE``: the shortfall may have been the tenant's peak, which
        has since passed — retry, but only after re-budgeting (the caller's
        job; see ``mint_budget``). ``CRASHED``: an UNCLASSIFIED death, worth
        exactly one more attempt. ``ABANDONED`` is not a failure at all, and
        ``REFUSED`` is deterministic — re-running it buys a second billed
        compile for the same named refusal.

        pgw#816: a crash the child CLASSIFIED for itself is deterministic too.
        A ``status="failed"`` report means the child caught the exception,
        named it, and wrote it down — from the same request file against the
        same on-disk inputs, so attempt 2 re-runs the identical failure (the
        first delegated mint in production burned two 8.5 s attempts on one
        load error this way). Only a death the child could NOT classify — no
        report at all: signal, OOM-killer, the parent's stall kill, a spawn
        failure — can plausibly differ next time. Resource shortfalls exit
        ``EXIT_RESOURCE`` and never reach here.
        """
        if self.status == RESOURCE:
            return True
        if self.status != CRASHED:
            return False
        return self.report is None or self.report.status != "failed"

    def line(self) -> str:
        """The one structured line this outcome logs and puts on the wire."""
        parts = [f"mint_process status={self.status}"]
        if self.exit_code is not None:
            parts.append(f"exit={self.exit_code}")
        if self.last_phase:
            parts.append(f"phase={self.last_phase}")
        parts.append(f"elapsed={self.elapsed_s:.0f}s")
        if self.report is not None and self.report.peak_vram_bytes:
            parts.append(
                f"child_peak_vram={self.report.peak_vram_bytes / (1 << 30):.2f}GiB")
        # pgw#1080: the fake-export footprint travels on the SAME line the hub
        # already reads, so "the child held no weights" is an observation
        # anyone can make from the wire instead of a claim in a design doc.
        if self.report is not None and self.report.structure_only_components:
            parts.append(
                "structure_only="
                + ",".join(self.report.structure_only_components))
            parts.append(
                f"virtual_weights={self.report.virtual_param_bytes / (1 << 30):.2f}GiB")
            parts.append(
                f"export_peak={self.report.export_peak_bytes / (1 << 20):.1f}MiB")
            if self.report.warm_proof_values:
                parts.append(
                    f"warm_proof_values={self.report.warm_proof_values} "
                    f"peak={self.report.warm_proof_peak_bytes / (1 << 20):.1f}MiB")
        if self.report is not None and self.report.structure_refusal:
            parts.append(
                f"structure_refusal={self.report.structure_refusal[:200]!r}")
        if self.detail:
            parts.append(f"detail={self.detail[:400]!r}")
        return " ".join(parts)


def frame_line(
    phase: str = "", step: int = 0, total: int = 0, note: str = "",
) -> str:
    """One wire line for a progress frame — the child's half of the protocol,
    kept here so both sides share one encoder."""
    body = msgspec.json.encode(
        MintFrame(phase=phase, step=step, total=total, note=note[:400]))
    return FRAME_PREFIX + body.decode() + "\n"


def _decode_report(path: Path) -> Optional[MintReport]:
    try:
        return msgspec.json.decode(path.read_bytes(), type=MintReport)
    except Exception:
        return None


def _tree_cpu_seconds(pid: int) -> Optional[float]:
    """The child's process-tree CPU seconds — INCLUDING descendants that have
    already exited.

    Inductor forks its own compile workers and a ``g++`` under each of those,
    so the tree — not the child alone — is what burns the CPU a mint is made
    of. The subtle half is the word *already*: on Linux a process's CPU leaves
    its own ``utime/stime`` and lands in its parent's ``cutime/cstime`` the
    instant the parent reaps it (recursively — a reaped subtree rolls all the
    way up). Summing only the live members' ``user + system`` therefore makes
    this quantity a SAWTOOTH: an entry compile child that FINISHES subtracts
    its entire lifetime CPU from the total.

    That is not a cosmetic wobble. ``_observe`` ratchets against a high-water
    mark, so a finishing entry digs a hole one whole entry deep, and the mint
    is killed for the crime of completing something (pgw#964; it killed
    pgw#868 attempt eighteen twice, byte-identically, on two independent pods).
    Adding the reaped-children counters makes the total monotonic for as long
    as ``pid`` lives, which is exactly as long as it is consulted.

    Returns ``None`` when the tree could not be sampled at all. A failure is
    NOT zero: reporting zero would manufacture the same cliff from a transient
    ``/proc`` race.
    """
    try:
        import psutil
    except Exception:  # pragma: no cover - psutil is a hard dep in practice
        return None
    try:
        proc = psutil.Process(pid)
        members = [proc] + proc.children(recursive=True)
    except Exception:
        return None
    total = 0.0
    for member in members:
        try:
            times = member.cpu_times()
        except Exception:
            continue
        # A member's own time, plus the time of every descendant IT has
        # already waited for. No double count: a process's CPU is in exactly
        # one of the two places, never both.
        total += float(times.user) + float(times.system)
        total += float(getattr(times, "children_user", 0.0) or 0.0)
        total += float(getattr(times, "children_system", 0.0) or 0.0)
    return total


def _work_root_mib(work_root: Path) -> Optional[float]:
    """MiB the child has written into its work root.

    Generated sources, compiled objects and the packed cell — not a stdio
    buffer. It grows in BURSTS: sources land, then a single-threaded ``g++``
    chews on them for minutes writing nothing. So its growth proves work and
    its silence proves nothing, which is why ``_observe`` treats it as an
    independent positive signal that can never, on its own, vote to kill.
    """
    total = 0
    try:
        for path in work_root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return None
    return total / (1 << 20)


#: The measured signals, sampled together. Deliberately a PAIR and never a
#: sum: they have different failure modes and different units, and adding them
#: lets a fall in one cancel a rise in the other. That is precisely how a
#: healthy mint died — a reaped entry's CPU drop swallowed the work-root bytes
#: its successor was writing.
def _evidence(pid: int, work_root: Path) -> Tuple[Optional[float], Optional[float]]:
    return _tree_cpu_seconds(pid), _work_root_mib(work_root)


def child_argv(request_path: Path, *, python: str = "") -> Sequence[str]:
    return [python or sys.executable, "-m", MINT_CHILD_MODULE, str(request_path)]


def _read_phase_snapshot(path: str) -> Dict[str, Any]:
    """The child's live phase table, or ``{}`` when there is none.

    Never raises: a mint's outcome must not depend on whether its telemetry
    file parsed.
    """
    if not path:
        return {}
    try:
        table = msgspec.json.decode(Path(path).read_bytes())
    except (OSError, msgspec.DecodeError, ValueError):
        return {}
    return dict(table) if isinstance(table, dict) else {}


def write_request(workdir: Path, request: MintRequest) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / REQUEST_NAME
    path.write_bytes(msgspec.json.encode(request))
    return path


def child_env(
    request: MintRequest, *, base: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The child's environment.

    The seal is re-established by the child from this env, so it must be the
    parent's env — a child that sealed differently would stamp a cell the
    parent's ``verify()`` rejects. Two deliberate additions:

    * ``CUDA_VISIBLE_DEVICES`` pins the child to the mint device, so a
      library that reads ``cuda:0`` cannot wander onto a card serving
      another group;
    * ``GEN_WORKER_MINT_CHILD=1`` is how any code that must not run twice on
      one pod (publishers, boot records, hub transports) can tell.
    """
    env = dict(os.environ if base is None else base)
    env["GEN_WORKER_MINT_CHILD"] = "1"
    if request.device >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(request.device)
    return env


async def _pump_frames(
    reader: Optional[asyncio.StreamReader],
    on_frame: Optional[Callable[[MintFrame], None]],
    state: Dict[str, Any],
) -> None:
    if reader is None:
        return
    while True:
        raw = await reader.readline()
        if not raw:
            return
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if not line.startswith(FRAME_PREFIX):
            logger.debug("mint child: %s", line[:500])
            continue
        try:
            frame = msgspec.json.decode(
                line[len(FRAME_PREFIX):].encode(), type=MintFrame)
        except Exception:
            logger.debug("mint child: unparsable frame %s", line[:500])
            continue
        if frame.phase:
            state["phase"] = frame.phase
        if on_frame is not None:
            try:
                on_frame(frame)
            except Exception:
                logger.exception("mint frame callback failed")


async def _pump_stderr(
    reader: Optional[asyncio.StreamReader], state: Dict[str, Any],
) -> None:
    if reader is None:
        return
    tail = bytearray()
    while True:
        chunk = await reader.read(4096)
        if not chunk:
            state["stderr"] = tail.decode("utf-8", "replace")
            return
        tail.extend(chunk)
        if len(tail) > _STDERR_TAIL_BYTES:
            del tail[:-_STDERR_TAIL_BYTES]
        state["stderr"] = tail.decode("utf-8", "replace")


async def _observe(
    pid: int,
    work_root: Path,
    window: SilenceWindow,
    on_evidence: Optional[Callable[[float], None]],
    interval_s: float,
) -> str:
    """Watch the child's MEASURED progress. Returns the stall reason, or ''
    when cancelled because the child exited.

    Each signal keeps its OWN high-water mark and an advance in EITHER is
    progress (pgw#964). Two properties fall out, and both are load-bearing:

    * a signal that cannot be sampled this poll contributes nothing rather
      than a cliff, and
    * a signal that is merely quiet — work-root bytes during a long ``g++`` —
      cannot cancel one that is moving. Only the absence of BOTH advances is
      silence, which is the only thing this window is entitled to judge.
    """
    cpu, mib = _evidence(pid, work_root)
    window.touch()
    while True:
        await asyncio.sleep(interval_s)
        now_cpu, now_mib = _evidence(pid, work_root)
        advanced = False
        if now_cpu is not None and (cpu is None or now_cpu - cpu >= _EVIDENCE_EPS):
            cpu, advanced = now_cpu, True
        if now_mib is not None and (mib is None or now_mib - mib >= _EVIDENCE_EPS):
            mib, advanced = now_mib, True
        if advanced:
            window.touch()
            if on_evidence is not None:
                try:
                    on_evidence((cpu or 0.0) + (mib or 0.0))
                except Exception:
                    logger.exception("mint evidence callback failed")
        elif window.stalled():
            return (
                f"the mint process made no measured progress for "
                f"{window.silent_for():.0f}s (window {window.window_s:.0f}s): "
                f"process-tree CPU {'unreadable' if now_cpu is None else f'{now_cpu:.1f}s'} "
                f"and work-root "
                f"{'unreadable' if now_mib is None else f'{now_mib:.1f}MiB'} "
                f"both flat")


def _terminate_group(pid: int, *, grace_s: float = 10.0) -> None:
    """SIGTERM the child's process GROUP, then SIGKILL.

    The group, not the process: inductor's compile workers are children of
    the child, and a mint that left them behind would keep burning the pod's
    CPU against a cell nobody will adopt.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        deadline = time.monotonic() + (grace_s if sig is signal.SIGTERM else 5.0)
        while time.monotonic() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG) != (0, 0):
                    return
            except ChildProcessError:
                return
            except OSError:
                break
            time.sleep(0.05)


async def run_mint(
    request: MintRequest,
    *,
    workdir: Path,
    on_frame: Optional[Callable[[MintFrame], None]] = None,
    on_evidence: Optional[Callable[[float], None]] = None,
    abandon: Optional[asyncio.Event] = None,
    evidence_window_s: float = _EVIDENCE_WINDOW_S,
    observe_interval_s: float = _OBSERVE_INTERVAL_S,
    python: str = "",
    env: Optional[Mapping[str, str]] = None,
) -> MintOutcome:
    """Build one cell in a child process. Never raises for a failed mint.

    A failed mint is an OUTCOME, not an exception, because the caller's
    correct response is identical in every branch: keep serving eager, report
    the failure, and leave the cell absent. The one thing this function must
    never do is take the worker with it — so the only exception it propagates
    is ``asyncio.CancelledError`` (the parent's own shutdown), and even that
    reaps the child's process group on the way out.
    """
    request_path = write_request(workdir, request)
    report_path = Path(request.report)
    work_root = Path(request.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"phase": "", "stderr": ""}
    started = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        *child_argv(request_path, python=python),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(child_env(request) if env is None else env),
        start_new_session=True,  # own process group -> group-wide reaping
    )
    logger.info(
        "mint_process: spawned pid=%s family=%s key=%s device=%s cap=%s",
        proc.pid, request.family, request.arm_token[:12] or "-",
        request.device, request.vram_cap_bytes or "none")

    frames = asyncio.ensure_future(_pump_frames(proc.stdout, on_frame, state))
    errs = asyncio.ensure_future(_pump_stderr(proc.stderr, state))
    watch = asyncio.ensure_future(_observe(
        proc.pid, work_root,
        SilenceWindow(evidence_window_s), on_evidence, observe_interval_s))
    reap = asyncio.ensure_future(proc.wait())
    stop = (
        asyncio.ensure_future(abandon.wait()) if abandon is not None
        else None)
    waits = [reap, watch] + ([stop] if stop is not None else [])

    killed_reason = ""
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        if not reap.done():
            if watch.done() and watch.result():
                killed_reason = watch.result()
            elif stop is not None and stop.done():
                killed_reason = ABANDONED
            await asyncio.to_thread(_terminate_group, proc.pid)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(reap), timeout=_REAP_GRACE_S)
    except asyncio.CancelledError:
        await asyncio.to_thread(_terminate_group, proc.pid)
        raise
    finally:
        for fut in (frames, errs, watch, reap, stop):
            if fut is not None and not fut.done():
                fut.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(frames, errs, watch, return_exceptions=True)

    code = reap.result() if reap.done() and not reap.cancelled() else None
    elapsed = time.monotonic() - started
    report = _decode_report(report_path)
    stderr_tail = str(state.get("stderr") or "")[-_STDERR_TAIL_BYTES:]
    phase = str(state.get("phase") or (report.phase if report else ""))
    # pgw#848: a child the parent KILLED wrote no report, and everything it
    # measured is on disk in the snapshot. Read unconditionally — a report and
    # a snapshot never disagree, and when both exist the report wins because
    # the child reached its own terminus.
    partial = _read_phase_snapshot(request.phases_snapshot)

    def _out(status: str, detail: str, artifact: Optional[Path] = None) -> MintOutcome:
        outcome = MintOutcome(
            status=status, detail=detail, artifact=artifact, report=report,
            exit_code=code, stderr_tail=stderr_tail, last_phase=phase,
            elapsed_s=elapsed, partial_phases=partial)
        logger.info("%s", outcome.line())
        return outcome

    if killed_reason == ABANDONED:
        return _out(ABANDONED, "the parent abandoned this mint")
    if killed_reason:
        return _out(CRASHED, killed_reason)

    if code == EXIT_MINTED:
        artifact = Path(report.artifact) if report and report.artifact else None
        if artifact is None or not artifact.is_file():
            return _out(
                CRASHED,
                "the mint process exited 0 but wrote no artifact "
                f"(report={'present' if report else 'absent'})")
        return _out(MINTED, report.detail if report else "", artifact)
    if code == EXIT_REFUSED:
        return _out(REFUSED, (report.detail if report else "") or stderr_tail[-400:])
    if code == EXIT_RESOURCE:
        return _out(RESOURCE, (report.detail if report else "") or stderr_tail[-400:])
    if code == EXIT_BAD_REQUEST:
        # A wiring defect on OUR side. Loud and terminal: a retry would just
        # re-send the same malformed request.
        return _out(
            REFUSED,
            "the mint request was rejected as malformed (worker bug): "
            + ((report.detail if report else "") or stderr_tail[-400:]))
    if code is not None and code < 0:
        return _out(
            CRASHED,
            f"the mint process died on signal {-code} "
            f"({signal.Signals(-code).name if -code in set(int(s) for s in signal.Signals) else '?'})")
    return _out(CRASHED, f"the mint process exited {code}: {stderr_tail[-400:]}")


__all__ = [
    "ABANDONED",
    "CRASHED",
    "CompileCellSpec",
    "EXIT_BAD_REQUEST",
    "EXIT_MINTED",
    "EXIT_REFUSED",
    "EXIT_RESOURCE",
    "FRAME_PREFIX",
    "MAX_ATTEMPTS",
    "PHASES_SNAPSHOT_NAME",
    "MINTED",
    "MINT_CHILD_MODULE",
    "MintFrame",
    "MintOutcome",
    "MintReport",
    "MintRequest",
    "REFUSED",
    "REPORT_NAME",
    "REQUEST_NAME",
    "RESOURCE",
    "child_argv",
    "frame_line",
    "child_env",
    "run_mint",
    "write_request",
]
