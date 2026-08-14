"""pgw#809: compile a cell's entries K-wide instead of one at a time.

A pgw#758 cell is N independent graph-class entries; an sdxl cell is 18.
``aot_mint`` exports them from the live pipeline (serial by construction —
one pipeline, one card, and the branch arm is toggled once for the whole
branchless group) and then AOTI-compiles each one at ~420 s. The compiles
share nothing: each is ``aot_compile(ep.module(), ...)`` over its own
ExportedProgram, producing its own loose files, combined afterwards by
``package_aoti``. Serially that is ~2 h; K-wide it is ``ceil(N/K)`` x
per-entry.

Why processes, and not threads
------------------------------
MEASURED, not assumed: four concurrent ``aot_compile`` calls in ONE process
(torch 2.13.0+cu130) produced ONE usable result and three distinct internal
failures — ``AssertionError: CURRENT_PATCHER is None in finally block``,
``KeyError: 'custom'`` in ``fx.traceback.annotate``, and a fake-tensor
propagation crash. Inductor's compile path keeps process-global mutable
state (``torch.fx.traceback`` current-meta stack, dynamo's patcher). A
thread pool here is not slower, it is WRONG, and it fails nondeterministically
— the worst shape. So the unit of parallelism is an OS process, which is also
what pgw#784 already established for the mint itself.

The handoff — and why there is no longer a program in it (pgw#1215/pgw#1216)
---------------------------------------------------------------------------
There used to be one. A child cannot inherit an exported program (``fork`` is
banned after CUDA init, pgw#784), so it arrived on disk: ``torch.export.save``
in the parent, ``torch.export.load`` in the child. That pair was defended as
affordable — byte-exact, and off the critical path because saves overlap
compiles. **The defence priced the wrong half.** The child's ``load`` is not
overlapped with anything: it is the first 36.04 s (median, P0-E §5c) of every
child's serial life, ~22 min of a 36-class sdxl mint.

So the program does not cross. The child receives the SHARE — which declared
graph classes are its — plus the four facts it needs to build the weight-free
pipeline itself (``function``, ``modules``, ``slots``, ``cfg``, exactly what
``boot_key.TraceJob`` carries), traces its rows with
``aot_mint.trace_for_key(compile_now=True)``, and packs each one with
``aot_mint.pack_graph_classes``. One address space from trace to artifact.

The gates travel with the program, because they always ran beside it: every
package-side gate (``program_package_drift``, ``eliminated_constants``,
``input_contract``) is inside ``pack_graph_classes``, so it now runs in the
child, against the program the child itself traced. That is strictly tighter
than the old split — the parent used to gate ITS program against the CHILD's
package, which could only ever catch divergence, and divergence is exactly
what stops being possible when there is one program.

What this does NOT change
-------------------------
Cell identity. Parallelism is not sealed (pgw#757 established
``compile_threads`` as non-identity by the same argument, and the digest check
is re-run here): the pool changes WHEN entries compile, never what they
compile. Assembly is ordered by ENTRY NAME, not completion, so a cell minted
at K=4 is byte-identical to one minted at K=1.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple)

import msgspec

from . import aot_compile_spans, aot_device_lock, aot_resume, env_seal
from . import compile_posture, kernel_path
from .child_contract import CompileSpec, MintSlot
from .compile_posture import (
    USER_MACHINE_RSS_RESERVE_BYTES, CompilePosture)
from .postmortem import cpu_quota_cores
from .stall import SilenceWindow
import hashlib

logger = logging.getLogger(__name__)

ENTRY_CHILD_MODULE = "gen_worker.aot_compile_child"

#: Report file each entry child writes before exiting.
ENTRY_REPORT_NAME = "report.json"

#: pgw#840: the directory that must be on the child's path for
#: ``-m gen_worker.aot_compile_child`` to mean THIS gen_worker.
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)

#: The three modules that define the parent/child contract: the child's own
#: entrypoint, this module (the job/report structs) and the span partition.
_CONTRACT_MODULES = (
    "aot_compile_child.py", "aot_compile_pool.py", "aot_compile_spans.py")


def _code_digest() -> str:
    """A digest of the parent/child contract source, taken AT IMPORT.

    pgw#840: the pool spawns ``sys.executable -m gen_worker.aot_compile_child``
    and lets the child's import system decide which ``gen_worker`` that is. It
    can legitimately be a different one — a ``PYTHONPATH`` entry, a ``gen_worker``
    in the cwd, a second checkout, a stale wheel in the interpreter's
    site-packages, or the same tree edited between the parent's import and the
    child's spawn. The child then compiles the very files the cell publishes
    with code the parent never ran, and the parent believes the report.

    Taken at import (not at read time) on purpose: a tree edited mid-run must
    compare the code each process is actually EXECUTING, not what its files say
    afterwards.
    """

    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _CONTRACT_MODULES:
        try:
            digest.update(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:  # zipimport / frozen: no source to compare
            return ""
    return digest.hexdigest()[:16]


#: Computed once per process, so parent and child each carry their OWN.
CODE_DIGEST = _code_digest()

# ---------------------------------------------------------------------------
# Width policy (pgw#809 constraint 1: NEVER os.cpu_count, NEVER a hardcoded K)
# ---------------------------------------------------------------------------

#: Cores the pool must leave to the serving process. A mint runs on a SERVING
#: pod under the pgw#784 contract ("the worker is available the entire time"),
#: so the eager forward, the 10 s heartbeat and the transport all have to keep
#: their share. Two is the measured floor for a worker that is serving: one
#: for the asyncio loop carrying the beat, one for the eager forward's host
#: side. Below that the pool is not allowed to run at all (K falls to 1, which
#: is the in-process serial path and adds nothing).
SERVING_HEADROOM_CPUS = 2

#: Cores one entry child occupies ON AVERAGE over its life. NOT
#: ``compile_threads``: pgw#793 measured that an AOTI compile is dominated by
#: two strictly SINGLE-THREADED phases — inductor's Python codegen (25 %) and
#: the one g++ invocation on the wrapper TU (46 %) — with ``compile_threads``
#: parallelism confined to the Triton kernel compiles in between. So the
#: instantaneous ask is 1 core for ~71 % of the entry and up to
#: ``compile_threads`` for the rest; 2 is that average, rounded up.
#: ``K * compile_threads`` is deliberately ALLOWED to exceed the budget —
#: clamping on the peak would size the pool for a phase that is a quarter of
#: the work and leave the box idle through the other three quarters.
CPUS_PER_ENTRY_WORKER = 2

#: Hard ceiling regardless of how fat the pod is.
#:
#: §4.24, re-derived pgw#1035. The previous defense was twenty lines pricing
#: this against REGIONAL mints — a kind pgw#846 retired — and it turned on
#: ``aot_mint._block_device_fraction``, a function that no longer exists. A
#: constant defended by a deleted function and a retired feature is undefended.
#:
#: THE THREAT this bounds, named: K entry children compile CONCURRENTLY into
#: ONE shared inductor cache directory, each linking its own wrapper TU. Past
#: some K the contention is not the compile — it is the cache directory's
#: write amplification, the page cache thrashing between N saved programs, and
#: N concurrent ``cc1plus`` link steps on a box whose disk the serving process
#: also uses. Nothing here has ever been measured failing at 8; what HAS been
#: measured is the other side of the trade, and it is what fixes the number.
#:
#: WHY 8 AND NOT MORE: K only buys whole ROUNDS. sdxl's 18 entries take
#: ceil(18/8) = 3 rounds; K=6 also takes 3. The next round-boundary win for 18
#: entries is K=9, and the serial terms K never touches (export, package,
#: pack) already dominate what that would save. 8 is one step above the point
#: where rounds stop improving for the largest real family.
#:
#: WHAT WOULD FALSIFY IT: a family whose entry count makes ceil(N/K) strictly
#: decrease past K=8 AND whose per-entry compile still dominates the serial
#: terms. Measure ``mint_phases`` on that family before raising this; do not
#: raise it because a pod looks idle.
MAX_ENTRY_WORKERS = 8

#: Host RAM the pool must leave alone: the serving process's own resident set
#: is already counted (we read AVAILABLE, not total), this is the margin on
#: top so that a tenant request arriving mid-mint does not meet an OOM killer.
ENTRY_RSS_RESERVE_BYTES = 4 * 1024**3

#: Per-entry peak RSS assumed before this (family, lane) has banked one.
#:
#: §4.24, re-derived pgw#1035 against the row it was accused of contradicting.
#: THE THREAT: K entry children whose summed peak RSS exceeds available host
#: RAM, and the OOM killer takes the SERVING process mid-request — the mint's
#: whole premise is that the worker keeps serving (pgw#784).
#:
#: WHY THIS IS NOT pgw#877 #5's REFUSE-TO-WIDEN CASE, which deleted the DEVICE
#: twin of this constant. Two differences, both load-bearing:
#:
#: * That branch fires when the per-entry device footprint is genuinely
#:   UNKNOWN — no number, from anywhere, for the entry about to run. This one
#:   is a MEASURED number: 2.09 GiB, on the real sdxl wrapper TU (codegen holds
#:   the generated source plus inductor's IR; ``cc1plus`` on the wrapper TU is
#:   the peak), carried at a 1.43x margin. "Not yet banked for this
#:   (family, lane)" is not "unmeasured".
#: * Overshooting the card is a hard CUDA OOM with no soft landing.
#:   Overshooting host RAM lands first in ``ENTRY_RSS_RESERVE_BYTES`` (4 GiB,
#:   deliberately larger than one entry) and then in reclaim, which is why host
#:   RAM has been the LOOSEST of the three bounds on every pod measured.
#:
#: Refusing to widen here would therefore not buy safety; it would pin K=1 on
#: every cold pod's first mint, which is every mint of a new (family, lane).
#:
#: WHAT WOULD FALSIFY IT: an entry child measured above 3 GiB peak RSS, or any
#: mint-time host OOM — the answer then is to make the FIRST entry serial and
#: bank its peak, not to guess a larger constant.
#:
#: Banked per (family, lane) once measured (``mint_workers.compiled_graph_peak_rss``).
#: pgw#1175: this is now the ONLY per-entry footprint K divides by.
DEFAULT_ENTRY_PEAK_RSS_BYTES = 3 * 1024**3

_KILL_GRACE_S = 10.0
_POLL_S = 0.25

#: pgw#1243: how long ONE compile child's measured evidence — its process
#: tree's CPU seconds plus the bytes it has written — may fail to advance
#: before the pool calls it wedged. A SILENCE window, never a compile budget:
#: an entry that spends forty minutes inside one `aot_compile` advances this
#: every poll and is never touched. It exists because the drain loop below had
#: no give-up test of ANY kind — it polled `proc.poll()` forever — and the
#: three-tier stack's own window (which used to cover this whole tree from the
#: mint child) went away with the middle tier.
_ENTRY_SILENCE_WINDOW_S = 300.0

#: Evidence advance (CPU-seconds + MiB written) that counts as progress.
_ENTRY_EVIDENCE_EPS = 0.05


@dataclass(frozen=True)
class CpuFacts:
    """What the pod's cores actually are, and which reading said so.

    pgw#842: a provider ADVERTISES vCPUs (RunPod's ``host_vcpus``) and the
    kernel ENFORCES a quota, and the two are routinely different numbers. The
    pool must size on the enforced one — and must SAY which one it read, or a
    K that came out narrow than the advertisement is indistinguishable from a
    bug in the formula.
    """

    vcpus: int
    basis: str
    os_cpu_count: int
    affinity_cpus: int
    quota_cores: float

    def facts(self) -> Dict[str, Any]:
        return {
            "vcpus": int(self.vcpus),
            "cpu_basis": self.basis,
            "os_cpu_count": int(self.os_cpu_count),
            "affinity_cpus": int(self.affinity_cpus),
            "quota_cores": round(float(self.quota_cores), 2),
        }


@dataclass(frozen=True)
class MemoryFacts:
    """Host RAM the pool may take, and the reading that bounded it."""

    available_bytes: int
    basis: str
    host_available_bytes: int
    cgroup_available_bytes: int
    cgroup_reclaimable_bytes: int

    def facts(self) -> Dict[str, Any]:
        return {
            "available_bytes": int(self.available_bytes),
            "mem_basis": self.basis,
            "host_available_bytes": int(self.host_available_bytes),
            "cgroup_available_bytes": int(self.cgroup_available_bytes),
            "cgroup_reclaimable_bytes": int(self.cgroup_reclaimable_bytes),
        }


@dataclass(frozen=True)
class PoolWidth:
    """The chosen K and every input that chose it — so a mint's telemetry can
    answer "why this width" without re-deriving anything."""

    workers: int
    entries: int
    vcpus: int
    cpu_workers: int
    mem_workers: int
    available_bytes: int
    per_entry_rss_bytes: int
    device_lock: bool
    reason: str
    #: pgw#842: the constraint that ACTUALLY held K down, by name, plus the
    #: readings each bound was taken from. A width narrower than the pod could
    #: carry is a performance defect, and it has to be legible from one record
    #: rather than inferred by diffing two pods that no longer exist.
    binding: str = ""
    ceiling: int = MAX_ENTRY_WORKERS
    #: The caller's own cap (``entry_workers(limit=)``), 0 when uncapped. An
    #: INPUT that chose K, carried so an operator who forced the serial path
    #: can see from the record that they did.
    limit: int = 0
    cpu: Optional[CpuFacts] = None
    memory: Optional[MemoryFacts] = None
    #: ``"measured"`` here is literal: the value is one entry child's VmHWM
    #: summed over its real descendant tree (``_peak_rss_bytes``), banked by
    #: the serving parent (``mint_workers.record_compiled_graph_peak_rss``).
    per_entry_rss_basis: str = "default"
    #: §4.30 / pgw#1137: whose MACHINE this is. Distinct from the goals above,
    #: which say what the pod was bought to do — a K held down for a human at
    #: a keyboard and a K held down for a tenant are different decisions and a
    #: reader must be able to tell them apart. ``FLEET`` on every pod.
    posture: CompilePosture = compile_posture.FLEET

    @property
    def underwidth(self) -> int:
        """Workers the pod would have run had the binding constraint not
        bound — 0 when K is already the most this cell can use."""
        return max(0, min(self.entries, self.ceiling) - self.workers)

    def facts(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "entry_workers": int(self.workers),
            "entries": int(self.entries),
            "vcpus": int(self.vcpus),
            "cpu_workers": int(self.cpu_workers),
            "mem_workers": int(self.mem_workers),
            "available_bytes": int(self.available_bytes),
            "per_entry_rss_bytes": int(self.per_entry_rss_bytes),
            "device_lock": bool(self.device_lock),
            "binding": self.binding,
            "ceiling": int(self.ceiling),
            "limit": int(self.limit),
            "underwidth": int(self.underwidth),
            "per_entry_rss_basis": self.per_entry_rss_basis,
            "width_reason": self.reason,
            **self.posture.facts(),
        }
        for block in (self.cpu, self.memory):
            if block is not None:
                out.update(block.facts())
        return out


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # A cgroup v1 "unlimited" is a huge sentinel, not a limit.
    return value if 0 <= value < (1 << 62) else None


def _cgroup_reclaimable_bytes(stat: Path) -> int:
    """File pages this cgroup is charged for that the kernel will hand back
    under pressure — page cache, and reclaimable slab.

    pgw#842: ``memory.current`` counts page cache. A mint reads GBs (weights,
    the toolchain the seal hashes, every staged program) and every one of
    those pages inflates ``current`` until something needs the memory. Sizing
    a pool on ``max - current`` therefore shrinks the pool in proportion to
    how much I/O the pod has already done — a bound that moves with history
    instead of with the box. Subtracting what is reclaimable is the same
    working-set definition every container runtime uses.
    """
    total = 0
    try:
        lines = stat.read_text().splitlines()
    except OSError:
        return 0
    wanted = {
        "inactive_file", "slab_reclaimable",          # cgroup v2
        "total_inactive_file", "total_slab_reclaimable",  # cgroup v1
    }
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in wanted:
            try:
                total += int(parts[1])
            except ValueError:
                continue
    return total


def memory_facts(
    *,
    meminfo: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> MemoryFacts:
    """Host RAM this process may actually take, cgroup-aware.

    ``MemAvailable`` is the host's answer and a container's limit is not; the
    narrower of the two is the only honest one — the same rule
    ``effective_cpu_count`` applies to cores. Both readings are kept, so the
    telemetry can say WHICH one bounded the pool.

    The paths are arguments so a test can drive this function against a real
    (synthetic) cgroup tree instead of re-implementing its arithmetic.
    """
    host = 0
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        host = 0
    cgroup = -1
    reclaimable = 0
    for limit_name, usage_name, stat_name in (
        ("memory.max", "memory.current", "memory.stat"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
         "memory/memory.stat"),
    ):
        limit = _read_int(cgroup_root / limit_name)
        used = _read_int(cgroup_root / usage_name)
        if limit is None or used is None or limit <= 0:
            continue
        reclaimable = _cgroup_reclaimable_bytes(cgroup_root / stat_name)
        working_set = max(0, used - reclaimable)
        cgroup = max(0, limit - working_set)
        break
    if host > 0 and cgroup >= 0:
        basis = "cgroup" if cgroup < host else "meminfo"
        return MemoryFacts(min(host, cgroup), basis, host, cgroup, reclaimable)
    if cgroup >= 0:
        return MemoryFacts(cgroup, "cgroup", host, cgroup, reclaimable)
    if host > 0:
        return MemoryFacts(host, "meminfo", host, -1, 0)
    return MemoryFacts(0, "unreadable", 0, -1, 0)


def cpu_facts() -> CpuFacts:
    """The pod's honest core count, and which of the three readings it is."""
    os_count = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os_count
    quota = cpu_quota_cores()
    quota_cores = float(quota) if quota is not None else -1.0
    candidates = [(os_count, "cpu_count"), (affinity, "affinity")]
    if quota is not None:
        candidates.append((max(1, int(quota + 0.5)), "quota"))
    vcpus, basis = min(candidates)
    return CpuFacts(max(1, vcpus), basis, os_count, affinity, quota_cores)


def _has_card() -> bool:
    """Is there a CUDA device at all — the ONLY question :func:`entry_workers`
    asks the card (pgw#1175).

    It gates the device-LOCK bound, which is about whether two concurrent
    autotunes would bake contention-chosen kernel configs into the artifact.
    That is a correctness property of the bytes, not a capacity question, so
    it needs presence and never size. Unreadable counts as present: refusing
    to widen is the conservative answer for a lock question.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return True


def entry_workers(
    entries: int,
    *,
    peak_rss_bytes: int = 0,
    vcpus: int = 0,
    available_bytes: int = -1,
    limit: int = 0,
    device_lock: Optional[bool] = None,
    posture: Optional[CompilePosture] = None,
) -> PoolWidth:
    """How many entries this pod may compile at once.

    **K = f(cores, one measured child RSS)** (§4.33, pgw#1175). Two bounds,
    both read off the HOST, and neither predicts VRAM:

    * **vCPU**, from :func:`cpu_facts` (cgroup quota AND affinity mask AND
      host cores, whichever is narrowest) minus
      :data:`SERVING_HEADROOM_CPUS`. ~94 % of an entry compile is ONE core of
      serial host work, so this bound is generous and scales near-perfectly.
    * **Host RAM** over one entry child's MEASURED peak RSS
      (``mint_workers.compiled_graph_peak_rss``, banked by the serving parent from a
      previous entry on this pod). Read via :func:`memory_facts`, whose cgroup
      half counts the WORKING SET rather than everything the pod has ever
      paged in.

    WHAT LEFT, AND WHY (pgw#1175). A third bound divided free VRAM by a
    per-entry device ask, and the only source that ask ever had in production
    was ``mint_budget.co_residency().need_bytes`` — the MINT CHILD's whole
    co-residency estimate, whose leading term was the PARENT's resident
    weights. Compiles are weight-free since ``fc77b923``; the estimate
    described a process that no longer exists, and the machinery built on top
    of it (a card census, a simultaneity budget, a per-spawn re-ask and a
    mid-mint re-widen) was all arithmetic over that one wrong number. An entry
    child that genuinely runs out of device memory dies in its own process and
    is classified there — the attempt is the signal, and it costs ~2 minutes.

    ``device_lock=False`` FORCES K=1 on a GPU cell: without torch's
    ``set_gpu_benchmark_lock_context`` hook the pool cannot stop two entries
    benchmarking at once, and a cell whose kernel configs were chosen under
    self-inflicted contention publishes under an unchanged key. Refusing to
    widen is the only safe answer. This is a CORRECTNESS bound on the artifact
    and has nothing to do with capacity — it is why the card is still asked
    whether it exists, and never how big it is.

    pgw#842: every bound records the READING behind it (:class:`CpuFacts`,
    :class:`MemoryFacts`) and the returned width names the constraint that
    actually bound. K is the mint's only multiplicative lever — two mints of
    one cell differed 5-vs-3 with nothing recorded to say why — so an
    unexplained K is a defect in itself.

    §4.30 / pgw#1137: both bounds are posture-aware. The ``goals`` above answer
    *"is there a tenant"*; ``posture`` answers *"is there a human"*, and on a
    user's own desktop the CPU budget is halved and the host-RAM reserve is
    doubled (:mod:`gen_worker.compile_posture` holds the derivation for both).
    The default posture is ``FLEET``.
    """
    entries = max(0, int(entries))
    if posture is None:
        posture = compile_posture.current()
    # §4.28 / pgw#1092: both of this policy's reserves are UNCONDITIONAL. They
    # used to be relaxed to zero on a pod holding no serve goal, and §4.28
    # deleted that pod class: the only mint left is the one a SERVING pod runs
    # in the background on a cell miss (pgw#784), so there is always a tenant
    # to protect. `SERVING_HEADROOM_CPUS` keeps cores for an eager forward and
    # a heartbeat; `ENTRY_RSS_RESERVE_BYTES` keeps host RAM so a request
    # arriving mid-mint does not meet the OOM killer.
    cpu_headroom = SERVING_HEADROOM_CPUS
    rss_reserve = posture.rss_reserve_bytes(ENTRY_RSS_RESERVE_BYTES)
    locked = aot_device_lock.supported() if device_lock is None \
        else bool(device_lock)
    if entries <= 1:
        # pgw#877: the entry count alone decides this, so no bound is READ —
        # and the row must not report unread bounds as zeros. It used to say
        # `available_bytes=0`: a row whose entire job is to explain K=1,
        # telling its reader the pod has no RAM. `-1` is this module's existing
        # "not read" (`cgroup_available_bytes`, `quota_cores`), and the bases
        # say so in words.
        return PoolWidth(
            workers=1, entries=entries, vcpus=0, cpu_workers=1, mem_workers=1,
            available_bytes=-1, per_entry_rss_bytes=0,
            per_entry_rss_basis="not-read",
            device_lock=locked, binding="entries", ceiling=1,
            limit=max(0, int(limit)), posture=posture,
            reason=(
                f"{entries} entr{'y' if entries == 1 else 'ies'}: serial "
                f"(no cpu/memory bound was read — the entry count "
                f"decides this width on its own)"))

    if vcpus > 0:
        cpu = CpuFacts(int(vcpus), "caller", int(vcpus), int(vcpus), -1.0)
    else:
        cpu = cpu_facts()
    vcpus = cpu.vcpus
    budget = posture.cpu_budget_cores(vcpus, headroom=cpu_headroom)
    cpu_workers = max(1, budget // CPUS_PER_ENTRY_WORKER)

    if available_bytes >= 0:
        memory = MemoryFacts(
            int(available_bytes), "caller", int(available_bytes), -1, 0)
    else:
        memory = memory_facts()
    avail = memory.available_bytes
    per_entry = int(peak_rss_bytes) if peak_rss_bytes > 0 \
        else DEFAULT_ENTRY_PEAK_RSS_BYTES
    if avail <= 0:
        # An unreadable host does not get to license a wide pool.
        mem_workers = 1
    else:
        mem_workers = max(
            1, int(max(0, avail - rss_reserve) // per_entry))

    # A caller cap NARROWS. `limit` above MAX_ENTRY_WORKERS is a caller
    # asking for more than the ceiling allows, and the ceiling wins.
    # ...and so does the POSTURE (§4.30): a user's machine caps at half the
    # fleet ceiling, and a caller cap below that still wins. Both narrow;
    # neither can widen.
    ceiling = posture.entry_ceiling(
        min(MAX_ENTRY_WORKERS, int(limit)) if limit > 0 else MAX_ENTRY_WORKERS)
    rss_basis = "measured" if peak_rss_bytes > 0 else "default"

    def _width(
        workers: int, *, binding: str, reason: str, lock: bool,
    ) -> PoolWidth:
        return PoolWidth(
            workers=workers, entries=entries, vcpus=vcpus,
            cpu_workers=cpu_workers, mem_workers=mem_workers,
            available_bytes=avail, per_entry_rss_bytes=per_entry,
            device_lock=lock,
            reason=reason, binding=binding, ceiling=ceiling, cpu=cpu,
            memory=memory,
            per_entry_rss_basis=rss_basis,
            limit=max(0, int(limit)), posture=posture)

    workers = max(1, min(cpu_workers, mem_workers, ceiling, entries))
    if workers > 1 and _has_card() and not locked:
        return _width(
            1, binding="device-lock", lock=False,
            reason=(
                "serial: this torch has no GPU-benchmark lock hook, so a wide "
                "pool would let entries benchmark against each other and bake "
                "contention-chosen kernel configs into a cell whose key would "
                "not move"))
    binding = min(
        (cpu_workers, "cpu"), (mem_workers, "host-memory"),
        (ceiling, "ceiling"),
        (entries, "entries"))[1]
    polite = (
        " [§4.30 user-machine: half the cores, "
        f"{USER_MACHINE_RSS_RESERVE_BYTES // 1024**3} GiB RAM left alone, "
        f"ceiling {ceiling}, nice {posture.nice_level()}]"
        if posture.user_machine else "")
    reason = (
        f"K={workers} ({binding}-bound{polite}): "
        f"{vcpus} vCPU ({cpu.basis}) -> "
        f"{cpu_workers}, {avail / 1024**3:.1f} GiB RAM ({memory.basis}) "
        f"/ {per_entry / 1024**3:.1f} GiB per entry ({rss_basis}) -> "
        f"{mem_workers}")
    return _width(workers, binding=binding, reason=reason, lock=locked)


# ---------------------------------------------------------------------------
# The wire: one job in, one report out (same shape as pgw#784's mint child)
# ---------------------------------------------------------------------------


class EntryJob(msgspec.Struct, frozen=True, kw_only=True):
    """One compile child's SHARE of a mint, as a file a human can re-run.

    pgw#1215 (th#1834 Phase 3 step 2b) turned this struct inside out. It used
    to name ONE already-exported graph class and the file its
    ``ExportedProgram`` had been ``torch.export.save``d to; the child
    ``torch.export.load``ed it back, at a **36.04 s median** (pgw#1216, P0-E
    §5c) — ~22 min of a 36-class sdxl mint spent deserializing what another
    process in the same pod had just serialized.

    So the program does not cross at all any more. This names what the child
    needs to BUILD the weight-free pipeline itself (``function`` / ``modules``
    / ``slots`` / ``cfg`` — the same four ``boot_key.TraceJob`` carries, and
    for the same reason) plus WHICH declared graph classes are its share.
    The child then traces, compiles and packs them in ONE address space.

    Three fields died with the round trip and are not coming back, because the
    thing they repaired is not happening: ``program`` (the staged file),
    ``symbol_values`` and ``symbol_labels`` (pgw#998 — the ShapeEnv values
    ``torch.export``'s save/load loses). See ``aot_compile_spans`` for the
    matching hole in the span partition.
    """

    #: WHAT to build. Identical for every child of one mint — the pool copies
    #: the caller's template and stamps only the share/location fields below.
    function: str = ""
    modules: Tuple[str, ...] = ()
    cfg: CompileSpec = msgspec.field(default_factory=CompileSpec)
    slots: Dict[str, MintSlot] = {}
    #: pgw#947's measured serving-kernel lane, stamped into every artifact this
    #: child packs. The parent measures it (only the loader can swap the
    #: linears) and the child cannot re-derive it, so it crosses.
    execution_lane: Optional[kernel_path.Verdict] = None

    #: WHICH classes. ``rows[i::K]`` over ``aot_mint.declared_class_rows`` —
    #: by INDEX and never by name, because the adapter fork is decided by the
    #: COMPOSED pipeline and no parent can enumerate the names to hand out.
    share: str = ""
    share_index: int = 0
    share_count: int = 1
    #: pgw#1215 step 4: declared graph classes this pod ALREADY HAS as packed
    #: artifacts, from an earlier attempt of the same mint. A child skips them
    #: before it exports — not after, and not at pack time — so a retry pays
    #: neither the trace nor the compile for work that is already on disk.
    #: This is the whole of what the deleted ``mint_delegate.build_cell``
    #: retry loop got wrong: it re-ran every attempt in a FRESH ``child-N``
    #: directory, so attempt 2 of a 36-class mint re-paid 35 finished classes
    #: to retry one. Named by CLASS, because that is what a child can match
    #: before it has traced anything; the artifacts themselves are addressed
    #: by ``cg-key-v1`` key and the supervisor reads the names back off them.
    have_classes: Tuple[str, ...] = ()

    #: WHERE. ``out_dir`` receives the packed artifacts, ``work`` the
    #: packaging scratch, ``report`` this child's one report file.
    out_dir: str = ""
    work: str = ""
    report: str = ""
    inductor_configs: Dict[str, Any] = {}
    cache_dir: str = ""
    device_lock: str = ""


class PackedGraphClass(msgspec.Struct, frozen=True, kw_only=True):
    """One graph class the child traced, compiled AND packed (pgw#1215).

    The child hands back an ARTIFACT, not loose inductor files: it holds the
    ``_MintedEntry`` row in its own address space, so it is the only process
    that can run ``aot_mint.pack_graph_classes`` over it. ``metadata`` is
    canonical JSON rather than a decoded map for the reason
    ``boot_key.TraceReport.blocks`` is: the parent hands it straight on, and a
    re-encode on either side is a place for two canonicalizations to disagree
    about the thing being hashed.
    """

    name: str
    key: str = ""
    artifact: str = ""
    metadata: str = ""
    #: This class's own ``export_s`` / ``compile_s`` and inductor phase split,
    #: straight off ``_export_entry``'s timings. Per CLASS, because a share is
    #: several classes and one number for the share answers nothing.
    spans: Dict[str, float] = {}


class EntryReport(msgspec.Struct, frozen=True, kw_only=True):
    entry: str
    status: str = ""
    #: pgw#1215: what this child produced, one row per graph class. Replaces
    #: the loose-file list — the files never leave the child now.
    classes: List[PackedGraphClass] = []
    #: How many classes the WHOLE declaration produced on this child's
    #: pipeline. Every child reports it, all must agree, and the union of the
    #: shares must be exactly that many — which proves the class set is whole
    #: without the parent ever enumerating it (``boot_key``'s rule, same
    #: sharding).
    declared_classes: int = 0
    detail: str = ""
    elapsed_s: float = 0.0
    peak_rss_bytes: int = 0
    #: pgw#868 A4: the child's DEVICE high-water, allocated and reserved.
    #: Defaulted so an older child's report still decodes. TELEMETRY (pgw#1175)
    #: — it rides the phase table to the hub as `peak_child_device_bytes`, and
    #: is the only honest answer to "what does one entry compile cost a card".
    #: It sizes nothing: K is f(cores, measured child RSS).
    peak_device_bytes: int = 0
    peak_device_reserved_bytes: int = 0
    #: Inductor's own phase split (lowering / codegen / host C++ compile+link)
    #: MEASURED IN THE CHILD. pgw#757's instrument-first deliverable is read
    #: from dynamo's in-process counters, which do not move in the parent once
    #: the compile leaves it — so without this the phase table silently goes
    #: dark the moment the pool turns on. Caught by pgw#758's own test.
    phases: Dict[str, float] = {}
    #: pgw#830: the child's COMPLETE wall partition — `phases` above measures
    #: only the inside of `aot_compile`, while the recorded `compile_s` is the
    #: parent's Popen-to-reap wall. Everything between those two definitions
    #: (interpreter boot, `import torch`, the seal, the device-lock install,
    #: and — until pgw#1215 deleted it — `torch.export.load` of the staged
    #: program) was the dark 44 %.
    spans: Dict[str, float] = {}
    #: Named, and deliberately NOT summed with `spans`: these nest inside
    #: partition members (triton keys inside codegen/host compile; a device
    #: benchmark-lock wait inside `aot_compile`). Adding them to the total was
    #: the second attribution bug.
    overlays: Dict[str, float] = {}
    spans_v: int = 0
    #: Wall-clock stamps that close the spans crossing the process boundary.
    module_import_epoch: float = 0.0
    run_start_epoch: float = 0.0
    report_epoch: float = 0.0
    #: Every `compilation_time_metrics` key that moved. The evidence for
    #: naming whatever `compile_other_s` turns out to contain — a residual
    #: nobody can look inside is only half an attribution.
    metrics_raw: Dict[str, float] = {}
    #: pgw#840: WHICH gen_worker compiled this entry. ``code_digest`` is the
    #: contract-source digest the child computed at ITS import; ``code_dir`` is
    #: where it imported from, which is the actionable half of the message. An
    #: empty digest means a child too old to answer — which is the case that
    #: went dark: it also predates the span table, so the parent absorbed its
    #: whole compile into ``reap_lag_s`` and reported a partition that did not
    #: close (the pgw#830 invariant, red on a tree nobody had changed).
    code_digest: str = ""
    code_dir: str = ""


COMPILED = "compiled"
REFUSED = "refused"

EXIT_COMPILED = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4


class EntryCompileFailed(RuntimeError):
    """One entry's compile failed. Carries the entry name — a pool of 18 that
    fails anonymously is undebuggable — and its CLASSIFICATION.

    pgw#848: ``resource`` is the difference between a mint that is retried at
    a narrower K and one that is never retried at all. Everything this class
    carried used to converge on ``MintRefused`` -> ``EXIT_REFUSED``, which
    ``mint_process`` documents as "typed, deterministic — terminal", so the
    ONE failure class a narrower pool would have fixed was the one class
    routed down the never-retry path. ``basis`` names how the verdict was
    reached, because a verdict from the kernel's own OOM counter and a
    verdict inferred from a signal are not the same evidence.
    """

    def __init__(
        self, entry: str, detail: str, *,
        resource: bool = False, basis: str = "", peak_rss_bytes: int = 0,
    ) -> None:
        super().__init__(detail)
        self.entry = entry
        self.detail = detail
        self.resource = bool(resource)
        self.basis = str(basis)
        self.peak_rss_bytes = int(peak_rss_bytes)


class EntryCompileAbandoned(RuntimeError):
    """The SUPERVISOR asked this pool to stop (pgw#1215 step 4).

    Not a failure: a co-tenancy decision — an adopt-on-arm, a vacate, a
    shutdown — took the card while the pool was working. The distinction is
    load-bearing at the terminus (`self_mint_abort` phase `abandoned_*` vs
    `error`), and it is the one thing the old three-tier stack could express
    that a `to_thread`-driven pool cannot express by task cancellation: a
    cancelled task does not reach the children, and children the parent has
    stopped waiting for keep compiling on a card somebody else now owns.
    """


@dataclass
class _Running:
    entry: str
    proc: subprocess.Popen
    job: EntryJob
    started: float
    stderr_path: Path
    #: pgw#848: THIS entry's own high-water, sampled while it lives. The
    #: pool-wide max cannot answer "how big was the one that died", and a
    #: child killed by the OOM killer writes no report — so the parent's
    #: live sample is the only measurement that survives it, and it is
    #: exactly the number the next attempt must size K against.
    peak_rss_bytes: int = 0
    #: Wall clock at the moment ``Popen`` returned — the other end of the
    #: child's boot span, which no monotonic clock can close across processes.
    spawn_epoch: float = 0.0
    #: pgw#1243: this child's own silence window, and the high-water marks it
    #: judges. Per ROW, because the pool's question is per row: one share
    #: finishing tells you nothing about whether a sibling has wedged, and a
    #: pool-wide signal lets a busy sibling vouch for a dead one.
    window: Optional[SilenceWindow] = None
    cpu_s: Optional[float] = None
    work_mib: Optional[float] = None
    #: True when the child wrote its report and the PARENT ended it, rather
    #: than the child exiting under its own power. The exit code is then the
    #: parent's signal and says nothing about the compile — read the report.
    reaped_at_terminus: bool = False


@dataclass
class PoolLedger:
    """pgw#830: where the POOL's seconds went, as against where the entries'
    seconds went. The two are different questions with opposite fixes.

    ``compile_s`` (the sum of entry walls) is SERIAL work: shrinking it means
    compiling less or compiling faster. ``pool_idle_s`` is SCHEDULING loss:
    workers with nothing to run. Attempt nine's 75 % pool efficiency at K=5 is
    the second number, and collapsing the entry count (pgw#829) moves it
    without touching a single compile — so a table that adds the two into one
    "dark 44 %" figure would send that work at the wrong target.

    The idle split is exact, not sampled: the loop charges every free-slot
    second to the state the parent was actually in.
    """

    workers: int = 1
    #: pgw#868 A4: the width the pool was CONSTRUCTED at, kept beside the one
    #: it finished at. Equal since pgw#1175 deleted the mid-mint re-widen;
    #: retained because the ledger's identity is stated over intervals and a
    #: future width change must stay readable as a delta from one row.
    workers_initial: int = 1
    wall_s: float = 0.0
    #: Sum of the per-entry Popen-to-reap walls (== the mint's ``compile_s``).
    busy_s: float = 0.0
    #: ``workers * wall_s`` — the seconds the pool COULD have compiled for.
    capacity_s: float = 0.0
    #: Free-slot seconds while the parent was inside ``torch.export.save``.
    #: The pool refills a freed slot only after staging the next program, so
    #: this is export-vs-compile SERIALIZATION, measured.
    idle_staging_s: float = 0.0
    #: pgw#1052: free-slot seconds while the parent was PULLING the entry
    #: source — which, on the overlapped mint path, is the export itself
    #: (``torch.export`` runs in the parent when the pool asks for the next
    #: entry). This is the producer-side half of the serialization
    #: ``idle_staging_s`` measures on the save side: it prices exactly how
    #: much the single-threaded producer starved the pool, which is the
    #: number that says when the fused export child (pgw#1000) is owed.
    idle_source_s: float = 0.0
    #: Free-slot seconds with nothing left to start: the straggler tail.
    idle_drain_s: float = 0.0
    #: Free-slot seconds inside ``Popen`` itself.
    idle_spawn_s: float = 0.0
    #: Free-slot seconds that were neither — the residual of the idle split.
    idle_other_s: float = 0.0
    #: Parent-serial staging cost, summed. Not idle time (it overlaps running
    #: children); the idle FRACTION of it is ``idle_staging_s``.
    stage_total_s: float = 0.0
    spawn_total_s: float = 0.0
    #: pgw#832: parent-serial cost of seeding the seal library memo, paid
    #: BEFORE the pool wall starts (so never part of the capacity identity).
    #: Near-zero in a sealed parent; one full hashing pass in a cold one —
    #: which is the pass every CHILD used to pay.
    seal_seed_s: float = 0.0
    entries: int = 0
    #: pgw#848 item 5: the resume bank's own row (``resume_root``, ``resumed``,
    #: ``resume_cold``, ``resume_refused`` by reason, ``resume_admit_s``).
    #: Empty on every mint that runs without a bank, so a pod with no resume
    #: root reads exactly as it did before. It rides the LEDGER rather than a
    #: second event because "N of 36 entries were recovered" is the first thing
    #: that explains a pool wall, and a reader who has to join two rows to
    #: learn it will price a resumed mint as a fast compile.
    resume: Dict[str, Any] = field(default_factory=dict)

    @property
    def idle_s(self) -> float:
        return round(
            self.idle_staging_s + self.idle_source_s + self.idle_drain_s
            + self.idle_spawn_s + self.idle_other_s, 3)

    @property
    def efficiency(self) -> float:
        return round(self.busy_s / self.capacity_s, 4) if self.capacity_s else 0.0

    def facts(self) -> Dict[str, Any]:
        return {
            "pool_wall_s": round(self.wall_s, 3),
            "pool_busy_s": round(self.busy_s, 3),
            "pool_capacity_s": round(self.capacity_s, 3),
            "pool_idle_s": self.idle_s,
            "pool_efficiency": self.efficiency,
            "idle_staging_s": round(self.idle_staging_s, 3),
            "idle_source_s": round(self.idle_source_s, 3),
            "idle_drain_s": round(self.idle_drain_s, 3),
            "idle_spawn_s": round(self.idle_spawn_s, 3),
            "idle_other_s": round(self.idle_other_s, 3),
            "stage_total_s": round(self.stage_total_s, 3),
            "spawn_total_s": round(self.spawn_total_s, 3),
            "seal_seed_s": round(self.seal_seed_s, 3),
            "pool_entries": int(self.entries),
            "pool_workers": int(self.workers),
            "pool_workers_initial": int(self.workers_initial),
            **dict(self.resume),
        }


def child_argv(job_path: Path, *, python: str = "") -> List[str]:
    return [python or sys.executable, "-m", ENTRY_CHILD_MODULE, str(job_path)]


def child_env(
    cache_dir: str, *, base: Optional[Mapping[str, str]] = None,
    seal_memo: str = "",
) -> Dict[str, str]:
    """The entry child's environment.

    The parent's env verbatim (the seal must not move between parent and
    child — pgw#784's rule, and here the child produces the very bytes the
    seal describes), plus:

    * ``TORCHINDUCTOR_CACHE_DIR`` pointing every child at ONE worker-local
      directory. This is a LOCATION, not a recipe: nothing in ``env_seal``
      digests it, and the compile's output is byte-identical wherever it
      lands (measured — the same graph hashes to the same cache subdirectory
      from any process). Sharing it is what lets children recover the
      cross-entry kernel dedup a serial loop got from one warm process, and
      it is how the parent reads the loose files the children produced.
    * ``GEN_WORKER_SEAL_LIB_MEMO`` (pgw#832), when the pool seeded one: the
      parent's toolchain digests, so the child's ``env_seal.establish()``
      stats instead of re-hashing 4 GB. A LOCATION of digests, never a
      recipe: the child verifies every file's (path, mtime_ns, size) itself
      and rehashes on any mismatch, so the seal value cannot move.
    * ``PYTHONPATH`` with the PARENT's own package root in front, and
      ``PYTHONSAFEPATH`` (pgw#840). ``-m gen_worker.aot_compile_child`` used to
      mean "whatever gen_worker this interpreter resolves": the cwd first, then
      any inherited ``PYTHONPATH``, then site-packages. On a box with more than
      one checkout that is a coin flip, and the child that wins compiles the
      files the cell publishes. Pinning the root makes the child the parent's
      OWN code by construction; ``PYTHONSAFEPATH`` removes the cwd, which would
      otherwise still outrank it. The digest check in ``_collect`` is the
      backstop that proves it rather than assuming it.
    """
    env = dict(os.environ if base is None else base)
    # (pgw#1030: the GEN_WORKER_AOT_ENTRY_CHILD marker is deleted — written
    # for four months, read by nothing.)
    if cache_dir:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    if seal_memo:
        env[env_seal.SEAL_LIB_MEMO_ENV] = str(seal_memo)
    env["PYTHONSAFEPATH"] = "1"
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and p != PACKAGE_ROOT]
    env["PYTHONPATH"] = os.pathsep.join([PACKAGE_ROOT] + existing)
    return env


def _terminate_group(proc: subprocess.Popen, *, grace_s: float = _KILL_GRACE_S) -> None:
    """SIGTERM then SIGKILL the child's process GROUP.

    The GROUP, not the process: an entry child spawns inductor's own compile
    workers and g++ underneath it, and a pool that killed only the direct
    children would leave orphan cc1plus processes burning a serving pod's CPU
    against a cell nobody will adopt. Every child is started with
    ``start_new_session=True`` precisely so this call has a group to aim at.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)


#: ``prctl(2)`` PR_SET_PDEATHSIG.
_PR_SET_PDEATHSIG = 1


def arm_parent_death_signal() -> bool:
    """Ask the kernel to SIGKILL THIS process when its parent dies.

    Called by the entry child on itself, never by the parent through
    ``preexec_fn``. Two reasons, and the second is the expensive one:

    * ``preexec_fn`` runs in the forked child of a process that has threads,
      where only async-signal-safe work is legal — the hazard is real enough
      that CPython documents it.
    * Passing ``preexec_fn`` also FORCES ``fork()`` instead of
      ``posix_spawn()`` for every spawn. The mint child has live gRPC threads,
      and gRPC installs ``pthread_atfork`` handlers; making the pool the only
      thing in the worker that forks a threaded process is a large blast
      radius for a one-line guarantee. Arming from the child costs a
      microscopic race (the parent dying between ``exec`` and this call) and
      buys back the ordinary spawn path.

    Why it exists at all: the pool runs INSIDE pgw#784's mint child, and the
    serving worker reaps that child by process GROUP when a mint is abandoned.
    Entry children hold their OWN session — so the worker's group kill does
    not reach them, and without this an abandoned mint would leave K compiles
    burning a serving pod's CPU with nothing left to notice.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    except Exception:  # noqa: BLE001  (non-Linux, or no libc — group kill only)
        return False
    return True


def _read_report(path: Path) -> Optional[EntryReport]:
    try:
        return msgspec.json.decode(path.read_bytes(), type=EntryReport)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError):
        return None


def _vmhwm_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _descendants(root: int) -> List[int]:
    """``root`` and every process under it, TRANSITIVELY.

    pgw#848: the previous reading walked ``/proc/<pid>/task/<pid>/children``
    once — direct children of the main thread, one level, and only that
    thread's. MEASURED on a real AOTI compile: the entry child's direct
    children are ``g++`` (a driver that allocates nothing) and inductor's
    ``async_compile`` subprocess workers; **``cc1plus`` — the 2.04 GiB — is at
    depth 2**, and ``as``/``collect2``/``ld`` sit beside it. So the one number
    the pool's memory bound exists to measure was, by construction, the one
    number this function could not see.
    """
    out: List[int] = []
    stack = [int(root)]
    while stack:
        pid = stack.pop()
        out.append(pid)
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                stack.extend(
                    int(p) for p in (task / "children").read_text().split())
        except (OSError, ValueError):
            continue
    return out


def cgroup_oom_kills(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """How many processes this cgroup's memory limit has killed, ever.

    pgw#848: the kernel's own counter, and the only NON-inferential evidence
    that a dead entry child died of memory. A SIGKILL with no report is the
    OOM killer far more often than anything else — the pool has said so in
    ``_exit_note`` since pgw#809 — but "far more often" is not a
    classification, and the retry policy branches on it. ``-1`` = unreadable,
    which is honest and is reported as such rather than folded into 0.
    """
    for name in ("memory.events", "memory/memory.oom_control"):
        path = cgroup_root / name
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("oom_kill", "oom_kill_disable"):
                if parts[0] == "oom_kill":
                    try:
                        return int(parts[1])
                    except ValueError:
                        return -1
    return -1


def _peak_rss_bytes(proc: subprocess.Popen) -> int:
    """The child tree's high-water RSS, read from the kernel while it lives.

    Summed across the tree because the members are concurrent: the
    interpreter holds the loaded program while its compiler holds the TU. A
    concurrent pool slot must reserve the pair, not the larger of the two.
    """
    return sum(_vmhwm_bytes(pid) for pid in _descendants(proc.pid))


def _tree_cpu_seconds(pid: int) -> Optional[float]:
    """CPU seconds burned by a child's whole process tree, reaped members
    included (pgw#964, ported to the pool by pgw#1243).

    The reaped counters are what make it monotonic: a member's CPU leaves its
    own ``utime/stime`` and enters its parent's ``cutime/cstime`` the instant
    the parent waits for it, so a plain live-member sum falls into a hole one
    finished sub-process deep — and a supervisor comparing against a
    high-water mark reads that as death. ``None`` (never 0) when the tree
    cannot be sampled: an absent measurement is no evidence, not a zero.
    """
    try:
        import psutil
    except Exception:  # pragma: no cover — psutil is a hard dep in practice
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
        total += float(times.user) + float(times.system)
        total += float(getattr(times, "children_user", 0.0) or 0.0)
        total += float(getattr(times, "children_system", 0.0) or 0.0)
    return total


def _dir_mib(root: Path) -> Optional[float]:
    """MiB a child has written into its scratch — generated sources, objects,
    the packed artifact. It grows in BURSTS (sources land, then a single
    ``cc1plus`` chews for minutes writing nothing), so its growth proves work
    and its silence proves nothing: an independent positive signal that can
    never, on its own, vote to condemn."""
    total = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return None
    return total / (1 << 20)


class EntryCompilePool:
    """Trace, compile and pack a family's declared graph classes K-wide, out
    of process.

    Not a general executor: it exists to hold pgw#809's three invariants —
    named failure, group-wide sibling teardown, and assembly by graph-class
    NAME rather than completion order.

    pgw#1215 changed what a child IS. It used to receive one already-exported
    ``ExportedProgram`` on disk; it now receives a SHARE of the declaration
    (``rows[i::K]``) and builds its own weight-free pipeline, so the process
    that traces a graph class is the process that compiles and packs it. The
    parent therefore stages nothing, holds no program, and produces nothing
    the children consume — which is why there is no producer iterator here any
    more: K children are dispatched once, and the loop only supervises.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        width: PoolWidth,
        inductor_configs: Optional[Mapping[str, Any]] = None,
        cache_dir: str = "",
        python: str = "",
        entry_silence_window_s: float = _ENTRY_SILENCE_WINDOW_S,
    ) -> None:
        self.workdir = Path(workdir)
        #: pgw#1243: how long ONE share may make no measured progress. A
        #: parameter so a tape can drive the window rather than wait it out;
        #: production never passes it.
        self.entry_silence_window_s = float(entry_silence_window_s)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.width = width
        #: pgw#842/th#1359: the width facts as last EMITTED, so a re-emit
        #: happens only when they actually moved.
        self._emitted_width_facts: Optional[Dict[str, Any]] = None
        self._emit_width("construction")
        self.inductor_configs = dict(inductor_configs or {})
        # pgw#848 item 5, NARROWED by pgw#1215 to the one half that survives
        # the keystone: the bank is opened for its CACHE DIRECTORY and its
        # ledger row, never for file admission any more. Admission was
        # `bank.admit(entry, program)` — it re-derived the graph hash from the
        # ExportedProgram THIS attempt exported, and the parent no longer
        # exports anything, so the identity it compared against cannot be
        # computed here. What survives is strictly the better half: the
        # inductor cache stays scoped to the MINT rather than the attempt, so
        # a killed mint's next attempt still hits torch's own FX graph cache.
        # ⚠️ OWED (step 3/4): re-home file-level resume at the graph-class
        # artifact, which is where pgw#1176 already made durability live.
        self.bank = aot_resume.open_bank(
            inductor_configs=self.inductor_configs)
        # The inductor cache follows the bank when there is one. A per-attempt
        # cache dir is why a killed mint got not even a cache hit on retry;
        # inductor's key is the graph, not the process (measured — see this
        # module's header), so widening the scope from one attempt to one mint
        # cannot change what is produced.
        self.cache_dir = str(
            cache_dir
            or (self.bank.cache_dir if self.bank is not None else "")
            or (self.workdir / "inductor-cache"))
        # pgw#809: ONE lock file for the whole pool. Every entry child routes
        # its inductor GPU benchmarks through it, so no two entries ever time
        # a kernel on the card at the same moment.
        self.device_lock_path = self.workdir / aot_device_lock.LOCK_NAME
        # pgw#832: the parent's toolchain digests, seeded once per pool so N
        # short-lived children stop re-paying a multi-GB SHA-256 pass each.
        # Emptied if seeding fails — children then rehash in full (safe path).
        self.seal_memo = str(self.workdir / "seal-lib-memo.json")
        self.seal_seed_s = 0.0
        self.python = python
        self.peak_rss_bytes = 0
        #: pgw#877 #2: the widest DEVICE high-water any entry child reported.
        #: MEASUREMENT ONLY (pgw#1175) — it rides the phase table to the hub as
        #: `peak_child_device_bytes` and decides nothing here. What one entry
        #: child really costs a card is exactly the question P0-E/P0-F ask; it
        #: is not, and was never, a licence to divide free VRAM by it.
        self.peak_device_bytes = 0
        #: pgw#1205: the same reading, kept PER ENTRY instead of collapsed.
        #: `peak_device_bytes` above answers "how big was the biggest compile"
        #: — one number for a whole cell, which is the wrong granularity for
        #: the only question anyone asks of it ("what does THIS graph class
        #: cost a card"). Both survive: the max is what the existing phase-table
        #: field publishes, and these rows are what gets banked with their
        #: provenance. entry name -> (allocated, reserved).
        self.entry_device_peaks: Dict[str, Tuple[int, int]] = {}
        self.peak_concurrency = 0
        # pgw#848: the kernel's OOM-kill counter as it stood before this pool
        # ran. A DELTA over the pool's own wall is evidence; the absolute
        # number is a pod's whole history and means nothing here.
        self.oom_kills_at_start = cgroup_oom_kills()
        #: Set when an entry died of memory, so the mint's aborted phase
        #: table carries the actionable half rather than a bare "refused".
        self.oom_entry = ""
        self.oom_basis = ""
        self.entry_seconds: Dict[str, float] = {}
        self.entry_phases: Dict[str, Dict[str, float]] = {}
        #: share -> how many graph classes the WHOLE declaration produced on
        #: that child's pipeline. The evidence `_assert_shares_whole` reads.
        self.entry_declared: Dict[str, int] = {}
        #: pgw#1215: share -> the graph classes a REFUSING child had already
        #: packed before it refused. They exist on disk; recording them is how
        #: "this share produced nothing" and "this share produced most of a
        #: cell and then hit one bad class" stop reading the same.
        self.refused_classes: Dict[str, List[PackedGraphClass]] = {}
        #: pgw#1215: graph class -> that class's OWN `export_s`/`compile_s`
        #: and inductor phase split, as the child measured them. The pool's
        #: other tables are per SHARE, and a share is several classes — the
        #: only granularity anybody asks about a compile is the class.
        self.class_spans: Dict[str, Dict[str, float]] = {}
        # pgw#830: parent-side per-share spans (writing the job + spawn) and
        # the pool-level idle split. Kept separate from `entry_phases` because
        # they are NOT inside `compile_s`: they happen in the parent while
        # other children run, so summing them into the compile total would
        # invent seconds nobody spent compiling. Since pgw#1215 the "stage"
        # is a few hundred bytes of JSON rather than a multi-GB
        # `torch.export.save`, and it is still measured — a cost that stops
        # being measured is a cost nobody can prove went away.
        self.entry_stage_seconds: Dict[str, float] = {}
        self.entry_spawn_seconds: Dict[str, float] = {}
        self.entry_overlays: Dict[str, Dict[str, float]] = {}
        self.entry_metrics_raw: Dict[str, Dict[str, float]] = {}
        self.ledger = PoolLedger(
            workers=int(width.workers), workers_initial=int(width.workers))

    # -- dispatch ---------------------------------------------------------

    def _stage(self, template: EntryJob, index: int, count: int) -> Tuple[EntryJob, Path]:
        """Write ONE child's job file: the caller's recipe plus this share.

        pgw#1215: what this used to do was ``torch.export.save`` a multi-GB
        ExportedProgram (~16 s at 2.5 GB) so a child could ``torch.export.load``
        it back at a 36.04 s median. Both halves are gone. The share is named
        by INDEX into ``aot_mint.declared_class_rows``' order, never by class
        NAME, because the adapter fork is decided by the COMPOSED pipeline and
        no parent can enumerate the names to hand out — the same rule
        ``boot_key`` shards by.
        """
        share = f"share-{index:03d}"
        slot = self.workdir / share
        slot.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        job = msgspec.structs.replace(
            template,
            share=share,
            share_index=index,
            share_count=count,
            out_dir=str(template.out_dir or (self.workdir / "artifacts")),
            work=str(slot / "work"),
            report=str(slot / ENTRY_REPORT_NAME),
            inductor_configs=dict(self.inductor_configs),
            cache_dir=self.cache_dir,
            device_lock=str(self.device_lock_path),
        )
        job_path = slot / "job.json"
        job_path.write_bytes(msgspec.json.encode(job))
        self.entry_stage_seconds[share] = round(time.monotonic() - t0, 3)
        self.ledger.stage_total_s = round(
            self.ledger.stage_total_s + self.entry_stage_seconds[share], 3)
        return job, job_path

    def _spawn(self, job: EntryJob, job_path: Path) -> _Running:
        stderr_path = job_path.parent / "stderr.log"
        handle = stderr_path.open("wb")
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                child_argv(job_path, python=self.python),
                stdout=subprocess.DEVNULL,
                stderr=handle,
                env=child_env(self.cache_dir, seal_memo=self.seal_memo),
                start_new_session=True,   # own group -> group-wide reaping
            )
        finally:
            handle.close()
        started, spawn_epoch = time.monotonic(), time.time()
        self.entry_spawn_seconds[job.share] = round(started - t0, 3)
        self.ledger.spawn_total_s = round(
            self.ledger.spawn_total_s + (started - t0), 3)
        logger.info(
            "aot-pool: %s (rows[%d::%d]) -> pid %s",
            job.share, job.share_index, job.share_count, proc.pid)
        return _Running(
            entry=job.share, proc=proc, job=job,
            started=started, stderr_path=stderr_path, spawn_epoch=spawn_epoch)

    # -- the run ----------------------------------------------------------

    def compile(
        self, template: EntryJob,
        *, on_share: Optional[Callable[[str, int, int], None]] = None,
        should_abandon: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, PackedGraphClass]:
        """Dispatch this mint's declared classes K-wide and collect what the
        children packed. ``{graph class name: PackedGraphClass}``.

        ``template`` carries the WHAT (function, modules, slots, cfg,
        execution lane, out_dir); this method stamps the WHICH (share index of
        K) and the WHERE (work, report, cache, device lock). Every child runs
        the same recipe over a disjoint share, which is why the result is a
        union rather than a merge — a class produced twice is a defect, and it
        is refused by name below rather than silently last-writer-wins.

        Raises :class:`EntryCompileFailed` naming the FIRST share to fail,
        after tearing down every sibling group. Returns a dict ordered by
        graph-class NAME, never by completion, so what gets packed cannot
        depend on which child finished first.

        ``on_share(name, done, total)`` (pgw#824) fires as each share lands.
        This loop is the longest wire-silent stretch of a mint. Progress
        reporting is best-effort by construction: a raising callback must never
        cost the mint the classes it already has.

        ``should_abandon()`` (pgw#1215 step 4) is polled in the same drain loop
        and raises :class:`EntryCompileAbandoned`, so the ``finally`` below
        group-kills every live child. The supervisor drives this pool from a
        worker thread and cancelling that thread's task would reach nothing —
        the children would keep compiling on a card the supervisor has already
        given back.
        """
        width = max(1, int(self.width.workers))
        running: List[_Running] = []
        done: Dict[str, PackedGraphClass] = {}
        by_share: Dict[str, List[str]] = {}
        failure: Optional[EntryCompileFailed] = None
        #: Every child reports how many classes the WHOLE declaration produced
        #: on its own pipeline. They must agree, and the union of the shares
        #: must be exactly that many — which proves the class set is whole
        #: without the parent having enumerated it.
        declared: Dict[str, int] = {}
        # pgw#832: seed BEFORE the pool wall starts, so the cost is its own
        # named line (`seal_seed_s`) and never inside the capacity identity.
        self._seed_seal_memo()
        self.ledger.entries = width

        def _cb(name: str) -> None:
            if on_share is not None:
                try:
                    on_share(name, len(by_share), width)
                except Exception:  # noqa: BLE001 — telemetry never fails a mint
                    logger.debug(
                        "entry-pool progress callback failed", exc_info=True)

        # pgw#830: exact idle accounting. Every wall second is multiplied by
        # the number of FREE worker slots at that moment and charged to
        # whatever the parent was doing — so `pool_idle_s` is not a residual
        # anybody has to trust, it is a sum with named causes.
        pool_t0 = mark = time.monotonic()

        def charge(bucket: str, free: int) -> None:
            nonlocal mark
            now = time.monotonic()
            if free > 0:
                setattr(self.ledger, bucket,
                        getattr(self.ledger, bucket) + (now - mark) * free)
            # pgw#868 A4: capacity is ACCUMULATED at the width that was live
            # for each interval, not multiplied out at the end. K is fixed for
            # the life of a pool since pgw#1175, but the identity (busy + idle
            # == capacity) is stated over intervals so it stays true whatever
            # K does.
            self.ledger.capacity_s += (now - mark) * self.width.workers
            mark = now

        try:
            for index in range(width):
                free = width - len(running)
                job, job_path = self._stage(template, index, width)
                charge("idle_staging_s", free)
                running.append(self._spawn(job, job_path))
                charge("idle_spawn_s", width - len(running))
            while running:
                free = width - len(running)
                if should_abandon is not None and should_abandon():
                    raise EntryCompileAbandoned(
                        f"the supervisor abandoned this mint with "
                        f"{len(running)} of {width} share(s) still compiling; "
                        f"{len(done)} graph class(es) are already packed and "
                        f"stay on disk")
                finished = self._reap(running)
                if finished is None:
                    time.sleep(_POLL_S)
                    charge("idle_drain_s", free)
                    continue
                charge("idle_drain_s", free)
                running.remove(finished)
                try:
                    packed = self._collect(finished)
                except EntryCompileFailed as exc:
                    failure = exc
                    break
                by_share[finished.entry] = [row.name for row in packed]
                declared[finished.entry] = self.entry_declared.get(
                    finished.entry, 0)
                for packed_row in packed:
                    if packed_row.name in done:
                        failure = EntryCompileFailed(
                            finished.entry,
                            f"{finished.entry}: graph class "
                            f"{packed_row.name!r} was "
                            f"packed by two shares — the declaration's row "
                            f"order is not stable across this pool's "
                            f"children, so rows[i::{width}] does not "
                            f"partition it and some class is missing "
                            f"entirely")
                        break
                    done[packed_row.name] = packed_row
                if failure is not None:
                    break
                _cb(finished.entry)
                # Collection and pgw#824's progress callback both run with the
                # slot ALREADY FREE, so they are charged as idle rather than
                # left outside the split — a callback that blocked would
                # otherwise vanish from a ledger whose whole point is that
                # nothing vanishes.
                charge("idle_other_s", width - len(running))
            if failure is not None:
                raise failure
            self._assert_shares_whole(
                declared, done, width, have=len(template.have_classes))
        finally:
            self.ledger.wall_s = round(time.monotonic() - pool_t0, 3)
            self.ledger.busy_s = round(sum(self.entry_seconds.values()), 3)
            # Closed at the LIVE width for the final interval, then rounded —
            # `charge` has been accumulating it all along (see there).
            charge("idle_other_s", 0)
            self.ledger.capacity_s = round(self.ledger.capacity_s, 3)
            for row in running:
                _terminate_group(row.proc)
            # Re-emit only if the width moved since construction (no-op
            # otherwise); the ledger row keeps carrying the timing facts.
            self._emit_width("terminus")
            self._emit_ledger()
        return {name: done[name] for name in sorted(done)}

    def _assert_shares_whole(
        self, declared: Mapping[str, int], done: Mapping[str, Any], width: int,
        *, have: int = 0,
    ) -> None:
        """The shares must reconstruct the WHOLE declared class set.

        pgw#1089's proof, applied at the compile seam: the parent never
        enumerated the classes (it holds no pipeline), so the only evidence
        that ``rows[i::K]`` partitioned them is that every child reported the
        same declared count and the union has exactly that many rows. Without
        this a child whose share came back empty — a stale declaration, a
        shard-index bug, a family whose fork differs per child — publishes a
        SHORT cell that verifies, arms, and is missing a class.

        ``have`` (pgw#1215 step 4) is how many classes the children were told
        to SKIP because this pod already holds their artifacts. They are part
        of the whole set and are counted as such — the proof is over coverage,
        not over this attempt's work, so a retry that compiles one class and
        skips 35 is exactly as whole as a first attempt that compiled 36.
        """
        counts = {int(v) for v in declared.values() if int(v) > 0}
        if not counts:
            raise EntryCompileFailed(
                "pool",
                f"none of the {width} compile child(ren) reported how many "
                f"graph classes this family declares, so there is no evidence "
                f"the shares cover it")
        if len(counts) > 1:
            raise EntryCompileFailed(
                "pool",
                f"the compile children disagree about how many graph classes "
                f"this family declares ({sorted(counts)!r} across "
                f"{sorted(declared)!r}) — they composed different pipelines, "
                f"so their shares do not partition one declaration")
        want = counts.pop()
        if len(done) + int(have) != want:
            raise EntryCompileFailed(
                "pool",
                f"the {width} share(s) packed {len(done)} graph class(es) "
                + (f"beside {have} already held " if have else "")
                + f"but every child reported {want} declared — "
                f"rows[i::{width}] did not partition the declaration and this "
                f"cell would be short")

    def _refresh_resume_facts(self) -> None:
        """Keep the LIVE ledger carrying the bank's row.

        pgw#848 refreshes `progress.pool_ledger` on every completed entry, so
        the phase snapshot an abandoned mint leaves behind already carries K
        and its binding. The resume row belongs in the same place for the same
        reason: an abandoned attempt's most useful fact is how much of it the
        NEXT one will not have to repeat.
        """
        if self.bank is not None:
            self.ledger.resume = self.bank.facts()

    def _seed_seal_memo(self) -> None:
        """pgw#832: write the parent's toolchain digests where every entry
        child can verify-and-reuse them instead of re-hashing ~4 GB apiece.

        Near-free in a sealed parent (warm cache -> stats only); pays the
        full pass exactly once in a cold one — instead of once per ENTRY.
        Failure is not a mint problem (children fall back to the full rehash,
        which is the safe path), but a systematically unusable snapshot is a
        cost regression somebody must see, so it emits the typed event the
        hub already ingests rather than a bare log."""
        t0 = time.monotonic()
        try:
            count = env_seal.write_library_memo(Path(self.seal_memo))
        except Exception as exc:  # noqa: BLE001 — the fallback path is safe
            detail = f"{type(exc).__name__}: {exc}"
            self.seal_memo = ""
            logger.warning(
                "aot-pool: pgw#832 seal memo seeding failed (%s) — every "
                "entry child will re-pay the full toolchain hash", detail)
            try:
                from . import activity as activity_mod

                activity_mod.emit_event(
                    activity_mod.KIND_AOT_MINT,
                    "seal library memo seeding failed (pgw#832): "
                    f"{detail} — entry children fall back to a full "
                    "per-child toolchain rehash (correct, but re-pays a "
                    "multi-GB SHA-256 pass once per entry)",
                    phase="pool",
                )
            except Exception:  # pragma: no cover — telemetry never fails a mint
                logger.debug("aot-pool: seed-failure event failed", exc_info=True)
            return
        self.seal_seed_s = round(time.monotonic() - t0, 3)
        self.ledger.seal_seed_s = self.seal_seed_s
        logger.info(
            "aot-pool: pgw#832 seal memo seeded (%d lib(s), %.2fs) at %s",
            count, self.seal_seed_s, self.seal_memo)

    def _emit_width(self, when: str) -> None:
        """Emit K, its binding and the underwidth THE MOMENT THEY ARE DECIDED.

        th#1359: these facts are settled at pool construction, before the first
        entry compiles — and they were reported with the phase table, which is
        flushed at the TERMINUS. Three mints have now died before producing
        them: pgw#846 attempt sixteen emitted exactly one row
        (``status=abandoned total_s=1741.33``), zero ``entry:`` rows and no
        ``pool`` row, and 29 minutes of measurement went with it. A datum that
        only survives a successful run is not a measurement of a regime where
        runs keep dying. Moving the emission — not adding a harness read —
        makes it robust to the mint dying, which on this program's record is
        the way to bet.

        Emitted at construction AND on any later change, never replacing a
        late-and-complete row with an early-and-stale one: if K is ever
        re-sized, the changed facts emit again and the last row is the true
        one. An early row that lies would be worse than a late row that is
        missing.
        """
        try:
            from . import activity as activity_mod

            facts = dict(self.width.facts())
            if facts == self._emitted_width_facts:
                return
            first = self._emitted_width_facts is None
            self._emitted_width_facts = dict(facts)
            activity_mod.emit_event(
                activity_mod.KIND_AOT_MINT,
                f"pool width ({when}) {facts}",
                phase="pool",
            )
            logger.info("aot-pool: pgw#842 width (%s) %s", when, facts)
            if not first:
                logger.warning(
                    "aot-pool: pool width CHANGED after construction — the "
                    "row above supersedes the earlier one")
        except Exception:  # pragma: no cover — telemetry never fails a mint
            logger.debug("aot-pool: width emission failed", exc_info=True)

    def _emit_ledger(self) -> None:
        """The pool's own typed event. Separate from the mint's roll-up on
        purpose: pool idle is not a mint phase, and a reader who groups them
        together would price a scheduling loss as compile work."""
        try:
            from . import activity as activity_mod

            facts = self.ledger.facts()
            activity_mod.emit_event(
                activity_mod.KIND_AOT_MINT,
                f"pool ledger (pgw#830) {facts}",
                phase="pool",
                duration_ms=int(round(self.ledger.wall_s * 1000)),
            )
            logger.info("aot-pool: pgw#830 ledger %s", facts)
        except Exception:  # pragma: no cover — telemetry never fails a mint
            logger.debug("aot-pool: ledger emission failed", exc_info=True)

    def _reap(self, running: Sequence[_Running]) -> Optional[_Running]:
        # Observed concurrency, not intended: the ONLY load-independent
        # evidence that the pool actually overlapped rather than looping.
        self.peak_concurrency = max(self.peak_concurrency, len(running))
        for row in running:
            # Sample while it is alive: /proc vanishes at exit, and VmHWM is
            # the only free high-water mark the kernel keeps. Per ROW as well
            # as pool-wide (pgw#848) — a child the OOM killer takes writes no
            # report, so this sample is the only measurement of it that ever
            # exists, and it is what sizes the next attempt's K.
            row.peak_rss_bytes = max(
                row.peak_rss_bytes, _peak_rss_bytes(row.proc))
            self.peak_rss_bytes = max(self.peak_rss_bytes, row.peak_rss_bytes)
            if row.proc.poll() is not None:
                return row
            # pgw#1243: A CHILD'S TERMINUS IS ITS REPORT, NOT ITS EXIT.
            # `aot_compile_child.run` writes the report as its last statement
            # and returns; everything after is interpreter teardown, and a
            # process that has just traced and AOTI-compiled has plenty that
            # can hang there — a non-daemon thread, a subproc pool, CUDA. Two
            # production mints packed and reported an entire cell and then sat
            # in `finalize` for 78.9 and 62 minutes while the tier above them
            # waited on an exit that never came. Everything this pool needs is
            # in the report; the corpse is not part of the contract.
            if _read_report(Path(row.job.report)) is not None:
                logger.info(
                    "aot-pool: %s wrote its report and did not exit — reaping "
                    "its group; the report is the terminus (pgw#1243)",
                    row.entry)
                row.reaped_at_terminus = True
                _terminate_group(row.proc)
                return row
            self._judge_entry_liveness(row)
        return None

    def _judge_entry_liveness(self, row: _Running) -> None:
        """Condemn a share that has stopped making MEASURED progress.

        pgw#1243. Until this existed the drain loop had no give-up test at
        all: `proc.poll()` forever, `time.sleep(_POLL_S)` forever. The
        three-tier stack used to get this for free — the mint child's own
        supervisor watched this whole process tree — and pgw#1215 step 4
        deleted that tier without moving the watch down with it.

        Progress is MEASURED (process-tree CPU plus bytes written), never a
        clock and never a frame the child could print while wedged, and the two
        signals keep separate high-water marks so a quiet one cannot cancel a
        moving one (pgw#964). A child inside a forty-minute `aot_compile`
        advances this every poll and is never touched.
        """
        if row.window is None:
            row.window = SilenceWindow(self.entry_silence_window_s)
        cpu = _tree_cpu_seconds(row.proc.pid)
        mib = _dir_mib(Path(row.job.work))
        advanced = False
        if cpu is not None and (
                row.cpu_s is None or cpu - row.cpu_s >= _ENTRY_EVIDENCE_EPS):
            row.cpu_s, advanced = cpu, True
        if mib is not None and (
                row.work_mib is None
                or mib - row.work_mib >= _ENTRY_EVIDENCE_EPS):
            row.work_mib, advanced = mib, True
        if advanced:
            row.window.touch()
            return
        if not row.window.stalled():
            return
        raise EntryCompileFailed(
            row.entry,
            f"{row.entry} (rows[{row.job.share_index}::"
            f"{row.job.share_count}]): the compile child made no measured "
            f"progress for {row.window.silent_for():.0f}s (window "
            f"{row.window.window_s:.0f}s) and wrote no report — process-tree "
            f"CPU "
            f"{'unreadable' if row.cpu_s is None else f'{row.cpu_s:.1f}s'} "
            f"and its work dir "
            f"{'unreadable' if row.work_mib is None else f'{row.work_mib:.1f}MiB'} "
            f"are both flat. It is wedged, not compiling; this build FAILS and "
            f"this worker keeps serving eager",
            peak_rss_bytes=row.peak_rss_bytes)

    def observe_entry_device(self, report: EntryReport) -> None:
        """Bank one entry child's DEVICE high-water (pgw#877 #2).

        RESERVED in preference to allocated, on the child's own argument:
        allocated is what the compile needed, reserved is what the caching
        allocator HELD and therefore what a concurrent sibling actually cannot
        have — and K is a question about siblings. A child too old to report
        reserved still contributes its allocated figure rather than nothing.
        """
        allocated = max(0, int(report.peak_device_bytes or 0))
        reserved = max(0, int(report.peak_device_reserved_bytes or 0))
        peak = reserved or allocated
        if peak > 0:
            self.peak_device_bytes = max(self.peak_device_bytes, peak)
        # pgw#1205: and the per-class row, both readings kept apart. Maxed per
        # field so a retry of the same entry widens rather than replaces.
        entry = str(report.entry or "").strip()
        if entry and (allocated > 0 or reserved > 0):
            held_a, held_r = self.entry_device_peaks.get(entry, (0, 0))
            self.entry_device_peaks[entry] = (
                max(held_a, allocated), max(held_r, reserved))

    def _collect(self, row: _Running) -> List[PackedGraphClass]:
        elapsed = time.monotonic() - row.started
        reap_epoch = time.time()
        self.entry_seconds[row.entry] = round(elapsed, 2)
        code = row.proc.returncode
        report = _read_report(Path(row.job.report))
        if row.reaped_at_terminus and report is not None:
            # pgw#1243: this child reached its own terminus and then failed to
            # die, so the parent ended it — the exit code below is the
            # PARENT's signal and says nothing about the compile. The report
            # carries the same classification the exit code does, so read it
            # there and let the ladder run unchanged. Without this a share
            # that packed every one of its graph classes would be reported as
            # a signal death and its artifacts thrown away.
            code = EXIT_COMPILED if report.status == COMPILED else EXIT_REFUSED
        if report is not None:
            # pgw#877: banked BEFORE any gate can raise, and on the failure
            # path too — pgw#848's rule for the host half applies unchanged
            # here: the attempt that FAILED is exactly the attempt whose
            # measurement the next one has to size against.
            self.observe_entry_device(report)
            self._verify_child_code(row, report)
        # An EMPTY share is legitimate and must not read as a failure: the
        # parent sizes K from an EXPECTED class count and never enumerates the
        # real one, so a declaration with fewer classes than the pool has
        # children genuinely leaves a child with nothing to do. Whether the
        # shares together cover the declaration is `_assert_shares_whole`'s
        # question, asked once over every child's reported count — asking it
        # here, per child, would refuse the legitimate case and give the
        # illegitimate one the wrong name.
        if code == EXIT_COMPILED and report is not None:
            if report.peak_rss_bytes:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(report.peak_rss_bytes))
            missing = [
                c.artifact for c in report.classes
                if not c.artifact or not Path(c.artifact).exists()]
            if missing:
                raise EntryCompileFailed(
                    row.entry,
                    f"{row.entry}: child reported {len(report.classes)} packed "
                    f"graph class(es) but {len(missing)} artifact(s) do not "
                    f"exist (first: {missing[0] or '<no path>'}) — the pool's "
                    f"out_dir {row.job.out_dir!r} is not visible to this "
                    f"process")
            self.entry_declared[row.entry] = int(report.declared_classes or 0)
            self.entry_phases[row.entry] = self._close_entry_partition(
                row, report, elapsed=elapsed, reap_epoch=reap_epoch)
            self.entry_overlays[row.entry] = dict(report.overlays or {})
            self.entry_metrics_raw[row.entry] = dict(report.metrics_raw or {})
            # pgw#1205's per-class row, at the granularity the child measured
            # it: one share is several graph classes, and "how big was the
            # biggest compile in this SHARE" answers nothing anybody asks.
            for packed in report.classes:
                self.class_spans[packed.name] = dict(packed.spans or {})
            logger.info(
                "aot-pool: %s packed %d graph class(es) in %.1fs spans=%s",
                row.entry, len(report.classes), elapsed,
                self.entry_phases[row.entry])
            self._refresh_resume_facts()
            return list(report.classes)
        # pgw#1215: a share that REFUSED at class k still packed k-1
        # artifacts, and they are on disk. Their measurement is banked before
        # the raise for pgw#848's reason — the attempt that FAILED is exactly
        # the attempt the next one has to size against — and `refused_classes`
        # names what exists so a caller is never told the share produced
        # nothing when it produced most of a cell.
        if report is not None and report.classes:
            for packed in report.classes:
                self.class_spans[packed.name] = dict(packed.spans or {})
            self.refused_classes[row.entry] = list(report.classes)
            self.entry_declared[row.entry] = int(report.declared_classes or 0)
        detail = report.detail if report is not None else ""
        if not detail:
            detail = _stderr_tail(row.stderr_path)
        resource, basis = self._memory_verdict(code, report)
        if resource:
            self.oom_entry, self.oom_basis = row.entry, basis
        raise EntryCompileFailed(
            row.entry,
            f"{row.entry} (rows[{row.job.share_index}::"
            f"{row.job.share_count}]): compile child exited {code} after "
            f"{elapsed:.0f}s ({_exit_note(code)}): {detail or 'no detail'}"
            + (
                f" [{len(self.refused_classes.get(row.entry) or ())} graph "
                f"class(es) from this share ARE packed and on disk]"
                if self.refused_classes.get(row.entry) else "")
            + (
                f" [pgw#848 classification: MEMORY SHORTFALL, basis={basis}; "
                f"this share's measured high-water was "
                f"{row.peak_rss_bytes / 1024**3:.2f} GiB and the pool ran "
                f"K={self.width.workers} against a "
                f"{self.width.per_entry_rss_bytes / 1024**3:.2f} GiB/entry "
                f"({self.width.per_entry_rss_basis}) ask — this is retryable "
                f"at a narrower K, NOT a deterministic refusal]"
                if resource else ""),
            resource=resource, basis=basis,
            peak_rss_bytes=row.peak_rss_bytes)

    def _memory_verdict(
        self, code: Optional[int], report: Optional[EntryReport],
    ) -> Tuple[bool, str]:
        """Did this entry die of MEMORY, and on what evidence?

        pgw#848. Two bases, and they are not equivalent:

        * ``cgroup`` — the kernel's own ``memory.events`` ``oom_kill`` counter
          moved while this pool ran. That is a fact, not an inference.
        * ``sigkill`` — the child was SIGKILLed and wrote no report, with no
          usable counter to corroborate it. The pool has documented this
          shape as "the OOM killer far more often than a compiler bug" since
          pgw#809; one retry at a narrower K is the right response to a
          "far more often", and a wrong guess costs one retry rather than a
          permanently unmintable cell.

        A child that wrote a report classified ITSELF and is believed: a
        named refusal is deterministic no matter how it exited.
        """
        if report is not None:
            return False, ""
        if code is None or code >= 0 or -int(code) != int(signal.SIGKILL):
            return False, ""
        now = cgroup_oom_kills()
        if now >= 0 and self.oom_kills_at_start >= 0 \
                and now > self.oom_kills_at_start:
            return True, "cgroup"
        return True, "sigkill"

    def _verify_child_code(self, row: _Running, report: EntryReport) -> None:
        """pgw#840: the child that compiled this entry must BE the parent.

        Not a telemetry check. The child produces the loose files
        ``package_aoti`` packs and the cell publishes, while every gate runs in
        the parent against the parent's program — an assignment that is only
        sound while both are the same code. A skewed child was invisible: it
        compiled successfully, returned files that exist, and differed only in
        what it reported. MEASURED on this box: of 236 preserved entry reports,
        150 were written by a child that predates pgw#830's span table, several
        of them under a parent that had it (the pool workdir holds the pgw#832
        seal memo only a post-pgw#832 parent writes). That is the whole of
        pgw#840: the invariant went red on a tree nobody had changed, because
        the child was not from that tree.

        Refused, not warned: an artifact compiled by unknown code must not be
        packed into a cell whose identity claims the parent's.
        """
        if not CODE_DIGEST:
            return  # no source to compare (zipimport) — cannot prove either way
        if report.code_digest == CODE_DIGEST:
            return
        raise EntryCompileFailed(
            row.entry,
            f"entry {row.entry!r}: the compile child ran a DIFFERENT "
            f"gen_worker than this parent — child code "
            f"{report.code_digest or '<too old to report one>'} from "
            f"{report.code_dir or '<unknown>'}, parent code {CODE_DIGEST} from "
            f"{PACKAGE_ROOT}. `python -m {ENTRY_CHILD_MODULE}` resolves "
            f"whatever the interpreter's path yields (a second checkout, an "
            f"inherited PYTHONPATH, a stale wheel, or this tree edited between "
            f"the parent's import and this spawn), and that child compiled the "
            f"files this cell would publish while every gate ran against the "
            f"parent's program")

    def _close_entry_partition(
        self, row: _Running, report: EntryReport, *,
        elapsed: float, reap_epoch: float,
    ) -> Dict[str, float]:
        """pgw#830: close the outermost partition — the one that spans the
        process boundary and therefore could never be closed inside the child.

        ``compile_s = child_boot_s + child_wall_s + reap_lag_s
        + parent_other_s``, where ``child_boot_s`` is interpreter startup plus
        this package's import (paid once per ENTRY, because the pool's unit of
        parallelism is a process that exits) and ``reap_lag_s`` is the child's
        exit plus the parent's poll granularity.

        pgw#1099: ``reap_lag_s`` is a MEASURED span and may never double as
        the partition's catch-all. When a child reports no epochs the split is
        unknowable, so the unclaimed remainder lands in ``parent_other_s`` —
        the declared residual — and ``reap_lag_s`` stays 0. Overloading it
        cost a real investigation: pgw#1085 §5c read a 259.6 s median off the
        residual branch as poll lag and pgw#1099 was filed against a lever
        that did not exist.
        """
        spans = dict(report.spans or {})
        spans["compile_s"] = round(elapsed, 3)
        if report.run_start_epoch and row.spawn_epoch:
            spans["child_boot_s"] = round(
                report.run_start_epoch - row.spawn_epoch, 3)
        if report.module_import_epoch and row.spawn_epoch:
            # Split of `child_boot_s`, reported as an overlay-style detail:
            # interpreter exec + gen_worker import, before any of the child's
            # own code runs. It is what a persistent worker would delete.
            spans["child_interp_s"] = round(
                report.module_import_epoch - row.spawn_epoch, 3)
        if report.report_epoch:
            spans["reap_lag_s"] = round(reap_epoch - report.report_epoch, 3)
        # The outer partition's residual: recorded on EVERY entry (0.0 when the
        # named members closed it), so `check` covers it and `dark_fraction`
        # counts it instead of a measured span silently absorbing the gap.
        spans["child_boot_s"] = spans.get("child_boot_s", 0.0)
        spans["reap_lag_s"] = spans.get("reap_lag_s", 0.0)
        spans["parent_other_s"] = round(
            spans["compile_s"] - spans["child_boot_s"]
            - float(spans.get("child_wall_s", 0.0))
            - spans["reap_lag_s"], 3)
        violations = aot_compile_spans.check(spans)
        if violations:
            # Named, loud, and non-fatal: an attribution defect must never
            # fail a mint, and must never be silent either (pgw#824's class).
            logger.warning(
                "aot-pool: pgw#830 attribution defect on entry %r: %s",
                row.entry, "; ".join(violations))
        spans["child_interp_s"] = spans.get("child_interp_s", 0.0)
        # Parent-side work for THIS entry. Prefixed, and listed in
        # `aot_compile_spans.SUBSPANS`, because it is not inside `compile_s`:
        # staging overlaps other children, so summing it into the compile
        # total would invent seconds nobody spent compiling. Its idle FRACTION
        # is `ledger.idle_staging_s`, which is a pool number, not an entry one.
        spans["parent_stage_s"] = self.entry_stage_seconds.get(row.entry, 0.0)
        spans["parent_spawn_s"] = self.entry_spawn_seconds.get(row.entry, 0.0)
        return spans

def _stderr_tail(path: Path, limit: int = 2048) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return raw[-limit:].decode("utf-8", "replace").strip()


def _exit_note(code: Optional[int]) -> str:
    if code is None:
        return "still running"
    if code == EXIT_REFUSED:
        return "named refusal"
    if code == EXIT_BAD_JOB:
        return "malformed job — a wiring defect"
    if code < 0:
        name = f"SIG{-code}"
        try:
            name = signal.Signals(-code).name
        except ValueError:
            pass
        if -code in (signal.SIGKILL, signal.SIGSEGV):
            return (
                f"killed by {name} — on a mint this is the OOM killer far "
                f"more often than a compiler bug; the pool's width is "
                f"memory-bounded for exactly this reason")
        return f"killed by {name}"
    return "crashed"


__all__ = [
    "CODE_DIGEST",
    "COMPILED",
    "CPUS_PER_ENTRY_WORKER",
    "DEFAULT_ENTRY_PEAK_RSS_BYTES",
    "ENTRY_CHILD_MODULE",
    "ENTRY_REPORT_NAME",
    "ENTRY_RSS_RESERVE_BYTES",
    "EXIT_BAD_JOB",
    "EXIT_COMPILED",
    "EXIT_REFUSED",
    "EntryCompileFailed",
    "EntryCompilePool",
    "arm_parent_death_signal",
    "EntryJob",
    "EntryReport",
    "MAX_ENTRY_WORKERS",
    "PACKAGE_ROOT",
    "REFUSED",
    "SERVING_HEADROOM_CPUS",
    "CpuFacts",
    "MemoryFacts",
    "PoolLedger",
    "PoolWidth",
    "child_argv",
    "child_env",
    "cpu_facts",
    "entry_workers",
    "memory_facts",
]
