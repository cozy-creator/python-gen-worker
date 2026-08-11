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

The handoff, and its cost
-------------------------
A child cannot inherit the exported program (``fork`` is banned after CUDA
init, pgw#784), so it arrives on disk: ``torch.export.save`` in the parent,
``torch.export.load`` in the child. Two facts make that affordable:

* **It is byte-exact.** A compile after the roundtrip produces a
  ``wrapper.cpp`` byte-identical to the in-process compile, and lands under
  the SAME inductor cache hash — the cache key is the graph, not the process.
* **It is off the critical path.** Only the FIRST save is serial; every later
  save overlaps a child that is already compiling, and a save (~16 s at 2.5 GB)
  is 4 % of a compile (~420 s). The parent is never the bottleneck.

The parent keeps its own in-memory program regardless — every package-side
gate (``program_package_drift``, ``eliminated_constants``, ``input_contract``)
runs against the parent's real program and the child's package, so a child
that diverged is caught by gates that already exist, named by entry.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional,
    Sequence, Tuple)

import msgspec

from . import aot_shape_hints

from . import aot_compile_spans, aot_device_lock, aot_resume, env_seal
from . import mint_budget, worker_goals
from .worker_goals import WorkerGoals
from .postmortem import cpu_quota_cores
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
#: mint-time host OOM. Either means this bound, not the device bound, was
#: binding — and the answer then is to make the FIRST entry serial and bank its
#: peak, not to guess a larger constant.
#:
#: Banked per (family, lane) once measured, exactly like
#: ``mint_budget.record_child_peak`` banks the device peak.
DEFAULT_ENTRY_PEAK_RSS_BYTES = 3 * 1024**3

#: Host RAM a FORGE pod leaves alone (th#1359). Not a tenant reserve — there
#: is no tenant — but the OS, the page cache and the mint child's own
#: supervisor are real on any pod. A quarter of the serving reserve.
FORGE_RSS_RESERVE_BYTES = 1 * 1024**3

#: VRAM the pool must leave to the tenant. The mint's whole premise is that
#: the worker keeps serving (pgw#784), so the eager forward's weights AND its
#: activation peak stay untouchable; this is the margin ON TOP of the free
#: figure, because "free right now" is measured between tenant forwards.
DEVICE_RESERVE_BYTES = 2 * 1024**3

#: pgw#877 #5: there is NO per-entry device default, deliberately.
#: ``DEFAULT_ENTRY_DEVICE_BYTES = 8 GiB`` sat here and was UNREACHABLE:
#: ``aot_mint._entry_device_bytes`` returns 0 only when ``co_residency`` is
#: unprobed, and ``_probe_free_device_bytes`` returns 0 under exactly those
#: conditions, so ``free_vram <= 0`` short-circuited before the fallback could
#: be consulted. A constant nothing can reach still reads as a policy, and
#: this one read as "8 GiB is a reasonable entry". Deleted; a readable card
#: with no footprint now refuses to widen instead, because the failure mode
#: of guessing here is an OOM on paid work.

#: Programs staged AHEAD of the running set. The export loop hands the pool
#: every entry at once; staging them all would put ~46 GB of exported programs
#: on disk for an 18-entry sdxl cell. One spare per pool is enough to keep a
#: freed slot from waiting on a multi-GB write.
INFLIGHT_PROGRAM_SLACK = 1

_KILL_GRACE_S = 10.0
_POLL_S = 0.25


#: pgw#842: how many times the device bound samples free VRAM before it
#: believes a number, and how far apart. The pool shares the card with a
#: SERVING process by construction (pgw#784), so a single ``mem_get_info``
#: can land inside a tenant forward and read that forward's activation set as
#: "gone". :data:`DEVICE_RESERVE_BYTES` already reserves the tenant's peak;
#: subtracting an in-flight peak on top of it charges the same bytes twice —
#: measured as a 5 -> 3 width swing on identical work. The MAX over a short
#: window is the steady figure the reserve was written against. Three samples
#: over 0.1 s costs nothing against a mint that runs for minutes.
DEVICE_FREE_SAMPLES = 3
DEVICE_FREE_SAMPLE_GAP_S = 0.05

#: Entry children whose DEVICE high-water must be in hand before the pool
#: re-derives its own K from them (:meth:`EntryCompilePool._rewiden`).
#:
#: One is an anecdote. The first round's children start together, so they are
#: also the ones that miss the shared PCH and the autotune cache, and a width
#: re-derived from a single unrepresentative child is exactly the "5-vs-3 with
#: nothing recorded to say why" defect pgw#842 closed. Two is the smallest
#: number that is a measurement rather than a sample, and on a 36-entry sdxl
#: cell it arrives after ~6 % of the work — early enough that the other 94 %
#: is compiled at the measured width.
REWIDEN_MIN_SAMPLES = 2


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
class DeviceFacts:
    """Free VRAM the pool may divide, and every sample behind it."""

    free_bytes: int
    basis: str
    samples: Tuple[int, ...]

    def facts(self) -> Dict[str, Any]:
        return {
            "free_device_bytes": int(self.free_bytes),
            "device_basis": self.basis,
            "free_device_samples": [int(x) for x in self.samples],
        }


@dataclass(frozen=True)
class CardCensus:
    """Who holds the card, taken BEFORE the pool spawns its first child.

    pgw#992: the one reading that makes a simultaneity bound computable. At
    pool construction no entry child exists, so everything on the device that
    is not this process is, by elimination, the RESIDENT co-tenant — the
    eager-serving parent the pgw#784 contract keeps alive through the mint.
    Taken later the same subtraction would be meaningless, because the pool's
    own children would be inside it.
    """

    total_bytes: int
    free_bytes: int
    own_reserved_bytes: int
    basis: str

    @property
    def resident_other_bytes(self) -> int:
        """The co-tenant's occupancy. Never negative: a driver that reports
        `free + own > total` is reporting something this bound must not turn
        into free capacity."""
        return max(0, self.total_bytes - self.free_bytes - self.own_reserved_bytes)

    @property
    def readable(self) -> bool:
        return self.basis == "sampled" and self.total_bytes > 0

    def facts(self) -> Dict[str, Any]:
        return {
            "card_total_bytes": int(self.total_bytes),
            "card_free_at_open_bytes": int(self.free_bytes),
            "card_own_at_open_bytes": int(self.own_reserved_bytes),
            "card_resident_other_bytes": int(self.resident_other_bytes),
            "card_census_basis": self.basis,
        }


def card_census(device: int = -1) -> CardCensus:
    """One (total, free, own-reserved) reading of the mint's card.

    All three at the same moment on purpose: the subtraction that names the
    co-tenant is only sound if its terms describe one instant.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return CardCensus(0, 0, 0, "absent")
        dev = torch.cuda.current_device() if device < 0 else int(device)
        free, total = torch.cuda.mem_get_info(dev)
        return CardCensus(
            int(total), int(free), int(torch.cuda.memory_reserved(dev)),
            "sampled")
    except Exception:  # noqa: BLE001 — an unreadable card licenses nothing
        return CardCensus(0, 0, 0, "unreadable")


def own_reserved_now(device: int = -1) -> int:
    """This process's CURRENT reserved bytes (-1 = unreadable).

    pgw#1053: the post-release floor. Distinct from
    :func:`own_device_high_water` on purpose — the high-water answers "what
    did this process ever hold" and can only be re-baselined through
    ``reset_peak_memory_stats``; this answers "what does it hold NOW", which
    is what the released budget re-derives from.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return -1
        dev = torch.cuda.current_device() if device < 0 else int(device)
        return int(torch.cuda.memory_reserved(dev))
    except Exception:  # noqa: BLE001 — unreadable regrants nothing
        return -1


def own_device_high_water(device: int = -1) -> int:
    """This process's own device high-water (0 = unreadable).

    RESERVED, not allocated — the caching allocator's held blocks are exactly
    what a co-resident child cannot have, which is the question a simultaneity
    bound asks. The mint child's resident pipeline is the largest single
    consumer on the card (16.20 GiB of 44.39 on the pgw#992 pod), and it is the
    one consumer this process can measure exactly.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        dev = torch.cuda.current_device() if device < 0 else int(device)
        return max(int(torch.cuda.max_memory_reserved(dev)),
                   int(torch.cuda.memory_reserved(dev)))
    except Exception:  # noqa: BLE001
        return 0


@dataclass(frozen=True)
class PoolWidth:
    """The chosen K and every input that chose it — so a mint's telemetry can
    answer "why this width" without re-deriving anything."""

    workers: int
    entries: int
    vcpus: int
    cpu_workers: int
    mem_workers: int
    device_workers: int
    available_bytes: int
    free_device_bytes: int
    per_entry_rss_bytes: int
    per_entry_device_bytes: int
    device_lock: bool
    reason: str
    #: pgw#842: the constraint that ACTUALLY held K down, by name, plus the
    #: readings each bound was taken from. A width narrower than the pod could
    #: carry is a performance defect, and it has to be legible from one record
    #: rather than inferred by diffing two pods that no longer exist.
    binding: str = ""
    ceiling: int = MAX_ENTRY_WORKERS
    #: The caller's own cap (``entry_workers(limit=)``), 0 when uncapped. It is
    #: an INPUT that chose K and was the one input this record did not carry —
    #: which mattered the moment :meth:`EntryCompilePool._rewiden` began
    #: re-deriving K mid-mint: an operator who forced the serial path must not
    #: have it widened back out from under them by a later measurement.
    limit: int = 0
    cpu: Optional[CpuFacts] = None
    memory: Optional[MemoryFacts] = None
    device: Optional[DeviceFacts] = None
    #: ``"estimated"`` when the caller handed a per-entry device ask,
    #: ``"default"`` when the width fell back to a constant.
    #:
    #: It read ``"measured"`` until pgw#877, and that overstated it. The only
    #: caller is ``aot_mint._entry_device_bytes``, which returns
    #: ``mint_budget.co_residency().need_bytes`` — the MINT CHILD's resident
    #: set times :data:`~gen_worker.mint_budget._UNMEASURED_ACTIVATION_FRACTION`
    #: plus two flat constants, i.e. ~56 % of the number was never observed and
    #: no entry child was ever watched. ``EntryReport.peak_device_bytes`` is
    #: the observation, and nothing reads it yet; until something does, this
    #: axis has no ``"measured"`` value to report and must not claim one.
    per_entry_device_basis: str = "default"
    #: ``"measured"`` here is literal: the value is one entry child's VmHWM
    #: summed over its real descendant tree (``_peak_rss_bytes``), banked by
    #: the serving parent (``mint_budget.record_entry_peak_rss``).
    per_entry_rss_basis: str = "default"
    #: pgw#930 (§1.17): the two goals, reported INDEPENDENTLY. A K of 1 on a
    #: pod holding a serve goal and a K of 1 on a mint-only pod are different
    #: defects, and the row has to say which one it is. This was one boolean
    #: named `forge`, which could not describe a pod holding both goals — the
    #: exact case Paul's ruling requires to work.
    serve_goal: bool = True
    mint_goal: bool = False

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
            "device_workers": int(self.device_workers),
            "available_bytes": int(self.available_bytes),
            "free_device_bytes": int(self.free_device_bytes),
            "per_entry_rss_bytes": int(self.per_entry_rss_bytes),
            "per_entry_device_bytes": int(self.per_entry_device_bytes),
            "device_lock": bool(self.device_lock),
            "binding": self.binding,
            "ceiling": int(self.ceiling),
            "limit": int(self.limit),
            "underwidth": int(self.underwidth),
            "per_entry_device_basis": self.per_entry_device_basis,
            "per_entry_rss_basis": self.per_entry_rss_basis,
            "serve_goal": bool(self.serve_goal),
            "mint_goal": bool(self.mint_goal),
            "width_reason": self.reason,
        }
        for block in (self.cpu, self.memory, self.device):
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


class DeviceProbeError(RuntimeError):
    """The card could not be read (pgw#940).

    Distinct from "there is no card", which is a 0 return. Collapsing the two
    is the whole defect: `except Exception: return 0` made a transient
    `mem_get_info` failure, a post-fork CUDA-context error and a flapping
    `is_available()` indistinguishable from a CPU-only pod — and every caller
    read that shared zero as the permissive case.
    """


def _probe_free_device_bytes(device: int = -1) -> int:
    """One reading of free VRAM on the mint's card, 0 when there is no card.

    Reads the ALLOCATOR's view of free plus what this process has reserved
    but not allocated, exactly as ``mint_budget`` does — a cached block the
    tenant is not using is free to nobody but this process, and pretending
    otherwise is how a mint OOMs a live request.

    Raises :class:`DeviceProbeError` when a card is present but unreadable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
    except Exception as exc:  # noqa: BLE001
        # Even "is there a card" did not answer. That is unreadable, not
        # absent: a pod with no torch at all fails at import long before here.
        raise DeviceProbeError(f"cuda availability unreadable: {exc}") from exc
    try:
        dev = torch.cuda.current_device() if device < 0 else int(device)
        free, _total = torch.cuda.mem_get_info(dev)
        reserved = int(torch.cuda.memory_reserved(dev))
        allocated = int(torch.cuda.memory_allocated(dev))
        return int(free) + max(0, reserved - allocated)
    except Exception as exc:  # noqa: BLE001
        raise DeviceProbeError(f"free VRAM unreadable: {exc}") from exc


def device_facts(
    device: int = -1,
    *,
    samples: int = DEVICE_FREE_SAMPLES,
    gap_s: float = DEVICE_FREE_SAMPLE_GAP_S,
    probe: Optional[Callable[[int], int]] = None,
) -> DeviceFacts:
    """Free VRAM the pool may divide — the STEADY figure, not an instant.

    pgw#842: the mint shares the card with the serving process, so one
    ``mem_get_info`` taken while a tenant forward holds its activation set
    reads several GiB below the steady free figure — and the pool then
    subtracts :data:`DEVICE_RESERVE_BYTES` for that same tenant peak on top.
    The MAX over a short window is the reading the reserve was written
    against; every sample is kept so the choice is auditable.
    """
    read = probe if probe is not None else _probe_free_device_bytes
    taken: List[int] = []
    for i in range(max(1, int(samples))):
        if i and gap_s > 0:
            time.sleep(gap_s)
        try:
            value = int(read(device))
        except DeviceProbeError as exc:
            # pgw#940: "no card" and "unreadable card" are different facts and
            # the pool must decide them differently. Only the caught type is
            # narrowed here — anything else still propagates, because a probe
            # that raises something unexpected is not a measurement outcome.
            logger.warning("free-VRAM probe failed: %s", exc)
            return DeviceFacts(0, "unreadable", tuple(taken))
        taken.append(value)
        if value <= 0:
            # No card. Sampling an absence is not evidence of a size.
            return DeviceFacts(0, "absent", tuple(taken))
    return DeviceFacts(max(taken), "sampled", tuple(taken))


def entry_workers(
    entries: int,
    *,
    peak_rss_bytes: int = 0,
    device_bytes: int = 0,
    vcpus: int = 0,
    available_bytes: int = -1,
    free_vram_bytes: int = -1,
    device_basis: str = "",
    limit: int = 0,
    device_lock: Optional[bool] = None,
    goals: Optional[WorkerGoals] = None,
) -> PoolWidth:
    """How many entries this pod may compile at once.

    Derived, never configured, from THREE bounds:

    * **VRAM — the one that actually binds.** An AOTI compile benchmarks
      kernels on the card, so every concurrent entry child holds its own
      weight copy, activation set and CUDA context. On a 24 GB card with the
      tenant's model resident that is K=2-3 whatever the CPU says. Read via
      :func:`device_facts` — the STEADY free figure, never one sample.
    * **vCPU**, from :func:`cpu_facts` (cgroup quota AND affinity mask AND
      host cores, whichever is narrowest) minus
      :data:`SERVING_HEADROOM_CPUS`. ~94 % of an entry compile is ONE core of
      serial host work, so this bound is generous and scales near-perfectly.
    * **Host RAM**, the loosest of the three: the wrapper ``cc1plus`` peaks at
      ~2.1 GiB, so a pod that has VRAM for K has RAM for K several times over.
      Read via :func:`memory_facts`, whose cgroup half counts the WORKING SET
      rather than everything the pod has ever paged in.

    ``device_lock=False`` FORCES K=1 on a GPU cell: without torch's
    ``set_gpu_benchmark_lock_context`` hook the pool cannot stop two entries
    benchmarking at once, and a cell whose kernel configs were chosen under
    self-inflicted contention publishes under an unchanged key. Refusing to
    widen is the only safe answer.

    pgw#842: every bound records the READING behind it (:class:`CpuFacts`,
    :class:`MemoryFacts`, :class:`DeviceFacts`) and the returned width names
    the constraint that actually bound. K is the mint's only multiplicative
    lever — two mints of one cell differed 5-vs-3 with nothing recorded to
    say why — so an unexplained K is a defect in itself.
    """
    entries = max(0, int(entries))
    if goals is None:
        goals = worker_goals.current()
    # pgw#930 (§1.17): THREE of this policy's terms are tenant reserves, and a
    # pod with no SERVE goal has no tenant. `SERVING_HEADROOM_CPUS` keeps cores
    # for an eager forward and a heartbeat; `DEVICE_RESERVE_BYTES` keeps VRAM
    # for the tenant's peak; `ENTRY_RSS_RESERVE_BYTES` keeps host RAM so a
    # request arriving mid-mint does not meet the OOM killer. A pod holding no
    # serve goal receives no tenant dispatch, so all three protect nobody — and
    # on pgw#846's attempts fourteen and fifteen the VRAM reserve alone held
    # the pool at K=1 on a host that could have run it 127 CPU-side.
    #
    # These three used to be `0 if forge else X` — strictly two-valued, keyed
    # on an exclusive mode. Keyed on the SERVE GOAL they compose: a pod serving
    # one small model while driving a scheduled mint keeps every reserve, which
    # is the case the mode could not express.
    #
    # A SMALL host-RAM reserve survives regardless: the OS, the page cache and
    # the mint child's own supervisor are real on a mint-only pod too. It is
    # the TENANT's share that goes, not prudence.
    reserve = goals.tenant_reserve_applies()
    cpu_headroom = SERVING_HEADROOM_CPUS if reserve else 0
    device_reserve = DEVICE_RESERVE_BYTES if reserve else 0
    rss_reserve = ENTRY_RSS_RESERVE_BYTES if reserve else FORGE_RSS_RESERVE_BYTES
    locked = aot_device_lock.supported() if device_lock is None \
        else bool(device_lock)
    if entries <= 1:
        # pgw#877: the entry count alone decides this, so no bound is READ —
        # and the row must not report unread bounds as zeros. It used to say
        # `available_bytes=0, free_device_bytes=0`: a row whose entire job is
        # to explain K=1, telling its reader the pod has no RAM and no card.
        # `-1` is this module's existing "not read" (`cgroup_available_bytes`,
        # `quota_cores`), and the bases say so in words.
        return PoolWidth(
            workers=1, entries=entries, vcpus=0, cpu_workers=1, mem_workers=1,
            device_workers=1, available_bytes=-1, free_device_bytes=-1,
            per_entry_rss_bytes=0, per_entry_device_bytes=0,
            per_entry_rss_basis="not-read", per_entry_device_basis="not-read",
            device_lock=locked, binding="entries", ceiling=1,
            limit=max(0, int(limit)),
            serve_goal=goals.serve, mint_goal=goals.mint,
            reason=(
                f"{entries} entr{'y' if entries == 1 else 'ies'}: serial "
                f"(no cpu/memory/device bound was read — the entry count "
                f"decides this width on its own)"))

    if vcpus > 0:
        cpu = CpuFacts(int(vcpus), "caller", int(vcpus), int(vcpus), -1.0)
    else:
        cpu = cpu_facts()
    vcpus = cpu.vcpus
    budget = vcpus - cpu_headroom
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

    if free_vram_bytes >= 0:
        device = DeviceFacts(
            int(free_vram_bytes),
            "caller" if free_vram_bytes > 0 else "absent",
            (int(free_vram_bytes),))
    else:
        device = device_facts()
    free_vram = device.free_bytes
    per_device = max(0, int(device_bytes))
    if free_vram <= 0 and device.basis == "unreadable":
        # pgw#940. This branch used to be shared with "no card" and yielded
        # MAX_ENTRY_WORKERS for both, so a GPU pod whose probe raised compiled
        # eight children device-UNBOUNDED. The old comment defended it by
        # asserting `_probe_free_device_bytes` fails only when there is no
        # card — which was false, and is now true by construction: an
        # unreadable card raises `DeviceProbeError` and lands HERE, closed,
        # beside its two sibling branches. The stake is stated at :192-201 —
        # "the failure mode of guessing here is an OOM on paid work."
        device_workers = 1
    elif free_vram <= 0:
        # No card at all: the device bound is DROPPED and K falls to whichever
        # of cpu/mem/ceiling binds. Right for a CPU-only cell, which is not
        # device-bound at all.
        device_workers = MAX_ENTRY_WORKERS
    elif per_device <= 0:
        # pgw#877 #5: a card we CAN read but have no per-entry footprint for.
        # Unreachable in production today (see the note where the deleted
        # default used to live), and stated explicitly rather than left to a
        # constant: an unmeasured footprint does not license concurrency on
        # the card it cannot describe.
        device_workers = 1
    else:
        device_workers = max(
            1, int(max(0, free_vram - device_reserve) // per_device))

    # A caller cap NARROWS. `limit` above MAX_ENTRY_WORKERS is a caller
    # asking for more than the ceiling allows, and the ceiling wins.
    ceiling = min(MAX_ENTRY_WORKERS, int(limit)) if limit > 0 \
        else MAX_ENTRY_WORKERS
    # pgw#877: ONE definition of each basis. It was written twice — once for
    # the row, once for the reason string — which is how the reason could
    # have drifted from the field it explains.
    # THREE values, because there are three provenances and collapsing them is
    # the defect this issue is named for:
    #   "measured"   — a real entry child was watched (pgw#877 #1/#2)
    #   "estimated"  — `co_residency().need_bytes`, ~56 % never observed
    #   "unmeasured" — a readable card and no footprint at all -> K=1
    # The caller states which; a bare non-zero ask cannot tell them apart, and
    # "the caller handed me a number" is precisely the overstatement that made
    # `"measured"` meaningless in the first place.
    device_basis = str(device_basis or "").strip() or (
        "estimated" if device_bytes > 0 else "unmeasured")
    rss_basis = "measured" if peak_rss_bytes > 0 else "default"

    def _width(
        workers: int, *, binding: str, reason: str, lock: bool,
    ) -> PoolWidth:
        return PoolWidth(
            workers=workers, entries=entries, vcpus=vcpus,
            cpu_workers=cpu_workers, mem_workers=mem_workers,
            device_workers=device_workers, available_bytes=avail,
            free_device_bytes=free_vram, per_entry_rss_bytes=per_entry,
            per_entry_device_bytes=per_device, device_lock=lock,
            reason=reason, binding=binding, ceiling=ceiling, cpu=cpu,
            memory=memory, device=device,
            per_entry_device_basis=device_basis,
            per_entry_rss_basis=rss_basis,
            limit=max(0, int(limit)),
            serve_goal=goals.serve, mint_goal=goals.mint)

    workers = max(
        1, min(cpu_workers, mem_workers, device_workers, ceiling, entries))
    if workers > 1 and free_vram > 0 and not locked:
        return _width(
            1, binding="device-lock", lock=False,
            reason=(
                "serial: this torch has no GPU-benchmark lock hook, so a wide "
                "pool would let entries benchmark against each other and bake "
                "contention-chosen kernel configs into a cell whose key would "
                "not move"))
    binding = min(
        (cpu_workers, "cpu"), (mem_workers, "host-memory"),
        (device_workers, "vram"), (ceiling, "ceiling"),
        (entries, "entries"))[1]
    reason = (
        f"K={workers} ({binding}-bound, goals="
        f"{'serve+mint' if goals.serve and goals.mint else 'serve' if goals.serve else 'mint' if goals.mint else 'none'}): "
        f"{vcpus} vCPU ({cpu.basis}) -> "
        f"{cpu_workers}, {avail / 1024**3:.1f} GiB RAM ({memory.basis}) -> "
        f"{mem_workers}, {free_vram / 1024**3:.1f} GiB VRAM ({device.basis}) "
        f"/ {per_device / 1024**3:.1f} GiB per entry "
        f"({device_basis}) -> "
        f"{device_workers}")
    return _width(workers, binding=binding, reason=reason, lock=locked)


# ---------------------------------------------------------------------------
# The wire: one job in, one report out (same shape as pgw#784's mint child)
# ---------------------------------------------------------------------------


class EntryJob(msgspec.Struct, frozen=True, kw_only=True):
    """One entry's compile, as a file a human can re-run by hand."""

    entry: str
    program: str
    report: str
    inductor_configs: Dict[str, Any] = {}
    cache_dir: str = ""
    device_lock: str = ""
    #: pgw#998: the tracing process's ShapeEnv symbol values. `torch.export`'s
    #: round trip rebuilds `var_to_val` keyed by size EXPRESSIONS, so a
    #: derived symbol (`multiple_of` -> `2*s18`) leaves every extent that is
    #: not literally one of those keys — a matmul M that multiplies two of
    #: them — unrealizable, and inductor dies with `('unexpected None!',
    #: 512*s18*s57)`. The parent is the only process that knows these, so it
    #: sends them rather than letting the child infer them.
    symbol_values: Dict[str, int] = {}
    #: pgw#998: `{symbol: the dim name the AUTHOR wrote}`. Debug surfaces do
    #: not survive serialization, so a child that has to refuse can only say
    #: `512*s18*s57` unless the parent tells it these.
    symbol_labels: Dict[str, str] = {}


class EntryReport(msgspec.Struct, frozen=True, kw_only=True):
    entry: str
    status: str = ""
    files: List[str] = []
    detail: str = ""
    elapsed_s: float = 0.0
    peak_rss_bytes: int = 0
    #: pgw#868 A4: the child's DEVICE high-water, allocated and reserved.
    #: Defaulted so an older child's report still decodes. No longer telemetry
    #: only: `EntryCompilePool._rewiden` divides the pool's own free-VRAM
    #: reading by THIS, once two children have reported one, in place of the
    #: `mint_budget.co_residency` estimate the pool was constructed with.
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
    #: `torch.export.load` of the staged program) was the dark 44 %.
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


@dataclass
class _Running:
    entry: str
    proc: subprocess.Popen
    job: EntryJob
    program_path: Path
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
    #: it finished at. When ``_rewiden`` replaces an estimated per-entry device
    #: ask with a measured one, ``workers`` moves and this does not — so the
    #: prize is readable as a delta from one row, without a second mint to
    #: compare against.
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


class EntryCompilePool:
    """Compile N exported programs K-wide, out of process.

    Not a general executor: it exists to hold pgw#809's three invariants —
    entry-named failure, group-wide sibling teardown, and assembly by entry
    NAME rather than completion order.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        width: PoolWidth,
        inductor_configs: Optional[Mapping[str, Any]] = None,
        cache_dir: str = "",
        python: str = "",
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.width = width
        #: pgw#842/th#1359: the width facts as last EMITTED, so a re-emit
        #: happens only when they actually moved.
        self._emitted_width_facts: Optional[Dict[str, Any]] = None
        #: pgw#992: who else is on the card, read before the first child exists.
        #: `_rewiden` cannot compute a simultaneity bound without it, and a
        #: census it cannot read refuses the widen rather than assuming an
        #: empty card.
        self.census = card_census()
        #: pgw#1053: the OWN term's floor in the simultaneity budget. Seeded
        #: from the construction census (the resident pipeline), and
        #: re-baselined by :meth:`note_residents_released` when the mint
        #: parent provably hands its residents back — the ONE way the budget's
        #: own term is ever allowed to shrink. Without an explicit floor,
        #: ``max(own_device_high_water(), census.own_reserved_bytes)`` pins
        #: the budget to the construction-time pipeline forever, and a release
        #: frees bytes the arithmetic can never grant.
        self._own_floor_bytes = int(self.census.own_reserved_bytes)
        #: pgw#1053: bytes the release handed back, added to the CONSTRUCTION
        #: free reading when K is re-derived. ``_rewiden`` deliberately never
        #: re-probes a card K children are sitting on; the release's gain is
        #: instead reconstructed as "the construction free figure plus what
        #: the residents measurably returned", which children cannot distort.
        self._free_gain_bytes = 0
        #: The terms of the last simultaneity decision, merged into the width
        #: row so a future OOM names WHICH term was wrong instead of leaving a
        #: reader to diff two pods that no longer exist.
        self.simultaneity: Dict[str, Any] = {}
        self._apply_simultaneity_bound()
        self._emit_width("construction")
        self.inductor_configs = dict(inductor_configs or {})
        # pgw#848 item 5: the crash-only half. `bank` is None whenever the
        # process was not given a stable root (`aot_resume.set_root`, the one
        # production wiring — pgw#1030 deleted the redundant `resume_dir`
        # param), and then this class behaves exactly as it did before — no
        # admission pass, no hashing, no copies.
        self.bank = aot_resume.open_bank(
            inductor_configs=self.inductor_configs)
        #: entry -> the graph hash re-derived at admission, so `_collect` can
        #: bank the finished files under an identity the parent computed from
        #: the program it exported (never one read back from an artifact).
        self._entry_graph: Dict[str, str] = {}
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
        #: pgw#868 A4 measured this and left it telemetry-only; nothing read
        #: it, so the per-entry device ask stayed the mint child's whole
        #: co-residency estimate. This is the number that ends that.
        self.peak_device_bytes = 0
        #: How many entry children have contributed one. `_rewiden` refuses to
        #: act on fewer than :data:`REWIDEN_MIN_SAMPLES`.
        self.device_samples = 0
        #: The width the pool was CONSTRUCTED with — AFTER the pgw#992
        #: simultaneity bound, so the record `_rewiden` re-derives against is
        #: the one the pool actually ran. `_rewiden` re-uses this record's own
        #: free-VRAM and host-RAM readings — the same question, a measured
        #: divisor — so the initial row has to survive being superseded.
        self.width_initial = self.width
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
        # pgw#830: parent-side per-entry spans (staging + spawn) and the
        # pool-level idle split. Kept separate from `entry_phases` because
        # they are NOT inside `compile_s`: staging happens in the parent while
        # other children run, so summing it into the compile total would
        # invent seconds nobody spent compiling.
        self.entry_stage_seconds: Dict[str, float] = {}
        self.entry_spawn_seconds: Dict[str, float] = {}
        self.entry_overlays: Dict[str, Dict[str, float]] = {}
        self.entry_metrics_raw: Dict[str, Dict[str, float]] = {}
        self.ledger = PoolLedger(
            workers=int(width.workers), workers_initial=int(width.workers))

    # -- staging ----------------------------------------------------------

    def _stage(self, entry: str, program: Any, index: int) -> Tuple[EntryJob, Path]:
        import torch

        slot = self.workdir / f"entry-{index:03d}"
        slot.mkdir(parents=True, exist_ok=True)
        program_path = slot / "program.pt2"
        t0 = time.monotonic()
        torch.export.save(program, program_path)
        self.entry_stage_seconds[entry] = round(time.monotonic() - t0, 3)
        self.ledger.stage_total_s = round(
            self.ledger.stage_total_s + self.entry_stage_seconds[entry], 3)
        logger.info(
            "aot-pool: staged %r (%.1f MB) in %.1fs",
            entry, program_path.stat().st_size / 1e6, time.monotonic() - t0)
        job = EntryJob(
            entry=entry,
            program=str(program_path),
            report=str(slot / ENTRY_REPORT_NAME),
            inductor_configs=dict(self.inductor_configs),
            cache_dir=self.cache_dir,
            device_lock=str(self.device_lock_path),
            symbol_values=aot_shape_hints.symbol_values(program),
            symbol_labels=aot_shape_hints.symbol_labels(program),
        )
        job_path = slot / "job.json"
        job_path.write_bytes(msgspec.json.encode(job))
        return job, job_path

    def _spawn(self, job: EntryJob, job_path: Path, program_path: Path) -> _Running:
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
        self.entry_spawn_seconds[job.entry] = round(started - t0, 3)
        self.ledger.spawn_total_s = round(
            self.ledger.spawn_total_s + (started - t0), 3)
        logger.info("aot-pool: entry %r -> pid %s", job.entry, proc.pid)
        return _Running(
            entry=job.entry, proc=proc, job=job, program_path=program_path,
            started=started, stderr_path=stderr_path, spawn_epoch=spawn_epoch)

    # -- the run ----------------------------------------------------------

    def compile(
        self, entries: Iterable[Tuple[str, Any]],
        *, on_entry: Optional[Callable[[str, int, int], None]] = None,
        expected_total: int = 0,
    ) -> Dict[str, List[str]]:
        """``[(entry, ExportedProgram)] -> {entry: [file, ...]}``.

        Raises :class:`EntryCompileFailed` naming the FIRST entry to fail,
        after tearing down every sibling group. Returns a dict ordered by
        entry NAME, never by completion, so the packaged cell cannot depend
        on which child finished first.

        ``entries`` may be a SEQUENCE (every entry already exported — the
        pre-pgw#1052 shape, unchanged) or an ITERATOR that produces entries
        as they become ready. Pulling from an iterator runs the PRODUCER's
        own work on this thread — on the overlapped mint path that is a
        ``torch.export`` of the next declared row — while the children keep
        compiling in their own processes. That single sentence is the whole
        of pgw#1052: the phases were sequential by code, not by necessity.
        The iterator contract is deliberately narrow — ``(name, program)``
        pairs, staged internally — so a later producer that ships structure
        instead of a full program (pgw#1056's fake-weight mint) changes
        ``_stage``, not this orchestration. ``expected_total`` is the
        producer's best count for progress reporting; the ledger records the
        REAL count once the source is exhausted.

        ``on_entry(name, done, total)`` (pgw#824) fires as each entry lands.
        This loop is the longest wire-silent stretch of a mint — an 18-entry
        sdxl cell spends the bulk of its wall clock right here — and until now
        it reported nothing between "compiling" and "packed". Progress
        reporting is best-effort by construction: a raising callback must never
        cost the mint the entries it already has.
        """
        staged: List[Tuple[EntryJob, Path]] = []
        running: List[_Running] = []
        done: Dict[str, List[str]] = {}
        # One program staged AHEAD of the running set, and no more. Staging is
        # a multi-GB write (~16 s at 2.5 GB) and a freed slot that had to wait
        # for one would idle a core through every round; one spare removes
        # that without turning an 18-entry sdxl cell into ~46 GB on disk.
        failure: Optional[EntryCompileFailed] = None
        # pgw#832: seed BEFORE the pool wall starts, so the cost is its own
        # named line (`seal_seed_s`) and never inside the capacity identity.
        self._seed_seal_memo()
        streamed = not isinstance(entries, (list, tuple))
        if streamed:
            source: Iterator[Tuple[str, Any]] = iter(entries)
            total = max(0, int(expected_total))
            pulled = 0
        else:
            pending = [(i, name, prog)
                       for i, (name, prog) in enumerate(entries)]
            total = len(pending)
            self.ledger.entries = total
            # pgw#848 item 5: admission BEFORE the wall on the sequence path.
            # It is parent-serial and occupies no worker slot, so charging it
            # to the pool's capacity would price a recovered 626 s entry as
            # pool idle. (The streamed path admits per pull instead — the
            # entry does not exist before the pull, and the pull is already
            # inside the wall by construction.)
            pending = self._admit_banked(pending, done, on_entry, total)
            source = iter([(name, prog) for _i, name, prog in pending])
            pulled = total - len(pending)
        exhausted = False

        def _known_total() -> int:
            if not streamed:
                return total
            return pulled if exhausted else max(total, pulled)

        def _cb(name: str) -> None:
            if on_entry is not None:
                try:
                    on_entry(name, len(done), _known_total())
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
            # for each interval, not multiplied out at the end. `_rewiden` can
            # move K mid-pool, and `wall_s * final_workers` would then price
            # the whole run at a width the first entries never had — turning
            # the efficiency identity (busy + idle == capacity) into a
            # residual nobody can trust, which is the one thing this ledger
            # exists not to be.
            self.ledger.capacity_s += (now - mark) * self.width.workers
            mark = now

        try:
            while True:
                # Re-read every round: `_rewiden` can widen K after an entry
                # lands, and a staging cap frozen at the construction width
                # would starve the slots it just opened.
                staged_cap = max(1, self.width.workers + INFLIGHT_PROGRAM_SLACK)
                # SPAWN first: a freed slot takes already-staged work before
                # the parent disappears into a source pull that, on the
                # overlapped path, can be minutes of export.
                while staged and not failure \
                        and len(running) < self.width.workers \
                        and self._spawn_admitted(len(running)):
                    job, job_path = staged.pop(0)
                    free = self.width.workers - len(running)
                    running.append(
                        self._spawn(job, job_path, Path(job.program)))
                    charge("idle_spawn_s", free)
                # PULL one entry when there is stage room. The pull IS the
                # producer's work (pgw#1052); the fresh program spawns at the
                # top of the next iteration.
                if not exhausted and not failure \
                        and len(staged) + len(running) < staged_cap:
                    free = self.width.workers - len(running)
                    try:
                        name, program = next(source)
                    except StopIteration:
                        exhausted = True
                        self.ledger.entries = pulled
                        charge("idle_source_s", free)
                        continue
                    charge("idle_source_s", free)
                    pulled += 1
                    if streamed:
                        self.ledger.entries = pulled
                        if self.bank is not None:
                            # pgw#848 item 5 on the streamed path: per-pull
                            # admission, same order-of-operations safety (the
                            # graph hash is re-derived from THIS export).
                            admission = self.bank.admit(name, program)
                            if admission.ok:
                                done[name] = list(admission.files)
                                self._refresh_resume_facts()
                                _cb(name)
                                continue
                    free = self.width.workers - len(running)
                    staged.append(self._stage(name, program, pulled - 1))
                    # The freed slot waits for the NEXT program to be written
                    # before it can be refilled: export-vs-compile
                    # serialization, charged where it happens.
                    charge("idle_staging_s", free)
                    continue
                if not running:
                    if failure is not None or (exhausted and not staged):
                        break
                    continue
                free = self.width.workers - len(running)
                # Nothing left to start: the free slots are the straggler
                # tail, which is a SCHEDULING loss and not a compile cost.
                # pgw#829's entry collapse moves this number; nothing about
                # the compiler does. A slot held by the LIVE simultaneity
                # bound (`_spawn_admitted`) lands in `idle_other_s` with the
                # holding terms recorded on `simultaneity`.
                bucket = "idle_drain_s" if exhausted and not staged \
                    else "idle_other_s"
                finished = self._reap(running)
                if finished is None:
                    time.sleep(_POLL_S)
                    charge(bucket, free)
                    continue
                charge(bucket, free)
                running.remove(finished)
                try:
                    done[finished.entry] = self._collect(finished)
                except EntryCompileFailed as exc:
                    failure = exc
                    break
                # pgw#868 A4: this entry's own DEVICE high-water is now banked
                # (`_collect` -> `observe_entry_device`). Ask K again with the
                # measurement in place of the estimate the pool was built on.
                self._rewiden()
                _cb(finished.entry)
                # Collection (report read, program unlink) and pgw#824's
                # progress callback both run with the slot ALREADY FREE, so
                # they are charged as idle rather than left outside the split
                # — a callback that blocked would otherwise vanish from a
                # ledger whose whole point is that nothing vanishes.
                charge("idle_other_s", self.width.workers - len(running))
            if failure is not None:
                raise failure
        finally:
            self.ledger.wall_s = round(time.monotonic() - pool_t0, 3)
            self.ledger.busy_s = round(sum(self.entry_seconds.values()), 3)
            # Closed at the LIVE width for the final interval, then rounded —
            # `charge` has been accumulating it all along (see there).
            charge("idle_other_s", 0)
            self.ledger.capacity_s = round(self.ledger.capacity_s, 3)
            for row in running:
                _terminate_group(row.proc)
            self._sweep()
            # Re-emit only if the width moved since construction (no-op
            # otherwise); the ledger row keeps carrying the timing facts.
            self._emit_width("terminus")
            self._emit_ledger()
        return {name: done[name] for name in sorted(done)}

    def _admit_banked(
        self,
        pending: List[Tuple[int, str, Any]],
        done: Dict[str, List[str]],
        on_entry: Optional[Callable[[str, int, int], None]],
        total: int,
    ) -> List[Tuple[int, str, Any]]:
        """pgw#848 item 5: hand back the entries a previous attempt finished.

        The order of operations is the safety property: the graph hash is
        re-derived from the ExportedProgram THIS attempt exported and handed to
        the bank, which compares it against what it recorded. An entry is never
        admitted because a file exists at a path — under pgw#846 that is how a
        stale artifact gets packed into a cell that verifies, arms, and is
        wrong.

        Every refusal falls through to a normal compile. A bank must never be
        able to cost a mint a cell.
        """
        if self.bank is None:
            return pending
        remaining: List[Tuple[int, str, Any]] = []
        for index, name, program in pending:
            admission = self.bank.admit(name, program)
            if not admission.ok:
                remaining.append((index, name, program))
                continue
            done[name] = list(admission.files)
            if on_entry is not None:
                try:
                    on_entry(name, len(done), total)
                except Exception:  # noqa: BLE001 — telemetry never fails a mint
                    logger.debug(
                        "entry-pool progress callback failed", exc_info=True)
        self._refresh_resume_facts()
        if self.bank.resumed:
            logger.info(
                "aot-resume: %d of %d entr%s re-admitted from %s in %.2fs — "
                "this attempt compiles %d",
                len(self.bank.resumed), total,
                "y" if total == 1 else "ies", self.bank.root,
                self.bank.admit_s, len(remaining))
        return remaining

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

            facts = {**self.width.facts(), **self.census.facts(),
                     **self.simultaneity}
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
        return None

    def observe_entry_device(self, report: EntryReport) -> None:
        """Bank one entry child's DEVICE high-water (pgw#877 #2).

        RESERVED in preference to allocated, on the child's own argument:
        allocated is what the compile needed, reserved is what the caching
        allocator HELD and therefore what a concurrent sibling actually cannot
        have — and K is a question about siblings. A child too old to report
        reserved still contributes its allocated figure rather than nothing.
        """
        peak = int(report.peak_device_reserved_bytes or 0) \
            or int(report.peak_device_bytes or 0)
        if peak > 0:
            self.peak_device_bytes = max(self.peak_device_bytes, peak)
            self.device_samples += 1

    def entry_budget_bytes(
        self, ask: int,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """Bytes K entry children may hold AT THEIR SIMULTANEOUS PEAKS, with
        every term named. ``(budget, terms)``, where ``None`` means a term was
        unreadable — distinct from a budget that is readable and NEGATIVE,
        which is a card already oversubscribed by its residents and a perfectly
        computable answer of "one child, and only because one is the floor".
        Conflating the two would let an oversubscribed card take the
        unreadable branch and keep whatever width it had.

        pgw#992 — the defect this replaces. A4 divided the pool's
        free-VRAM SAMPLE by the measured per-entry peak: ``29.5 GiB / 6.02 GiB
        -> K=4``. On the real path that killed the first AOT mint ever to reach
        the compile phase, deterministically, on entry 2 of 36::

            44.39 GiB card, 2.69 MiB free, OOM on a 14 MiB alloc
              9.54 GiB  eager-serving parent   (resident, by pgw#784's contract)
             16.20 GiB  mint child's pipeline  (resident, this process)
             18.61 GiB  four entry children    (4 x ~6 GiB, as measured)

        Two facts make the division wrong, and neither is a shortage of card:

        * **"free right now" is not a simultaneous budget.** The sample is
          taken before the widened children exist and prices none of their
          growth. Its own comment already says it is read *between tenant
          forwards*.
        * **The residents keep growing against the same card.** They were
          14.9 GiB at the sample and 25.7 GiB at the OOM. A momentary reading
          cannot bound a future peak.

        So the budget is taken against the CARD, not against a moment:

            budget = total
                   - resident co-tenant (census, measured before child one)
                   - this process's own device high-water
                   - the tenant's forward reserve, when a serve goal exists

        Every term is an observation, and every term is read from the DEVICE.
        Summing what the pool believes is loaded would not do: on the z-image
        pod, 16.2 GiB was free on an 80 GB card whose static slot sum is
        53.3 GiB — ~9 GiB of CUDA context, allocator fragmentation and child
        overhead that no catalog arithmetic can see.

        Indifferent to ``ask``'s BASIS, deliberately. The estimate is not safer
        than the measurement (it only happened to be larger on one pod) and the
        measurement is not more dangerous than the estimate; a bound that
        preferred either would be a statement about the divisor, and the
        divisor is not what was wrong.

        Deliberately NOT a larger :data:`DEVICE_RESERVE_BYTES` (§4.24): padding
        a constant moves the same unpriced simultaneity somewhere else and
        re-fires on the next card.
        """
        terms: Dict[str, Any] = {
            "simultaneity_ask_bytes": int(ask),
            "simultaneity_basis": self.census.basis,
        }
        if not self.census.readable or ask <= 0:
            terms["simultaneity_verdict"] = "unreadable — no widen"
            return None, terms
        # pgw#1053: the floor is `_own_floor_bytes`, not the construction
        # census verbatim — identical until `note_residents_released`
        # re-baselines it, which the caller may only do after a real release
        # plus `reset_peak_memory_stats` (so the high-water restarts from the
        # released level rather than remembering the pipeline).
        own_peak = max(own_device_high_water(), int(self._own_floor_bytes))
        if self._free_gain_bytes > 0:
            # pgw#1053: the release rides every later decision row — a widen
            # or a hold after it must say the budget it ran against moved.
            terms["residents_released_bytes"] = int(self._free_gain_bytes)
        reserve = DEVICE_RESERVE_BYTES \
            if WorkerGoals(serve=self.width.serve_goal,
                           mint=self.width.mint_goal).tenant_reserve_applies() \
            else 0
        budget = (self.census.total_bytes - self.census.resident_other_bytes
                  - own_peak - reserve)
        terms.update({
            "simultaneity_own_peak_bytes": int(own_peak),
            "simultaneity_tenant_reserve_bytes": int(reserve),
            "simultaneity_budget_bytes": int(budget),
            "simultaneity_k_cap": int(max(0, budget) // ask),
        })
        return budget, terms

    def _apply_simultaneity_bound(self) -> None:
        """Narrow the CONSTRUCTED width to what the card can hold at once.

        pgw#992, second reading. The first version of this fix capped only
        ``_rewiden``, which would have held the L40S at K=2 — and the z-image
        contrast specimen shows why that is not the invariant. Same
        ``_rewiden`` code, a different pod:

            free_device 16.2 GiB / per_entry 25.0 GiB (**estimated**) -> K=1,
            underwidth=3

        The ESTIMATE accidentally protected that pod; the L40S died because the
        MEASURED peak shrank the denominator. So the bound cannot be "prefer
        the measured peak" or "distrust the measured peak" — either one is a
        statement about the DIVISOR, and the divisor is not what was wrong.
        **The threat is K children's simultaneous peak against the residents'
        future peaks, and it has to be read from the DEVICE regardless of which
        basis supplies the per-entry figure.** That makes it a bound on every
        width this pool ever runs, not a patch on the one path that widens.

        Measured on the same z-image pod, and the reason this reads the card
        rather than adding up what the pool thinks is loaded: 16.2 GiB free on
        an 80 GB card whose static slot sum is 53.3 GiB — **~9 GiB of CUDA
        context, allocator fragmentation and child overhead that no catalog
        arithmetic can see**.

        Never below 1: K=1 is the in-process serial path, it is what the pool
        degrades TO, and a bound that could forbid it would forbid minting at
        all. Never above what the caller already chose — this only narrows.
        """
        ask = int(self.width.per_entry_device_bytes or 0)
        budget, terms = self.entry_budget_bytes(ask)
        self.simultaneity = terms
        if budget is None or ask <= 0:
            return
        capped = max(1, int(terms["simultaneity_k_cap"]))
        if capped >= self.width.workers:
            return
        logger.warning(
            "aot-pool: pgw#992 narrowing the CONSTRUCTED K %d -> %d — the "
            "card holds %.2f GiB, a %.2f GiB co-tenant and a %.2f GiB "
            "resident set, leaving %.2f GiB for entries at %.2f GiB each",
            self.width.workers, capped, self.census.total_bytes / 1024**3,
            self.census.resident_other_bytes / 1024**3,
            terms["simultaneity_own_peak_bytes"] / 1024**3,
            budget / 1024**3, ask / 1024**3)
        self.width = replace(
            self.width, workers=capped, binding="simultaneity",
            reason=(f"K={capped} (simultaneity-bound): {self.width.reason} — "
                    f"narrowed to what the card holds at once"))

    def note_residents_released(self) -> None:
        """pgw#1053: the mint parent PROVABLY handed its residents back.

        Called by the mint after the last row exports, once it has dropped its
        pipeline and the retained programs' weight aliases, run
        ``empty_cache()`` AND ``reset_peak_memory_stats()`` — the reset is
        load-bearing: it is what lets ``own_device_high_water`` restart from
        the released level instead of remembering the pipeline forever.

        Two things happen, both measured rather than asserted:

        * the OWN floor of the simultaneity budget re-baselines to what this
          process holds NOW (the delta is exactly what the driver got back);
        * K is re-derived through the SAME pgw#992-bounded path a measured
          entry peak takes — with the construction ask when fewer than
          :data:`REWIDEN_MIN_SAMPLES` children have reported, because the
          release moved the BUDGET, not the divisor, and holding a freed card
          at the resident-priced K would be pgw#842's underwidth defect with a
          receipt attached.

        A no-op on a cardless pool (nothing was priced off a card) and after a
        release that freed nothing (the floor only ever re-baselines DOWN).
        """
        if not self.census.readable:
            return
        now = own_reserved_now()
        if now < 0:
            return
        freed = max(0, int(self._own_floor_bytes) - now)
        if freed <= 0:
            return
        self._own_floor_bytes = now
        self._free_gain_bytes += freed
        logger.info(
            "aot-pool: pgw#1053 residents released %.2f GiB back to the "
            "simultaneity budget (own floor %.2f GiB)",
            freed / 1024**3, now / 1024**3)
        self._rewiden(trigger="release")

    def _spawn_admitted(self, running_count: int) -> bool:
        """May a (``running_count + 1``)-th child exist on this card NOW?

        pgw#992 continued under pgw#1052. The construction-time simultaneity
        bound prices the residents as they stood when the pool was BUILT — and
        the overlapped mint builds the pool before the export phase, whose own
        device growth (measured 9.0 -> 15.6 GiB reserved across a 36-row sdxl
        export) arrives after that reading. This re-asks the SAME budget with
        the live own high-water before every spawn: a spawn the budget cannot
        hold WAITS — for a sibling to exit, or for the pgw#1053 release to
        grow the budget — instead of being priced against a reading it
        postdates. Nothing narrows and nothing is killed; a spawn is deferred
        to a boundary the card admits. Floor 1: the serial path must always
        be reachable or the pool deadlocks against its own bound.
        """
        if running_count <= 0:
            return True
        ask = int(self.width.per_entry_device_bytes or 0)
        if ask <= 0 or not self.census.readable:
            return True
        budget, terms = self.entry_budget_bytes(ask)
        if budget is None:
            return True
        k_cap = max(1, int(terms["simultaneity_k_cap"]))
        if running_count < k_cap:
            return True
        terms["simultaneity_spawn_held_at"] = int(running_count)
        self.simultaneity = terms
        return False

    def _rewiden(self, *, trigger: str = "measured") -> None:
        """Re-derive K from what the entry children MEASURED (pgw#868 A4),
        bounded by what the CARD can hold at once (pgw#992).

        pgw#809 sizes the pool before a single entry has run, and the only
        per-entry device figure available then is
        ``mint_budget.co_residency().need_bytes`` — the MINT CHILD's whole
        co-residency estimate (a full weight copy, an activation set the
        estimate never observed, and two flat constants), used as ONE entry
        child's ask. pgw#877 measured that ~56 % of it was never observed and
        renamed its basis from ``"measured"`` to ``"estimated"`` for exactly
        that reason. Every entry child since has reported what it actually
        peaked at, :meth:`observe_entry_device` has banked it, and nothing
        read it: an sdxl cell spent 34 more entries at a width chosen by an
        estimate its own first two entries had already disproved.

        This asks the SAME question against the SAME readings — only the
        divisor changes, from an estimate to an observation. That is the
        pgw#847 shape (delete a guess, keep the computation), not a new
        policy: no reserve moves, no ceiling moves, no bound is invented.

        Fail-closed, in six directions:

        * **the grant is capped by the card's simultaneous budget**
          (:meth:`entry_budget_bytes`) — the one that was missing, and the one
          that killed pgw#868 A1's first real AOT compile;
        * it never NARROWS. Children are already running against the wider
          number, and a mid-flight dip is the reading ``device_facts`` takes a
          max over precisely so it cannot be acted on;
        * it never exceeds the caller's own ``limit`` — an operator who forced
          the serial path keeps it;
        * it refuses on fewer than :data:`REWIDEN_MIN_SAMPLES` reports, on a
          zero peak, and on a pool whose device lock is absent (``K=1`` there
          is a safety width, not a resource one);
        * it re-derives against ``width_initial``'s OWN free-VRAM and host-RAM
          figures rather than re-probing a card that K running children are
          sitting on — re-probing would read their footprints as absent
          capacity and narrow, which is the opposite of the truth. That
          argument is right about the FREE figure and was never a licence to
          skip the card-wide bound above;
        * anything raising leaves the width exactly as it was.

        It changes no artifact. K is not an input to codegen, kernel selection
        or the traced graph — the device lock already serializes the one thing
        that is shared (``benchmark_all_configs``) — so this is a PROCESS
        change under pgw#846's rule, and the emitted files are untouched.

        ``trigger="release"`` (pgw#1053) is the residents-release re-ask: the
        budget's own term just shrank by a MEASURED amount, so the same
        derivation runs even before :data:`REWIDEN_MIN_SAMPLES` children have
        reported — the divisor then stays the CONSTRUCTION ask on the
        construction basis (never a new guess), and only the budget moved.
        """
        released = trigger == "release"
        if not self.width.device_lock:
            return
        measured = (self.device_samples >= REWIDEN_MIN_SAMPLES
                    and self.peak_device_bytes > 0)
        if not measured and not released:
            return
        base = self.width_initial
        if base.free_device_bytes <= 0:
            # The initial width never read the card (`entries <= 1`, or an
            # absent probe). There is nothing to re-divide, and inventing a
            # free figure here is the guess this method exists to delete.
            return
        try:
            if measured:
                ask = mint_budget.entry_device_ask(int(self.peak_device_bytes))
                basis = "measured"
            else:
                ask = int(base.per_entry_device_bytes or 0)
                basis = base.per_entry_device_basis
            if ask <= 0:
                return
            # pgw#992: the cap comes FIRST and is recorded whether or not the
            # widen happens — a refused widen is the interesting row.
            budget, terms = self.entry_budget_bytes(ask)
            self.simultaneity = terms
            if budget is None:
                logger.info(
                    "aot-pool: pgw#992 declining to widen from K=%d — the "
                    "card census is %s, so K children's simultaneous peak "
                    "cannot be priced", self.width.workers, self.census.basis)
                self._emit_width("simultaneity bound unreadable")
                return
            k_cap = int(terms["simultaneity_k_cap"])
            wider = entry_workers(
                base.entries,
                limit=base.limit,
                device_bytes=ask,
                device_basis=basis,
                # pgw#1053: the construction reading plus what the release
                # measurably returned — never a re-probe of a card K running
                # children are sitting on (their footprints would read as
                # absent capacity).
                free_vram_bytes=int(
                    base.free_device_bytes + self._free_gain_bytes),
                available_bytes=int(base.available_bytes),
                vcpus=int(base.vcpus),
                peak_rss_bytes=int(self.peak_rss_bytes or 0),
                device_lock=base.device_lock,
                # Reconstructed from the record rather than re-read: the
                # re-derivation must run the policy the pool was BUILT under,
                # not whatever a later `install()` published.
                goals=WorkerGoals(
                    serve=base.serve_goal, mint=base.mint_goal),
            )
        except Exception:  # noqa: BLE001 — a re-derivation never fails a mint
            logger.debug("aot-pool: width re-derivation failed", exc_info=True)
            return
        granted = min(wider.workers, k_cap)
        if granted < wider.workers:
            # The row pgw#992 exists to produce: A4's own arithmetic said one
            # thing, the card said another, and the card wins BY NAME.
            logger.warning(
                "aot-pool: pgw#992 capping K %d -> %d (A4 asked for %d) — the "
                "card holds %.2f GiB, a %.2f GiB co-tenant and a %.2f GiB "
                "resident pipeline, leaving %.2f GiB for entries at %.2f "
                "GiB each",
                self.width.workers, granted, wider.workers,
                self.census.total_bytes / 1024**3,
                self.census.resident_other_bytes / 1024**3,
                terms["simultaneity_own_peak_bytes"] / 1024**3,
                budget / 1024**3, ask / 1024**3)
        if granted <= self.width.workers:
            # Nothing to grant. Emit anyway: "the pool did NOT widen, and here
            # is the bound that stopped it" is the row a later OOM needs.
            self._emit_width("simultaneity bound held K")
            return
        wider = replace(wider, workers=granted)
        logger.info(
            "aot-pool: K %d -> %d (%s) — per-entry device ask %.2f GiB (%s, "
            "%d report(s)) against the %.2f GiB (%s) the pool was sized "
            "with, within a %.2f GiB simultaneous budget (pgw#992/pgw#1053)",
            self.width.workers, wider.workers, trigger, ask / 1024**3,
            basis, self.device_samples,
            base.per_entry_device_bytes / 1024**3,
            base.per_entry_device_basis, budget / 1024**3)
        self.width = wider
        self.ledger.workers = wider.workers
        self._emit_width(
            "re-derived from measured entry peaks" if trigger == "measured"
            else "re-derived after residents release (pgw#1053)")

    def _collect(self, row: _Running) -> List[str]:
        elapsed = time.monotonic() - row.started
        reap_epoch = time.time()
        self.entry_seconds[row.entry] = round(elapsed, 2)
        code = row.proc.returncode
        report = _read_report(Path(row.job.report))
        # The program is the biggest thing on disk and is dead the moment the
        # child exits; drop it before the next stage runs.
        with_suppress_unlink(row.program_path)
        if report is not None:
            # pgw#877: banked BEFORE any gate can raise, and on the failure
            # path too — pgw#848's rule for the host half applies unchanged
            # here: the attempt that FAILED is exactly the attempt whose
            # measurement the next one has to size against.
            self.observe_entry_device(report)
            self._verify_child_code(row, report)
        if code == EXIT_COMPILED and report is not None and report.files:
            if report.peak_rss_bytes:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(report.peak_rss_bytes))
            missing = [f for f in report.files if not Path(f).exists()]
            if missing:
                raise EntryCompileFailed(
                    row.entry,
                    f"entry {row.entry!r}: child reported {len(report.files)} "
                    f"compiled file(s) but {len(missing)} do not exist "
                    f"(first: {missing[0]}) — the pool's shared inductor cache "
                    f"dir {self.cache_dir!r} is not visible to this process")
            self.entry_phases[row.entry] = self._close_entry_partition(
                row, report, elapsed=elapsed, reap_epoch=reap_epoch)
            self.entry_overlays[row.entry] = dict(report.overlays or {})
            self.entry_metrics_raw[row.entry] = dict(report.metrics_raw or {})
            logger.info(
                "aot-pool: entry %r compiled in %.1fs (%d file(s)) spans=%s",
                row.entry, elapsed, len(report.files),
                self.entry_phases[row.entry])
            if self.bank is not None:
                # pgw#848 item 5: banked HERE, the moment the entry is finished
                # and verified, never at the end of the pool — a mint that is
                # SIGKILLed at entry 30 of 36 runs no `finally`, and an
                # end-of-run bank would be exactly the thing the crash takes.
                self.bank.put(
                    row.entry, self.bank.graphs.get(row.entry, ""),
                    list(report.files))
                self._refresh_resume_facts()
            return list(report.files)
        detail = report.detail if report is not None else ""
        if not detail:
            detail = _stderr_tail(row.stderr_path)
        resource, basis = self._memory_verdict(code, report)
        if resource:
            self.oom_entry, self.oom_basis = row.entry, basis
        raise EntryCompileFailed(
            row.entry,
            f"entry {row.entry!r}: compile child exited {code} after "
            f"{elapsed:.0f}s ({_exit_note(code)}): {detail or 'no detail'}"
            + (
                f" [pgw#848 classification: MEMORY SHORTFALL, basis={basis}; "
                f"this entry's measured high-water was "
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

        ``compile_s = child_boot_s + child_wall_s + reap_lag_s``, where
        ``child_boot_s`` is interpreter startup plus this package's import
        (paid once per ENTRY, because the pool's unit of parallelism is a
        process that exits) and ``reap_lag_s`` is the child's exit plus the
        parent's poll granularity.

        A child that predates this instrumentation reports no epochs; rather
        than invent a split, the whole wall lands in the residual and the
        invariant check says so out loud.
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
        if "child_boot_s" not in spans or "reap_lag_s" not in spans:
            spans["child_boot_s"] = spans.get("child_boot_s", 0.0)
            spans["reap_lag_s"] = round(
                spans["compile_s"] - spans["child_boot_s"]
                - float(spans.get("child_wall_s", 0.0)), 3)
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

    def _sweep(self) -> None:
        """Every staged program, gone. The loose compiled files stay: they
        live in the inductor cache dir and are what ``package_aoti`` reads."""
        for slot in sorted(self.workdir.glob("entry-*")):
            with_suppress_unlink(slot / "program.pt2")


def with_suppress_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


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
    "FORGE_RSS_RESERVE_BYTES",
    "EXIT_BAD_JOB",
    "EXIT_COMPILED",
    "EXIT_REFUSED",
    "EntryCompileFailed",
    "EntryCompilePool",
    "arm_parent_death_signal",
    "EntryJob",
    "EntryReport",
    "DEVICE_RESERVE_BYTES",
    "MAX_ENTRY_WORKERS",
    "PACKAGE_ROOT",
    "REFUSED",
    "SERVING_HEADROOM_CPUS",
    "CpuFacts",
    "DeviceFacts",
    "MemoryFacts",
    "PoolLedger",
    "PoolWidth",
    "child_argv",
    "child_env",
    "cpu_facts",
    "device_facts",
    "entry_workers",
    "memory_facts",
]
