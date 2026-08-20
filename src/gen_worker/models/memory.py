"""VRAM/memory decisions for the models layer (#358/#366).

One low-VRAM decider for the whole worker, driven by FREE VRAM only (never
total capacity), plus size/measurement probes used by residency accounting.

Ladder (auto mode, least-aggressive first):

  off           : no optimizations (pipeline on CUDA as-is)
  vae_only      : VAE slicing only (th#1107: the pipeline FITS here — tiling
                  and attention slicing are VRAM tools and are reserved for
                  the rungs below, where the model genuinely does not fit)
  model_offload : VAE slicing + tiling + attention slicing +
                  ``enable_model_cpu_offload()``                (~10% slower)
  group_offload : leaf-level group offload with CUDA streams   (~25% slower)
  sequential    : ``enable_sequential_cpu_offload()``          (~50%+ slower)
  cpu           : the whole pipeline on the host, no device at all (~40x)

``cpu`` is the bottom rung and it EXECUTES (pgw#1315). It is never selected by
``auto`` — it is where a cardless pod starts and where a reactive descent that
exhausted every offload rung ends. §1.35 amendment 2: *"even a pod without a
GPU, heck; we can run it CPU only"*.

Upstream foot-gun: ``enable_sequential_cpu_offload`` must NOT be called on a
pipeline already moved to CUDA; ``apply_low_vram_config`` moves it back first.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import msgspec

from .. import activity as activity_mod
from .. import measured_posture as posture_mod
from . import machine_fit
from ..api.errors import HostRamMoveRefusedError
from ..component_vocab import component_vocabulary
from .partial_resident import PARTIAL_RESIDENT_DEVICE_ATTR
from .structure_only import STAMP as _STRUCTURE_ONLY
import asyncio
from ..hostfacts import cuda_ready
from .. import hostfacts

_LOG = logging.getLogger(__name__)

_GIB = 1024 ** 3

Mode = str  # "auto" | "off" | "vae_only" | "model_offload" | "group_offload" | "sequential" | "cpu"

_VALID_MODES: tuple[str, ...] = (
    "auto", "off", "vae_only", "partial_resident", "partial_stream",
    "model_offload", "group_offload", "sequential", "cpu",
)

_DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB = 8.0
_DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB = 6.0
# Safety margin below free VRAM reserved for activations.
_DEFAULT_SAFETY_MARGIN_GB = 2.0
# Free headroom beyond the requirement above which "off" beats "vae_only".
_DEFAULT_OFF_HEADROOM_GB = 8.0
_GGUF_RESIDENT_MARGIN_GB = 0.5

# Sentinel attribute set on pipelines to make apply_low_vram_config idempotent.
_COZY_MODE_ATTR = "_cozy_low_vram_mode"

#: Set on a module that reached a pipeline through `provision.load_slot`'s
#: `components=` — i.e. one loaded ONCE and aliased into every lane sharing its
#: content address (gw#479). It records what HAPPENED to the object, not a
#: property anyone declared about it.
SHARED_COMPONENT_ATTR = "_cozy_shared_component"


def mark_shared_components(components: Optional[Dict[str, Any]]) -> int:
    """Mark preloaded shared modules so no offload rung may hook them (ie#721).

    A shared module is ALIASED into several pipelines. Moving it to the host
    leaves every co-resident consumer on the device, and the failure is a fatal
    `mat1 is on cuda:0, mat2 on cpu` in the middle of a generate — measured on
    krea-2, qwen-image, z-image and hidream-o1-image (ie#480 finding 12).

    The invariant `provision`'s docstring states, enforced from a MEASURED fact
    — this object was injected as a shared component — instead of from an
    author's word. th#1867 deleted `Resources.strict_vram`, which was a
    card-size claim in softer words but was also this invariant's only enforcer.

    Returns how many modules were marked, so a caller can assert on it.
    """
    n = 0
    for mod in (components or {}).values():
        # Only real modules can be hooked by an offload rung; a path string or
        # a config object cannot, so marking it would be noise.
        if mod is None or not hasattr(mod, "parameters"):
            continue
        try:
            setattr(mod, SHARED_COMPONENT_ATTR, True)
        except Exception:  # pragma: no cover - exotic __slots__ objects
            continue
        n += 1
    return n


def shared_component_names(pipeline: Any) -> List[str]:
    """Names of this pipeline's components carrying the shared mark.

    Read off the live objects, so it reflects what was actually injected rather
    than what a manifest said would be.
    """
    out: List[str] = []
    comps = getattr(pipeline, "components", None)
    if not isinstance(comps, dict):
        return out
    for name, mod in comps.items():
        if mod is not None and getattr(mod, SHARED_COMPONENT_ATTR, False):
            out.append(str(name))
    return sorted(out)

# Authors declare ``Resources(vram_gb=X)`` as the TOTAL VRAM of the smallest
# card they target ("runs on a 24 GB card") — a placement recommendation, not
# measurable free bytes. The platform reserves this much for the fixed
# driver/framebuffer/CUDA-context overhead when comparing the recommendation
# against probed VRAM, so vram_gb=24 serves on a 24 GB card (~23.6 GB free).
GPU_VRAM_OVERHEAD_GB = 1.0


# th#1867 deleted `effective_vram_requirement_gb` with the declaration it
# translated. `GPU_VRAM_OVERHEAD_GB` survives below because the other use
# subtracts it from a MEASURED total, which is arithmetic on a fact.

# The ladder itself lives in rung.py (pgw#1206 A2): one ordered Rung, one
# walk, one price. This module keeps the probes and the appliers.
from .rung import (
    PLACEMENT_LADDER,
    RUN_CPU,
    descend as _descend_rung,
    price as _rung_price,
    touches_host_ram,
    transition_line,
)


def is_cuda_oom(exc: Optional[BaseException]) -> bool:
    """CUDA allocator exhaustion in any of its shapes: torch.cuda.OutOfMemoryError
    (class name match — no torch import needed) plus the allocator's RuntimeError
    flavors ("CUDA error: out of memory", CUBLAS/CUDNN alloc failures).

    pgw#1499 adds ``torch.AcceleratorError`` — the ASYNCHRONOUS shape. A kernel
    that runs out of memory device-side surfaces later, on whatever call next
    synchronizes, as an AcceleratorError carrying cudaErrorMemoryAllocation
    (code 2). Missing it made a real OOM read as an unclassified crash, so no
    ladder ever ran. It also leaves the context's error state poisoned, which
    is what :func:`discard_cuda_async_error` clears — every caller of this
    predicate is about to retry something, so the clear belongs here rather
    than in each of them.
    """
    if exc is None:
        return False
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    text = str(exc).lower()
    if type(exc).__name__ == "AcceleratorError" and (
        getattr(exc, "error_code", None) == 2 or "out of memory" in text
    ):
        discard_cuda_async_error()
        return True
    if isinstance(exc, RuntimeError):
        return (
            "out of memory" in text
            or "cuda oom" in text
            or "cublas_status_alloc_failed" in text
            or "cudnn_status_alloc_failed" in text
        )
    return False


def discard_cuda_async_error() -> None:
    """Flush a poisoned asynchronous CUDA error so the PROCESS survives it.

    An async device-side failure sticks to the context: the next launch
    re-reports it, and a worker that has already caught and handled the OOM
    then dies on an error it has no live cause for. A trivial kernel plus a
    synchronize forces the sticky error out where it can be swallowed once,
    deliberately, right here — we already learned the fact from the synchronous
    return. Always safe to call; no-op without CUDA.
    """
    try:
        import torch

        if not cuda_ready():
            return
        a = torch.tensor([1], dtype=torch.uint8, device="cuda")
        b = torch.tensor([1], dtype=torch.uint8, device="cuda")
        _ = a + b
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — dumping the error IS the point
        pass


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


#: Why a free-VRAM reading is zero. pgw#940: `except Exception: return 0.0`
#: made "this host has no CUDA" and "the probe raised on a host that does"
#: the same value, and every caller that had to DECIDE something read the
#: shared zero as the permissive case.
VRAM_NO_CUDA = "no_cuda"
VRAM_UNREADABLE = "unreadable"


@dataclass(frozen=True)
class VramReading:
    """Free VRAM, and — when there is none to report — why."""

    gb: float
    #: "" when `gb` is a real measurement, else VRAM_NO_CUDA / VRAM_UNREADABLE.
    reason: str = ""

    @property
    def measured(self) -> bool:
        return not self.reason


def available_vram(device_index: int = 0) -> VramReading:
    """Currently-free VRAM on the selected CUDA device, with its zero-cause.

    The one probe every free-VRAM question in this module is answered from.
    A caller that only wants the number keeps calling
    :func:`get_available_vram_gb`; a caller that must DECIDE with it reads
    ``reason`` and says out loud what it does when the card is unreadable.
    """
    if not hostfacts.cuda_ready():
        return VramReading(0.0, VRAM_NO_CUDA)
    free = hostfacts.free_vram_bytes(device_index)
    if free is None:
        return VramReading(0.0, VRAM_UNREADABLE)
    return VramReading(float(free) / float(1024**3))


def process_ceiling_vram(device_index: int = 0) -> VramReading:
    """What THIS process may occupy on the card at its peak, with its
    zero-cause — driver-free plus what it already holds.

    The reading a *whole working set* fit is decided against: resident weights
    plus the activations they will produce must land under it.
    :func:`available_vram` answers "what would a NEW process get", which is
    the wrong denominator for a process budgeting against its own weights.

    pgw#1558: an endpoint that computes this itself (``free + allocated`` off a
    raw ``mem_get_info``) cannot distinguish a CPU host from a wedged card, and
    ``minimax-h3`` demonstrably could not — it named every zero "UNREADABLE,
    not merely unmeasured" including on a machine with no GPU at all. That is
    pgw#940's misattribution, one repo over; ``reason`` is how it stays fixed.
    """
    if not hostfacts.cuda_ready():
        return VramReading(0.0, VRAM_NO_CUDA)
    ceiling = hostfacts.process_ceiling_bytes(device_index)
    if ceiling is None:
        return VramReading(0.0, VRAM_UNREADABLE)
    return VramReading(float(ceiling) / float(1024**3))


def get_available_vram_gb(device_index: int = 0) -> float:
    """Currently-free VRAM on the selected CUDA device. 0.0 if no CUDA.

    Reporting shape: it deliberately collapses "no card" and "unreadable"
    into one number, which is right for a log line and wrong for a decision.
    Anything that PLACES a model reads :func:`available_vram` instead.
    """
    return available_vram(device_index).gb


def get_total_vram_gb(device_index: int = 0) -> float:
    """TOTAL VRAM of the selected CUDA device — a per-SKU constant
    (pgw#750: deterministic placement inputs). 0.0 if no CUDA."""
    total = hostfacts.total_vram_bytes(device_index)
    return float(total) / float(1024**3) if total is not None else 0.0


def get_available_ram_gb() -> float:
    """Effective available host RAM: min(meminfo available, cgroup headroom)."""
    return probe_host_ram().available_gb


def get_total_ram_gb() -> float:
    """Effective total host RAM: min(meminfo total, cgroup limit)."""
    return probe_host_ram().total_gb


# pgw#973 (§4.24): ONE owner for the host-RAM floor. `residency` and `staging`
# each declared `_RAM_FLOOR_GB = 8.0` / `_RAM_FLOOR_FRACTION = 0.2` AND
# re-derived the same min/max expression, with staging's comment promising it
# was "kept numerically identical" — a promise nothing enforced. They cannot
# import each other (`residency -> pinned_swap -> staging` already exists, so
# the reverse edge closes a cycle); both already import this module.
#
# The threat (gw#407): a warm/pinned host tier that eats the host's working set
# pushes it into reclaim-thrash, and a thrashing host stalls the whole process
# INCLUDING the gRPC keepalive acks — the hub then disconnects the worker,
# which is the livelock. Nothing else prevents it: the tiers allocate against
# free RAM, and free RAM is exactly what a page cache makes look available.
_RAM_FLOOR_GB = 8.0
#: Small hosts (dev boxes) would be gated out entirely by a flat 8 GiB, so the
#: floor is adaptive below 40 GiB total.
_RAM_FLOOR_FRACTION = 0.2


def effective_ram_floor_gb(total_gb: Optional[float] = None) -> float:
    """Host RAM this process must leave alone, in GiB.

    ``total_gb`` is the caller's already-resolved host total; omit it to read
    one here. Callers pass their own so a per-group RAM share (procsplit) or a
    test's substitution is honoured by the ONE policy rather than by a copy.
    """
    total = get_total_ram_gb() if total_gb is None else float(total_gb)
    if total <= 0:
        return _RAM_FLOOR_GB
    return min(_RAM_FLOOR_GB, max(1.0, total * _RAM_FLOOR_FRACTION))


# ---------------------------------------------------------------------------
# Cgroup-aware host-RAM probes (th#721): RunPod GPU pods land on lottery-RAM
# hosts and the container is cgroup-limited below /proc/meminfo — psutil alone
# over-reports and the kernel SIGKILLs at the cgroup ceiling.
# ---------------------------------------------------------------------------

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
# v1 "unlimited" sentinel territory (kernel reports ~0x7ffffffffffff000).
_CGROUP_UNLIMITED = 1 << 60


class HostRam(msgspec.Struct, frozen=True, kw_only=True):
    """Effective host-RAM view: meminfo capped by the cgroup memory limit."""

    total_gb: float
    available_gb: float
    meminfo_total_gb: float
    meminfo_available_gb: float
    cgroup_limit_gb: Optional[float]  # None = no cgroup cap
    source: str  # "cgroup" | "meminfo"
    # Clean cgroup page cache credited back into available_gb (pgw#752).
    reclaimable_file_gb: float = 0.0
    # pgw#783: compute children sharing this container's memory cgroup. 1 for
    # every pod that exists today; > 1 once the execution group is an OS
    # process and G of them share one cap.
    siblings: int = 1


def _read_cgroup_int(path: Path) -> Optional[int]:
    """Parse a cgroup memory file; None for missing / 'max' / v1 sentinel."""
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
    return value if 0 <= value < _CGROUP_UNLIMITED else None


def cgroup_memory_limit_bytes(
    root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> Optional[int]:
    """Effective cgroup memory limit for this process; None when uncapped.

    v2: tightest ``memory.max`` on the root->self chain (covers both private
    and host cgroup namespaces); v1 fallback: ``memory/memory.limit_in_bytes``.
    """
    limits = [
        v for node in hostfacts.cgroup_nodes(root, proc_self_cgroup)
        if (v := _read_cgroup_int(node / "memory.max")) is not None
    ]
    if limits:
        return min(limits)
    return _read_cgroup_int(root / "memory" / "memory.limit_in_bytes")


def cgroup_memory_current_bytes(
    root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> Optional[int]:
    """Current cgroup memory usage; reads the deepest available counter."""
    for node in reversed(hostfacts.cgroup_nodes(root, proc_self_cgroup)):
        v = _read_cgroup_int(node / "memory.current")
        if v is not None:
            return v
    return _read_cgroup_int(root / "memory" / "memory.usage_in_bytes")


def _read_cgroup_stat(path: Path) -> Optional[Dict[str, int]]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    values: Dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if value >= 0:
            values[parts[0]] = value
    return values


def _cgroup_reclaimable_file_bytes(
    root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> int:
    """Clean page cache this cgroup can give back without an OOM kill.

    pgw#752: BOTH file LRUs count. A page read or written seconds ago lands on
    the ACTIVE file LRU; the kernel demotes active -> inactive and reclaims it
    under pressure, so an admission decision that treats active_file as
    consumed memory is not conservative, it is wrong — it charges the incoming
    model's own freshly-downloaded snapshot pages against the room needed to
    load that same snapshot (measured: a 251GB wan-2.2 pod reported 71.5GiB
    available while ~180GiB of it was clean snapshot cache).

    Excluded because the kernel cannot simply drop them: shmem/tmpfs (no
    backing file) and pages still dirty or under writeback.
    """
    stats: Optional[Dict[str, int]] = None
    for node in reversed(hostfacts.cgroup_nodes(root, proc_self_cgroup)):
        stats = _read_cgroup_stat(node / "memory.stat")
        if stats is not None:
            break
    if stats is None:
        stats = _read_cgroup_stat(root / "memory" / "memory.stat") or {}

    values: Dict[str, int] = stats

    def stat(*names: str) -> int:
        for name in names:
            for key in (name, f"total_{name}"):
                if key in values:
                    return values[key]
        return 0

    cache = stat("inactive_file") + stat("active_file")
    pinned = stat("shmem") + stat("file_dirty", "dirty") + stat(
        "file_writeback", "writeback")
    return max(0, cache - pinned)


def _host_ram_share(ram: HostRam, siblings: int) -> HostRam:
    """This process's SHARE of a container it splits with G-1 siblings.

    pgw#783: once the execution group is an OS process, G children sit in ONE
    memory cgroup — and every one of them reads the WHOLE container's cap here.
    Left alone, G children each believe the whole pod's RAM is theirs, so
    pgw#763's host-move guard and the residency demote floor become G times too
    permissive on precisely the pods most likely to OOM: four children each
    admitting a 50 GiB move against a 60 GiB cap, and the kernel settles it.

    The rule is deliberately the conservative one: a child may claim no more
    than its share of the cap AND no more than actually exists. The sum of G
    children's claims is then bounded by the cap, which is the property the
    guard needs. Unchanged (and byte-identical) at ``siblings == 1``, which is
    every pod today.
    """
    if siblings <= 1:
        return ram
    share_total = ram.total_gb / siblings
    return msgspec.structs.replace(
        ram,
        total_gb=share_total,
        available_gb=min(ram.available_gb, share_total),
        cgroup_limit_gb=(
            None if ram.cgroup_limit_gb is None else ram.cgroup_limit_gb / siblings
        ),
        siblings=siblings,
    )


def probe_host_ram(
    *,
    root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
    meminfo: Optional[Path] = None,
    siblings: Optional[int] = None,
) -> HostRam:
    """The ONE effective host-RAM view: ``/proc/meminfo`` min'd with the
    cgroup cap, with clean page cache credited back.

    Every consumer that must answer "will this fit in host RAM" reads THIS —
    the host-move guard, the residency demote floor, the staging admission and
    (since pgw#897) the AOT compile pool, which used to parse ``/proc/meminfo``
    and ``memory.stat`` itself with a NARROWER reclaimable definition and no
    sibling divisor.

    ``siblings`` defaults to the compute-child count this process shares its
    cgroup with (pgw#783; 1 unless the process split is running G groups).
    """
    if siblings is None:
        from ..procsplit import host_siblings

        siblings = host_siblings()
    info = hostfacts.meminfo_kb(meminfo)
    meminfo_total = float(info.get("MemTotal", 0)) * 1024.0 / float(_GIB)
    meminfo_available = float(info.get("MemAvailable", 0)) * 1024.0 / float(_GIB)
    limit = cgroup_memory_limit_bytes(root, proc_self_cgroup)
    if limit is None:
        return _host_ram_share(HostRam(
            total_gb=meminfo_total,
            available_gb=meminfo_available,
            meminfo_total_gb=meminfo_total,
            meminfo_available_gb=meminfo_available,
            cgroup_limit_gb=None,
            source="meminfo",
        ), siblings)
    limit_gb = float(limit) / float(_GIB)
    current = cgroup_memory_current_bytes(root, proc_self_cgroup)
    # #543: memory.current includes filesystem page cache. Model
    # downloads/loads fill a pod's cgroup with clean file cache that outlives
    # the pipeline objects. Only memory the kernel CANNOT hand back on demand
    # constrains the next load, so the working set is current minus every
    # reclaimable clean page (pgw#752 — crediting only the inactive LRU
    # double-charged the incoming snapshot's own cache).
    reclaimable = _cgroup_reclaimable_file_bytes(root, proc_self_cgroup)
    working_set = max(0, (current or 0) - reclaimable)
    cg_avail_gb = max(0.0, float(limit - working_set) / float(_GIB))
    total = min(meminfo_total, limit_gb) if meminfo_total > 0 else limit_gb
    avail = min(meminfo_available, cg_avail_gb) if meminfo_available > 0 else cg_avail_gb
    constrained = meminfo_total <= 0 or limit_gb < meminfo_total
    return _host_ram_share(HostRam(
        total_gb=total,
        available_gb=avail,
        meminfo_total_gb=meminfo_total,
        meminfo_available_gb=meminfo_available,
        cgroup_limit_gb=limit_gb,
        source="cgroup" if constrained else "meminfo",
        reclaimable_file_gb=float(reclaimable) / float(_GIB),
    ), siblings)


_ram_budget_logged = False


def log_ram_budget_once(*, floor_gb: float) -> None:
    """One prominent boot line naming the derived warm-RAM-tier budget and its
    source (cgroup cap vs /proc/meminfo) — DEGRADED_MODE-style greppability."""
    global _ram_budget_logged
    if _ram_budget_logged:
        return
    _ram_budget_logged = True
    ram = probe_host_ram()
    budget = max(0.0, ram.total_gb - floor_gb)
    parts = [
        f"RAM_BUDGET={budget:.1f}GiB",
        f"source={ram.source}",
        f"total_gb={ram.total_gb:.1f}",
        f"floor_gb={floor_gb:.1f}",
    ]
    if ram.cgroup_limit_gb is not None:
        parts.append(f"cgroup_limit_gb={ram.cgroup_limit_gb:.1f}")
    if ram.source == "cgroup":
        parts.append(f"meminfo_total_gb={ram.meminfo_total_gb:.1f}")
        _LOG.warning(
            "%s: container RAM capped below host /proc/meminfo; warm RAM tier "
            "sized to the cgroup limit (excess pipelines spill to disk)",
            " ".join(parts),
        )
        return
    _LOG.info(" ".join(parts))


def cuda_allocated_bytes(device_index: Optional[int] = None) -> int:
    """``torch.cuda.memory_allocated`` (0 without CUDA). Deltas across a load
    are the measured VRAM footprint reported in ModelEvent.vram_bytes."""
    try:
        import torch

        if cuda_ready():
            return int(torch.cuda.memory_allocated(device_index))
    except Exception:
        pass
    return 0


# pgw#740: a HINT order, never the whole world. Fixed component lists are how
# Wan's `transformer_2` (the second MoE expert) and LTX's `connectors` came to
# be silently skipped by the offload loops — a live memory bug for those
# families. Enumeration is generic; this only decides who goes first.
#
# B5: read from the ONE vocabulary at call time. Offload priority is
# largest-first — denoisers, then VAEs, then text encoders — which is not
# vocabulary declaration order, so the groups are concatenated explicitly.
# `connectors` is no longer named here: an endpoint that carries it declares
# it (declare_components) and lands in this hint; one that does not is still
# enumerated generically and simply sorts alphabetically after the hint.
def _component_order_hint() -> tuple[str, ...]:
    vocab = component_vocabulary()
    return tuple(dict.fromkeys(
        vocab.denoisers + vocab.vaes + vocab.text_encoders
        + vocab.auxiliaries + vocab.extras
    ))


def _module_attributes(pipeline: Any) -> List[tuple[str, Any]]:
    """Every parameter-bearing module hanging off ``pipeline``, discovered
    rather than listed (pgw#740). Deterministic order: the hint order first,
    then everything else alphabetically."""
    found: dict[str, Any] = {}
    for name in dir(pipeline):
        if name.startswith("_"):
            continue
        try:
            value = getattr(pipeline, name)
        except Exception:  # noqa: BLE001 — properties may raise on a half-built pipe
            continue
        if value is None or not hasattr(value, "parameters"):
            continue
        found[name] = value
    ordered: List[tuple[str, Any]] = [
        (name, found.pop(name)) for name in _component_order_hint()
        if name in found
    ]
    ordered.extend(sorted(found.items()))
    return ordered


def _named_components(pipeline: Any) -> List[tuple[str, Any]]:
    out: List[tuple[str, Any]] = []
    raw = getattr(pipeline, "components", None)
    if isinstance(raw, dict):
        out.extend(raw.items())
    else:
        out.extend(_module_attributes(pipeline))
    if not out and hasattr(pipeline, "parameters"):
        out.append(("", pipeline))  # bare nn.Module
    return out


def _iter_components(pipeline: Any) -> List[Any]:
    return [c for _, c in _named_components(pipeline)]


def device_mismatches(obj: Any, device: str) -> List[tuple[str, str, str]]:
    """Every parameter/buffer of ``obj``'s module components that is NOT on
    ``device``'s device type, as ``(component, tensor, actual_device)``.

    The paranoid post-move walk (gw#409): a pipeline ``.to()`` that raises or
    skips mid-way leaves a mixed-device pipeline that fatals mid-denoise
    ("Expected all tensors to be on the same device"); this surfaces the miss
    at move time instead. [] without torch / for tensor-less objects.

    VIRTUAL-BY-DESIGN TENSORS ARE NOT MISPLACED (pgw#1124). A pgw#1080
    structure-only component is composed ON the compute device with fake
    parameters and declines ``_apply`` outright (``_freeze_placement``), so
    counting it here made a CPU rollback impossible to satisfy — and
    ``place_pipeline``'s OOM demotion turned a recoverable ladder step into
    the fatal ``CUDA OOM left the pipeline mixed-device``, deterministically,
    on every boot-trace child of two live families. Such a component is
    skipped whole: its fake parameters allocate nothing and its real buffers
    are part of the graph being traced, so MOVING them would be the defect.
    A fake tensor is exempt wherever it is found, for the same reason — and so
    is a wrapper SUBCLASS over fake data (pgw#1198), which is what a
    ``setup()``-time quantizer leaves behind and which an ``isinstance``
    against ``FakeTensor`` cannot see. A META tensor is exempt only inside a
    structure-only component — elsewhere it is an unmaterialized load, which
    :func:`meta_tensors` exists to report."""
    try:
        import torch

        from ..meta_instantiation import is_virtual

        target = torch.device(device).type
    except Exception:
        return []
    out: List[tuple[str, str, str]] = []
    for cname, comp in _named_components(obj):
        if comp is None or not hasattr(comp, "named_parameters"):
            continue
        if getattr(comp, _STRUCTURE_ONLY, False):
            continue
        try:
            named = list(comp.named_parameters())
            if hasattr(comp, "named_buffers"):
                named.extend(comp.named_buffers())
        except Exception:
            continue
        for tname, t in named:
            if not isinstance(t, torch.Tensor):
                continue
            # Allocates nothing => not misplaced. Asked of the STORAGE, so a
            # wrapper subclass over fake data answers like the fake it wraps
            # (pgw#1198). A META tensor is the one virtual thing this walk must
            # still report: outside a structure-only component it is an
            # unmaterialized load, which `meta_tensors` reads out of here.
            if t.device.type != "meta" and is_virtual(t):
                continue
            if t.device.type != target:
                out.append((cname, tname, str(t.device)))
    return out


def meta_tensors(obj: Any) -> List[tuple[str, str]]:
    """Parameters/buffers left unmaterialized by a failed low-memory load."""
    return [
        (component, name)
        for component, name, device in device_mismatches(obj, "cpu")
        if device == "meta"
    ]


def repair_device_placement(obj: Any, device: str) -> List[tuple[str, str, str]]:
    """Targeted ``.to(device)`` on each component holding off-device tensors,
    then re-walk. Returns the remaining mismatches ([] = fully repaired)."""
    missed = device_mismatches(obj, device)
    if not missed:
        return []
    bad = {c for c, _, _ in missed}
    for cname, comp in _named_components(obj):
        if cname not in bad:
            continue
        try:
            comp.to(device)
        except Exception as exc:
            _LOG.warning("device repair: %s.to(%s) failed: %s", cname or "obj", device, exc)
    remaining = device_mismatches(obj, device)
    if not remaining:
        return []
    # ESCALATION (pgw#1558). A second `.to(device)` is not a different act from
    # the first, so a component whose `.to()` is a NO-OP — a torchao/quantized
    # wrapper, an accelerate-hooked module, anything that overrides `_apply` —
    # survives the retry unchanged and the caller is told "unrepairable". It is
    # not: rebinding `tensor.data` moves the storage regardless of what the
    # module's `.to()` chose to do, and `minimax-h3` has been carrying exactly
    # this escalation privately since a stuck 27 GiB text encoder OOM'd a
    # denoise that had already "evicted" it.
    # Only the tensors `device_mismatches` NAMED — it is the walk that already
    # exempts virtual/structure-only/meta tensors, and force-moving one of
    # those is the pgw#1124 defect with a bigger hammer.
    wanted = {(c, t) for c, t, _ in remaining}
    still = {c for c, _, _ in remaining}
    forced = 0
    for cname, comp in _named_components(obj):
        if cname not in still:
            continue
        for owner, tname, t, slot in _owned_tensors(comp):
            if (cname, tname) not in wanted:
                continue
            leaf = tname.rsplit(".", 1)[-1]
            try:
                moved = t.data.to(device)
            except Exception as exc:  # noqa: BLE001 — report what is left, never raise
                _LOG.warning(
                    "device repair: %s.%s could not be read onto %s: %s",
                    cname or "obj", tname, device, exc,
                )
                continue
            try:
                # In-place first: this preserves tensor IDENTITY, which is what
                # `Module._apply` itself does and what anything holding a
                # reference to the parameter (an installed hook, a cached
                # module handle) depends on.
                t.data = moved
                forced += 1
                continue
            except Exception as identity_exc:  # noqa: BLE001
                _LOG.debug(
                    "device repair: %s.%s in-place rebind refused (%s); replacing the slot",
                    cname or "obj", tname, identity_exc,
                )
            try:
                # torch itself refuses some in-place rebinds across device
                # kinds. Replacing the slot on the OWNING module is the same
                # move by the other door, and is what `_apply` falls back to
                # for buffers.
                slot[leaf] = _like(t, moved)
                forced += 1
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "device repair: %s.%s tensor-wise move to %s failed: %s",
                    cname or "obj", tname, device, exc,
                )
    if forced:
        detail = (
            f"components={sorted(still) or ['obj']} did not follow `.to({device})`; "
            f"{forced} tensor(s) moved storage-wise"
        )
        _LOG.warning("device repair: %s", detail)
        # Hub-visible. A component that ignores `.to()` is the shape that
        # OOM'd a denoise which had already "evicted" a 27 GiB text encoder —
        # recovering it silently means the next one is diagnosed from scratch.
        try:
            from .. import activity as _activity

            _activity.emit_event(
                _activity.KIND_RESIDENCY_FAULT, detail=detail, phase="evict_incomplete",
            )
        except Exception:  # noqa: BLE001 — an instrument must never fail a move
            _LOG.debug("device repair: fault event not emitted", exc_info=True)
    return device_mismatches(obj, device)


def _like(old: Any, moved: Any) -> Any:
    """``moved``, wrapped back into a Parameter when ``old`` was one."""
    try:
        import torch

        if isinstance(old, torch.nn.Parameter):
            return torch.nn.Parameter(moved, requires_grad=old.requires_grad)
    except Exception:  # noqa: BLE001
        pass
    return moved


def _owned_tensors(comp: Any) -> List[tuple[Any, str, Any, Any]]:
    """``(owner_module, dotted_name, tensor, slot_dict)`` for every parameter
    and buffer under ``comp``, where ``slot_dict`` is the owner's own
    ``_parameters``/``_buffers`` mapping. The owner is what a placement repair
    needs and ``named_parameters()`` does not give it. [] for a non-module."""
    out: List[tuple[Any, str, Any, Any]] = []
    try:
        import torch
    except Exception:  # noqa: BLE001
        return out
    if comp is None or not hasattr(comp, "named_modules"):
        return out
    try:
        modules = list(comp.named_modules())
    except Exception:  # noqa: BLE001
        return out
    for prefix, module in modules:
        for slot in (
            getattr(module, "_parameters", None),
            getattr(module, "_buffers", None),
        ):
            if not isinstance(slot, dict):
                continue
            for leaf, t in list(slot.items()):
                if not isinstance(t, torch.Tensor):
                    continue
                out.append((module, f"{prefix}.{leaf}" if prefix else leaf, t, slot))
    return out


def _iter_tensors(comp: Any) -> Iterable[tuple[str, Any]]:
    """``(name, tensor)`` over one module's parameters and buffers, [] for a
    non-module. Never raises."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if comp is None or not hasattr(comp, "named_parameters"):
        return []
    try:
        named = list(comp.named_parameters())
        if hasattr(comp, "named_buffers"):
            named.extend(comp.named_buffers())
    except Exception:  # noqa: BLE001
        return []
    return [(n, t) for n, t in named if isinstance(t, torch.Tensor)]


def tensor_storage_bytes(t: Any) -> int:
    """Bytes the tensor OCCUPIES, not what its logical dtype implies.

    pgw#1558, lifted from ``minimax-h3`` where it was written because the
    plain formula lied in production: a torchao tensor subclass reports the
    LOGICAL dtype it emulates, so ``numel() * element_size()`` prices a
    per-row fp8 weight at bf16 — "the census row two independent readers read
    as *no fp8 here*". torch's own ``__tensor_flatten__`` names the real
    inner tensors, so the walk descends into them and sums what they hold.

    This is the byte question every residency estimate in this module asks, so
    it is asked here once: an endpoint that quantizes and then reads
    :func:`estimate_cuda_resident_gb` used to be told its fp8 DiT still cost
    what its bf16 checkpoint did.
    """
    data = getattr(t, "data", t)
    flatten = getattr(data, "__tensor_flatten__", None)
    if flatten is not None:
        try:
            names, _meta = flatten()
            return sum(tensor_storage_bytes(getattr(data, n)) for n in names)
        except Exception:  # noqa: BLE001 — a census must never raise
            pass
    try:
        return int(data.numel()) * int(data.element_size())
    except Exception:  # noqa: BLE001
        return 0


def tensor_dtype_label(t: Any) -> str:
    """``torch.bfloat16``, or ``Float8Tensor[torch.float8_e4m3fn]`` for a
    quantized subclass — never an emulated dtype without naming the subclass
    that emulates it. The reporting twin of :func:`tensor_storage_bytes`
    (pgw#1558, same origin, same reason: a histogram reading
    ``bfloat16=37.4GiB`` over a quantized DiT says this lane is bf16)."""
    data = getattr(t, "data", t)
    flatten = getattr(data, "__tensor_flatten__", None)
    if flatten is None:
        return str(getattr(data, "dtype", "?"))
    try:
        names, _meta = flatten()
        inner = getattr(data, names[0])
        return f"{type(data).__name__}[{inner.dtype}]"
    except Exception:  # noqa: BLE001 — a census must never raise
        return f"{type(data).__name__}[{getattr(data, 'dtype', '?')} logical]"


def resident_census(obj: Any) -> List[tuple[str, int]]:
    """``(component, cuda-resident bytes)`` for every component of ``obj`` that
    holds anything on the card, largest first — what to print when an OOM has
    to name its holder. Storage-priced (:func:`tensor_storage_bytes`), so a
    quantized component is reported at what it actually occupies."""
    rows: List[tuple[str, int]] = []
    for cname, comp in _named_components(obj):
        on_cuda = sum(
            tensor_storage_bytes(t)
            for _n, t in _iter_tensors(comp)
            if t.device.type == "cuda"
        )
        if on_cuda:
            rows.append((cname, on_cuda))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def _sum_tensor_bytes(objs: Iterable[Any], *, cuda_only: bool) -> int:
    """Weight bytes across ``objs``' components, each storage counted once.

    A VIRTUAL tensor (pgw#1080's fake parameters) declares a shape and a device
    and holds no storage, and this walk treats it as such in two places
    (pgw#1128):

    * it is never ``data_ptr()``-ed. Every FakeTensor answers ``0``, so the
      storage dedupe collapsed a whole structure-only tree into its first
      tensor and understated the rest; and torch deprecated the call outright
      (*"Accessing the data pointer of FakeTensor is deprecated and will
      error"*), so the walk that measures a virtual structure would eventually
      raise inside the ``except`` that hides every failure as ``0.0``.
    * it is never RESIDENT. ``cuda_only`` asks what occupies the card right
      now, and a fake parameter occupies nothing, whatever device it claims.
      Booking it made a structure-only pipeline look like its own weights were
      already paid for (``select_auto_mode``'s net requirement fell to zero,
      so every rung read "off").

    It still COUNTS toward the requirement (``cuda_only=False``): the shape and
    dtype it declares are the bytes a real load will go on to allocate. The two
    estimates below are the two questions, and virtuality answers them
    differently. (Until pgw#1199 the mint child was a second such allocator —
    it materialised random values for the pgw#984 proof. It does not any more:
    the proof runs on the resident parent, so nothing downstream of a
    structure-only build allocates a checkpoint.)
    """
    total = 0
    #: ``("ptr", data_ptr)`` for a tensor with storage — shared storages are
    #: counted ONCE — and ``("obj", id)`` for one without, which has no storage
    #: identity to share.
    seen: set[tuple[str, int]] = set()
    for obj in objs:
        for c in _iter_components(obj):
            total += _module_bytes(c, cuda_only=cuda_only, seen=seen)
    return total


def _module_bytes(c: Any, *, cuda_only: bool, seen: set[tuple[str, int]]) -> int:
    """One module's own parameters and buffers, storage-priced and deduped
    against ``seen``. The shared inner walk of :func:`_sum_tensor_bytes` and
    :func:`module_storage_bytes`."""
    import torch

    from ..meta_instantiation import is_virtual

    if c is None or not hasattr(c, "parameters"):
        return 0
    tensors = list(c.parameters())
    if hasattr(c, "buffers"):
        tensors.extend(c.buffers())
    total = 0
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        # pgw#1198: asked of the STORAGE. A `setup()`-time quantizer
        # leaves a wrapper subclass over fake data, which is a
        # FakeTensor to nobody and occupies the card to nothing.
        virtual = t.device.type != "meta" and is_virtual(t)
        if cuda_only and (virtual or t.device.type != "cuda"):
            continue
        key: tuple[str, int]
        if virtual:
            key = ("obj", id(t))
        else:
            try:
                key = ("ptr", t.data_ptr())
            except Exception:
                key = ("obj", id(t))
        if key in seen:
            continue
        seen.add(key)
        # pgw#1558: STORAGE bytes, not the logical dtype's. A virtual
        # tensor has no storage to descend into and is priced by what
        # it DECLARES, which is what a real load will go on to
        # allocate; everything else is priced by what it holds, so a
        # quantized subclass stops being booked at the dtype it
        # emulates.
        total += t.numel() * t.element_size() if virtual else tensor_storage_bytes(t)
    return total


def module_storage_bytes(module: Any, *, cuda_only: bool = False) -> int:
    """Bytes ONE module's own parameters and buffers occupy, shared storages
    counted once. 0 for a non-module.

    pgw#1558. :func:`estimate_pipeline_size_gb` is the wrong tool for a single
    component: it enumerates COMPONENTS OF the object it is handed, and handed
    a bare denoiser it finds that denoiser's submodule attributes and misses
    every parameter held on the root. This asks the module itself, which is
    what "how big is this one component" means — the question a residency
    schedule asks about each of its stage residents.
    """
    return _module_bytes(module, cuda_only=cuda_only, seen=set())


def estimate_pipeline_size_gb(pipeline: Any) -> float:
    """Total weight bytes of a pipeline regardless of device — the *requirement*
    estimate the offload ladder compares against free VRAM. Tensors that share
    storage (shared components) are counted once, and a virtual (fake)
    parameter counts for the bytes it declares, because that is what a real
    load will allocate. 0.0 without torch."""
    try:
        return float(_sum_tensor_bytes([pipeline], cuda_only=False)) / float(1024**3)
    except Exception:
        return 0.0


def estimate_cuda_resident_gb(*objects: Any) -> float:
    """CUDA-resident bytes across the given pipelines/modules, shared storages
    counted once — the *residency accounting* estimate (#358: CPU-offloaded
    pipelines must not be booked as full VRAM; shared components once).

    pgw#1128: a VIRTUAL tensor is not resident. A pgw#1080 structure-only
    component's parameters are fake ON the compute device by construction, and
    booking their declared bytes as occupied VRAM is the same category error
    pgw#1124 fixed in ``device_mismatches`` — a structure that allocates
    nothing read as a card that was already full of it."""
    try:
        return float(_sum_tensor_bytes(objects, cuda_only=True)) / float(1024**3)
    except Exception:
        return 0.0


def release_cached_vram() -> None:
    """Hand the allocator's cached-but-unused blocks back to the driver.
    ``empty_cache`` and NOTHING else — always safe to call.

    pgw#1558. This is the fragmentation discipline a residency schedule runs
    between stages, and it is deliberately not :func:`flush_memory`: that one
    also resets the peak counters, which is exactly what an endpoint measuring
    its own activation across a stage boundary must not do (the same reason
    ``aflush_memory`` defaults ``reset_peak=False``). Nor does it ``gc``,
    which is seconds on a large pipeline and pointless between two stages of
    one request.
    """
    try:
        import torch

        if cuda_ready():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        _LOG.debug("release_cached_vram: empty_cache failed", exc_info=True)


def flush_memory() -> None:
    """gc + empty_cache + reset_peak_memory_stats. Always safe to call."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if cuda_ready():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        pass


async def aflush_memory(*, collect: bool = True, reset_peak: bool = False) -> None:
    """Async twin of :func:`flush_memory` for the executor's teardown paths
    (pgw#657 — three hand-rolled copies of this lived inline).

    Both steps are blocking C calls (a full gc pass over a torn-down pipeline
    is seconds), so each rides ``asyncio.to_thread`` — that, not the body, is
    why this could not simply call ``flush_memory``.

    ``reset_peak`` defaults FALSE and the executor never sets it: pgw#652's
    activation learning reads ``max_memory_allocated`` per request, so a
    teardown that quietly reset the peak would zero the measurement the
    admission ladder is built on.
    """

    if collect:
        await asyncio.to_thread(gc.collect)
    try:
        import torch
    except Exception:  # noqa: BLE001 — torch-free installs (cozy-local CLI)
        return
    if not cuda_ready():
        return
    try:
        await asyncio.to_thread(torch.cuda.empty_cache)
        if reset_peak:
            await asyncio.to_thread(torch.cuda.reset_peak_memory_stats)
    except Exception:
        _LOG.debug("aflush_memory: CUDA cache flush failed", exc_info=True)


def release_unused_pinned_host_cache() -> int:
    """Return unused PyTorch pinned-host blocks to the OS, best effort.

    Pinned swap keeps live RAM-tier weights checked out from PyTorch's host
    allocator.  Once an owner is torn down those tensors are gone, but the
    allocator caches their blocks process-wide; ordinary ``gc.collect()`` and
    ``torch.cuda.empty_cache()`` release neither.  Under measured host-memory
    pressure callers may use this after dropping every object owner.  Active
    blocks (including surviving RAM-tier models) remain owned and untouched.

    The result is the allocator's observed decrease in owned host bytes.  Zero
    means no bytes were released or the installed PyTorch lacks this API.
    """
    try:
        gc.collect()
        import torch

        if not cuda_ready():
            return 0
        # Pinned frees can remain stream-dependent.  Finish prior CUDA work so
        # the host allocator can distinguish inactive blocks before flushing.
        torch.cuda.synchronize()
        try:
            before = int(torch.cuda.memory.host_memory_stats().get(
                "allocated_bytes.current", 0))
        except Exception:
            before = 0

        accelerator = getattr(torch, "accelerator", None)
        accelerator_memory = getattr(accelerator, "memory", None)
        empty_host_cache = getattr(
            accelerator_memory, "empty_host_cache", None)
        if not callable(empty_host_cache):
            # PyTorch exposed the same allocator operation privately before
            # torch.accelerator.memory.empty_host_cache became public.
            empty_host_cache = getattr(
                getattr(torch, "_C", None), "_host_emptyCache", None)
        if not callable(empty_host_cache):
            return 0
        empty_host_cache()
        try:
            after = int(torch.cuda.memory.host_memory_stats().get(
                "allocated_bytes.current", before))
        except Exception:
            return 0
        return max(0, before - after)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Mode selection (auto) — the ONE low-VRAM decider, free-VRAM inputs only
# ---------------------------------------------------------------------------


def select_auto_mode(
    *,
    pipeline: Any,
    available_vram_gb: Optional[float] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
    total_vram_gb: Optional[float] = None,
) -> str:
    """Pick the least-aggressive ladder step that keeps the pipeline in memory.

    FIT decisions (resident vs offload rungs) are made against FREE VRAM
    (what is actually available right now) — a second model on an occupied
    card must see the reduced free space. The resident REFINEMENT (off vs
    vae_only) is made against TOTAL capacity, a per-SKU constant, so the
    traced graph specialization and mint object set are deterministic per SKU
    (pgw#750).

    ``peak_vram_gb`` is the endpoint's DECLARED per-request peak
    (``Resources.peak_vram_per_request_gb``, #339); when provided the fit
    requirement becomes ``max(model_gb, peak_vram_gb)``.

    pgw#1025: every comparison against LIVE FREE VRAM uses the requirement
    NET of what this pipeline already holds on the card. ``avail`` has
    already been reduced by the gw#479 shared components resident on CUDA,
    so comparing it against the pipeline's TOTAL weight bytes counts those
    bytes twice — measured at ~7.85 GB for z-image and ~15.5 GB for
    qwen-image, enough to push a second shared lane off the resident rung
    entirely (th#1867 deleted the ``strict_vram`` refusal that used to sit here).
    The per-SKU refinement below keeps the GROSS requirement on purpose.
    """
    avail = available_vram_gb if available_vram_gb is not None else get_available_vram_gb()
    # "How much is free" and "why is it zero" are two questions, and the
    # second is only worth asking when the first answers zero — which also
    # keeps a caller-supplied or stubbed figure authoritative all the way
    # through, exactly as before.
    zero_cause = "" if avail > 0.0 else available_vram().reason
    if avail <= 0.0:
        # pgw#940. `return "off"` for BOTH zero-causes: "off" means fully
        # resident, no offload at all — the single most memory-hungry rung on
        # the ladder — so a GPU host whose probe raised loaded a pipeline that
        # needed `group_offload` fully resident and OOMed during load. The two
        # causes are different facts and get different answers.
        #
        # No CUDA: "off" is still correct and is not a placement claim at all
        # — there is no card to offload FROM, and the CPU path ignores the
        # rung. Unreadable: the deepest rung this ladder has, matching the
        # unknown-model-size branch below, which has always descended to
        # `group_offload` rather than up to `off`. A rung of performance is
        # the price; an OOM on paid tenant work is what it buys off.
        if zero_cause == VRAM_UNREADABLE:
            _LOG.warning(
                "free VRAM unreadable; selecting %s rather than the resident "
                "rung — an unmeasured card does not license full residency "
                "(pgw#940)", "group_offload")
            return "group_offload"
        return "off"

    model_gb = model_size_gb if model_size_gb is not None else estimate_pipeline_size_gb(pipeline)
    requirement = model_gb
    if peak_vram_gb is not None and peak_vram_gb > 0.0:
        requirement = max(model_gb, float(peak_vram_gb))
    margin = _DEFAULT_SAFETY_MARGIN_GB
    # What is already ON the card: free VRAM has paid for it, so the
    # incremental cost of placing this pipeline is the rest.
    fit_requirement = max(0.0, requirement - estimate_cuda_resident_gb(pipeline))

    if requirement > 0.0:
        usable = max(0.0, avail - margin)
        # Fits (measured weights + margin — a quantized pipeline measures its
        # REAL post-quant size): resident, never offloaded. gw#521: the old
        # absolute low-free-VRAM rule group-offloaded pipelines the emergency
        # rung had just shrunk to fit, making the rung pointless on exactly
        # the cards it exists for.
        if fit_requirement <= usable:
            # pgw#750: BOTH branches are RESIDENT — this refinement only
            # toggles VAE slicing, which changes the traced decode graph
            # class and hence the compiled object set a mint proves. Key it
            # on the card's TOTAL capacity (a per-SKU constant), never live
            # free VRAM: the live margin hovered at this threshold and
            # split one identical L4 fleet's mints 6/13 into off/vae_only
            # cohorts (VRAM-posture-dependent proof_failed roulette). The
            # FIT decisions above/below stay free-VRAM-based (safety); a
            # card that later runs tight degrades reactively down
            # PLACEMENT_LADDER instead of choosing a nondeterministic mint
            # posture up front. pgw#1025 does NOT touch this comparison: it
            # is the GROSS requirement against a per-SKU constant, and
            # netting out live residency here would restore exactly the
            # nondeterminism pgw#750 removed.
            total = total_vram_gb if total_vram_gb is not None \
                else get_total_vram_gb()
            if total > 0.0:
                sku_usable = max(
                    0.0, total - GPU_VRAM_OVERHEAD_GB - margin)
                if (sku_usable - requirement) >= _DEFAULT_OFF_HEADROOM_GB:
                    return "off"
                return "vae_only"
            # No total-capacity probe: the only input left is live free VRAM,
            # so this one takes the NET requirement like the fit test above.
            if (usable - fit_requirement) >= _DEFAULT_OFF_HEADROOM_GB:
                return "off"
            return "vae_only"
        # Doesn't fit: very low free VRAM needs the aggressive rung.
        if avail <= _DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB:
            return "group_offload"
        return "model_offload"

    # Unknown model size: conservative free-VRAM thresholds.
    if avail <= _DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB:
        return "group_offload"
    if avail <= _DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB:
        return "model_offload"
    return "vae_only"


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _call_if_present(obj: Any, method: str, **kwargs: Any) -> bool:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return False
    try:
        fn(**kwargs) if kwargs else fn()
        return True
    except TypeError:
        try:
            fn()
            return True
        except Exception:
            return False
    except Exception as exc:
        _LOG.debug("low_vram: %s() raised %s", method, exc)
        return False


def _to_host(pipeline: Any) -> None:
    """Move the pipeline to host RAM — best effort, with ONE failure that is
    not best effort.

    pgw#1315: ``HostRamMoveRefusedError`` used to be swallowed into a DEBUG
    line, after which ``place_pipeline``'s rollback check resurfaced it as a
    generic ``RuntimeError("… mixed-device … rollback failed")``. That erased
    the one case where ``GEN_WORKER_HOST_MOVE_GUARD`` legitimately stops a
    degrade — it refuses only a move that would SIGKILL the worker anyway — and
    made it indistinguishable from a bug in our own rollback.

    Every OTHER move failure stays swallowed, deliberately: this is a hygiene
    step, and ``repair_device_placement`` is what decides whether the pipeline
    actually came back coherent. Promoting all of them to fatal would turn
    recoverable descents into refusals.
    """
    try:
        if callable(getattr(pipeline, "to", None)):
            pipeline.to("cpu")
    except HostRamMoveRefusedError:
        raise
    except Exception as exc:
        _LOG.debug("low_vram: move-to-cpu failed: %s", exc)


def _move_pipeline_to_cpu(pipeline: Any) -> None:
    """Roll a partially-promoted pipeline back to the host. A no-op without
    CUDA, where nothing was ever promoted off it."""
    if not cuda_ready():
        return
    _to_host(pipeline)


def _apply_vae_and_attention(
    pipeline: Any, applied: Dict[str, bool], *, no_reactive_ladder: bool = False
) -> None:
    """VAE/attention memory savers.

    **A PLACEMENT RUNG ANSWERS "WHERE DO THE WEIGHTS LIVE" AND NOTHING ELSE**
    (pgw#1570). Tiled decode and attention slicing are ACTIVATION tools on a
    different axis, and they cost real latency: tiled decode re-runs the VAE
    per tile and blends 25% overlaps, and ``enable_attention_slicing()``
    becomes a diffusers ``SlicedAttnProcessor`` — ``baddbmm`` + ``softmax`` in
    a python loop over ``batch*heads`` chunks, materializing the full NxN score
    matrix — **in place of** ``AttnProcessor2_0`` (torch SDPA -> flash /
    mem-efficient). That is a per-STEP tax on every request for the life of
    the process, and it was being levied as a side effect of moving weights.

    Measured A/B/A, RTX 4070 Laptop, SDXL 1024^2, 20 steps, CFG, bf16, eager,
    on the ``model_offload`` rung, per-step off the sampler's own loop rate:
    **sliced 1.10 s/step (two arms, bracketing) against SDPA's 0.827 — 1.33x**,
    warm round trip 26.1/33.5 s against 23.6 s. **Peak VRAM moved 5932 ->
    5956 MiB: the whole tax bought 24 MiB.** SDPA's workspace is nothing at a
    4096-token sequence, and the rung had already freed 1.6 GiB by evicting the
    text encoders. This term was 84% of the round-trip gap [[va#3]] measured
    against ComfyUI on this card, which runs SDPA on the same weights.

    th#1107 cut these off the ``vae_only`` rung on exactly this argument and
    stopped there. The rest of the cut is here: **nothing is applied
    proactively on any rung that has a reactive ladder.** pgw#1499's
    ``oom_ladder`` is installed on EVERY rung by :func:`apply_low_vram_config`
    and applies both — tiles on a decode OOM, slices on a denoise-step OOM —
    when an op actually does not fit, confessing each time. A card that never
    needs them never pays.

    ``no_reactive_ladder=True`` is the ONE exception and it names its reason:
    the ``cpu`` rung, where the constraint is host RAM and there is no CUDA OOM
    for ``oom_ladder`` to catch, so the savers must be armed up front.

    VAE *slicing* stays unconditional — it is a no-op at batch 1 and does not
    touch the denoise loop.
    """
    if not _call_if_present(pipeline, "enable_vae_slicing"):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and _call_if_present(vae, "enable_slicing"):
            applied["vae_slicing"] = True
    else:
        applied["vae_slicing"] = True

    if not no_reactive_ladder:
        return

    if not _call_if_present(pipeline, "enable_vae_tiling"):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and _call_if_present(vae, "enable_tiling"):
            applied["vae_tiling"] = True
    else:
        applied["vae_tiling"] = True

    if _call_if_present(pipeline, "enable_attention_slicing"):
        applied["attention_slicing"] = True


def _dtype_fragile_vae(pipeline: Any) -> Optional[Any]:
    """The pipeline's VAE when ``config.force_upcast`` is set (SDXL family).
    Such a VAE mutates dtype at decode time (``upcast_vae`` -> fp32 -> back);
    hook-managed weights miss the runtime cast and decode fatals with
    Half/float mismatches (gw#441/gw#469). It must stay resident on the
    execution device — never hook-managed by any offload rung."""
    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "parameters"):
        return None
    if bool(getattr(getattr(vae, "config", None), "force_upcast", False)):
        return vae
    return None


def unhookable_components(pipeline: Any) -> List[str]:
    """Components no offload rung may hook — the UNION of two independent
    reasons, either of which alone is sufficient:

    * **dtype-fragile** (gw#441/gw#469): a ``force_upcast`` VAE mutates dtype at
      decode; hook-managed weights miss the runtime cast and decode fatals.
    * **content-shared** (gw#479/ie#721): the module is aliased into other
      pipelines, so moving it to the host strands its co-resident consumers on
      the device — a fatal ``mat1 is on cuda:0, mat2 on cpu`` mid-generate.

    Deliberately a UNION and not a precedence: a component can be one, the
    other, or both, and each reason ALONE is enough. A proof that severs only
    one of two independently sufficient terms says nothing about the other, so
    both arms are red-proven separately.
    """
    names: List[str] = list(shared_component_names(pipeline))
    if _dtype_fragile_vae(pipeline) is not None and "vae" not in names:
        names.append("vae")
    return sorted(names)


def _pin_unhookable_components(
    pipeline: Any, applied: Dict[str, bool], log: logging.Logger,
) -> None:
    """Keep dtype-fragile and content-shared components out of the diffusers
    CPU-offload hooks. ``_exclude_from_cpu_offload`` is honored by BOTH the
    model and sequential rungs, which move excluded components to the execution
    device themselves."""
    names = unhookable_components(pipeline)
    if not names:
        return
    excl = list(getattr(pipeline, "_exclude_from_cpu_offload", None) or [])
    for name in names:
        if name not in excl:
            excl.append(name)
    try:
        pipeline._exclude_from_cpu_offload = excl
    except Exception:
        return
    if _dtype_fragile_vae(pipeline) is not None:
        applied["vae_resident"] = True
        log.info(
            "low_vram: force_upcast vae stays resident (excluded from offload "
            "hooks — dtype-safety, gw#441/gw#469)"
        )
    shared = shared_component_names(pipeline)
    if shared:
        applied["shared_resident"] = True
        log.info(
            "low_vram: %d content-shared component(s) stay resident (%s) — "
            "moving one to the host strands its co-resident consumers on the "
            "device (gw#479/ie#721)", len(shared), ", ".join(shared),
        )


def _execution_device() -> Any:
    """The device an offload rung onloads TO. Index 0, like every other rung
    here (``enable_model_cpu_offload(gpu_id=0)``)."""
    return "cuda:0"


#: Where the last residency reserve came from, for the confession to state.
_LAST_RESERVE: Dict[str, Any] = {}


def _plan_partial_resident(
    pipeline: Any, log: logging.Logger, *, min_moved_bytes: int = 0,
    peak_vram_gb: Optional[float] = None, model_size_gb: Optional[float] = None,
) -> Any:
    """The pgw#1577 component-residency plan, or None to keep ``model_offload``.

    None is the answer whenever the subset search cannot beat the coarse rung —
    the denoiser alone over budget, an unmeasurable tree, no CUDA. It is never
    an exception: this sits on the load path of every offloaded pipeline, and a
    planner that raises would take out placements that work today.
    """
    try:
        from .partial_resident import (
            PARTIAL_RESIDENT_RESERVE_GB,
            plan_for_pipeline,
        )

        free_gb = get_available_vram_gb()
        if free_gb <= 0.0:
            return None
        # pgw#1595/#1586 item 5. THE RESERVE MUST COME FROM THE REQUEST, NOT A
        # CONSTANT. `PARTIAL_RESIDENT_RESERVE_GB` was derived from ONE workload
        # shape (pgw#1570's 20-step 1024^2 SDXL); a 28-step job overran it and
        # thrashed the allocator at 6.6 MB free. The endpoint's DECLARED
        # per-request peak is already in this function's caller and was being
        # dropped on the floor — `select_auto_mode` gets it, this planner did
        # not. `peak_vram_gb` is a TOTAL requirement, so the activation share is
        # what it asks for beyond the weights; the constant stays as a FLOOR for
        # endpoints that declare nothing or under-declare.
        reserve_gb = PARTIAL_RESIDENT_RESERVE_GB
        # NOT A COSMETIC FIELD. Measured 2026-08-20: ZERO of the 26 shipped
        # endpoints declare `peak_vram_per_request_gb`, so on every real serve
        # today this reserve is an ASSUMPTION carried over from one workload
        # shape, not a measurement of this one. pgw#1595's 28-step job overran
        # it. Until endpoints declare, the honest thing the log can do is say
        # which of the two it used.
        reserve_source = "default"
        declared = 0.0
        if peak_vram_gb is not None and peak_vram_gb > 0.0:
            weights_gb = (
                model_size_gb if model_size_gb is not None
                else estimate_pipeline_size_gb(pipeline)
            )
            declared = max(0.0, float(peak_vram_gb) - float(weights_gb))
            reserve_source = "declared" if declared > reserve_gb else "default"
            if declared > reserve_gb:
                log.info(
                    "low_vram: partial_resident reserve %.2f -> %.2f GiB, from "
                    "the endpoint's declared per-request peak (%.2f GiB total, "
                    "%.2f GiB of weights)",
                    reserve_gb, declared, float(peak_vram_gb), float(weights_gb),
                )
                reserve_gb = declared
        _LAST_RESERVE.clear()
        _LAST_RESERVE.update(reserve_gb=reserve_gb, reserve_source=reserve_source)
        plan = plan_for_pipeline(
            pipeline,
            budget_bytes=int(max(0.0, free_gb - reserve_gb) * _GIB),
            free_bytes=int(free_gb * _GIB),
            sizer=lambda m: module_storage_bytes(m),
            forced_resident=unhookable_components(pipeline),
            min_moved_bytes=min_moved_bytes,
        )
    except Exception as exc:
        log.warning(
            "low_vram: could not plan component residency (%s: %s); keeping "
            "model_offload", type(exc).__name__, exc,
        )
        return None
    if not plan.fits:
        log.info("low_vram: partial_resident declined — %s", plan.refusal)
        return None
    if not plan.offloaded:
        # Nothing to evict means the pipeline fits resident, which is a fit
        # decision `select_auto_mode` already made against a different margin.
        # Do not overrule it from here.
        return None
    log.info("low_vram: upgrading model_offload -> partial_resident: %s",
             plan.summary())
    return plan


#: The typed phase a `partial_stream` arming FAILURE confesses under. A rung
#: that could not arm and fell through to a coarser one is a placement the
#: operator asked for and did not get — pgw#1497 measured that exact silence on
#: the card (a component-vocabulary AttributeError, a warning, a pipeline that
#: served on `model_offload`, and nothing off the pod said so).
PARTIAL_STREAM_UNARMED_PHASE = "partial_stream_unarmed"

#: Set on a pipeline the `partial_stream` rung armed: its
#: :class:`~gen_worker.models.stream_residency.StreamedResidency`, so the tail
#: can be trimmed or promoted later without rediscovering the tree.
STREAM_RESIDENCY_ATTR = "_cozy_stream_residency"


def stream_residency_of(pipeline: Any) -> Any:
    """The `partial_stream` handle armed on ``pipeline``, or None."""
    return getattr(pipeline, STREAM_RESIDENCY_ATTR, None)


def _apply_partial_stream(
    pipeline: Any,
    applied: Dict[str, Any],
    *,
    budget_bytes: Any,
    log: logging.Logger,
    device: str = "cuda",
) -> bool:
    """Arm pgw#1497's per-leaf budgeted residency. False = could not.

    ``budget_bytes`` is the DEVICE bytes this pipeline's weights may occupy,
    and it is the caller's — the residency lease's — number. Nothing here
    estimates it, and the whole rung refuses rather than invent one.

    The dtype-fragile and content-shared union is excluded exactly as every
    other rung excludes it: those components are handed to the ring by nobody
    and stay wherever they are.
    """
    def _unarmed(reason: str) -> bool:
        """The rung did not arm. Say so on BOTH channels and fall through.

        A warning alone was the defect: the pipeline still served, on a
        coarser rung than the budget asked for, and nothing off the pod
        recorded it. Placement the operator did not get is a degradation.
        """
        _confess_serve_degrade(
            phase=PARTIAL_STREAM_UNARMED_PHASE,
            line=transition_line(
                event="refused", phase="load", from_rung="partial_stream",
                to_rung="model_offload", detail=reason,
            ),
            detail=(
                f"pipeline={type(pipeline).__name__}: the partial_stream rung "
                f"could NOT arm under its {budget_bytes} byte budget "
                f"({reason}); this pipeline falls through to a COARSER rung "
                f"and the per-leaf budget it was admitted for is not being "
                f"honoured."
            ),
            log=log,
        )
        return False

    try:
        from .stream_residency import MemoryBudget, StreamedResidency
    except Exception as exc:  # noqa: BLE001 — torch-less host
        return _unarmed(f"{type(exc).__name__}: {exc}")

    excluded = set(unhookable_components(pipeline))
    # `_named_components` answers the pipeline's whole COMPONENT vocabulary,
    # and a real sd1.5 pipeline puts a CLIPTokenizer, a scheduler and a feature
    # extractor in it. Measured on the card: without this filter the rung
    # raised `CLIPTokenizer has no attribute named_modules` and fell through to
    # `model_offload` — silently, because the fall-through is a warning and the
    # pipeline still served.
    roots = [
        (name, module)
        for name, module in _named_components(pipeline)
        if name not in excluded and hasattr(module, "named_modules")
    ]
    if not roots:
        # A bare module (a lane ModuleDict, a test tree) is its own root.
        roots = [(type(pipeline).__name__, pipeline)] if hasattr(
            pipeline, "named_modules"
        ) else []
    if not roots:
        return _unarmed("no hookable nn.Module tree on this pipeline")

    # The excluded components are kept OUT of the ring, and that makes placing
    # them this rung's job — exactly as `_apply_group_offload` keeps its own
    # `exclude_modules` resident. MEASURED on the 4070: sd1.5's VAE is
    # `force_upcast`, so it is dtype-fragile and excluded, and with nobody
    # moving it the first decode died with `Input type (torch.cuda.HalfTensor)
    # and weight type (torch.HalfTensor) should be the same`. An exclusion is a
    # statement about HOOKS, never about residency.
    components = getattr(pipeline, "components", None)
    for name in sorted(excluded):
        module = components.get(name) if isinstance(components, dict) else None
        if module is None or not hasattr(module, "to"):
            continue
        try:
            module.to(device)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "low_vram: partial_stream could not keep excluded component %r "
                "on %s (%s: %s); it will not serve",
                name, device, type(exc).__name__, exc,
            )

    try:
        residency = StreamedResidency(
            roots, device=device, budget_bytes=MemoryBudget.of(budget_bytes)
        )
        plan = residency.engage()
    except Exception as exc:  # noqa: BLE001
        return _unarmed(f"{type(exc).__name__}: {exc}")

    try:
        # Before the handle is visible, not after: the device repair below
        # READS it, and diffusers' `__setattr__` treats attribute names as
        # component registrations, so a failed set must not go unnoticed.
        setattr(pipeline, STREAM_RESIDENCY_ATTR, residency)
    except Exception:  # noqa: BLE001
        log.warning(
            "low_vram: could not stamp the partial_stream handle on %s; "
            "`pipeline.device` will keep answering with the host while the "
            "tail is parked there",
            type(pipeline).__name__,
        )
    # The rung parks leaves on the host, so `pipeline.device` stops answering
    # with the execution device. Install the repair HERE too, not only on the
    # `apply_low_vram_config` path, so a direct caller gets a coherent
    # pipeline rather than an embedding that dies on a host-side index.
    install_execution_device_fallback()
    applied["partial_stream"] = True
    applied["stream_budget_bytes"] = int(plan.budget_bytes)
    # The RAM half of the assigned pair, REPORTED — enforcement is the named
    # pgw#1497 follow-up. `host_bytes` is what the pinned tail costs the host.
    applied["stream_ram_budget_bytes"] = int(plan.ram_budget_bytes)
    applied["stream_host_bytes"] = int(plan.host_bytes)
    applied["stream_host_fits"] = bool(plan.host_fits)
    applied["stream_resident_bytes"] = int(plan.resident_bytes)
    applied["stream_streamed_bytes"] = int(plan.streamed_bytes)
    applied["stream_window_bytes"] = int(plan.window_bytes)
    applied["stream_resident_leaves"] = len(plan.all_resident)
    applied["stream_streamed_leaves"] = len(plan.streamed)

    if not plan.streamed:
        # The budget held the whole tree. That is not a degradation and must
        # not be reported as one — the rung armed and then had nothing to do.
        log.info(
            "low_vram: partial_stream armed on %s and streams nothing — the "
            "%.2f GiB budget holds all %d leaves; serving fully resident",
            type(pipeline).__name__, budget_bytes / _GIB, len(plan.all_resident),
        )
        return True

    if not plan.fits:
        # The confession the `fits` property exists for: even the streaming
        # window is over the lease. It serves, and it says so.
        log.warning(
            "low_vram: partial_stream is OVER ITS LEASE on %s — %.2f GiB of "
            "in-flight cast window against a %.2f GiB budget. It serves; the "
            "budget arithmetic upstream is what needs looking at.",
            type(pipeline).__name__, plan.window_bytes / _GIB,
            budget_bytes / _GIB,
        )
    log.warning(
        "DEGRADED_MODE=engaged model=%s phase=load rung=resident->"
        "partial_stream: %d of %d leaves (%.2f GiB) rest in pinned host RAM "
        "and cast per forward; %.2f GiB stays resident under a %.2f GiB "
        "budget, %.2f GiB reserved for the in-flight cast window",
        type(pipeline).__name__, len(plan.streamed),
        len(plan.streamed) + len(plan.all_resident),
        plan.streamed_bytes / _GIB, plan.resident_bytes / _GIB,
        budget_bytes / _GIB, plan.window_bytes / _GIB,
    )
    return True


def _apply_group_offload(
    pipeline: Any,
    applied: Dict[str, bool],
    *,
    offload_to_disk_path: Optional[str],
) -> bool:
    try:
        import torch
    except Exception:
        return False
    if not cuda_ready():
        return False

    kwargs: Dict[str, Any] = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": True,
    }
    if offload_to_disk_path:
        kwargs["offload_to_disk_path"] = offload_to_disk_path

    fragile_vae = _dtype_fragile_vae(pipeline)
    # ie#721: the group rung takes the SAME union the hook-based rungs do.
    # `exclude_modules` keeps them out of the group hooks; the caller must then
    # put them ON the device itself, exactly as the fragile-VAE path always has.
    excluded = unhookable_components(pipeline)
    shared = shared_component_names(pipeline)

    def _keep_excluded_resident() -> None:
        comps = getattr(pipeline, "components", None)
        for name in excluded:
            mod = comps.get(name) if isinstance(comps, dict) else None
            if mod is None:
                mod = getattr(pipeline, name, None)
            if mod is None or not hasattr(mod, "to"):
                continue
            try:
                mod.to("cuda")
            except Exception as exc:
                _LOG.warning("low_vram: resident move failed for %s: %s", name, exc)
        if fragile_vae is not None:
            applied["vae_resident"] = True
            _LOG.info(
                "low_vram: force_upcast vae stays resident under group offload "
                "(dtype-safety, gw#441/gw#469)"
            )
        if shared:
            applied["shared_resident"] = True
            _LOG.info(
                "low_vram: %d content-shared component(s) stay resident under "
                "group offload (%s) — gw#479/ie#721",
                len(shared), ", ".join(shared),
            )

    fn = getattr(pipeline, "enable_group_offload", None)
    if callable(fn):
        try:
            if excluded:
                fn(**kwargs, exclude_modules=list(excluded))
            else:
                fn(**kwargs)
            _keep_excluded_resident()
            applied["group_offload"] = True
            if offload_to_disk_path:
                applied["disk_offload_path"] = True
            return True
        except Exception as exc:
            _LOG.debug("low_vram: pipeline.enable_group_offload failed: %s", exc)

    any_applied = False
    apply_group_offloading: Any = None
    try:
        from diffusers.hooks import apply_group_offloading
    except Exception:
        apply_group_offloading = None

    # pgw#740: enumerate what the pipeline ACTUALLY carries. The old fixed list
    # skipped Wan's transformer_2 and LTX's connectors silently — they stayed
    # fully resident while the caller believed group offload had been applied.
    skipped: List[str] = []
    for attr, mod in _module_attributes(pipeline):
        if mod is None:
            continue
        # DELIBERATELY resident, not accidentally uncovered — so it is not added
        # to `skipped` below, whose warning is about the silent kind. ie#721
        # widened this from "the fragile vae" to every unhookable component.
        if attr in excluded:
            continue
        mod_fn = getattr(mod, "enable_group_offload", None)
        if callable(mod_fn):
            try:
                mod_fn(**kwargs)
                any_applied = True
                continue
            except Exception as exc:
                _LOG.debug("low_vram: %s.enable_group_offload failed: %s", attr, exc)
        if apply_group_offloading is not None:
            try:
                apply_group_offloading(
                    mod,
                    onload_device=kwargs["onload_device"],
                    offload_type="block_level",
                    num_blocks_per_group=2,
                    **({"offload_to_disk_path": offload_to_disk_path} if offload_to_disk_path else {}),
                )
                any_applied = True
            except Exception as exc:
                _LOG.debug("low_vram: apply_group_offloading(%s) failed: %s", attr, exc)
                skipped.append(attr)
        else:
            skipped.append(attr)

    # Fail-loud doctrine: a component that stays fully resident while the
    # caller believes it was offloaded is exactly the silence this rule exists
    # to remove. WARNING, naming every one of them.
    if skipped:
        _LOG.warning(
            "low_vram: group offload did NOT cover %d component(s): %s — they "
            "stay fully resident; VRAM planning for this pipeline is wrong by "
            "their size", len(skipped), ", ".join(sorted(skipped)))

    if any_applied:
        _keep_excluded_resident()
        applied["group_offload"] = True
        if offload_to_disk_path:
            applied["disk_offload_path"] = True
    return any_applied


def _apply_gguf_dequant_ahead(
    pipeline: Any,
    applied: Dict[str, Any],
    *,
    budget: Any,
    log: logging.Logger,
) -> None:
    """Spend the residency lease's SURPLUS decoding GGML weights once, at load.

    Paul's three-tier ruling (2026-08-19): quantized-resident and
    fully-dequantized are the two ENDPOINTS OF ONE DIAL, and which end a worker
    sits at is the lease's answer, not the loader's. A worker handed surplus
    memory should use it — one decode pass beats paying a per-forward decode
    every step for the life of the endpoint — and a constrained worker pays per
    forward. ``gguf_torch.dequant_ahead`` graduates weights LARGEST FIRST, so
    the transient headroom the fit plan must reserve falls as the dial turns up.

    The surplus is ``lease VRAM − what the pipeline already costs``, and it is
    capped by what the card ACTUALLY has free, exactly as the ``partial_stream``
    upgrade caps its budget: a lease written when the card was emptier is not a
    licence to allocate bytes that are gone. Nothing is ever re-quantized, so
    this is one-way and the ladder still only ever SELECTS artifacts.

    Runs BEFORE the rung is chosen, because turning the dial changes the
    resident footprint the chooser is looking at.

    ``budget`` is pgw#1497's admission shape — a bare VRAM int or the
    {VRAM, RAM} :class:`~gen_worker.models.stream_residency.MemoryBudget` pair.
    Only the VRAM half means anything here: decoding ahead moves nothing to the
    host, so this dial never spends the RAM half.
    """
    try:
        from .gguf_torch import dequant_ahead, gguf_leaves, peak_transient_bytes
        from .stream_residency import MemoryBudget
    except Exception:  # noqa: BLE001 — torch-less host
        return

    lease = MemoryBudget.of(budget).vram_bytes
    if lease <= 0:
        return

    denoisers = [
        module for _, module in _named_components(pipeline)
        if hasattr(module, "named_modules") and gguf_leaves(module)
    ]
    if not denoisers and hasattr(pipeline, "named_modules") and gguf_leaves(pipeline):
        denoisers = [pipeline]
    if not denoisers:
        return

    import torch

    dtype = torch.bfloat16
    resident = int(_sum_tensor_bytes([pipeline], cuda_only=False))
    headroom = int(max(0.0, get_available_vram_gb() - _DEFAULT_SAFETY_MARGIN_GB) * _GIB)
    surplus = max(0, min(lease, headroom) - resident)
    materialized = 0
    for denoiser in denoisers:
        done = dequant_ahead(denoiser, surplus_bytes=float(surplus), dtype=dtype)
        materialized += len(done)
        # Each pass spends its own share; charge it so two denoisers cannot
        # both spend the whole surplus.
        after = int(_sum_tensor_bytes([pipeline], cuda_only=False))
        surplus = max(0, surplus - (after - resident))
        resident = after

    applied["gguf_dequant_ahead"] = materialized
    applied["gguf_quantized_bytes"] = sum(
        int(_gguf_quantized_bytes(d)) for d in denoisers)
    applied["gguf_peak_transient_bytes"] = max(
        (int(peak_transient_bytes(d, dtype=dtype)) for d in denoisers), default=0)
    log.info(
        "low_vram: gguf dequant-ahead materialized %d weight(s) under a "
        "%.2f GiB lease; %.2f GiB still resides as ggml blocks, largest "
        "per-forward decode %.3f GiB",
        materialized, lease / _GIB,
        applied["gguf_quantized_bytes"] / _GIB,
        applied["gguf_peak_transient_bytes"] / _GIB,
    )


def _gguf_quantized_bytes(model: Any) -> int:
    from .gguf_torch import quantized_bytes

    return quantized_bytes(model)


def _gguf_resident_override(
    pipeline: Any, effective: Mode, log: logging.Logger,
) -> Mode:
    """Keep a selected GGUF rung resident when its remaining weights fit."""
    if not touches_host_ram(effective) or not getattr(
        pipeline, "_cozy_gguf_quant", None
    ):
        return effective
    total = estimate_pipeline_size_gb(pipeline)
    remaining = max(0.0, total - estimate_cuda_resident_gb(pipeline))
    available = get_available_vram_gb()
    if total > 0.0 and remaining + _GGUF_RESIDENT_MARGIN_GB <= available:
        log.info(
            "low_vram: GGUF pipeline fits resident (%.2f GB remaining, "
            "%.2f GB free); using vae_only instead of %s",
            remaining,
            available,
            effective,
        )
        return "vae_only"
    return effective


def place_pipeline(
    pipeline: Any,
    *,
    logger: Optional[logging.Logger] = None,
    mode: Mode = "auto",
    ref: str = "",
) -> Dict[str, Any]:
    """Worker-owned placement + offload policy for a freshly-loaded pipeline.

    ``mode="auto"`` runs the one low-VRAM decider against free VRAM: plenty of
    headroom puts the whole pipeline on CUDA; tighter budgets step down the
    offload ladder. Callers with plan-time knowledge (a ServePlan offload
    verdict, a learned degraded floor) pass ``mode`` explicitly so a doomed
    fully-resident attempt is never paid (ie#369). Endpoints never write
    device/offload code — the worker calls this around ``setup()`` injection.
    No-op without CUDA.

    A CUDA OOM during placement is a ladder transition, not a failure
    (gw#463): flush, demote one offload rung, retry — down to sequential.
    The result dict carries ``oom_demotions``/``requested_mode`` when that
    happened so the caller can record + report the degradation.

    th#1867 deleted the ``strict_vram`` opt-out (th#1043/th#1107) that used to
    turn a CPU-touching rung into a hard refusal here. It was the author
    declaring a card requirement in softer words (§2.4 ruling 4), and it made
    THIS function — the one place that actually MEASURES the pipeline against
    the card — refuse on a declaration instead. The descent is now bounded only
    by the ladder itself (``models/rung``), and a binding that genuinely cannot
    survive host-resident weights is a defect with an owner in OUR code, not a
    card size to declare around (§1.35 amendment 2).
    """
    log = logger or _LOG
    if not cuda_ready():
        # pgw#1315: the CPU rung is APPLIED here, not merely described — a bare
        # `{"mode": "cpu"}` leaves `low_vram_mode()` reading `""`, so a later
        # transition cannot tell the bottom rung from an unprepped object. Plan time
        # and the reactive descent stamp ONE token.
        return apply_low_vram_config(pipeline, mode="cpu", logger=log)
    effective = select_auto_mode(pipeline=pipeline) if mode == "auto" else mode
    if mode == "auto":
        effective = _gguf_resident_override(pipeline, effective, log)
    requested = effective
    # th#1871 P1 (pgw#1225) §6.6 item 2/3: the fit numbers this decision was
    # made on, kept instead of discarded. They are locals of the decision one
    # frame up (`select_auto_mode`) and were the only quantified answer to "why
    # is this pipeline offloaded" anywhere on the pod — and only on the REACTIVE
    # OOM path did anything downstream even hear that it had happened. A
    # measured descent taken BEFORE any OOM is the largest silent degradation
    # the census found, and it is this branch.
    fit_needed_gb = 0.0
    fit_available_gb = 0.0
    demotions = 0
    while True:
        try:
            # Measured at the rung being ATTEMPTED, and only the first time a
            # CPU-touching one is: on a reactive descent the interesting numbers
            # are the ones that made the resident attempt fail, not the ones
            # left after it has already spilled.
            if fit_needed_gb <= 0.0 and touches_host_ram(effective):
                fit_needed_gb = estimate_pipeline_size_gb(pipeline)
                fit_available_gb = get_available_vram_gb()
            if effective in ("off", "vae_only") and callable(getattr(pipeline, "to", None)):
                pipeline.to("cuda")
            applied = apply_low_vram_config(pipeline, mode=effective, logger=log)
            if demotions:
                applied["oom_demotions"] = demotions
                applied["requested_mode"] = requested
            if fit_needed_gb > 0.0:
                applied["fit_needed_gb"] = fit_needed_gb
                applied["fit_available_gb"] = fit_available_gb
            # DIAGNOSTIC ONLY — deliberately NOT an input to any budget check.
            # Which components this placement forced to stay resident, so an
            # operator asking "why is this pod's footprint higher than the rung
            # implies" can read the answer instead of investigating it. It sits
            # with `oom_demotions` / `fit_needed_gb` as reporting, not contract.
            #
            # It was briefly designed as a published input for the adoption
            # headroom check (pgw#1255). That seam is RETIRED and must not be
            # rebuilt: pgw#1265 checks `have >= need` where `have` is the
            # DRIVER'S FREE-MEMORY FIGURE read at the decision point, so every
            # byte these exclusions hold on the card is already subtracted by
            # OBSERVATION — as is a sibling instance's weights, which no
            # published set could have described. A measured quantity that
            # already includes the effect beats two modules agreeing to exchange
            # a computed one. Wiring this back in would reintroduce exactly the
            # second derivation that seam existed to prevent.
            #
            # Always present, `[]` included, so a reader can tell "nothing was
            # excluded" from "this build predates the field".
            applied["resident_excluded"] = unhookable_components(pipeline)
            return applied
        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise
            nxt_rung = _descend_rung(effective)
            nxt = nxt_rung.name if nxt_rung is not None else None
            if nxt is None:
                raise
            # ``pipeline.to('cuda')`` may have moved only a prefix of the
            # component graph before the allocator raised. Offload hooks must
            # start from a coherent CPU object; attaching them to that partial
            # move creates the mixed-device fatal seen on live SDXL.
            try:
                _move_pipeline_to_cpu(pipeline)
            except HostRamMoveRefusedError as refused:
                # pgw#1315: the guard refusing this rollback and our rollback
                # being BROKEN are different facts, and the generic
                # mixed-device error below reported them as the same one. The
                # guard is correct — it refuses only a move that would SIGKILL
                # the worker — so its verdict travels, carrying the OOM that
                # provoked the rollback as its cause.
                raise refused from exc
            missed = repair_device_placement(pipeline, "cpu")
            if missed:
                raise RuntimeError(
                    "CUDA OOM left the pipeline mixed-device and CPU rollback "
                    f"failed ({missed[:5]!r})"
                ) from exc
            flush_memory()
            log.warning(transition_line(
                event="engaged", model=ref, phase="load",
                from_rung=effective, to_rung=nxt,
                needed_gb=estimate_pipeline_size_gb(pipeline),
                free_gb=get_available_vram_gb(),
                detail=f"CUDA OOM during placement ({type(exc).__name__}); retrying offloaded",
            ))
            demotions += 1
            effective = nxt


#: The phase token every pipeline-level offload activation reports under.
#: Countable hub-side in `worker_activity_events`.
OFFLOAD_ENGAGED_PHASE = "cpu_offload_engaged"

#: pgw#1315: the machine is below a lane's DECLARED minimum and it serves
#: anyway. Same confession home, its own phase token — and the token IS the
#: machine-readable cause (`measured_posture.REASON_BELOW_DECLARED_MINIMUM`),
#: so the reason has one spelling on both carriers.
UNDER_MINIMUM_PHASE = posture_mod.REASON_BELOW_DECLARED_MINIMUM


def _confess_serve_degrade(
    *, phase: str, line: str, detail: str, log: logging.Logger,
) -> None:
    """THE seam every degraded-posture confession leaves this pod through.

    Loud line for a human, typed `serve_degrade` event for the hub, derived
    from the SAME numbers so the two cannot disagree. Never a gate — every
    caller has already decided to serve. pgw#1312 built this for offload
    activation and pgw#1315 puts the under-minimum warning through it rather
    than beside it: a second emitter is a second answer, and the one the hub
    banks is never the one the operator read.
    """
    log.warning(line)
    activity_mod.emit_event(
        activity_mod.KIND_SERVE_DEGRADE, detail=detail, phase=phase)


def report_under_minimum(
    shortfalls: "Sequence[machine_fit.Shortfall]",
    *,
    scope: str,
    posture: str,
    lane: str = "",
    logger: Optional[logging.Logger] = None,
) -> str:
    """This machine is under a DECLARED minimum, and it serves anyway.

    Returns the warning text the caller puts on `ServePlan.warning` (which
    reaches the hub as `FnDegraded.reason`), having already emitted the typed
    event — one derivation, two carriers.

    It names the TERM, the declared floor, this machine's measured fact and
    the posture taken, and it names NO card: th#1867 deleted
    `FnDegraded.recommended_vram_gb` because the worker's suggestion was the
    author's own guess handed back, and only the hub knows the catalog.
    """
    rows = tuple(shortfalls)
    if not rows:
        return ""
    detail = machine_fit.summarize(rows)
    # `scope` names the function; each ROW names its own lane (function-scope
    # rows have none), so the head states the lane the facts PICKED rather
    # than attributing every shortfall to it.
    head = (
        f"running BELOW a declared minimum for {scope}"
        + (f" (lane picked: `{lane}`)" if lane else "")
        + f": {detail}. The request still serves at posture `{posture}`; "
        "this pod is serving DEGRADED and a declared floor is not being met."
    )
    _confess_serve_degrade(
        phase=UNDER_MINIMUM_PHASE,
        line=transition_line(
            event="planned", phase="requirements",
            from_rung="declared_minimum", to_rung=posture, detail=head,
        ),
        detail=head,
        log=logger or _LOG,
    )
    return head


# pgw#1425: `report_unevidenced_serving_facts` is DELETED, not baselined.
# It confessed that a DECLARED serving contract could not be checked against a
# checkpoint's stamped facts. pgw#1373 deleted the catalog/declaration
# architecture under Paul's hardcut mandate, so nothing stamps those facts and
# `@entrypoint` declares no serving contract to check them against — the
# question this answered no longer exists. The v2 degrade channel is
# `serving/placement.warn_if_degraded`, taken at residency-admit and stained
# onto every request the instance serves.


def attention_kernel_census(pipeline: Any) -> str:
    """The attention processor classes actually LIVE on the denoiser, counted.

    pgw#1570: `attention_slicing` was applied by a placement rung and nothing
    ever said which kernel that left running. The flag name is not the answer —
    diffusers turns `enable_attention_slicing()` into a `SlicedAttnProcessor`
    (`baddbmm`+`softmax`, chunked python loop, full NxN scores materialized) in
    place of `AttnProcessor2_0` (torch SDPA -> flash/mem-efficient), and the two
    differ by more than a memory saving. A rung that changes the kernel must
    NAME the kernel: "which one runs" is a fact to be read off the object, never
    inferred from the flag that set it.
    """
    denoiser = getattr(pipeline, "unet", None) or getattr(pipeline, "transformer", None)
    getter = getattr(denoiser, "attn_processors", None) if denoiser is not None else None
    if not getter:
        return "attention=unknown(no attn_processors)"
    counts: Dict[str, int] = {}
    for proc in getter.values():
        name = type(proc).__name__
        counts[name] = counts.get(name, 0) + 1
    return "attention=" + ",".join(
        f"{n}x{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def component_placement_census(pipeline: Any) -> str:
    """Where each weight-bearing component's tensors actually LIVE, read off
    the objects.

    pgw#1577, and the same lesson pgw#1570 learned about attention: the rung a
    log names is the decision, not the outcome. ``_pin_unhookable_components``
    has been setting ``_exclude_from_cpu_offload`` and reporting
    ``vae_resident`` since gw#441 — and diffusers consults that list ONLY for
    components absent from ``model_cpu_offload_seq``, where SDXL's ``vae`` is
    not, so the claim was decorative and nothing could see that. A census can.
    """
    parts: List[str] = []
    for name, comp in _named_components(pipeline):
        if not hasattr(comp, "parameters"):
            continue
        devices: Dict[str, int] = {}
        try:
            for t in comp.parameters(recurse=True):
                key = str(t.device)
                devices[key] = devices.get(key, 0) + 1
        except Exception:  # noqa: BLE001 - a census must not fail a placement
            continue
        if not devices:
            continue
        where = "/".join(sorted(devices, key=lambda d: -devices[d]))
        parts.append(f"{name}@{where}")
    return "placement=" + (",".join(parts) if parts else "unreadable")


def _report_offload_engaged(
    pipeline: Any, rung: str, applied: Dict[str, Any], log: logging.Logger,
    *, plan_free_gb: Optional[float] = None,
) -> None:
    """THE offload-activation confession — one home for every route into a
    CPU-touching rung (pgw#1312).

    Paul, 2026-08-17, killing the CPU-offload env veto: *"We ALWAYS allow
    CPU-offload, and encourage it — but when it happens we warn LOUDLY so the
    error can be caught (we don't want to serve degraded in production)."*
    That veto answered "may this host offload" — not a question, and a logic
    gate in an env var. The question that IS one is "did it happen", and only
    the OOM-triggered descent was answering it: a rung chosen against free VRAM
    before any OOM applied the same hooks and said nothing off the pod. Every
    route reaches `apply_low_vram_config`, so the answer belongs here.

    Loud line for a human, typed `serve_degrade` event for the hub — derived
    from the same numbers so the two cannot disagree. Never a gate: the
    placement has already succeeded when this runs.
    """
    needed_gb = estimate_pipeline_size_gb(pipeline)
    # pgw#1595. THE LINE STATES THE DECISION, SO IT MUST STATE THE DECISION'S
    # INPUT. This used to re-read free VRAM HERE — after placement — so a rung
    # chosen against 7.3 GiB printed `free_gb=0.4` beside its own name, and a
    # whole issue was filed against the wrong cause on the strength of it. The
    # plan-time figure is authoritative when the rung recorded one; the
    # post-placement figure is still reported, as `free_after_gb`, because "what
    # the card looks like now" is a real and different fact.
    free_after_gb = get_available_vram_gb()
    free_gb = free_after_gb if plan_free_gb is None else plan_free_gb
    _confess_serve_degrade(
        phase=OFFLOAD_ENGAGED_PHASE,
        line=transition_line(
            event="engaged", phase="load", from_rung="resident", to_rung=rung,
            needed_gb=needed_gb, free_gb=free_gb,
            detail=f"CPU offload ENGAGED ({_applied_summary(applied)}); every "
                   f"forward on this pipeline now moves weights over PCIe; "
                   f"{attention_kernel_census(pipeline)}; "
                   f"{component_placement_census(pipeline)}; "
                   f"free_after_gb={free_after_gb:.1f}",
        ),
        detail=(
            f"pipeline={type(pipeline).__name__}: CPU offload ENGAGED at rung "
            f"`{rung}` ({_applied_summary(applied)}) — ~{needed_gb:.1f} GiB of "
            f"weights rest in host RAM against {free_gb:.1f} GiB free VRAM and "
            f"stream to the device per forward. Offload is always allowed and "
            f"the request still serves; this pod is serving DEGRADED and every "
            f"request on it pays the transfer."
        ),
        log=log,
    )


_execution_device_fallback_installed = False


def install_execution_device_fallback() -> bool:
    """Make ``pipeline.device`` never answer ``meta``. Idempotent.

    ``enable_sequential_cpu_offload`` leaves every module on the meta device,
    so diffusers' ``DiffusionPipeline.device`` — a PUBLIC property, and the one
    thing endpoint code is told to ask — answers ``meta``. Endpoint code that
    builds a generator on it then dies with ``RuntimeError: META device type
    not an accelerator`` BEFORE any image. Measured on the real sdxl endpoint
    at 1024^2 (pgw#1486): the bottom rung of this ladder, the one whose whole
    job is "always works", did not work at all.

    The endpoint is not at fault and must not be the fix. The worker tells
    authors *"PLACEMENT is decided here, by the worker, never by author code"*
    and `ctx.load`'s contract is that they never name a device — so asking the
    pipeline where it lives is the documented-correct thing to do, and
    `_execution_device` is a private diffusers attribute no endpoint should
    reach for. The rung that breaks the answer repairs the answer, once, for
    every endpoint.

    Patching a foreign class is the same shape as ``host_move_guard``'s patch
    of ``torch.nn.Module.to``, for the same reason: the call site is in author
    code this worker does not own and cannot edit.
    """
    global _execution_device_fallback_installed
    if _execution_device_fallback_installed:
        return True
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    except Exception:  # pragma: no cover - no diffusers, nothing to patch
        return False

    original_getter = DiffusionPipeline.device.fget  # type: ignore[attr-defined]
    # `_execution_device` ENDS in `return self.device` when no component
    # carries an accelerate hook — so a pipeline parked on `meta` with no hooks
    # (a derive shell, a half-armed rung) would recurse until the stack blew.
    # The guard is thread-local because two loads may run concurrently and a
    # flag on the pipeline would have to survive diffusers' own `__setattr__`,
    # which treats attribute names as component registrations.
    reentry = threading.local()

    def device(self: Any) -> Any:
        got = original_getter(self)
        # pgw#1497. The SAME repair, for the rung one line below `meta` in
        # severity. `partial_stream` parks leaves on the host, so the original
        # getter — which answers with the first parameter it finds — reports
        # `cpu` for a pipeline that executes on the card. The pipeline then
        # builds `input_ids` on the host and the first embedding dies with
        # `index is on cpu, different from other tensors on cuda:0`. MEASURED
        # on the 4070: sd1.5 at a 5% budget and SDXL armed from the host, both
        # of them at exactly the budgets the rung exists for. This is the rung
        # breaking a public answer, so this is where it is repaired.
        armed = getattr(self, STREAM_RESIDENCY_ATTR, None)
        armed_device = getattr(armed, "device", None) if armed is not None else None
        if armed_device is not None and getattr(armed, "plan", None) is not None:
            return armed_device
        # pgw#1577. The SAME repair for `partial_resident`, which breaks the
        # same public answer for the same reason: an evicted text encoder is
        # parked on the host and the original getter answers with whichever
        # component it reaches first. The rung records the device its resident
        # set actually executes on, and that is the honest answer.
        resident_device = getattr(self, PARTIAL_RESIDENT_DEVICE_ATTR, None)
        if resident_device is not None:
            return resident_device
        if getattr(got, "type", None) != "meta":
            return got
        if getattr(reentry, "active", False):
            return got
        reentry.active = True
        try:
            # diffusers' own accelerate-aware answer to the same question:
            # where the next forward will actually run.
            resolved = self._execution_device
        except Exception:  # pragma: no cover - upstream shape changed
            return got
        finally:
            reentry.active = False
        # No hook to read: `meta` really is the honest answer, and inventing a
        # device here would send tensors somewhere the weights are not.
        return got if getattr(resolved, "type", None) == "meta" else resolved

    DiffusionPipeline.device = property(device)  # type: ignore[method-assign,assignment]
    _execution_device_fallback_installed = True
    _LOG.info(
        "low_vram: `pipeline.device` now falls back to the execution device "
        "when an offload rung parks modules on `meta` (pgw#1486)"
    )
    return True


def apply_low_vram_config(
    pipeline: Any,
    *,
    mode: Mode = "auto",
    logger: Optional[logging.Logger] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
    offload_to_disk_path: Optional[str] = None,
    stream_budget_bytes: int = 0,
    stream_ram_budget_bytes: int = 0,
) -> Dict[str, Any]:
    """Apply a low-VRAM configuration to a diffusers pipeline.

    ``mode="auto"`` runs :func:`select_auto_mode` against free VRAM. Returns a
    dict describing what was applied. Idempotent per pipeline object.

    ``stream_budget_bytes`` / ``stream_ram_budget_bytes`` are pgw#1497's
    ADMISSION PAIR — the {VRAM, RAM} profile this model was assigned (Paul,
    2026-08-19). The RAM half is REPORTED, not yet enforced (see
    :class:`~gen_worker.models.stream_residency.MemoryBudget`); the signature
    carries the pair now so enforcement is a change of behaviour, not shape.
    ``stream_budget_bytes`` is the VRAM half and is the ADMISSION number: the device bytes
    this pipeline's weights were leased. Passing it upgrades any offload rung
    ``auto`` would otherwise have chosen to ``partial_stream``, which moves
    only the bytes that did not fit instead of a whole component. Omitting it
    is not a smaller version of that — the rung REFUSES without it, because a
    budget nobody handed down is exactly the activation estimate this port
    exists to avoid.
    """
    log = logger or _LOG
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid low-VRAM mode: {mode!r}; expected one of {_VALID_MODES}")
    if mode == "partial_stream" and int(stream_budget_bytes) <= 0:
        raise ValueError(
            "partial_stream needs an explicit stream_budget_bytes: its resident "
            "set is sized by the RESIDENCY LEASE, never by an activation "
            "estimate (pgw#1497). A caller with no lease in hand wants "
            "model_offload."
        )

    # pgw#1499: arm the reactive in-rung ladders on EVERY rung, including the
    # resident ones. The rung answers "where do the weights live"; these answer
    # "this one op did not fit" — and the rung that most needs the second
    # answer is `off`, where nothing was pre-tiled because everything fitted.
    # Imported here, not at module scope: `oom_ladder` reads this module.
    from . import oom_ladder

    oom_ladder.install(pipeline, logger=log)

    prior = getattr(pipeline, _COZY_MODE_ATTR, None)
    if prior is not None:
        return {"mode": prior, "already_applied": True}

    # pgw#1586, CLOSING THE CLASS pgw#1595 OPENED. Free VRAM read ONCE here,
    # before anything is placed, and carried to every confession below.
    #
    # pgw#1595's fix threaded the plan-time figure into the `partial_resident`
    # confession ONLY, and left six sibling call sites — `model_offload`,
    # `sequential`, `partial_stream`, both `cpu` arms and the fall-through —
    # still re-reading free VRAM AT REPORT TIME, after placement. Within hours
    # the pgw#1548 lane read `free_gb=0.4` off a `model_offload` line on a card
    # with 7.9 GiB free at boot and reached for a boot-ordering cause, which is
    # the SAME wrong conclusion pgw#1595 was filed on. Fixing the one line that
    # had bitten me and leaving its five siblings was the defect; this is the
    # class.
    decision_free_gb = get_available_vram_gb()

    # pgw#1498's tier dial, BEFORE the rung is chosen: spending the lease's
    # surplus on decode-once weights changes the footprint every decision below
    # is made against. A no-op on any pipeline holding no ggml block bytes.
    gguf_dial: Dict[str, Any] = {}
    try:
        _apply_gguf_dequant_ahead(
            pipeline, gguf_dial,
            budget=stream_budget_bytes,
            log=log,
        )
    except Exception as exc:  # noqa: BLE001
        # A dial that cannot turn must never stop a load: the constrained tier
        # (decode every forward) is the correct, serving answer.
        log.warning(
            "low_vram: the gguf dequant-ahead dial could not turn (%s: %s); "
            "serving quantized-resident, decoding per forward",
            type(exc).__name__, exc)

    effective_mode = mode
    if effective_mode == "auto":
        effective_mode = select_auto_mode(
            pipeline=pipeline, model_size_gb=model_size_gb, peak_vram_gb=peak_vram_gb,
        )
        log.info("low_vram: auto-selected mode=%s", effective_mode)
    if effective_mode in ("group_offload", "sequential") and getattr(
        pipeline, "_cozy_gguf_quant", None
    ):
        log.warning(
            "low_vram: %s is unsupported for a GGUF pipeline; using "
            "model_offload",
            effective_mode,
        )
        effective_mode = "model_offload"
    if mode == "auto":
        effective_mode = _gguf_resident_override(pipeline, effective_mode, log)
    # pgw#1497. `select_auto_mode` cannot pick this rung — it is a proactive
    # decider and has no lease to read — so the UPGRADE happens here, where the
    # caller's lease number is in scope. Any offload rung the free-VRAM walk
    # chose becomes the fine-grained one, because the coarse rung's whole
    # disadvantage is that it moves a component when it needed to move a few
    # leaves. The budget is the SMALLER of the lease and what the card actually
    # has free: both are facts, and admitting more than the card holds would
    # OOM under a lease that was written when the card was emptier.
    if effective_mode in ("model_offload", "group_offload", "sequential") and int(
        stream_budget_bytes
    ) > 0:
        headroom = int(
            max(0.0, get_available_vram_gb() - _DEFAULT_SAFETY_MARGIN_GB) * _GIB
        )
        budget = min(int(stream_budget_bytes), headroom) if headroom else 0
        if budget > 0:
            log.info(
                "low_vram: upgrading %s -> partial_stream under a %.2f GiB "
                "budget (lease %.2f GiB, free-VRAM headroom %.2f GiB)",
                effective_mode, budget / _GIB,
                int(stream_budget_bytes) / _GIB, headroom / _GIB,
            )
            effective_mode = "partial_stream"
            stream_budget_bytes = budget
        else:
            log.info(
                "low_vram: keeping %s — the card has no free headroom to hold "
                "a partial_stream resident set (lease %.2f GiB)",
                effective_mode, int(stream_budget_bytes) / _GIB,
            )

    # pgw#1577. `model_offload` evicts EVERY component after EVERY request and
    # re-onloads the lot before the next one. Measured on the campaign card,
    # SDXL: 13 GiB of PCIe per request — 6.5 GiB out, 6.5 GiB back — to reclaim
    # the 1.2 GiB the pipeline was over budget by. Nothing about the placement
    # decision required that; the rung is simply all-or-nothing. So before
    # taking it, ask whether a SUBSET of components clears the budget, and keep
    # the denoiser on the card if one does. Paul, 2026-08-20: *"if the space is
    # available, and it helps us run faster, why wouldn't varena take it?"*
    #
    # ADMISSION-FIRST, and that is the whole safety argument (pgw#1560): the
    # plan is computed ONCE, here, from free VRAM and measured component sizes,
    # and the resident set never changes again. There is no eviction loop for a
    # non-raising allocator to thrash inside, and no except-OOM retry — an OOM
    # in a compiled graph is process death, so before the weights land is the
    # only honest place to decide.
    partial_resident_plan = None
    if effective_mode == "model_offload":
        partial_resident_plan = _plan_partial_resident(
            pipeline, log, peak_vram_gb=peak_vram_gb, model_size_gb=model_size_gb,
        )
        if partial_resident_plan is not None:
            effective_mode = "partial_resident"

    applied: Dict[str, Any] = {
        "mode": effective_mode,
        "vae_slicing": False,
        "vae_tiling": False,
        "attention_slicing": False,
        "partial_stream": False,
        "partial_resident": False,
        "model_offload": False,
        "group_offload": False,
        "sequential_offload": False,
        "disk_offload_path": False,
        "already_applied": False,
    }
    applied.update(gguf_dial)

    if effective_mode == "off":
        setattr(pipeline, _COZY_MODE_ATTR, "off")
        return applied

    if effective_mode == "cpu":
        # THE BOTTOM RUNG, AND IT RUNS (pgw#1315). No hook is armed: every
        # offload rung onloads to a device, and this rung's whole premise is
        # that there is no usable one — either the pod is cardless or every
        # rung above OOM'd. The savers still apply, because host RAM is now the
        # constraint and tiled decode is what keeps a large VAE inside it.
        _apply_vae_and_attention(pipeline, applied, no_reactive_ladder=True)
        _to_host(pipeline)
        flush_memory()
        setattr(pipeline, _COZY_MODE_ATTR, "cpu")
        # pgw#1312's one confession home. This rung is the LOUDEST degradation
        # the ladder has — ~40x — so it may not be the one route that reaches a
        # CPU-touching placement without saying so off the pod.
        _report_offload_engaged(pipeline, "cpu", applied, log,
                                plan_free_gb=decision_free_gb)
        return applied

    # Every rung reached from here has a live `oom_ladder` (armed above, on
    # every rung including the resident ones), so none of them arms the
    # activation savers proactively — see `_apply_vae_and_attention`. Only the
    # `cpu` rung, which returned earlier, has no CUDA OOM to react to.
    _apply_vae_and_attention(pipeline, applied)

    if effective_mode == "vae_only":
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: vae_only applied (%s)", _applied_summary(applied))
        return applied

    # Every rung from here down parks modules off the execution device, and
    # some park them on `meta`. Repair `pipeline.device` BEFORE any of them
    # arms, so no endpoint ever observes the meta answer (pgw#1486).
    install_execution_device_fallback()

    cuda_ok = cuda_ready()

    if not cuda_ok:
        # An offload rung was asked for on a host with no usable device. Every
        # such rung onloads to one, so none of them can be armed — but the
        # honest rung is the BOTTOM one, not `vae_only`. pgw#1315: stamping a
        # resident flavor here described a placement that was not taken, and
        # the ladder then read the pipeline as sitting two rungs above where it
        # actually was.
        log.warning(
            "low_vram: %s was requested and no CUDA device is usable; the CPU "
            "rung serves instead (~%.0fx a native run).",
            effective_mode, _rung_price(RUN_CPU),
        )
        applied["mode"] = "cpu"
        _to_host(pipeline)
        setattr(pipeline, _COZY_MODE_ATTR, "cpu")
        _report_offload_engaged(pipeline, "cpu", applied, log,
                                plan_free_gb=decision_free_gb)
        return applied

    if offload_to_disk_path is None and _should_auto_disk_offload():
        offload_to_disk_path = _default_disk_offload_path()
        if offload_to_disk_path:
            log.warning(
                "low_vram: CPU RAM tight (%.1f GB free); enabling disk offload at %s",
                get_available_ram_gb(), offload_to_disk_path,
            )

    if effective_mode == "partial_stream":
        from .stream_residency import MemoryBudget as _MemoryBudget

        if _apply_partial_stream(
            pipeline,
            applied,
            budget_bytes=_MemoryBudget(
                vram_bytes=int(stream_budget_bytes),
                ram_bytes=max(0, int(stream_ram_budget_bytes)),
            ),
            log=log,
        ):
            setattr(pipeline, _COZY_MODE_ATTR, "partial_stream")
            if applied.get("stream_streamed_leaves"):
                _report_offload_engaged(pipeline, "partial_stream", applied, log,
                                        plan_free_gb=decision_free_gb)
            return applied
        # It could not arm (no torch, no hookable tree, a meta/aliased leaf).
        # The next rung down is the honest answer, not a resident placement.
        log.warning(
            "low_vram: partial_stream did not arm; descending to model_offload"
        )
        applied["partial_stream"] = False
        effective_mode = "model_offload"
        applied["mode"] = effective_mode

    if effective_mode == "partial_resident" and partial_resident_plan is not None:
        from .partial_resident import (
            _MAX_PROBE_ATTEMPTS,
            PARTIAL_RESIDENT_UNARMED_PHASE,
            apply_component_residency,
        )

        # THE PROBE LOOP, and it is why this rung is admission-first WITHOUT
        # being estimate-first (pgw#1577). The planner's transient ceiling is
        # arithmetic over the two numbers it can read — component sizes and free
        # VRAM — and on the campaign card that arithmetic admitted a plan whose
        # onload then died 5 MiB short, because allocator fragmentation and a
        # co-tenant's share are in neither. So each plan is DONE once before it
        # is trusted, and a plan the card refuses is followed by the
        # next-cheapest one rather than by giving up on the rung. Bounded, and
        # every attempt is at load: no OOM can reach a request from here.
        plan = partial_resident_plan
        armed = False
        probe_facts: Dict[str, Any] = {}
        for _ in range(_MAX_PROBE_ATTEMPTS):
            armed = apply_component_residency(
                pipeline, plan, device=_execution_device(), log=log,
                free_bytes_now=lambda: int(get_available_vram_gb() * _GIB),
                facts=probe_facts,
            )
            if armed:
                break
            nxt = _plan_partial_resident(
                pipeline, log, min_moved_bytes=plan.offloaded_bytes,
                peak_vram_gb=peak_vram_gb, model_size_gb=model_size_gb,
            )
            if nxt is None:
                break
            plan = nxt
        if armed:
            applied["partial_resident"] = True
            applied["partial_resident_offloaded"] = list(plan.offloaded)
            applied["partial_resident_bytes"] = plan.offloaded_bytes
            # The decision's own arithmetic, on the LOUD line. pgw#1595: these
            # lived at INFO and vanished at the endpoint's WARNING level, so the
            # only surviving diagnostic was the misleading one.
            applied["plan_budget_gb"] = plan.budget_bytes / _GIB
            applied["reserve_gb"] = float(_LAST_RESERVE.get("reserve_gb", 0.0))
            applied["reserve_source"] = str(
                _LAST_RESERVE.get("reserve_source", "default"))
            applied["plan_peak_gb"] = plan.transient_peak_bytes / _GIB
            for _k, _lbl in (("attr_alloc_bytes", "attr_alloc_gb"),
                             ("attr_cache_bytes", "attr_cache_gb"),
                             ("attr_ctx_bytes", "attr_ctx_gb")):
                if _k in probe_facts:
                    applied[_lbl] = float(probe_facts[_k]) / _GIB
            probe_free = probe_facts.get("probe_free_bytes")
            if probe_free is not None:
                # Probe SUCCESS was `log.info` and therefore inaudible, which
                # makes "probe passed" and "probe never ran" the same picture
                # (pgw#1559 class). It is a number now, on the loud line.
                applied["probe_free_gb"] = float(probe_free) / _GIB
            setattr(pipeline, _COZY_MODE_ATTR, "partial_resident")
            _report_offload_engaged(
                pipeline, "partial_resident", applied, log,
                plan_free_gb=plan.free_bytes / _GIB,
            )
            return applied
        # Same discipline as `partial_stream`: a rung the operator was told
        # about and did not get is a placement lie. Confess and descend.
        log.warning(
            "low_vram: partial_resident did not arm; descending to "
            "model_offload (%s)", PARTIAL_RESIDENT_UNARMED_PHASE,
        )
        effective_mode = "model_offload"
        applied["mode"] = effective_mode

    if effective_mode == "model_offload":
        _pin_unhookable_components(pipeline, applied, log)
        ok = _call_if_present(pipeline, "enable_model_cpu_offload")
        if not ok:
            try:
                pipeline.enable_model_cpu_offload(gpu_id=0)
                ok = True
            except Exception as exc:
                log.warning("low_vram: enable_model_cpu_offload failed: %s", exc)
        applied["model_offload"] = ok
        setattr(pipeline, _COZY_MODE_ATTR, "model_offload")
        _report_offload_engaged(pipeline, "model_offload", applied, log,
                                plan_free_gb=decision_free_gb)
        return applied

    if effective_mode == "group_offload":
        ok = _apply_group_offload(pipeline, applied, offload_to_disk_path=offload_to_disk_path)
        if not ok:
            log.warning("low_vram: group_offload unavailable; falling back to sequential")
            effective_mode = "sequential"

    if effective_mode == "sequential":
        _pin_unhookable_components(pipeline, applied, log)
        _move_pipeline_to_cpu(pipeline)
        flush_memory()
        ok = _call_if_present(pipeline, "enable_sequential_cpu_offload")
        if not ok:
            try:
                pipeline.enable_sequential_cpu_offload(gpu_id=0)
                ok = True
            except Exception as exc:
                log.error("low_vram: enable_sequential_cpu_offload failed: %s", exc)
        applied["sequential_offload"] = ok
        applied["mode"] = "sequential"
        setattr(pipeline, _COZY_MODE_ATTR, "sequential")
        _report_offload_engaged(pipeline, "sequential", applied, log,
                                plan_free_gb=decision_free_gb)
        return applied

    setattr(pipeline, _COZY_MODE_ATTR, effective_mode)
    if touches_host_ram(effective_mode):
        _report_offload_engaged(pipeline, effective_mode, applied, log,
                                plan_free_gb=decision_free_gb)
    else:
        log.info("low_vram: %s applied (%s)", effective_mode, _applied_summary(applied))
    return applied


def _applied_summary(applied: Dict[str, Any]) -> str:
    """The engaged savers, and the numbers beside them — TYPED APART.

    pgw#1586 item 3: this used to print the KEY of anything truthy, so pgw#1577's
    two DATA entries rendered as if they were savers that engaged
    (``vae_slicing,partial_resident,partial_resident_offloaded,partial_resident_bytes``
    — four names for two techniques). The distinction is by TYPE, not by an
    allowlist, so the next data entry anyone adds cannot masquerade either.
    """
    names: List[str] = []
    values: List[str] = []
    for k, v in applied.items():
        if k in ("mode", "already_applied") or not v:
            continue
        if isinstance(v, bool):
            names.append(k)
        elif isinstance(v, float):
            values.append(f"{k}={v:.2f}")
        elif isinstance(v, (list, tuple)):
            values.append(f"{k}={'+'.join(str(x) for x in v)}")
        else:
            values.append(f"{k}={v}")
    return ",".join(names + values) or "none"


def _should_auto_disk_offload() -> bool:
    ram = get_available_ram_gb()
    return 0.0 < ram < 16.0


def _default_disk_offload_path() -> Optional[str]:
    try:
        p = "/tmp/cozy-offload"
        os.makedirs(p, exist_ok=True)
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OOM retry
# ---------------------------------------------------------------------------


def rearm_offload(pipeline: Any, mode: Mode = "model_offload") -> bool:
    """Serve-time offload fallback (gw#551): arm an offload rung on a
    pipeline that was already configured once (clears the idempotency stamp).
    Offload hooks must start from a coherent CPU object; the caller's failed
    promote already rolled the pipeline back to cpu — this re-verifies."""
    _move_pipeline_to_cpu(pipeline)
    if repair_device_placement(pipeline, "cpu"):
        return False
    flush_memory()
    try:
        delattr(pipeline, _COZY_MODE_ATTR)
    except AttributeError:
        pass
    applied = apply_low_vram_config(pipeline, mode=mode)
    return bool(
        applied.get("model_offload")
        or applied.get("group_offload")
        or applied.get("sequential_offload")
    )


def low_vram_mode(pipeline: Any) -> str:
    """The low-VRAM mode :func:`apply_low_vram_config` prepped this pipeline
    with ('' when never prepped). Part of the compile-cache graph key (gw#391):
    the flags are traced into the FX graphs."""
    return str(getattr(pipeline, _COZY_MODE_ATTR, "") or "")


_RESIDENT_MODES = ("off", "vae_only")


__all__ = [
    "apply_low_vram_config",
    "install_execution_device_fallback",
    "low_vram_mode",
    "rearm_offload",
    "place_pipeline",
    "touches_host_ram",
    "is_cuda_oom",
    "discard_cuda_async_error",
    "transition_line",
    "PLACEMENT_LADDER",
    "select_auto_mode",
    "device_mismatches",
    "repair_device_placement",
    "estimate_pipeline_size_gb",
    "estimate_cuda_resident_gb",
    "cuda_allocated_bytes",
    "get_available_vram_gb",
    "GPU_VRAM_OVERHEAD_GB",
    "get_available_ram_gb",
    "effective_ram_floor_gb",
    "get_total_ram_gb",
    "aflush_memory",
    "flush_memory",
    "release_cached_vram",
    "release_unused_pinned_host_cache",
    # pgw#1558 — the endpoint-facing mechanism surface.
    "available_vram",
    "process_ceiling_vram",
    "module_storage_bytes",
    "resident_census",
    "tensor_dtype_label",
    "tensor_storage_bytes",
    "VramReading",
    "VRAM_NO_CUDA",
    "VRAM_UNREADABLE",
]
