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

Upstream foot-gun: ``enable_sequential_cpu_offload`` must NOT be called on a
pipeline already moved to CUDA; ``apply_low_vram_config`` moves it back first.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import msgspec

from ..component_vocab import component_vocabulary
from .structure_only import STAMP as _STRUCTURE_ONLY
import asyncio

_LOG = logging.getLogger(__name__)

_GIB = 1024 ** 3

Mode = str  # "auto" | "off" | "vae_only" | "model_offload" | "group_offload" | "sequential"

_VALID_MODES: tuple[str, ...] = (
    "auto", "off", "vae_only", "model_offload", "group_offload", "sequential",
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

# Authors declare ``Resources(vram_gb=X)`` as the TOTAL VRAM of the smallest
# card they target ("runs on a 24 GB card") — a placement recommendation, not
# measurable free bytes. The platform reserves this much for the fixed
# driver/framebuffer/CUDA-context overhead when comparing the recommendation
# against probed VRAM, so vram_gb=24 serves on a 24 GB card (~23.6 GB free).
GPU_VRAM_OVERHEAD_GB = 1.0


# th#1867 deleted `effective_vram_requirement_gb`. It translated a DECLARED
# `vram_gb` recommendation into a probed-VRAM floor, and its only caller was
# `hub_policy.variant_fit`'s size arm — so with the declaration gone it had
# nothing left to translate. `GPU_VRAM_OVERHEAD_GB` survives below because the
# other use subtracts it from a MEASURED total, which is arithmetic on a fact.

# The ladder itself lives in rung.py (pgw#1206 A2): one ordered Rung, one
# walk, one price. This module keeps the probes and the appliers.
from .rung import (
    PLACEMENT_LADDER,
    descend as _descend_rung,
    touches_host_ram,
    transition_line,
)


def is_cuda_oom(exc: Optional[BaseException]) -> bool:
    """CUDA allocator exhaustion in any of its shapes: torch.cuda.OutOfMemoryError
    (class name match — no torch import needed) plus the allocator's RuntimeError
    flavors ("CUDA error: out of memory", CUBLAS/CUDNN alloc failures)."""
    if exc is None:
        return False
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return (
            "out of memory" in text
            or "cuda oom" in text
            or "cublas_status_alloc_failed" in text
            or "cudnn_status_alloc_failed" in text
        )
    return False


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
    try:
        import torch

        if not torch.cuda.is_available():
            return VramReading(0.0, VRAM_NO_CUDA)
        free, _total = torch.cuda.mem_get_info(device_index)
    except Exception as exc:
        _LOG.warning("free-VRAM probe failed: %s: %s", type(exc).__name__, exc)
        return VramReading(0.0, VRAM_UNREADABLE)
    return VramReading(float(free) / float(1024**3))


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
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        _free, total = torch.cuda.mem_get_info(device_index)
        return float(total) / float(1024**3)
    except Exception:
        return 0.0


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


def _v2_cgroup_nodes(root: Path, proc_self_cgroup: Path) -> List[Path]:
    """Cgroup-v2 dirs from root down to this process's own cgroup."""
    rel = ""
    try:
        for line in proc_self_cgroup.read_text().splitlines():
            if line.startswith("0::"):
                rel = line[3:].strip().strip("/")
                break
    except OSError:
        pass
    nodes = [root]
    node = root
    for part in Path(rel).parts:
        node = node / part
        nodes.append(node)
    return nodes


def cgroup_memory_limit_bytes(
    root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> Optional[int]:
    """Effective cgroup memory limit for this process; None when uncapped.

    v2: tightest ``memory.max`` on the root->self chain (covers both private
    and host cgroup namespaces); v1 fallback: ``memory/memory.limit_in_bytes``.
    """
    limits = [
        v for node in _v2_cgroup_nodes(root, proc_self_cgroup)
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
    for node in reversed(_v2_cgroup_nodes(root, proc_self_cgroup)):
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
    for node in reversed(_v2_cgroup_nodes(root, proc_self_cgroup)):
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
    siblings: Optional[int] = None,
) -> HostRam:
    """One truthful host-RAM snapshot: psutil meminfo min'd with the cgroup cap.

    ``siblings`` defaults to the compute-child count this process shares its
    cgroup with (pgw#783; 1 unless the process split is running G groups).
    """
    if siblings is None:
        from ..procsplit import host_siblings

        siblings = host_siblings()
    meminfo_total = meminfo_available = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        meminfo_total = float(vm.total) / float(_GIB)
        meminfo_available = float(vm.available) / float(_GIB)
    except Exception:
        pass
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

        if torch.cuda.is_available():
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
    return device_mismatches(obj, device)


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
    import torch

    from ..meta_instantiation import is_virtual

    total = 0
    #: ``("ptr", data_ptr)`` for a tensor with storage — shared storages are
    #: counted ONCE — and ``("obj", id)`` for one without, which has no storage
    #: identity to share.
    seen: set[tuple[str, int]] = set()
    for obj in objs:
        for c in _iter_components(obj):
            if c is None or not hasattr(c, "parameters"):
                continue
            tensors = list(c.parameters())
            if hasattr(c, "buffers"):
                tensors.extend(c.buffers())
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
                total += t.numel() * t.element_size()
    return total


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


def flush_memory() -> None:
    """gc + empty_cache + reset_peak_memory_stats. Always safe to call."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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

        if not torch.cuda.is_available():
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
    traced graph class and mint object set are deterministic per SKU
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


def _move_pipeline_to_cpu(pipeline: Any) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if callable(getattr(pipeline, "to", None)):
            pipeline.to("cpu")
    except Exception as exc:
        _LOG.debug("low_vram: move-to-cpu failed: %s", exc)


def _apply_vae_and_attention(
    pipeline: Any, applied: Dict[str, bool], *, memory_bound: bool = True
) -> None:
    """VAE/attention memory savers.

    th#1107: tiling and attention slicing are VRAM tools that cost real
    latency (tiled decode re-runs the VAE per tile and blends 25% overlaps;
    attention slicing replaces the fused SDPA/flash path with a chunked
    loop). ``vae_only`` is selected when the pipeline FITS and only headroom
    is tight, so applying them there taxes every request on a card that never
    needed them. ``memory_bound=False`` (the vae_only rung) keeps only VAE
    slicing, which is a no-op at batch 1.
    """
    if not _call_if_present(pipeline, "enable_vae_slicing"):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and _call_if_present(vae, "enable_slicing"):
            applied["vae_slicing"] = True
    else:
        applied["vae_slicing"] = True

    if not memory_bound:
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


def _pin_fragile_vae(pipeline: Any, applied: Dict[str, bool], log: logging.Logger) -> None:
    """Keep a force_upcast VAE out of the diffusers CPU-offload hooks.
    ``_exclude_from_cpu_offload`` is honored by BOTH the model and sequential
    rungs, which move excluded components to the execution device
    themselves."""
    if _dtype_fragile_vae(pipeline) is None:
        return
    excl = list(getattr(pipeline, "_exclude_from_cpu_offload", None) or [])
    if "vae" not in excl:
        excl.append("vae")
    try:
        pipeline._exclude_from_cpu_offload = excl
    except Exception:
        return
    applied["vae_resident"] = True
    log.info(
        "low_vram: force_upcast vae stays resident (excluded from offload "
        "hooks — dtype-safety, gw#441/gw#469)"
    )


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
    if not torch.cuda.is_available():
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

    def _keep_vae_resident() -> None:
        if fragile_vae is None:
            return
        try:
            fragile_vae.to("cuda")
        except Exception as exc:
            _LOG.warning("low_vram: resident vae move failed: %s", exc)
        applied["vae_resident"] = True
        _LOG.info(
            "low_vram: force_upcast vae stays resident under group offload "
            "(dtype-safety, gw#441/gw#469)"
        )

    fn = getattr(pipeline, "enable_group_offload", None)
    if callable(fn):
        try:
            if fragile_vae is not None:
                fn(**kwargs, exclude_modules=["vae"])
            else:
                fn(**kwargs)
            _keep_vae_resident()
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
        if attr == "vae" and fragile_vae is not None:
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
        _keep_vae_resident()
        applied["group_offload"] = True
        if offload_to_disk_path:
            applied["disk_offload_path"] = True
    return any_applied


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
    try:
        import torch

        if not torch.cuda.is_available():
            return {"mode": "cpu"}
    except Exception:
        return {"mode": "cpu"}
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
            _move_pipeline_to_cpu(pipeline)
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


class CpuOffloadForbidden(RuntimeError):
    """A CPU-offloading placement was attempted on a host that forbids it."""


_FORBID_CPU_OFFLOAD_ENV = "GEN_WORKER_FORBID_CPU_OFFLOAD"


def cpu_offload_forbidden() -> bool:
    """Whether this host refuses every CPU-touching placement.

    pgw#929 AMBIGUOUS #1 — this makes a DOCUMENTED CONTRACT TRUE rather than
    adding a knob. The workspace `CLAUDE.md` told operators and agents that
    ``GEN_WORKER_FORBID_CPU_OFFLOAD`` *"makes gen-worker raise on any
    CPU-touching placement"*, and the box exports it for that reason. Measured
    2026-08-03: it had exactly ONE reader in the tree,
    ``benchmarks/swap_latency.py`` (deleted by pgw#883), where it refused the
    swap-latency benchmark and nothing else. The real CPU-offload ladder below
    never consulted it, so the guard operators believed they had did not
    exist — stale prose reaching people as fact (C3). This function is now its
    only reader, and it is the placement boundary.

    It is a TRIPWIRE, not configuration: it carries no behaviour of its own and
    exists only to fire on a host that must never touch weights. Same shape as
    the C2PA key-material refusal, and it is read here rather than through
    `Settings` for the same reason a tripwire is not a setting — a control-plane
    box exports it box-wide with no worker config in sight. Recorded, with that
    argument, in `scripts/config_reads_allowlist.txt`.
    """
    return os.environ.get(_FORBID_CPU_OFFLOAD_ENV, "").strip() not in ("", "0", "false", "no")


def _refuse_cpu_offload(what: str) -> None:
    if cpu_offload_forbidden():
        raise CpuOffloadForbidden(
            f"refusing {what}: {_FORBID_CPU_OFFLOAD_ENV} is set on this host. "
            f"CPU-offloading placements move weights through host RAM, which "
            f"this machine forbids (weights-locality rule). Run it on a GPU pod."
        )


def apply_low_vram_config(
    pipeline: Any,
    *,
    mode: Mode = "auto",
    logger: Optional[logging.Logger] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
    offload_to_disk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a low-VRAM configuration to a diffusers pipeline.

    ``mode="auto"`` runs :func:`select_auto_mode` against free VRAM. Returns a
    dict describing what was applied. Idempotent per pipeline object.
    """
    log = logger or _LOG
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid low-VRAM mode: {mode!r}; expected one of {_VALID_MODES}")

    prior = getattr(pipeline, _COZY_MODE_ATTR, None)
    if prior is not None:
        return {"mode": prior, "already_applied": True}

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

    applied: Dict[str, Any] = {
        "mode": effective_mode,
        "vae_slicing": False,
        "vae_tiling": False,
        "attention_slicing": False,
        "model_offload": False,
        "group_offload": False,
        "sequential_offload": False,
        "disk_offload_path": False,
        "already_applied": False,
    }

    if effective_mode == "off":
        setattr(pipeline, _COZY_MODE_ATTR, "off")
        return applied

    _apply_vae_and_attention(
        pipeline, applied, memory_bound=effective_mode != "vae_only"
    )

    if effective_mode == "vae_only":
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: vae_only applied (%s)", _applied_summary(applied))
        return applied

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: CUDA unavailable, stopping at vae_only")
        return applied

    if offload_to_disk_path is None and _should_auto_disk_offload():
        offload_to_disk_path = _default_disk_offload_path()
        if offload_to_disk_path:
            log.warning(
                "low_vram: CPU RAM tight (%.1f GB free); enabling disk offload at %s",
                get_available_ram_gb(), offload_to_disk_path,
            )

    if effective_mode == "model_offload":
        _refuse_cpu_offload("enable_model_cpu_offload")
        _pin_fragile_vae(pipeline, applied, log)
        ok = _call_if_present(pipeline, "enable_model_cpu_offload")
        if not ok:
            try:
                pipeline.enable_model_cpu_offload(gpu_id=0)
                ok = True
            except Exception as exc:
                log.warning("low_vram: enable_model_cpu_offload failed: %s", exc)
        applied["model_offload"] = ok
        setattr(pipeline, _COZY_MODE_ATTR, "model_offload")
        log.info("low_vram: model_offload applied (%s)", _applied_summary(applied))
        return applied

    if effective_mode == "group_offload":
        ok = _apply_group_offload(pipeline, applied, offload_to_disk_path=offload_to_disk_path)
        if not ok:
            log.warning("low_vram: group_offload unavailable; falling back to sequential")
            effective_mode = "sequential"

    if effective_mode == "sequential":
        _refuse_cpu_offload("enable_sequential_cpu_offload")
        _pin_fragile_vae(pipeline, applied, log)
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
        log.info("low_vram: sequential_offload applied (%s)", _applied_summary(applied))
        return applied

    setattr(pipeline, _COZY_MODE_ATTR, effective_mode)
    log.info("low_vram: %s applied (%s)", effective_mode, _applied_summary(applied))
    return applied


def _applied_summary(applied: Dict[str, Any]) -> str:
    keys = [k for k, v in applied.items() if v and k not in ("mode", "already_applied")]
    return ",".join(keys) or "none"


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
    "low_vram_mode",
    "rearm_offload",
    "place_pipeline",
    "touches_host_ram",
    "is_cuda_oom",
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
    "release_unused_pinned_host_cache",
]
