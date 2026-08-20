"""ONE home for every primitive fact this worker observes about its own host.

pgw#896/pgw#897. The rule is not "fewer lines": it is that a **second answer
to the same question is unrepresentable**. Before this module the platform
answered "what does this host have" in many places at once — nine free-VRAM
formulas (ten ``mem_get_info`` call sites), 74 raw ``torch.cuda.is_available()``
predicates, two cgroup CPU-quota readers with different kernel coverage, four
independent ``min()`` reductions over the same three candidates, and four
parsers of ``/proc/meminfo`` — and the sites disagreed in ways only a real pod
could show.

What lives here, and what does NOT:

* **Primitive observations only.** A named formula reads the host and returns
  a number (or ``None`` for "no reading"). It never decides. Admission,
  placement, pool sizing and refusals stay with their owners, which read from
  here.
* **Four named VRAM formulas, not nine** — :func:`free_vram_bytes` (what the
  driver says is free right now), :func:`total_vram_bytes` (the card's
  nameplate as measured), :func:`headroom_bytes` (driver-free plus the
  allocator cache THIS process can hand back) and
  :func:`process_ceiling_bytes` (driver-free plus what this process already
  holds — the ceiling a whole working set fits under). They are different
  quantities with different right answers, so they are four names rather than
  one ambiguous "free". The fourth was added by pgw#1558, not invented there:
  an endpoint was already computing it, off in its own repo, with its own
  ``mem_get_info`` call.
* **Never aggregate across cards here.** VRAM is not fungible between cards; a
  producer that sums or maxes hands its consumer a number no single job can
  have. Aggregation is a consumer's explicit, argued act.

``import torch`` is deliberately lazy in every function: the control parent
must stay torch-free, and torch-free contexts (tools, tests, the CPU lane)
import this module freely.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Is there a usable CUDA device?
# ---------------------------------------------------------------------------

#: A card this process can use. The accelerator vocabulary is `cuda`/`none`
#: only — "cpu" is an oxymoron and is not a state of this axis.
DEVICE_PRESENT = "cuda"
#: This host genuinely has no card. A CPU-lane pod, a laptop, a CI runner.
DEVICE_ABSENT = "none"
#: There is a card (or a driver that owns one) and it would not answer. The
#: state the weak predicate cannot express: `torch.cuda.is_available()` is
#: False for BOTH of the two above, so every site that read it alone reported
#: a wedged H100 as a machine with no GPU — and zeros are what the fleet
#: places on.
DEVICE_UNREADABLE = "unreadable"


class CudaState(msgspec.Struct, frozen=True):
    """Which of the three CUDA states this host is in, and the evidence."""

    state: str
    #: `cuda_probe.classify_probe_failure` vocabulary; "" when present.
    probe_class: str = ""
    detail: str = ""

    @property
    def present(self) -> bool:
        return self.state == DEVICE_PRESENT

    @property
    def absent(self) -> bool:
        return self.state == DEVICE_ABSENT

    @property
    def unreadable(self) -> bool:
        return self.state == DEVICE_UNREADABLE


_cuda_state: Optional[CudaState] = None


def cuda_ready() -> bool:
    """Does torch have a usable CUDA device right now?

    The ONLY ``torch.cuda.is_available()`` call site in ``src/`` — fenced by
    ``tests/test_hostfacts_pgw896.py``. Deliberately uncached and deliberately
    unchanged in value from the raw predicate: it tracks the live device and
    every caller that had one keeps its exact behaviour.

    It is a CAPABILITY question ("may I take the GPU branch?"), for which
    degrading on a card that will not answer is correct. A caller that must
    tell "this host has no card" from "this host's card would not answer" —
    anything REPORTING to the fleet, anything condemning a SKU — calls
    :func:`cuda_state` instead, because this predicate cannot express it.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return False


def cuda_state(*, device_index: int = 0) -> CudaState:
    """The three-valued CUDA verdict, measured once per process.

    Built on ``cuda_probe.probe_cuda`` (allocate, op, synchronize, free) —
    the strong answer — and on the settled ``CARDLESS_PROBE_CLASSES``
    discriminator: only a host with a driver TO fail can answer
    ``driver_too_old``/``cuda_error``, so a broken diagnostic cannot buy an
    absent card's exemption (pgw#1120).

    Cached because it allocates. Nothing on a hot path needs it.
    """
    global _cuda_state
    if _cuda_state is not None:
        return _cuda_state
    from .cuda_probe import CARDLESS_PROBE_CLASSES, classify_probe_failure, probe_cuda

    result = probe_cuda(device_index)
    if result.ok:
        _cuda_state = CudaState(DEVICE_PRESENT)
        return _cuda_state
    klass = classify_probe_failure(result.reason)
    _cuda_state = CudaState(
        DEVICE_ABSENT if klass in CARDLESS_PROBE_CLASSES else DEVICE_UNREADABLE,
        probe_class=klass,
        detail=result.reason,
    )
    return _cuda_state


def reset_cuda_state() -> None:
    """Forget the cached verdict (tests; a re-probe after a device reset)."""
    global _cuda_state
    _cuda_state = None


# ---------------------------------------------------------------------------
# VRAM — three named formulas
# ---------------------------------------------------------------------------


def _mem_get_info(device: Optional[int]) -> Optional[Tuple[int, int]]:
    """``(free, total)`` driver bytes for one card, or None for no reading.

    The ONLY ``torch.cuda.mem_get_info`` call site in ``src/`` — fenced by
    ``tests/test_hostfacts_pgw896.py``. ``None`` and ``0`` are different
    answers here and stay different all the way out: a caller that reads an
    unreadable card as "0 bytes free" refuses work a healthy card would take,
    and one that reads it as "plenty" OOMs.
    """
    if not cuda_ready():
        return None
    try:
        import torch

        free, total = torch.cuda.mem_get_info(
            torch.cuda.current_device() if device is None else int(device)
        )
        return int(free), int(total)
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info failed: %s: %s", type(exc).__name__, exc)
        return None


def free_vram_bytes(device: Optional[int] = None) -> Optional[int]:
    """Driver-free bytes on ONE card right now. ``None`` = no reading.

    The instantaneous reading. It excludes the allocator cache this process
    holds, so it is the right input for "what would a NEW process get" and the
    wrong one for "may I allocate here" — that is :func:`headroom_bytes`.
    """
    reading = _mem_get_info(device)
    return None if reading is None else reading[0]


def total_vram_bytes(device: Optional[int] = None) -> Optional[int]:
    """Total bytes on ONE card, honestly measured (an H100-80GB reports
    79.19 GiB — declare that, not the marketing number). ``None`` = no
    reading."""
    reading = _mem_get_info(device)
    return None if reading is None else reading[1]


def headroom_bytes(device: Optional[int] = None) -> Optional[int]:
    """Bytes a new allocation on ONE card could actually get. ``None`` = no
    reading.

    Driver-free PLUS the allocator cache this process can return: cached
    blocks are headroom, and counting only ``free`` refuses a card that is
    holding a large idle cache.
    """
    reading = _mem_get_info(device)
    if reading is None:
        return None
    free, _total = reading
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
        reclaimable = max(
            0,
            int(torch.cuda.memory_reserved(index))
            - int(torch.cuda.memory_allocated(index)),
        )
    except Exception:  # noqa: BLE001
        reclaimable = 0
    return int(free) + reclaimable


def process_ceiling_bytes(device: Optional[int] = None) -> Optional[int]:
    """Bytes THIS process may occupy on ONE card at its peak. ``None`` = no
    reading.

    Driver-free PLUS what this process has ALREADY allocated — the ceiling a
    whole working set (resident weights + the activations they will produce)
    has to fit under. :func:`headroom_bytes` answers the different question
    "may I allocate this NEXT block", and deliberately excludes the process's
    own weights because they are not available for the next block; a fit
    decision about the whole set must include them, or it charges its own
    resident weights against itself twice.

    pgw#1558: this is the fourth named formula, and it exists because it was
    already being computed outside this module. ``minimax-h3`` carried
    ``free + allocated`` as ``_driver_usable_gib`` with its own raw
    ``mem_get_info`` call, which is exactly the drift pgw#896 abolished — the
    endpoint could not tell "no card" from "card would not answer" and named
    every zero UNREADABLE. Here the two stay different: ``None``.
    """
    reading = _mem_get_info(device)
    if reading is None:
        return None
    free, _total = reading
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
        allocated = int(torch.cuda.memory_allocated(index))
    except Exception:  # noqa: BLE001
        allocated = 0
    return int(free) + allocated


def device_count() -> int:
    """Cards this process can see. 0 when there is no CUDA."""
    if not cuda_ready():
        return 0
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def device_identity(device: Optional[int] = None) -> Tuple[str, str]:
    """``(name, sm)`` for one card — ``("", "")`` off-GPU."""
    if not cuda_ready():
        return "", ""
    name = sm = ""
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
    except Exception:  # noqa: BLE001
        return "", ""
    # Asked independently: a driver that will not answer one of them must not
    # erase the other, and the two feed different fleet decisions.
    try:
        name = str(torch.cuda.get_device_properties(index).name)
    except Exception:  # noqa: BLE001
        pass
    try:
        major, minor = torch.cuda.get_device_capability(index)
        sm = f"sm_{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return name, sm


# ---------------------------------------------------------------------------
# CPU — one quota reader, one reduction, one rounding rule
# ---------------------------------------------------------------------------

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
#: The one path any host-RAM reading is taken from.
PROC_MEMINFO = Path("/proc/meminfo")


def cgroup_nodes(
    root: Optional[Path] = None, proc_self_cgroup: Optional[Path] = None
) -> List[Path]:
    """Cgroup-v2 dirs from ``root`` down to this process's own cgroup."""
    root = _CGROUP_ROOT if root is None else root
    proc_self_cgroup = (
        _PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup)
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


def _read_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _deepest(
    name: str, root: Optional[Path], proc_self_cgroup: Optional[Path]
) -> Optional[Path]:
    for node in reversed(cgroup_nodes(root, proc_self_cgroup)):
        candidate = node / name
        if candidate.exists():
            return candidate
    return None


def cpu_quota(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> Optional[float]:
    """Cores this cgroup may use, or ``None`` when uncapped/unreadable.

    One reader with BOTH properties the two it replaces had one each of: it
    walks the cgroup node chain to the DEEPEST ``cpu.max`` (a fixed
    ``/sys/fs/cgroup/cpu.max`` reads ``max`` in a nested cgroup that really is
    capped) AND it falls back to v1's ``cpu.cfs_quota_us``/``cfs_period_us``
    (some fleets mount only the v1 controller).

    ``os.cpu_count()`` reports the HOST's cores — 32 on a pod that owns 4 —
    so anything sizing work by core count reads this, never that.
    """
    raw = _read_text(_deepest("cpu.max", root, proc_self_cgroup))
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
        if parts:
            return None  # an explicit "max" is uncapped, not unreadable
    try:
        v1 = (_CGROUP_ROOT if root is None else root) / "cpu"
        quota = int((v1 / "cpu.cfs_quota_us").read_text().strip())
        period = int((v1 / "cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_quota_raw(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> Optional[str]:
    """The verbatim ``cpu.max`` line, for a postmortem to echo. Same file, same
    reader as :func:`cpu_quota` — a diagnostic that opens the file itself is a
    second reader that can disagree with the one the pod acts on."""
    return _read_text(_deepest("cpu.max", root, proc_self_cgroup))


class CpuAllowance(msgspec.Struct, frozen=True):
    """The narrowest true bound on this process's CPU, and which one bound."""

    #: The fractional allowance. NOTHING rounds this.
    cores: float
    #: "quota" | "affinity" | "cpu_count"
    basis: str
    os_count: int
    affinity: int
    #: The cgroup quota, or -1.0 when uncapped/unreadable.
    quota_cores: float

    @property
    def whole_cores(self) -> int:
        """The integer allowance: ``floor``, never ``int(x + 0.5)``.

        Rounding a CPU quota UP is always wrong — it sizes a thread pool above
        the quota and the kernel throttles it. A 2.5-core pod gets 2.
        """
        return max(1, int(math.floor(self.cores)))


def cpu_allowance(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> CpuAllowance:
    """The ONE ``min()`` over cgroup quota, affinity mask and host core count.

    There were FOUR of these. Two rounded the quota half-up, one kept the
    fraction, and ``boot_key`` floored it while ignoring the affinity mask
    entirely: on a ``cpu.max = 250000 100000`` pod the fleet planned against 3
    while torch ran 2.5 cores worth of threads.
    """
    os_count = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os_count
    quota = cpu_quota(root=root, proc_self_cgroup=proc_self_cgroup)
    candidates: List[Tuple[float, str]] = [
        (float(os_count), "cpu_count"),
        (float(affinity), "affinity"),
    ]
    if quota is not None and quota > 0:
        candidates.append((float(quota), "quota"))
    cores, basis = min(candidates)
    return CpuAllowance(
        cores=max(0.0, cores),
        basis=basis,
        os_count=os_count,
        affinity=affinity,
        quota_cores=float(quota) if quota is not None else -1.0,
    )


# ---------------------------------------------------------------------------
# Host RAM
# ---------------------------------------------------------------------------


def meminfo_kb(path: Optional[Path] = None) -> Dict[str, int]:
    """``/proc/meminfo`` as ``{key: kB}`` — one parser, in kB as the file is.

    Callers that want an effective, cgroup-capped RAM view call
    ``models.memory.probe_host_ram`` instead; this is the raw file, for
    postmortems and for the OOM-rank denominator.
    """
    out: Dict[str, int] = {}
    try:
        raw = (PROC_MEMINFO if path is None else path).read_text()
    except OSError:
        return out
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith(":"):
            continue
        try:
            out[parts[0][:-1]] = int(parts[1])
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# The measured host, as one immutable struct
# ---------------------------------------------------------------------------


class HostFacts(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the fleet is told about this host, measured ONCE.

    A handful of these floats become FLEET-WIDE verdicts: ``HardwareUnsuitable``
    fences a machine, ``HostCanary`` condemns a SKU, ``gpu_name`` chooses which
    verdict key gets written. So there is exactly one producer
    (:func:`gen_worker.hostfacts.probe_hardware`, run by
    ``procsplit.measure`` before any tenant code is imported) and exactly one
    consumer that puts it on the wire
    (``procsplit.parent.ParentControl._parent_resources``). A second builder
    is a second answer, and the one the hub receives is never the one the pod
    acts on — that was pgw#898.
    """

    gpu_count: int = 0
    #: Card 0's TOTAL bytes. Not summed across cards: the hub admits one job
    #: to one group, so a sum promises room no job can have.
    vram_total_bytes: int = 0
    #: Free bytes the hub may admit against — the least-roomy group's, never
    #: a sum and never card 0's when a wider topology is delivered.
    vram_free_bytes: int = 0
    gpu_name: str = ""
    gpu_sm: str = ""
    torch_version: str = ""
    #: The CUDA runtime torch was BUILT against. Measured by
    #: `models.hub_policy.detect_worker_capabilities` and carried through to
    #: `gate_functions` (pgw#896).
    cuda_version: str = ""
    #: The HOST driver, read from NVML: it must stay readable when the CUDA
    #: runtime is not.
    driver_version: str = ""
    installed_libs: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {f: getattr(self, f) for f in self.__struct_fields__}


__all__ = [
    "DEVICE_ABSENT",
    "DEVICE_PRESENT",
    "DEVICE_UNREADABLE",
    "CpuAllowance",
    "CudaState",
    "HostFacts",
    "cgroup_nodes",
    "cpu_allowance",
    "cpu_quota",
    "cpu_quota_raw",
    "cuda_ready",
    "cuda_state",
    "device_count",
    "device_identity",
    "free_vram_bytes",
    "headroom_bytes",
    "PROC_MEMINFO",
    "meminfo_kb",
    "process_ceiling_bytes",
    "reset_cuda_state",
    "total_vram_bytes",
]
