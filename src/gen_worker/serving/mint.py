"""pgw#1371: the runtime background mint — a serving worker fills its own holes.

THE mint path, not a fallback: a worker that boots onto an unminted
(lane x sm) serves EAGER immediately and mints, on its own GPU, exactly the
graphs it is missing. Four properties, each load-bearing:

1. **HOLES ONLY.** The work-list is :attr:`EndpointHost.holes` — pgw#1372's
   ``AdoptSession`` handoff, canonical document order, one ``Hole`` per graph
   this (lane x sm) still needs. Document order IS mint order, which is what
   carries pgw#1384's default-parameter-first guarantee down here: whatever
   the derive put first is minted first, and this module never re-sorts.
   The whole-declaration sweep that took e2e#1892 run 7's pod down with it is
   not reachable from this entry at all.

2. **PER-GRAPH PUBLISH, AND THE ARM READS THE STORE BACK.** Each hole is
   compiled, published (local CAS + hub) and ARMED on its own, the moment its
   artifact lands — never a batch, never a seal at the end. A pod killed
   mid-mint keeps every completed graph and its successor mints only the
   remainder (pgw#1367 clause 4: partial-hit by construction). Since pgw#1573
   the bytes it arms are the bytes ``fetch_artifact`` hands back, not the
   directory the compile child wrote: one load path for a self-minted artifact
   and a hub-fetched one, which is what makes a publish/load format skew break
   immediately instead of on somebody else's pod a week later.

3. **CPU BUDGETED AGAINST THE SERVING PROCESS.** ``entry_workers <= vcpus -
   SERVING_RESERVE_CPUS``, the reserve non-zero and explicit, and the mint
   tree niced at ``compile_posture`` priority. Run 7 measured the cost of
   getting this wrong: a 36-class mint at 2-on-7 starved serving so badly
   that a 15m50s invocation never returned in 65 minutes.

4. **PROGRESS-GATED LIVENESS, AND IT REPORTS.** The guard reads what the mint
   has ACHIEVED — graphs completed, and the mint tree's own CPU — never a
   clock against the job. A mint that lands graphs is never condemned however
   long it takes; a mint that lands nothing AND burns no CPU is condemned
   inside its window. No window is ever enlarged to make a stall look healthy.
   And every verdict it reaches goes on the WIRE (pgw#1383): a serve pod's
   stdout goes nowhere (pgw#760), so a condemnation that only reaches
   ``logger.error`` is, from the hub's side, a mint that stopped emitting.

**THE PROPERTY THAT MADE THE OLD PATH INVISIBLE, stated so it is not
re-lost.** pgw#1383 root-caused pod ``j56tate13oav13``: the retired supervisor
ran its terminus adopt — an arm plus a numerics verify per graph specialization, on the
live pipeline — straight from a coroutine, and ``activity._emit``'s bound sink
ships every report as a task on THAT event loop. The seal phase it had just
declared was created and never ran; neither did any heartbeat or any abort. The
hub read ``inductor_compile step 2/2`` for thirty billed minutes because that
genuinely was the last thing to reach it. **The arm may never run on a loop
that also ships the worker's reports**, and here it does not: every arm runs on
a mint worker thread under ``arm_lock``, and the guard samples on its own
cadence beside them.
"""

from __future__ import annotations

import functools
import logging
import os
import queue
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import activity as activity_mod
from .. import compile_posture
from ..stall import SilenceWindow
from .mint_child import ContractModuleMissing, contract_digest

logger = logging.getLogger(__name__)


#: vCPUs the mint may never size itself onto. The serving process needs CPU for
#: its request loop, scheduling and decode, and it competes with AOTInductor on
#: the same box; e2e#1892 run 7 is the measurement of a zero reserve.
SERVING_RESERVE_CPUS = 2

#: How long the mint tolerates hearing NOTHING — no graph landed, no CPU burnt
#: — before condemning itself. A silence window over the channel, not a budget
#: for the work: a single inductor compile advances the CPU axis every sample.
DEFAULT_SILENCE_WINDOW_S = 600.0

#: Below this the CPU axis has not moved (sampling jitter, not work).
_CPU_EVIDENCE_EPS = 0.5

#: pgw#1383: the typed defect this worker reports about ITSELF when its own
#: mint stops making measured progress. A serve pod exposes no logs (pgw#760),
#: so `logger.error` is not a report — it is the mint dying in private, which
#: is the exact shape that cost `j56tate13oav13` thirty billed minutes and a
#: manual `DELETE /v1/admin/pods`. The hub's stall detector was right every
#: time it fired there; what it lacked was a fact from the worker to join to.
KIND_MINT_WEDGED = "self_mint_wedged"

#: A graph that is MINTED and PUBLISHED but did not arm on this pod's live
#: dispatch. Not a failure of the mint — the fleet has the bytes — but it is
#: why this pod keeps serving eager for a graph it just paid to compile, and
#: that is a wire fact rather than a pod-log line.
KIND_ARM_MISSED = "self_mint_arm_missed"


class MintCondemned(RuntimeError):
    """The mint made no measured progress inside its silence window."""


# ── the work-list ────────────────────────────────────────────────────────────


def hole_work_list(host: Any) -> Tuple[Any, ...]:
    """The ordered mint work-list off a booted :class:`EndpointHost`.

    Canonical document order, unchanged — pgw#1384's ask is satisfied by the
    DERIVE putting the default-parameter class first, and this function's job
    is to not disturb it. ``session.unclaimed`` is deliberately absent: an
    unclaimed record has no module to arm onto, so it is drift evidence, not
    mint work.
    """
    return tuple(getattr(host, "holes", ()) or ())


def entry_workers(
    holes: int,
    *,
    vcpus: Optional[int] = None,
    posture: Optional[compile_posture.CompilePosture] = None,
    reserve: int = SERVING_RESERVE_CPUS,
) -> int:
    """How wide the mint may run beside a LIVE serving process.

    ``entry_workers <= vcpus - reserve``, floored at one and capped by the
    work: a two-hole mint never opens seven workers. The posture applies its
    own ceiling on top (a user's machine takes the narrower bound), and the
    tree is niced besides — the reserve bounds the AVERAGE, priority decides
    who yields at microsecond granularity, and run 7 proved the reserve alone
    is not enough.
    """
    posture = posture or compile_posture.current()
    if vcpus is None:
        vcpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else (os.cpu_count() or 1)
    budget = posture.cpu_budget_cores(int(vcpus), headroom=max(0, int(reserve)))
    width = max(1, min(int(holes), budget))
    return max(1, posture.entry_ceiling(width))


# ── outcomes ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MintedHole:
    """One graph that went the whole way: compiled, published, armed."""

    graph: str
    target: str
    artifact: Path
    published: str = ""
    armed: bool = False
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class MintFailure:
    """One hole that did not land. Costs its own graph and nothing else."""

    graph: str
    target: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class MintOutcome:
    """What the background mint achieved, however it ended."""

    holes: int = 0
    entries: Tuple[MintedHole, ...] = ()
    failed: Tuple[MintFailure, ...] = ()
    width: int = 0
    condemned: str = ""
    elapsed_s: float = 0.0

    @property
    def landed(self) -> int:
        return len(self.entries)

    @property
    def complete(self) -> bool:
        return self.landed == self.holes and not self.failed

    def facts(self) -> Dict[str, Any]:
        return {
            "holes": self.holes,
            "landed": self.landed,
            "failed": len(self.failed),
            "width": self.width,
            "condemned": self.condemned,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# ── the progress-gated guard ─────────────────────────────────────────────────


class MintProgress:
    """What the mint has ACHIEVED, and the only give-up test over it.

    Two axes, each with its own high-water mark so a quiet one cannot cancel a
    moving one: graphs COMPLETED (the goal itself) and the MINT's own compile
    CPU (the evidence a compile is running even when no graph has landed yet —
    an inductor compile is tens of minutes of one graph). Neither is a clock
    against the job.

    The CPU axis is deliberately scoped to the mint's own workers, never to
    the process: a co-resident serving process burns CPU all day, and a
    process-wide reading would let ITS work vouch for a wedged mint. That is
    how the 2026-08-18 fleet pods stayed invisible — the pod looked busy
    because it WAS busy, serving.
    """

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_SILENCE_WINDOW_S,
        cpu_sample: Optional[Callable[[], Optional[float]]] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = SilenceWindow(window_s, now=now)
        # Reentrant: `judge` holds it and the CPU axis reads the worker set
        # under it. A plain Lock here deadlocks the guard on its first sample
        # — measured, and it hung a whole run.
        self._lock = threading.RLock()
        self._workers: set[int] = set()
        self._cpu_sample = cpu_sample if cpu_sample is not None else self._worker_cpu
        self.completed = 0
        self.cpu_s: Optional[float] = None
        self.condemned = ""

    @property
    def window_s(self) -> float:
        return self._window.window_s

    def register_worker(self, tid: int) -> None:
        """One mint worker's thread id — the CPU axis's whole population."""
        with self._lock:
            self._workers.add(int(tid))

    def _worker_cpu(self) -> Optional[float]:
        """The mint's OWN compile CPU: its threads plus their child processes.

        A real AOTI compile burns almost all of its CPU in CHILDREN — inductor's
        compile workers and the ``cc1plus`` under them — so a thread-only
        reading calls a flat-out mint idle. (Measured: it did, and condemned a
        healthy mint that was shelling out to a compiler.) Reaped children are
        counted through ``os.times``; live ones are read off ``/proc``.
        """
        with self._lock:
            tids = tuple(self._workers)
        if not tids:
            return None
        total = _thread_cpu_seconds(tids)
        if total is None:
            return None
        return total + _descendant_cpu_seconds() + _reaped_child_cpu_seconds()

    def landed(self, graph: str) -> None:
        """A graph completed — the goal itself advanced."""
        with self._lock:
            self.completed += 1
            self._window.touch()
        logger.info("runtime-mint: graph %s landed (%d complete)", graph, self.completed)

    def judge(self) -> None:
        """Condemn a mint that has achieved nothing and is burning no CPU.

        Called on a sampling cadence. The CPU axis is what keeps a healthy
        long compile alive; when BOTH axes are flat for the whole window the
        mint is wedged, and the honest response is to stop it and keep
        serving eager — not to widen the window.
        """
        with self._lock:
            if self.condemned:
                raise MintCondemned(self.condemned)
            cpu = self._cpu_sample()
            advanced = False
            if cpu is not None and (
                self.cpu_s is None or cpu - self.cpu_s >= _CPU_EVIDENCE_EPS
            ):
                self.cpu_s, advanced = cpu, True
            if advanced:
                self._window.touch()
                return
            if not self._window.stalled():
                return
            self.condemned = (
                f"the runtime mint made no measured progress for "
                f"{self._window.silent_for():.0f}s (window "
                f"{self._window.window_s:.0f}s): {self.completed} graph(s) "
                f"completed and mint-tree CPU "
                f"{'unreadable' if self.cpu_s is None else f'{self.cpu_s:.1f}s'} "
                f"is flat. It is wedged, not compiling; this mint stops and "
                f"the worker keeps serving eager"
            )
            raise MintCondemned(self.condemned)


_CLOCK_TICKS = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0


def _thread_cpu_seconds(tids: Sequence[int]) -> Optional[float]:
    """CPU seconds burnt by exactly these threads, off ``/proc/self/task``.

    Unreadable is ``None``, never zero: "I could not measure" and "it did no
    work" are different verdicts, and only the second may condemn.
    """
    total = 0.0
    seen = False
    for tid in tids:
        try:
            raw = Path(f"/proc/self/task/{int(tid)}/stat").read_text()
        except OSError:
            continue  # the thread exited; its work is already banked
        try:
            fields = raw[raw.rindex(")") + 2:].split()
            total += (float(fields[11]) + float(fields[12])) / _CLOCK_TICKS
        except (ValueError, IndexError):  # pragma: no cover — kernel format
            continue
        seen = True
    return total if seen else None


# ── the mint ─────────────────────────────────────────────────────────────────

# ── the artifact's own compatibility set ─────────────────────────────────────
#
# Paul, 2026-08-18 (compatibility sets, pt 4): the constraints are read OFF THE
# BUILT ARTIFACT, never off the environment's package list. An AOTI .so names
# its real dependencies in its own ELF ``DT_NEEDED`` entries — typically the
# CUDA math libraries inductor delegated to (cuBLAS, cuDNN), sometimes none at
# all when every kernel is embedded Triton/SASS. Embedded kernels are dep-free
# by construction, so there is nothing to record for them. Snapshotting the env
# would make an unrelated package bump able to reject bytes that still load.

#: ELF ``DT_NEEDED`` sonames that constrain NOTHING: the C/C++ runtime is
#: present on every image the fleet runs, and recording it would add a floor
#: nobody can act on.
_UNCONSTRAINING = (
    "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
    "libstdc++.so", "libgcc_s.so", "ld-linux", "libutil.so", "libresolv.so",
)


class ArtifactUnreadable(RuntimeError):
    """The built artifact could not be read for its own dependencies."""


#: Inside an AOTI ``.pt2`` package, compiled objects live under this prefix.
_AOTI_PREFIX = "model/data/aotinductor/"


def artifact_package(artifact: Path) -> Path:
    """The single FILE that IS this artifact — what gets published.

    ``Engine.compile`` resolves a minted key into ``destination`` and leaves an
    UNPACKED DIRECTORY there (``metadata.json`` + ``model.pt2``), so the thing
    the compiler hands back is not itself publishable: ``publish_artifact``
    requires a file, by design — a directory has no single digest to address
    it by.

    The package file is that digest's subject. Measured on a real sd1.5 mint:
    ``model.pt2`` is 10,817,910 B of stored-compression zip beside a 147,031 B
    ``metadata.json``.

    A path that is already a file passes through, so a compiler that returns
    the package directly needs no special case here.
    """
    path = Path(artifact)
    if path.is_file():
        return path
    if not path.is_dir():
        raise ArtifactUnreadable(f"{path} is neither a file nor a directory")
    packages = sorted(path.glob("*.pt2"))
    if len(packages) == 1:
        return packages[0]
    if not packages:
        raise ArtifactUnreadable(
            f"{path} is an unpacked artifact directory carrying no .pt2 "
            f"package (holds: {sorted(p.name for p in path.iterdir())})")
    raise ArtifactUnreadable(
        f"{path} holds {len(packages)} .pt2 packages "
        f"({sorted(p.name for p in packages)}); an artifact position is ONE "
        f"set of bytes and this cannot be resolved without guessing")


def compiled_object_bytes(artifact: Path) -> bytes:
    """The ELF bytes of the compiled object, wherever it actually lives.

    The publish target and the ELF-read target are DIFFERENT OBJECTS, and
    conflating them is what broke the runtime mint (pgw#1471): publishing wants
    the package, while ``DT_NEEDED`` lives on the compiled shared object INSIDE
    it. Pointing one path at both consumers forced a choice that made one of
    them wrong.

    So this reaches in. Measured layout of a real artifact::

        <destination>/
          metadata.json                        147,031 B
          model.pt2                         10,817,910 B   zip, stored
            └─ model/data/aotinductor/<graph>/
                 <hash>.wrapper.so            4,278,048 B   \\x7fELF

    A bare ELF file passes through unchanged, so a caller holding the object
    itself is unaffected.
    """
    path = Path(artifact)
    package = artifact_package(path)
    raw = package.read_bytes()
    if raw[:4] == b"\x7fELF":
        # Already the object — an older layout, or a caller who reached in
        # themselves. Nothing to unwrap.
        return raw
    if not zipfile.is_zipfile(package):
        raise ArtifactUnreadable(
            f"{package} is neither an ELF object nor a .pt2 package")
    with zipfile.ZipFile(package) as bundle:
        objects = [
            name for name in bundle.namelist()
            if name.startswith(_AOTI_PREFIX) and name.endswith(".so")
        ]
        if not objects:
            # NOT an error: an artifact whose kernels are all embedded
            # Triton/SASS names no shared object, and `needed_libraries`
            # already treats "constrains nothing" as a correct answer.
            return b""
        if len(objects) > 1:
            raise ArtifactUnreadable(
                f"{package} carries {len(objects)} compiled objects "
                f"({sorted(objects)}); one artifact position is one object and "
                f"choosing between them would be a guess")
        return bundle.read(objects[0])


def needed_libraries(artifact: Path) -> Tuple[str, ...]:
    """The artifact's own ELF ``DT_NEEDED`` sonames, in link order.

    A direct ELF read rather than a shell out to ``readelf``: the mint runs on
    a serving pod whose image is not guaranteed to carry binutils, and the
    manifest is not optional.

    pgw#1471: the bytes come from :func:`compiled_object_bytes`, which unwraps
    the ``.pt2`` package. The ELF assertion below STAYS exactly as it was — it
    is what caught this defect, and relaxing it to accept whatever the caller
    passed would have turned a loud failure into a manifest full of nothing.
    """
    raw = compiled_object_bytes(Path(artifact))
    if not raw:
        # No compiled object in the package: embedded-kernel artifact.
        return ()
    if raw[:4] != b"\x7fELF":
        raise ArtifactUnreadable(f"{artifact} is not an ELF object")
    if raw[4] != 2:  # ELFCLASS64
        raise ArtifactUnreadable(f"{artifact} is not a 64-bit ELF object")
    little = raw[5] == 1
    order = "little" if little else "big"

    def num(offset: int, size: int) -> int:
        return int.from_bytes(raw[offset:offset + size], order)  # type: ignore[arg-type]

    e_shoff, e_shentsize = num(0x28, 8), num(0x3A, 2)
    e_shnum = num(0x3C, 2)
    dynamic = strtab = None
    for index in range(e_shnum):
        base = e_shoff + index * e_shentsize
        sh_type = num(base + 4, 4)
        if sh_type == 6:  # SHT_DYNAMIC
            dynamic = (num(base + 0x18, 8), num(base + 0x20, 8))
        elif sh_type == 3 and strtab is None and num(base + 8, 8) & 0x2:
            # SHT_STRTAB carrying SHF_ALLOC: the dynamic string table.
            strtab = (num(base + 0x18, 8), num(base + 0x20, 8))
    if dynamic is None or strtab is None:
        return ()
    d_off, d_size = dynamic
    s_off, s_size = strtab
    names: List[str] = []
    for entry in range(d_off, d_off + d_size, 16):
        tag, value = num(entry, 8), num(entry + 8, 8)
        if tag == 0:  # DT_NULL
            break
        if tag != 1:  # DT_NEEDED
            continue
        start = s_off + value
        if not 0 <= value < s_size:
            continue
        end = raw.index(b"\x00", start)
        names.append(raw[start:end].decode("ascii", "replace"))
    return tuple(names)


def _reaped_child_cpu_seconds() -> float:
    """CPU of every child this process has already waited on."""
    try:
        times = os.times()
    except OSError:  # pragma: no cover
        return 0.0
    return float(times.children_user + times.children_system)


def _descendant_cpu_seconds() -> float:
    """CPU of the live child processes this process spawned.

    Direct children only: an inductor compile's tree is rooted at them, and
    their own descendants roll up through ``os.times`` the moment they are
    reaped. Unreadable contributes zero rather than failing the sample.
    """
    total = 0.0
    try:
        tasks = list(Path("/proc/self/task").iterdir())
    except OSError:  # pragma: no cover — Linux only
        return 0.0
    pids: set[int] = set()
    for task in tasks:
        try:
            pids.update(int(p) for p in (task / "children").read_text().split())
        except (OSError, ValueError):
            continue
    for pid in pids:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            total += (float(fields[11]) + float(fields[12])) / _CLOCK_TICKS
        except (OSError, ValueError, IndexError):
            continue
    return total


def _library_name(soname: str) -> str:
    """``libcublasLt.so.13`` -> ``libcublaslt``. The soname's stable identity.

    Case-folded because the manifest canonicalizes names on the way in, and a
    row that is emitted as ``libcublasLt`` but read back as ``libcublaslt``
    would silently never be checked. (Measured: it was.)
    """
    return soname.split(".so", 1)[0].lower()


def _torch_package_dir() -> Optional[Path]:
    """Where the torch package lives, WITHOUT importing it (pgw#1546).

    A torch already in ``sys.modules`` answers directly; otherwise
    ``find_spec`` locates the package without executing it — the reuse path
    publishes manifests for artifacts it never loads, and paying the full
    torch import to learn a directory made every warm publish carry it.
    """
    import sys

    module = sys.modules.get("torch")
    if module is not None:
        try:
            return Path(module.__file__).parent  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        for location in list(getattr(spec, "submodule_search_locations", None) or []):
            return Path(location)
    except Exception:  # noqa: BLE001 — a torchless env answers None
        pass
    return None


def _search_roots(artifact: Path) -> Tuple[Path, ...]:
    """Where a linked library may really live, nearest first.

    The artifact's own directory (a mint stages its libraries beside it), then
    every directory the loader has something open from — which on a pod, after
    CUDA is initialized, is exactly the set that matters.
    """
    roots: List[Path] = [Path(artifact).parent]
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.rsplit(" ", 1)[-1]
            if path.startswith("/") and ".so" in path:
                roots.append(Path(path).parent)
    except OSError:  # pragma: no cover — Linux only
        pass
    torch_dir = _torch_package_dir()  # wheels ship the CUDA libs beside torch
    if torch_dir is not None:
        roots.append(torch_dir / "lib")
        roots.append(torch_dir.parent / "nvidia")
    seen: Dict[str, Path] = {}
    for root in roots:
        seen.setdefault(str(root), root)
    return tuple(seen.values())


def _root_resolved_version(stem: str, root: Path) -> str:
    """The longest full-version tail resolvable for ``stem`` under one root."""
    best = ""
    try:
        candidates = (
            list(root.glob(f"{stem}.so*"))
            + list(root.glob(f"*/{stem}.so*"))
            # wheel-shipped CUDA libraries: nvidia/<lib>/lib/libX.so.N.N.N
            + list(root.glob(f"*/lib/{stem}.so*"))
        )
    except OSError:
        return best
    for path in candidates:
        try:
            real = os.path.realpath(path).rsplit("/", 1)[-1]
        except OSError:
            continue
        tail = real.split(".so.", 1)[-1] if ".so." in real else ""
        if len(tail) > len(best):
            best = tail
    return best


@functools.lru_cache(maxsize=None)
def _env_resolved_version(stem: str) -> str:
    """``_root_resolved_version`` over the ENV roots, cached per process.

    The env half of the search (loader maps, the torch/nvidia wheel trees) is
    the same answer for every artifact in a run, and the wheel-tree globs are
    what made it expensive: pgw#1546 measured 6.6 s of a 34 s warm publish in
    exactly these globs, repeated per artifact for the same four sonames.
    """
    best = ""
    for root in _search_roots(Path(os.devnull)):
        found = _root_resolved_version(stem, root)
        if len(found) > len(best):
            best = found
    return best


def _resolved_version(soname: str, artifact: Path) -> str:
    """The FULL version of the library this mint linked, off the real file.

    The soname pins the major; the file behind it carries the whole version
    (``libcublas.so.13`` -> ``libcublas.so.13.1.1.3``). Resolved on disk, and
    through symlinks, so the floor is the version that actually produced these
    bytes. When only the soname can be found, its major IS the honest answer
    and is recorded as such rather than invented more precisely.

    The artifact's own directory is probed fresh per call (a mint stages
    libraries beside its artifact); the env-level roots are one cached scan
    per soname per process.
    """
    stem = soname.split(".so", 1)[0]
    best = _root_resolved_version(stem, Path(artifact).parent)
    cached = _env_resolved_version(stem)
    if len(cached) > len(best):
        best = cached
    if best:
        return best
    return soname.split(".so.", 1)[-1] if ".so." in soname else ""


def _same_major_floor(version: str) -> str:
    """``13.1.1.3`` -> ``>=13.1.1.3,<14``. A floor, never a pin.

    Same-major because that is where the CUDA libraries keep ABI; a later
    patch or minor satisfies, an older one does not, and a major bump is a
    different library as far as these bytes are concerned.
    """
    if not version:
        return ""
    major = version.split(".", 1)[0]
    try:
        following = str(int(major) + 1)
    except ValueError:
        return f">={version}"
    return f">={version},<{following}"


def artifact_constraints(artifact: Path) -> Tuple[Tuple[str, str], ...]:
    """The compatibility set OF THESE BYTES: one same-major floor per linked
    library the artifact's own ELF names. Possibly empty — an artifact whose
    kernels are all embedded Triton/SASS genuinely constrains nothing."""
    rows: Dict[str, str] = {}
    for soname in needed_libraries(artifact):
        if any(soname.startswith(skip) for skip in _UNCONSTRAINING):
            continue
        floor = _same_major_floor(_resolved_version(soname, artifact))
        if floor:
            rows[_library_name(soname)] = floor
    return tuple(sorted(rows.items()))


def torch_shim_floor() -> Tuple[str, str]:
    """The AOTI shim this artifact was built against, as a same-major floor.

    Reads ``torch.__version__`` — a recording of the mint env, not a key axis
    — and deliberately not the dist-metadata reader the pgw#1489 guard fences
    to one diagnostic caller. The import is free in practice: every path that
    publishes has already imported torch through ``toolchain_digest``
    (pgw#1546).
    """
    import torch

    version = str(torch.__version__).split("+", 1)[0]
    return ("torch", _same_major_floor(version) or f">={version}")


def driver_floor() -> Optional[str]:
    """The CUDA line the mint's torch was built for, as a plain floor.

    ``None`` for a CPU-only torch. Read through
    :func:`gen_worker.env_identity._torch_cuda_line`, which answers
    ``torch.version.cuda`` from the generated constants module without
    executing the torch package (pgw#1546) — same value, no ~1.5 s import on
    the warm publish path.
    """
    try:
        from ..env_identity import _torch_cuda_line

        raw = _torch_cuda_line()
        return f">={raw}" if raw else None
    except Exception:  # noqa: BLE001 — a missing driver is not a mint failure
        return None


def artifact_manifest(env: Any, package: Path) -> Any:
    """The COMPATIBILITY SET of these bytes, read off the artifact itself.

    Paul, 2026-08-18: version constraints derived FROM THE ARTIFACT — its own
    ELF ``DT_NEEDED`` entries plus the AOTI shim it was built against — never a
    snapshot of the environment's package list. Each row is a same-major ``>=``
    floor at the version present at mint; the sm and the driver floor ride their
    own fields. An artifact whose kernels are all embedded Triton/SASS names no
    libraries at all, and that is a correct answer, not a recording bug.

    The mint always runs in the CURRENT env — by definition the newest the fleet
    runs — so there is deliberately no staleness logic here and no opportunistic
    re-mint.
    """
    from .._vendor.torchcg import RequirementsManifest

    rows = dict(artifact_constraints(package))
    name, floor = torch_shim_floor()
    rows[name] = floor
    return RequirementsManifest(
        include_set=tuple(sorted(rows.items())),
        sm_compiled=env.sm,
        cuda_floor=driver_floor(),
    )


def artifact_envelope(artifact: Path, workspace: Path) -> Path:
    """The tar+gzip ENVELOPE these bytes publish as (pgw#1561).

    An unpacked artifact DIRECTORY (``metadata.json`` + ``model.pt2``
    [+ ``constants.safetensors``]) is repacked with torchcg's own
    DETERMINISTIC ``pack_artifact`` — which also validates the metadata
    against the package on the way, so a publish is now a conformance proof
    rather than a byte copy, and a re-publish of the same members is
    byte-identical (idempotent ``present``, and the same object the engine
    cache already holds, so the CAS dedups the two bands for free).

    A FILE that already carries the gzip magic passes through as the envelope
    it is. ANY other file — above all the bare ``model.pt2`` ZIP — is a typed
    refusal: pgw#1561 is fourteen adoptions holing on ``cannot decompress``
    because the pre-envelope publisher banked exactly that ZIP, `SERVABLE` on
    its lips, and a publisher that can be handed a package and quietly bank it
    is the same defect waiting to be re-introduced.
    """
    from .._vendor.torchcg.artifact import pack_artifact

    source = Path(artifact)
    if source.is_file():
        with source.open("rb") as handle:
            if handle.read(2) == b"\x1f\x8b":
                return source
        raise ArtifactUnreadable(
            f"{source} is a file but not a compiled-graph envelope — a bare "
            f"package cannot be published (pgw#1561: the serving loader reads "
            f"the ENVELOPE, and a package file carries no metadata to build "
            f"one from). Hand the unpacked artifact DIRECTORY instead."
        )
    if not source.is_dir():
        raise ArtifactUnreadable(f"{source} is neither a file nor a directory")
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactUnreadable(
            f"{source} carries no metadata.json — not an unpacked "
            f"compiled-graph artifact; nothing publishable can be built from it"
        )
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    literals = source / "constants.safetensors"
    return pack_artifact(
        artifact_package(source),
        workspace / "compiled_graph.tar.gz",
        metadata,
        literals=literals if literals.is_file() else None,
    )


def publish_compiled(store: Any, graph: str, env: Any, artifact: Path) -> str:
    """Put one freshly-compiled artifact WHERE THE SERVING PATH READS IT.

    THE ONE PUBLISHER (pgw#1533). ``Engine.compile`` banks its output in
    torchcg's own engine store — that is the engine's compile cache, keyed by
    ``cg-key-v1`` and consulted by ``Engine.resolve`` so a re-compile is a
    reuse. It is NOT the band adoption reads. ``holes()``, ``has_artifact`` and
    every boot-time adopt address artifacts as ``(cg-graph-v1, cg-env-v2)`` in
    the ``GraphStore``, and NOTHING lands there unless somebody calls
    ``publish_artifact``.

    The runtime mint called it; ``gen-worker compile`` did not — so the CLI
    burned 26 minutes of real inductor work on this box, reported
    ``built=14 (of 14)`` with rc=0, and left the serving reader looking at 14
    holes. One compile path with two publishers was always going to drift; this
    is the single one, and both callers go through it.

    WHAT IS PUBLISHED IS THE ENVELOPE (pgw#1561). pgw#1471 published
    ``artifact_package`` — the bare ``model.pt2`` ZIP — and the adopt loader
    reads the tar+gzip envelope, so no artifact this function ever banked
    could be loaded by the path it was banked for: va#3 arm 2 measured
    ``0/14 armed``, every hole ``cannot decompress``. The envelope also
    carries what the bare package silently discarded — ``metadata.json`` (the
    artifact's own key, sm, toolchain, constants table) and any literal
    payload.
    """
    import tempfile

    source = Path(artifact)
    package = artifact_package(source)  # the ELF the manifest is read off
    manifest = artifact_manifest(env, package)
    with tempfile.TemporaryDirectory(prefix="pgw-envelope-") as raw:
        envelope = artifact_envelope(source, Path(raw))
        outcome = store.publish_artifact(graph, env, envelope, manifest)
    return str(getattr(outcome, "value", outcome) or "")


def satisfied(
    manifest: Any, present: Mapping[str, str]
) -> Tuple[bool, Tuple[str, ...]]:
    """Does ``present`` satisfy every floor this manifest states?

    The adoption-side check, stated here beside the emitter so the two halves
    of one contract cannot drift. Every row is ``>=X[,<Y]``: a missing library
    fails, an older version fails, a different major fails, anything else
    satisfies.
    """
    reasons: List[str] = []
    for name, constraint in manifest.include_set:
        have = present.get(name)
        if have is None:
            reasons.append(f"{name}: absent, artifact needs {constraint}")
            continue
        for clause in str(constraint).split(","):
            clause = clause.strip()
            if clause.startswith(">="):
                if _version(have) < _version(clause[2:]):
                    reasons.append(f"{name}: have {have}, artifact needs {clause}")
            elif clause.startswith("<"):
                if _version(have) >= _version(clause[1:]):
                    reasons.append(f"{name}: have {have}, artifact needs {clause}")
    return (not reasons), tuple(reasons)


def present_libraries(names: Sequence[str]) -> Dict[str, str]:
    """What version of each named library THIS host has, resolved on disk.

    The adoption-side mirror of the emitter: the manifest names libraries
    (``libcublas``), not packages (``nvidia-cublas``), because libraries are
    what the artifact links. Resolving both sides the same way is what keeps
    the emit and the check from drifting into two vocabularies.
    """
    present: Dict[str, str] = {}
    for name in names:
        if name == "torch":
            try:
                import torch

                present["torch"] = str(torch.__version__).split("+", 1)[0]
            except Exception:  # noqa: BLE001
                pass
            continue
        version = _resolved_version(f"{name}.so", Path("."))
        if version:
            present[name] = version
    return present


def assert_satisfied(manifest: Any, *, sm: str) -> None:
    """Refuse loudly when this host cannot satisfy an artifact's floors.

    Replaces the exact-env audit (Paul, 2026-08-18): an artifact is adoptable
    when the libraries it LINKS are present at or above the versions it was
    built against, same major. An unrelated package bump can no longer reject
    bytes that still load, and a genuinely older runtime still cannot.
    """
    from .._vendor.torchcg import EnvironmentMismatch

    stated = str(getattr(manifest, "sm_compiled", "") or "")
    if stated and sm and stated != sm:
        raise EnvironmentMismatch(
            f"artifact was compiled for {stated} but this host is {sm}")
    names = [name for name, _ in manifest.include_set]
    ok, reasons = satisfied(manifest, present_libraries(names))
    if not ok:
        raise EnvironmentMismatch(
            "this host does not satisfy the artifact's compatibility floors: "
            + "; ".join(reasons))


def _version(raw: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in str(raw).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


#: The field on a ``GraphRecord`` carrying the CAS digest of that graph's
#: SERIALIZED ``ExportedProgram`` (Paul, 2026-08-18). The derive lane emits
#: THE HOLE'S KEY IS ITS GRAPH HASH, not a blob digest (Paul, 2026-08-19:
#: "program blobs are ADDRESS-FREE — the identity is the contract, bytes are
#: local"). The document states the identity; every box resolves its own
#: serialized program under it.
PROGRAM_GRAPH_FIELD = "graph"


class MissingProgram(RuntimeError):
    """This box holds no serialized graph for the identity this hole names.

    NOT a derive defect and NOT a broken document: the ordinary cause is a box
    that has not derived this endpoint yet, and the remedy is local — derive
    (`gen-worker lock`, or `compile`) so the programs land in this machine's
    own store. Costs its own graph and never the boot.
    """


def program_graph(record: Any) -> str:
    """The GRAPH IDENTITY this hole must compile, as the document states it."""
    value = str(getattr(record, PROGRAM_GRAPH_FIELD, "") or "")
    if not value:
        raise MissingProgram(
            "a mint hole carries no graph identity at all, so nothing can be "
            "resolved for it — the release document is malformed"
        )
    return value


#: Compile one hole: (program blob on disk, record, destination) -> artifact.
#: The input is a BLOB, never a live module — the whole seam that keeps
#: minting free of author code, weights and tracing.
Compiler = Callable[[Path, Any, Path], Path]


def _child_compiler(
    *, cas: Path, target_arch: str, toolchain: Mapping[str, str],
) -> Compiler:
    """THE production compiler: one CHILD PROCESS per graph.

    Not a helper a caller opts into — it is what a mint runs by default. The
    child (:mod:`gen_worker.serving.mint_child`) does the ``torch.export.load``
    and the AOTI compile, because both hold the GIL for minutes and would
    starve the heartbeat and eager serving on the parent's loop (th#1299).
    It also makes a
    wedged compile KILLABLE, which is what turns the progress guard's verdict
    into an action.

    An explicit ``compiler`` is the local seam (no GPU, no inductor) and
    nothing else.
    """

    def compile_one(blob: Path, record: Any, destination: Path) -> Path:
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="runtime-mint-") as tmp:
            request = Path(tmp) / "request.json"
            result = Path(tmp) / "artifact.txt"
            request.write_text(json.dumps({
                "blob": str(blob),
                "graph": record.graph,
                "target": record.target,
                "ingress": record.ingress.as_dict(),
                "target_arch": target_arch,
                "toolchain": dict(toolchain),
                "cas": str(cas),
                "destination": str(destination),
                "result": str(result),
                # pgw#1444/pgw#840: THIS parent's contract source, so the
                # child can refuse before compiling if it is a different
                # gen_worker. The child recomputes its own and compares —
                # cheaper and stricter than the v1 shape (child reports,
                # parent judges after the work is already done).
                "contract": contract_digest(),
            }))
            proc = subprocess.run(
                [sys.executable, "-m", "gen_worker.serving.mint_child",
                 str(request)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                raise ChildCompileFailed(
                    f"the compile child for {record.graph} exited "
                    f"{proc.returncode}: {proc.stderr.strip()[-800:]}")
            return Path(result.read_text().strip())

    return compile_one


class ChildCompileFailed(RuntimeError):
    """A compile child died. Costs its own graph and nothing else."""


class MintNotServable(RuntimeError):
    """A graph compiled, published, and the store cannot hand it back.

    pgw#1573's red arm, and the only refusal that can catch a publish/read
    address split at the instant it is introduced. It costs its own graph:
    everything else this mint builds still lands, and the pod keeps serving
    eager for this one.
    """


# ── the mint ─────────────────────────────────────────────────────────────────


@dataclass
class BackgroundMint:
    """One serving worker's mint of its own holes.

    Every field is a seam the caller states rather than a global this module
    reaches for: the store it publishes to and fetches graph blobs from, and
    the compiler it runs. Nothing here reaches into the executor, and nothing
    here touches the author's instance — a mint's input is BYTES.
    """

    host: Any
    store: Any
    #: The local seam. ``None`` means "run the real inductor compile", which is
    #: built from ``engine`` + ``runtime`` below.
    compiler: Optional[Compiler]
    artifacts_dir: Path
    #: Where the production compile child admits its artifacts, the
    #: architecture it compiles for, and the toolchain it records. Together
    #: they ARE the production compiler; state them, not a factory.
    cas_dir: Optional[Path] = None
    target_arch: str = ""
    toolchain: Mapping[str, str] = field(default_factory=dict)
    #: Resolve one serialized graph BY GRAPH IDENTITY:
    #: (graph_hash, destination) -> path. Defaults to the store's own
    #: ``fetch_program``. Keyed by identity and not by a blob digest because a
    #: blob digest is machine-scoped (pgw#1462p2) and this box's bytes are the
    #: only ones it can use.
    program_source: Optional[Callable[[str, Path], Path]] = None
    posture: compile_posture.CompilePosture = compile_posture.FLEET
    reserve: int = SERVING_RESERVE_CPUS
    window_s: float = DEFAULT_SILENCE_WINDOW_S
    vcpus: Optional[int] = None
    on_landed: Optional[Callable[[MintedHole], None]] = None
    #: The work-list, READ ONCE by the caller that armed this mint (pgw#1564).
    #: ``None`` falls back to reading ``host.holes`` at run time — but a
    #: caller that already counted N holes and then lets `run` re-read a LIVE
    #: property is how a pod declared `completed 0/14` in 13 ms: the property
    #: answered empty on the second read and an empty run settles as complete.
    #: What was armed is what gets minted; each hole still re-checks presence
    #: under its lease, so a graph that landed in between costs a skip, never
    #: a rebuild.
    work_list: Optional[Tuple[Any, ...]] = None
    _progress: Optional[MintProgress] = None

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.store is None:
            # NOT optional any more (pgw#1573). The store is where a minted
            # artifact BECOMES servable — it is published there and armed from
            # there — so a mint with no store used to fall back to arming the
            # compiler's own output, which is the second load path this issue
            # exists to delete. A mint that cannot publish has nothing to
            # deliver and says so at construction, not eleven minutes of
            # inductor later.
            raise ValueError(
                "a mint needs the store it publishes into and arms from: pass "
                "`store=` (gen_worker.serving.mint_store.graph_store). Since "
                "pgw#1573 the mint arms the bytes the STORE hands back, never "
                "the directory the compiler wrote, so there is no storeless "
                "mint to fall back to")
        if self.compiler is None:
            if self.cas_dir is None or not self.target_arch:
                raise ValueError(
                    "a mint needs something to compile with: pass `cas_dir` + "
                    "`target_arch` (+ `toolchain`) for the real child-process "
                    "compile, or an explicit `compiler` seam")
            self.compiler = _child_compiler(
                cas=Path(self.cas_dir), target_arch=self.target_arch,
                toolchain=dict(self.toolchain))
        if self.program_source is None:
            fetch = getattr(self.store, "fetch_program", None)
            if fetch is not None:
                self.program_source = fetch

    # -- the run ------------------------------------------------------------

    def run(self) -> MintOutcome:
        """Mint every hole, publishing and arming each as it lands.

        Never raises for a mint failure: the worker must never die with its
        mint. A condemned mint returns what it had already banked.
        """
        started = time.monotonic()
        holes = (
            self.work_list if self.work_list is not None
            else hole_work_list(self.host)
        )
        if not holes:
            return MintOutcome(holes=0, width=0, elapsed_s=0.0)

        width = entry_workers(
            len(holes), vcpus=self.vcpus, posture=self.posture, reserve=self.reserve
        )
        progress = self._progress or MintProgress(window_s=self.window_s)
        self._progress = progress
        self._renice()
        logger.info(
            "runtime-mint: %d hole(s), %d worker(s) (reserve %d vCPU, nice %d)",
            len(holes), width, self.reserve, self.posture.nice_level(),
        )

        entries: List[MintedHole] = []
        failed: List[MintFailure] = []
        arm_lock = threading.Lock()
        banked = threading.Lock()
        stop = threading.Event()
        work: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        for hole in holes:
            work.put(hole)

        def one(hole: Any) -> None:
            record = hole.record
            began = time.monotonic()
            try:
                landed = self._mint_one(record, arm_lock)
            except Exception as exc:  # noqa: BLE001 — a hole costs its own graph
                logger.warning(
                    "runtime-mint: graph %s (%s) did not land: %s",
                    record.graph, record.target, exc, exc_info=True,
                )
                with banked:
                    failed.append(MintFailure(
                        graph=record.graph, target=record.target,
                        reason=type(exc).__name__, detail=str(exc)))
                return
            landed = MintedHole(
                graph=landed.graph, target=landed.target, artifact=landed.artifact,
                published=landed.published, armed=landed.armed,
                elapsed_s=time.monotonic() - began,
            )
            with banked:
                entries.append(landed)
            progress.landed(record.graph)
            if self.on_landed is not None:
                try:
                    self.on_landed(landed)
                except Exception:  # noqa: BLE001 — a sink never fails a mint
                    logger.debug("runtime-mint: on_landed sink failed", exc_info=True)

        idle = threading.Event()

        def worker() -> None:
            # The CPU axis's population: registered from INSIDE the thread, so
            # the guard measures the compiles and nothing else.
            progress.register_worker(threading.get_native_id())
            try:
                while not stop.is_set():
                    try:
                        hole = work.get_nowait()
                    except queue.Empty:
                        return
                    one(hole)
            finally:
                # Wake the guard the instant a worker finishes, so a mint that
                # completes never waits out a sampling tick to say so.
                idle.set()

        threads = [
            threading.Thread(
                target=worker, name=f"runtime-mint-{i}", daemon=True)
            for i in range(width)
        ]
        for thread in threads:
            thread.start()

        # THE GUARD RUNS ON ITS OWN CADENCE, beside the compiles — not between
        # them. A guard that only samples when a hole finishes cannot see the
        # case it exists for: every worker wedged inside a compile that will
        # never return. (Measured here: before this loop existed, a mint with
        # a 2s window sat wedged for the full 30s of a sleeping compiler and
        # was never condemned.) The cadence is derived from the window, so
        # shortening the window sharpens the guard instead of only shortening
        # a countdown.
        sample_s = max(0.05, min(self.window_s / 8.0, 15.0))
        while any(thread.is_alive() for thread in threads):
            try:
                progress.judge()
            except MintCondemned as exc:
                # Stop claiming holes and RETURN what is already banked. The
                # in-flight compile is a daemon thread: it cannot be
                # interrupted from here, and it must not be allowed to hold
                # the serving worker hostage while it is judged dead.
                logger.error("runtime-mint: %s", exc)
                # ...and SAY SO on the wire (pgw#1383). A serve pod's stdout
                # goes nowhere, so a condemnation nobody can read is a mint
                # that stopped emitting — which is precisely the defect the
                # hub could see and could not attribute.
                activity_mod.emit_event(
                    KIND_MINT_WEDGED,
                    f"{len(holes)} hole(s), {width} worker(s), "
                    f"{progress.completed} landed, "
                    f"{time.monotonic() - started:.0f}s elapsed: {exc}",
                    phase="no_measured_progress",
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                stop.set()
                break
            idle.wait(sample_s)
            idle.clear()
        else:
            for thread in threads:
                thread.join()

        return MintOutcome(
            holes=len(holes),
            entries=tuple(entries),
            failed=tuple(failed),
            width=width,
            condemned=progress.condemned,
            elapsed_s=time.monotonic() - started,
        )

    # -- one hole -----------------------------------------------------------

    def _mint_one(self, record: Any, arm_lock: threading.Lock) -> MintedHole:
        """Fetch, compile, publish, FETCH BACK, arm — for ONE graph.

        The order is the durability contract. The artifact is on disk before
        it is published; it is in the store before it is armed; so a kill at
        any point leaves the fleet strictly better off and never leaves a
        dispatcher pointed at bytes nobody else can fetch.

        Step one RESOLVES this box's serialized ``ExportedProgram`` for the
        graph IDENTITY the document names (Paul, 2026-08-19, address-free).
        The document pins the identity, not an address: `torch.export.save`
        emits different bytes on different machines for the same traced graph
        (pgw#1462p2 — 14/14 identities reproduce, 0/14 blob digests do), so an
        address in a shared document is unresolvable by construction. Matching
        on the identity is what makes a stamped release mintable on a box that
        did not produce it.

        No weights are read here, and a mint still cannot diverge from the
        graph the release shipped: the identity IS the graph.

        ## THE FETCH BACK IS THE POINT (pgw#1573)

        This used to arm ``artifact`` — the unpacked directory the compile
        child had just written — while publishing the ENVELOPE somewhere the
        arm never looked. Two producers, two formats, one dispatcher, and no
        run of this function ever read its own published bytes. That is
        exactly how pgw#1471's bare-``.pt2`` publish survived from the first
        publisher that ever existed until va#3 arm 2 fetched one on a pod and
        holed 14/14 on ``cannot decompress``: every local green was green over
        the directory, and the store's copy was never opened by anybody.

        So the mint now arms what the STORE hands back, through the same
        ``fetch_artifact`` boot adoption calls, into the same position boot
        adoption fetches to. A self-minted artifact and a hub-fetched artifact
        traverse one load path. A publisher that banks bytes the loader cannot
        read now breaks the mint's own arm, on the machine that made them,
        within seconds — instead of hiding for weeks behind a directory.
        """
        env = self.host.adoption.env
        # SCRATCH, and deliberately not the adopt position. `Engine.compile`
        # leaves an unpacked DIRECTORY at its destination while the store
        # copies out a FILE, and both used to be pointed at
        # `<artifacts_dir>/<env>/<graph>.so` — so a boot after a mint asked
        # `os.replace` to put a file where a directory stood.
        scratch = self.artifacts_dir / env.value / "build" / record.graph
        scratch.parent.mkdir(parents=True, exist_ok=True)
        #: Where boot adoption fetches to (torchcg `AdoptSession._adopt_module`)
        #: and therefore where this mint fetches to. One position, one shape.
        position = self.artifacts_dir / env.value / f"{record.graph}.so"

        graph = program_graph(record)
        if self.program_source is None:
            raise MissingProgram(
                f"graph {graph} has no program source on this mint (the store "
                f"exposes no `fetch_program` and no `program_source` was "
                f"stated)")
        blob = Path(self.program_source(
            graph, self.artifacts_dir / "programs" / f"{graph}.pt2"))

        assert self.compiler is not None  # __post_init__ guarantees it
        artifact = Path(self.compiler(blob, record, scratch))

        published = publish_compiled(self.store, record.graph, env, artifact)
        fetched = self.store.fetch_artifact(record.graph, env, position)
        if fetched is None:
            raise MintNotServable(
                f"graph {record.graph} compiled and published "
                f"({published or 'no ref reported'}) and the store answers a "
                f"MISS at ({record.graph[:16]}, {env.value}). The publish and "
                f"the read address different positions; nothing this mint "
                f"builds can ever be adopted."
            )

        armed = False
        try:
            with arm_lock:
                self.host.adoption.arm(record, Path(fetched))
            armed = True
        except Exception as exc:  # noqa: BLE001 — published beats armed
            # The graph IS minted and the fleet HAS it; only THIS pod's live
            # dispatch missed the swap. Stated, never escalated: a successor
            # adopts it at boot and this worker keeps serving eager for it.
            logger.warning(
                "runtime-mint: %s published but did not arm live: %s",
                record.graph, exc,
            )
            activity_mod.emit_event(
                KIND_ARM_MISSED,
                f"graph {record.graph} ({record.target}) is minted and in the "
                f"store, but this pod's live dispatch did not take it "
                f"({type(exc).__name__}: {exc}). A successor adopts it at "
                f"boot; this worker keeps serving eager for it.",
                phase=type(exc).__name__,
                graph_specialization=str(record.graph)[:300],
            )

        return MintedHole(
            graph=record.graph, target=record.target, artifact=Path(fetched),
            published=published, armed=armed,
        )

    # -- the serving reserve, at the scheduler -------------------------------

    def _renice(self) -> None:
        """Nice THIS mint's tree, never the serving process.

        ``compile_posture``'s own doctrine: priority says who yields, at
        microsecond granularity, in the kernel. The reserve bounds the
        average; this is what makes a co-resident serving process win every
        contended slice.
        """
        level = self.posture.nice_level()
        if level <= 0:
            return
        try:
            os.setpriority(os.PRIO_PROCESS, 0, level)
        except OSError:  # pragma: no cover — a restricted sandbox
            logger.debug("runtime-mint: could not nice the mint tree", exc_info=True)


def mint_holes(
    host: Any,
    *,
    store: Any,
    compiler: Optional[Compiler] = None,
    cas_dir: Optional[Path] = None,
    target_arch: str = "",
    toolchain: Optional[Mapping[str, str]] = None,
    artifacts_dir: Path,
    program_source: Optional[Callable[[str, Path], Path]] = None,
    posture: Optional[compile_posture.CompilePosture] = None,
    reserve: int = SERVING_RESERVE_CPUS,
    window_s: float = DEFAULT_SILENCE_WINDOW_S,
    vcpus: Optional[int] = None,
    on_landed: Optional[Callable[[MintedHole], None]] = None,
    work_list: Optional[Tuple[Any, ...]] = None,
) -> MintOutcome:
    """Run one serving worker's background mint over its own holes."""
    return BackgroundMint(
        host=host,
        store=store,
        compiler=compiler,
        artifacts_dir=Path(artifacts_dir),
        cas_dir=cas_dir,
        target_arch=target_arch,
        toolchain=dict(toolchain or {}),
        program_source=program_source,
        posture=posture or compile_posture.current(),
        reserve=reserve,
        window_s=window_s,
        vcpus=vcpus,
        on_landed=on_landed,
        work_list=work_list,
    ).run()


__all__ = [
    "BackgroundMint",
    "KIND_ARM_MISSED",
    "KIND_MINT_WEDGED",
    "Compiler",
    "DEFAULT_SILENCE_WINDOW_S",
    "MintCondemned",
    "MintFailure",
    "MintNotServable",
    "MintOutcome",
    "MintProgress",
    "MintedHole",
    "SERVING_RESERVE_CPUS",
    "ArtifactUnreadable",
    "ChildCompileFailed",
    "ContractModuleMissing",
    "MissingProgram",
    "PROGRAM_GRAPH_FIELD",
    "artifact_constraints",
    "artifact_manifest",
    "artifact_package",
    "publish_compiled",
    "assert_satisfied",
    "present_libraries",
    "driver_floor",
    "needed_libraries",
    "satisfied",
    "torch_shim_floor",
    "entry_workers",
    "hole_work_list",
    "mint_holes",
    "program_graph",
]
