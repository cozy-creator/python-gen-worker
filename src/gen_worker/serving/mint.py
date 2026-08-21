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


SERVING_RESERVE_CPUS = 2

DEFAULT_SILENCE_WINDOW_S = 600.0

_CPU_EVIDENCE_EPS = 0.5

KIND_MINT_WEDGED = "self_mint_wedged"

KIND_ARM_MISSED = "self_mint_arm_missed"


class MintCondemned(RuntimeError):
    """The mint made no measured progress inside its silence window."""


def hole_work_list(host: Any) -> Tuple[Any, ...]:
    """The ordered mint work-list off a booted :class:`EndpointHost`."""
    return tuple(getattr(host, "holes", ()) or ())


def entry_workers(
    holes: int,
    *,
    vcpus: Optional[int] = None,
    posture: Optional[compile_posture.CompilePosture] = None,
    reserve: int = SERVING_RESERVE_CPUS,
) -> int:
    """How wide the mint may run beside a LIVE serving process."""
    posture = posture or compile_posture.current()
    if vcpus is None:
        vcpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else (os.cpu_count() or 1)
    budget = posture.cpu_budget_cores(int(vcpus), headroom=max(0, int(reserve)))
    width = max(1, min(int(holes), budget))
    return max(1, posture.entry_ceiling(width))


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
    """One hole that did not land."""

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


class MintProgress:
    """What the mint has ACHIEVED, and the only give-up test over it."""

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_SILENCE_WINDOW_S,
        cpu_sample: Optional[Callable[[], Optional[float]]] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = SilenceWindow(window_s, now=now)
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
        """Condemn a mint that has achieved nothing and is burning no CPU."""
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
    total = 0.0
    seen = False
    for tid in tids:
        try:
            raw = Path(f"/proc/self/task/{int(tid)}/stat").read_text()
        except OSError:
            continue
        try:
            fields = raw[raw.rindex(")") + 2:].split()
            total += (float(fields[11]) + float(fields[12])) / _CLOCK_TICKS
        except (ValueError, IndexError):  # pragma: no cover — kernel format
            continue
        seen = True
    return total if seen else None


_UNCONSTRAINING = (
    "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
    "libstdc++.so", "libgcc_s.so", "ld-linux", "libutil.so", "libresolv.so",
)


class ArtifactUnreadable(RuntimeError):
    """The built artifact could not be read for its own dependencies."""


_AOTI_PREFIX = "model/data/aotinductor/"


def artifact_package(artifact: Path) -> Path:
    """The single FILE that IS this artifact — what gets published."""
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
    """The ELF bytes of the compiled object, wherever it actually lives."""
    path = Path(artifact)
    package = artifact_package(path)
    raw = package.read_bytes()
    if raw[:4] == b"\x7fELF":
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
            return b""
        if len(objects) > 1:
            raise ArtifactUnreadable(
                f"{package} carries {len(objects)} compiled objects "
                f"({sorted(objects)}); one artifact position is one object and "
                f"choosing between them would be a guess")
        return bundle.read(objects[0])


def needed_libraries(artifact: Path) -> Tuple[str, ...]:
    """The artifact's own ELF ``DT_NEEDED`` sonames, in link order."""
    raw = compiled_object_bytes(Path(artifact))
    if not raw:
        return ()
    if raw[:4] != b"\x7fELF":
        raise ArtifactUnreadable(f"{artifact} is not an ELF object")
    if raw[4] != 2:
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
        if sh_type == 6:
            dynamic = (num(base + 0x18, 8), num(base + 0x20, 8))
        elif sh_type == 3 and strtab is None and num(base + 8, 8) & 0x2:
            strtab = (num(base + 0x18, 8), num(base + 0x20, 8))
    if dynamic is None or strtab is None:
        return ()
    d_off, d_size = dynamic
    s_off, s_size = strtab
    names: List[str] = []
    for entry in range(d_off, d_off + d_size, 16):
        tag, value = num(entry, 8), num(entry + 8, 8)
        if tag == 0:
            break
        if tag != 1:
            continue
        start = s_off + value
        if not 0 <= value < s_size:
            continue
        end = raw.index(b"\x00", start)
        names.append(raw[start:end].decode("ascii", "replace"))
    return tuple(names)


def _reaped_child_cpu_seconds() -> float:
    try:
        times = os.times()
    except OSError:  # pragma: no cover
        return 0.0
    return float(times.children_user + times.children_system)


def _descendant_cpu_seconds() -> float:
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
    return soname.split(".so", 1)[0].lower()


def _torch_package_dir() -> Optional[Path]:
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
    roots: List[Path] = [Path(artifact).parent]
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.rsplit(" ", 1)[-1]
            if path.startswith("/") and ".so" in path:
                roots.append(Path(path).parent)
    except OSError:  # pragma: no cover — Linux only
        pass
    torch_dir = _torch_package_dir()
    if torch_dir is not None:
        roots.append(torch_dir / "lib")
        roots.append(torch_dir.parent / "nvidia")
    seen: Dict[str, Path] = {}
    for root in roots:
        seen.setdefault(str(root), root)
    return tuple(seen.values())


def _root_resolved_version(stem: str, root: Path) -> str:
    best = ""
    try:
        candidates = (
            list(root.glob(f"{stem}.so*"))
            + list(root.glob(f"*/{stem}.so*"))
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
    best = ""
    for root in _search_roots(Path(os.devnull)):
        found = _root_resolved_version(stem, root)
        if len(found) > len(best):
            best = found
    return best


def _resolved_version(soname: str, artifact: Path) -> str:
    stem = soname.split(".so", 1)[0]
    best = _root_resolved_version(stem, Path(artifact).parent)
    cached = _env_resolved_version(stem)
    if len(cached) > len(best):
        best = cached
    if best:
        return best
    return soname.split(".so.", 1)[-1] if ".so." in soname else ""


def _same_major_floor(version: str) -> str:
    if not version:
        return ""
    major = version.split(".", 1)[0]
    try:
        following = str(int(major) + 1)
    except ValueError:
        return f">={version}"
    return f">={version},<{following}"


def artifact_constraints(artifact: Path) -> Tuple[Tuple[str, str], ...]:
    """The compatibility set OF THESE BYTES: one same-major floor per linked library the artifact's own ELF names."""
    rows: Dict[str, str] = {}
    for soname in needed_libraries(artifact):
        if any(soname.startswith(skip) for skip in _UNCONSTRAINING):
            continue
        floor = _same_major_floor(_resolved_version(soname, artifact))
        if floor:
            rows[_library_name(soname)] = floor
    return tuple(sorted(rows.items()))


def torch_shim_floor() -> Tuple[str, str]:
    """The AOTI shim this artifact was built against, as a same-major floor."""
    import torch

    version = str(torch.__version__).split("+", 1)[0]
    return ("torch", _same_major_floor(version) or f">={version}")


def driver_floor() -> Optional[str]:
    """The CUDA line the mint's torch was built for, as a plain floor."""
    try:
        from ..env_identity import _torch_cuda_line

        raw = _torch_cuda_line()
        return f">={raw}" if raw else None
    except Exception:  # noqa: BLE001 — a missing driver is not a mint failure
        return None


def artifact_manifest(env: Any, package: Path) -> Any:
    """The COMPATIBILITY SET of these bytes, read off the artifact itself."""
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
    """Put one freshly-compiled artifact WHERE THE SERVING PATH READS IT."""
    import tempfile

    source = Path(artifact)
    package = artifact_package(source)
    manifest = artifact_manifest(env, package)
    with tempfile.TemporaryDirectory(prefix="pgw-envelope-") as raw:
        envelope = artifact_envelope(source, Path(raw))
        outcome = store.publish_artifact(graph, env, envelope, manifest)
    return str(getattr(outcome, "value", outcome) or "")


def satisfied(
    manifest: Any, present: Mapping[str, str]
) -> Tuple[bool, Tuple[str, ...]]:
    """Does ``present`` satisfy every floor this manifest states? The adoption-side check, stated here beside the emitter so the two halves of one contract cannot drift."""
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
    """What version of each named library THIS host has, resolved on disk."""
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
    """Refuse loudly when this host cannot satisfy an artifact's floors."""
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


PROGRAM_GRAPH_FIELD = "graph"


class MissingProgram(RuntimeError):
    """This box holds no serialized graph for the identity this hole names."""


def program_graph(record: Any) -> str:
    """The GRAPH IDENTITY this hole must compile, as the document states it."""
    value = str(getattr(record, PROGRAM_GRAPH_FIELD, "") or "")
    if not value:
        raise MissingProgram(
            "a mint hole carries no graph identity at all, so nothing can be "
            "resolved for it — the release document is malformed"
        )
    return value


Compiler = Callable[[Path, Any, Path], Path]


def _child_compiler(
    *, cas: Path, target_arch: str, toolchain: Mapping[str, str],
) -> Compiler:

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
    """A compile child died."""


class MintNotServable(RuntimeError):
    """A graph compiled, published, and the store cannot hand it back."""


@dataclass
class BackgroundMint:
    """One serving worker's mint of its own holes."""

    host: Any
    store: Any
    compiler: Optional[Compiler]
    artifacts_dir: Path
    cas_dir: Optional[Path] = None
    target_arch: str = ""
    toolchain: Mapping[str, str] = field(default_factory=dict)
    program_source: Optional[Callable[[str, Path], Path]] = None
    posture: compile_posture.CompilePosture = compile_posture.FLEET
    reserve: int = SERVING_RESERVE_CPUS
    window_s: float = DEFAULT_SILENCE_WINDOW_S
    vcpus: Optional[int] = None
    on_landed: Optional[Callable[[MintedHole], None]] = None
    work_list: Optional[Tuple[Any, ...]] = None
    _progress: Optional[MintProgress] = None

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.store is None:
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

    def run(self) -> MintOutcome:
        """Mint every hole, publishing and arming each as it lands."""
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
            progress.register_worker(threading.get_native_id())
            try:
                while not stop.is_set():
                    try:
                        hole = work.get_nowait()
                    except queue.Empty:
                        return
                    one(hole)
            finally:
                idle.set()

        threads = [
            threading.Thread(
                target=worker, name=f"runtime-mint-{i}", daemon=True)
            for i in range(width)
        ]
        for thread in threads:
            thread.start()

        sample_s = max(0.05, min(self.window_s / 8.0, 15.0))
        while any(thread.is_alive() for thread in threads):
            try:
                progress.judge()
            except MintCondemned as exc:
                logger.error("runtime-mint: %s", exc)
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

    def _mint_one(self, record: Any, arm_lock: threading.Lock) -> MintedHole:
        env = self.host.adoption.env
        scratch = self.artifacts_dir / env.value / "build" / record.graph
        scratch.parent.mkdir(parents=True, exist_ok=True)
        position = self.artifacts_dir / env.value / f"{record.graph}.so"

        graph = program_graph(record)
        if self.program_source is None:
            raise MissingProgram(
                f"graph {graph} has no program source on this mint (the store "
                f"exposes no `fetch_program` and no `program_source` was "
                f"stated)")
        blob = Path(self.program_source(
            graph, self.artifacts_dir / "programs" / f"{graph}.pt2"))

        assert self.compiler is not None
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

    def _renice(self) -> None:
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
