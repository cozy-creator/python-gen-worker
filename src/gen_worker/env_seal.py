"""Execution-environment seal — ERASE, IMPOSE, SEAL THE DECLARATION."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from . import (
    boot_phases, dist_records, guard_closure, host_isa, settings_authority,
    torch_capability,
)
import importlib.util

logger = logging.getLogger(__name__)

SEAL_VERSION = 4
SEAL_KEY = "env_seal"

SCRUB_PREFIXES = (
    "TORCH",
    "PYTORCH",
    "TRITON",
    "CUBLAS",
    "CUDNN",
    "NVIDIA_TF32",
    "OMP_",
    "MKL_",
    "NCCL_",
    "ATEN_",
    "AOT_INDUCTOR",
    "AOTINDUCTOR",
    "AOTI_",
    "AOT_PARTITIONER_DEBUG",
    "INDUCTOR_",
    "CUTLASS_", "CUTEDSL_",
    "KMP_",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "CUDA_PROFILE",
    "ENABLE_PERSISTENT_TMA_MATMUL",
    "ENABLE_TEMPLATE_TMA_STORE",
    "ENABLE_TMA_LOAD_FOR_TEMPLATE_EPILOGUE",
    "TENSORIFY_PYTHON_SCALARS",
    "FAKE_ALLOW_META",
    "FX_PATCH_GETITEM",
    "PARTITIONER_MEMORY_BUDGET_PARETO",
    "UNSAFE_SKIP_FSDP_MODULE_GUARDS",
)


class EnvSealError(RuntimeError):
    """The live settings drifted from the boot/declared state, or the host-ISA clamp could not be imposed."""


def scrub_env() -> List[str]:
    """ERASE every env var in the behavior namespaces."""
    erased = sorted(
        name for name in os.environ if name.startswith(SCRUB_PREFIXES))
    for name in erased:
        os.environ.pop(name, None)
    if erased:
        logger.info("env scrub (pgw#718): erased %d var(s): %s",
                    len(erased), ", ".join(erased))
    return erased


def effective_config() -> Dict[str, str]:
    """The LIVE values of the sealed config surface — the TRIPWIRE's config fact, never the seal's (the seal digests the declaration)."""
    base = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "hash_randomization": str(sys.flags.hash_randomization),
        **host_isa.effective(),
    }
    return {**settings_authority.torch_readback(), **base}


_PORTABLE_VOLATILE = ("aot_inductor.metadata",)


def inductor_config_digest() -> str:
    """Digest of torch's PORTABLE inductor config — the codegen surface a compiled graph's kernels were minted under (machine-specific entries excluded by torch itself, torch's own compile-side-effect ent..."""
    if not torch_capability.present():
        return torch_capability.ABSENT
    import torch._inductor.config as inductor_config

    portable = dict(inductor_config.save_config_portable())
    for key in _PORTABLE_VOLATILE:
        portable.pop(key, None)
    encoded = json.dumps(
        portable, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


_LIB_BASENAME_PREFIXES = (
    "libtorch", "libc10", "libcuda", "libcudart", "libcublas", "libcudnn",
    "libcufft", "libcusparse", "libcusolver", "libcupti", "libnvrtc",
    "libnvjitlink", "libtriton", "libnccl",
)

_DRIVER_LIB_BASENAME_PREFIXES = (
    "libcuda.so", "libcudadebugger", "libnvidia-", "libnvcuvid", "libnvoptix",
)


_MAPS_PATH = Path("/proc/self/maps")


@functools.lru_cache(maxsize=256)
def _lib_digest(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


SEAL_LIB_MEMO_ENV = "GEN_WORKER_SEAL_LIB_MEMO"

_MEMO_V = 1

_DISK_MEMO: Optional[Dict[str, str]] = None


def _memo_key(path: str, mtime_ns: int, size: int) -> str:
    return f"{path}\x00{mtime_ns}\x00{size}"


def _disk_memo() -> Dict[str, str]:
    global _DISK_MEMO
    if _DISK_MEMO is None:
        memo: Dict[str, str] = {}
        memo_path = os.environ.get(SEAL_LIB_MEMO_ENV, "")
        if memo_path:
            try:
                doc = json.loads(Path(memo_path).read_text())
                digests = doc.get("digests") if isinstance(doc, dict) else None
                if doc.get("memo_v") == _MEMO_V and isinstance(digests, dict):
                    memo = {str(k): str(v) for k, v in digests.items()}
            except (OSError, ValueError):
                memo = {}
        _DISK_MEMO = memo
    return _DISK_MEMO


class DigestSources(NamedTuple):
    """Where this process's library digests came from."""

    record: int
    memo: int
    hashed: int


_SOURCES = DigestSources(0, 0, 0)


def digest_sources() -> DigestSources:
    """Per-source counts of the library-identity pass so far."""
    return _SOURCES


def _identity_digest(path: str, mtime_ns: int, size: int) -> str:
    global _SOURCES
    recorded = dist_records.digest_for(path, mtime_ns, size)
    if recorded is not None:
        _SOURCES = _SOURCES._replace(record=_SOURCES.record + 1)
        return recorded
    memoized = _disk_memo().get(_memo_key(path, mtime_ns, size))
    if memoized is not None:
        _SOURCES = _SOURCES._replace(memo=_SOURCES.memo + 1)
        return memoized
    _SOURCES = _SOURCES._replace(hashed=_SOURCES.hashed + 1)
    return _lib_digest(path, mtime_ns, size)


def write_library_memo(path: Path) -> int:
    """Persist this process's toolchain digests for short-lived children."""
    digests: Dict[str, str] = {}
    for _base, lib_paths in sorted(_toolchain_lib_paths().items()):
        for lib_path in lib_paths:
            try:
                st = os.stat(lib_path)
                digests[_memo_key(lib_path, st.st_mtime_ns, st.st_size)] = (
                    _identity_digest(lib_path, st.st_mtime_ns, st.st_size))
            except OSError:
                continue
    encoded = json.dumps(
        {"memo_v": _MEMO_V, "digests": digests},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(encoded)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return len(digests)


class LoadedLibraries(NamedTuple):
    """The loader-map probe as a TRI-STATE."""

    readable: bool
    digests: Tuple[Tuple[str, str], ...]


def loaded_library_probe() -> LoadedLibraries:
    """(basename, content digest) of every relevant native library the LOADER actually mapped into this process (``/proc/self/maps``), plus whether that map was readable at all."""
    maps = _MAPS_PATH
    if not maps.is_file():
        return LoadedLibraries(False, ())
    paths: Dict[str, str] = {}
    try:
        for line in maps.read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) < 6 or not parts[5].startswith("/"):
                continue
            file_path = parts[5]
            base = os.path.basename(file_path)
            if ".so" not in base or not base.startswith(_LIB_BASENAME_PREFIXES):
                continue
            if base.startswith(_DRIVER_LIB_BASENAME_PREFIXES):
                continue
            paths[base] = os.path.realpath(file_path)
    except OSError:
        return LoadedLibraries(False, ())
    out: Dict[str, str] = {}
    for base in sorted(paths):
        try:
            st = os.stat(paths[base])
            out[base] = _identity_digest(
                paths[base], st.st_mtime_ns, st.st_size)
        except OSError:
            out[base] = "<unreadable>"
    return LoadedLibraries(True, tuple(sorted(out.items())))


def loaded_library_digests() -> Tuple[Tuple[str, str], ...]:
    """The mapped-library digests alone."""
    return loaded_library_probe().digests


_TOOLCHAIN_LIB_PACKAGES = ("torch", "triton", "nvidia")

_TOOLCHAIN_LIB_DIRS_OVERRIDE: Optional[Tuple[Path, ...]] = None


def _toolchain_lib_dirs() -> Tuple[Path, ...]:
    if _TOOLCHAIN_LIB_DIRS_OVERRIDE is not None:
        return _TOOLCHAIN_LIB_DIRS_OVERRIDE

    dirs: List[Path] = []
    for name in _TOOLCHAIN_LIB_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        roots = list(spec.submodule_search_locations or [])
        if not roots and spec.origin:
            roots = [os.path.dirname(spec.origin)]
        dirs.extend(Path(r) for r in roots if r)
    return tuple(dirs)


def _toolchain_lib_paths(
    all_copies: bool = False,
) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    for root in _toolchain_lib_dirs():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.so*")):
            base = path.name
            if ".so" not in base or not base.startswith(_LIB_BASENAME_PREFIXES):
                continue
            if base.startswith(_DRIVER_LIB_BASENAME_PREFIXES):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            bucket = paths.setdefault(base, [])
            if all_copies or not bucket:
                bucket.append(str(path))
    return paths


def toolchain_library_digests() -> Tuple[Tuple[str, str], ...]:
    """(basename, content digest) of every userspace toolchain native lib the python env SHIPS — deterministic in the installed content, independent of what has been dlopened so far."""
    out: Dict[str, str] = {}
    paths = _toolchain_lib_paths()
    for base in sorted(paths):
        try:
            st = os.stat(paths[base][0])
            out[base] = _identity_digest(
                paths[base][0], st.st_mtime_ns, st.st_size)
        except OSError:
            out[base] = "<unreadable>"
    return tuple(sorted(out.items()))


def _shipped_digest_sets() -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {}
    for base, lib_paths in _toolchain_lib_paths(all_copies=True).items():
        for lib_path in lib_paths:
            try:
                st = os.stat(lib_path)
                out.setdefault(base, []).append(_identity_digest(
                    lib_path, st.st_mtime_ns, st.st_size))
            except OSError:
                continue
    return {base: tuple(v) for base, v in out.items()}


_LIB_SNAPSHOT: Optional[Tuple[Tuple[str, str], ...]] = None

_LAST_LIBHASH_S: float = 0.0


def frozen_library_digests() -> Tuple[Tuple[str, str], ...]:
    global _LIB_SNAPSHOT, _LAST_LIBHASH_S
    if _LIB_SNAPSHOT is None:
        t0 = time.monotonic()
        with boot_phases.span(boot_phases.PHASE_LIB_MEMO) as sp:
            _LIB_SNAPSHOT = toolchain_library_digests()
            src = digest_sources()
            dists, indexed = dist_records.coverage()
            if not _LIB_SNAPSHOT:
                token = "no_libs"
            elif not src.hashed:
                token = "hit"
            elif not (src.record or src.memo):
                token = "miss"
            else:
                token = "partial"
            sp.classify(
                token,
                f"record={src.record} memo={src.memo} hashed={src.hashed} "
                f"libs={len(_LIB_SNAPSHOT)} dists={dists} recorded={indexed}")
        _LAST_LIBHASH_S = round(time.monotonic() - t0, 3)
    return _LIB_SNAPSHOT


_ESTABLISHED_OVERRIDES: Optional[Dict[str, str]] = None


def loaded_libs_digest() -> str:
    """Combined 16-hex digest of the BOOT-frozen loaded-library snapshot: toolchain CONTENT the dist-info RECORDs cannot see — the LD_PRELOAD/LD_LIBRARY_PATH substitution hole."""
    libs_encoded = json.dumps(
        dict(frozen_library_digests()), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(libs_encoded).hexdigest()[:16]


def declaration_digest() -> str:
    """16-hex digest of the settings DECLARATION this process was established with (``settings_authority.declaration()`` + the declared knob overrides)."""
    encoded = json.dumps(
        settings_authority.declaration(_ESTABLISHED_OVERRIDES),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def effective_seal() -> Dict[str, Any]:
    """The seal dict — a digest of the DECLARATION, recorded verbatim in compiled graph metadata."""
    return {
        "seal_v": SEAL_VERSION,
        **settings_authority.declaration(_ESTABLISHED_OVERRIDES),
        "loaded_libs": loaded_libs_digest(),
    }


def seal_digest(seal: Mapping[str, Any]) -> str:
    """The 16-hex digest of one seal dict — the artifact's ``env_seal`` wire fact (published identity-axis map + ``ArtifactIdentity``), not a key axis."""
    encoded = json.dumps(
        dict(seal), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def settings_readback() -> Dict[str, Any]:
    """The live settings state — the TRIPWIRE surface."""
    return {
        "posture": guard_closure.posture_snapshot(),
        "config": effective_config(),
        "dynamo": settings_authority.dynamo_readback(),
        "inductor": inductor_config_digest(),
    }


_BOOT_READBACK: Optional[Dict[str, Any]] = None


def _seal_diff(boot: Mapping[str, Any], live: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for fact in sorted(set(boot) | set(live)):
        b, n = boot.get(fact), live.get(fact)
        if isinstance(b, Mapping) and isinstance(n, Mapping):
            for sub in sorted(set(b) | set(n)):
                if b.get(sub) != n.get(sub):
                    out.append(
                        f"{fact}/{sub}: boot {b.get(sub)!r} != now {n.get(sub)!r}")
        elif b != n:
            out.append(f"{fact}: boot {b!r} != now {n!r}")
    return out


def assert_seal_unchanged(label: str = "") -> None:
    """Point-of-use enforcement: the LIVE settings must still be the BOOT settings — and boot verified those against the declaration, so a trip is a declaration mismatch by transitivity."""
    global _BOOT_READBACK
    live = settings_readback()
    if _BOOT_READBACK is None:
        _BOOT_READBACK = live
        return
    diffs = _seal_diff(_BOOT_READBACK, live)
    snapshot = dict(frozen_library_digests())
    if snapshot:
        probe = loaded_library_probe()
        if not probe.readable:
            diffs.append(
                f"loader map {_MAPS_PATH} unreadable: {len(snapshot)} sealed "
                "libraries could not be checked for substitution")
        current = dict(probe.digests)
        shipped = _shipped_digest_sets()
        for base in sorted(snapshot):
            now = current.get(base)
            if now is None:
                continue
            if "<unreadable>" in (snapshot[base], now):
                diffs.append(
                    f"loaded lib {base}: content unverified (boot "
                    f"{snapshot[base]}, now {now}) — no comparison happened")
                continue
            if now == snapshot[base]:
                continue
            if now in shipped.get(base, ()):
                continue
            diffs.append(
                f"loaded lib {base}: boot {snapshot[base]} != now {now} "
                "(native library substituted after boot)")
    if diffs:
        raise EnvSealError(
            "environment not verifiably unchanged since boot "
            f"({label or 'point-of-use'}): "
            + "; ".join(diffs))


LAST_ESTABLISH_SPANS: Dict[str, float] = {}


def establish(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The boot entry (entrypoint wiring): SCRUB the behavior namespaces, IMPOSE the declaration (env, torch flags + declared knobs, dynamo shape posture, host-ISA clamp, process posture), verify every re..."""
    with boot_phases.span(boot_phases.PHASE_ENV_ESTABLISH) as _sp:
        seal = _establish(overrides)
        _sp.note(
            f"digest={seal_digest(seal)} sm={seal.get('host_isa') or '-'} "
            + " ".join(f"{k}={v}" for k, v in sorted(LAST_ESTABLISH_SPANS.items())))
        return seal


def _establish(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    global _BOOT_READBACK, _ESTABLISHED_OVERRIDES

    spans: Dict[str, float] = {}
    marks = [time.monotonic()]

    def mark(name: str) -> None:
        marks.append(time.monotonic())
        spans[name] = round(marks[-1] - marks[-2], 3)

    scrub_env()
    settings_authority.impose_process_env()
    settings_authority.verify_interpreter_env()
    mark("seal_scrub_s")
    settings_authority.impose_torch(overrides)
    mark("seal_config_s")
    try:
        host_isa.impose()
    except host_isa.HostIsaError as exc:
        raise EnvSealError(str(exc)) from exc
    mark("seal_isa_s")
    settings_authority.impose_dynamo()
    mark("seal_dynamo_s")
    guard_closure.establish_posture()
    mark("seal_posture_s")
    cold_libs = _LIB_SNAPSHOT is None
    _ESTABLISHED_OVERRIDES = dict(overrides) if overrides else None
    _BOOT_READBACK = settings_readback()
    seal = effective_seal()
    mark("seal_effective_s")
    spans["seal_libhash_s"] = _LAST_LIBHASH_S if cold_libs else 0.0
    LAST_ESTABLISH_SPANS.clear()
    LAST_ESTABLISH_SPANS.update(spans)
    return seal


__all__ = [
    "LAST_ESTABLISH_SPANS",
    "DigestSources",
    "EnvSealError",
    "LoadedLibraries",
    "SCRUB_PREFIXES",
    "SEAL_KEY",
    "SEAL_LIB_MEMO_ENV",
    "SEAL_VERSION",
    "assert_seal_unchanged",
    "digest_sources",
    "effective_config",
    "effective_seal",
    "establish",
    "inductor_config_digest",
    "frozen_library_digests",
    "loaded_library_digests",
    "loaded_library_probe",
    "scrub_env",
    "seal_digest",
    "settings_readback",
    "write_library_memo",
]
