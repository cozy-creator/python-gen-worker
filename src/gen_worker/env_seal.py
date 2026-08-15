"""Execution-environment seal — ERASE, IMPOSE, SEAL THE DECLARATION.

Paul's env contract: the worker OWNS its process environment. It never audits
the world's env vars and refuses on surprises; it ERASES the behavior
namespaces wholesale and IMPOSES the declared configuration
(``settings_authority`` — the single writer of torch settings):

* :func:`scrub_env` — delete every var in the behavior namespaces
  (``TORCH*``/``PYTORCH*``/``TRITON*``/``CUBLAS*``/``CUDNN*``/
  ``NVIDIA_TF32*``/``OMP_*``/``MKL_*``), known or unknown; log the erased
  names; NEVER fail. Load-bearing order: the entrypoint calls it BEFORE
  torch imports (many vars are read at import/CUDA-init time). Plumbing
  (CUDA_VISIBLE_DEVICES, paths, credentials) is untouched.
* :func:`establish` — impose the declaration (env, torch flags + declared
  knobs, dynamo shape posture, host-ISA clamp, process posture), verify
  every read-back against it, fail closed on mismatch.
* :func:`effective_seal` — the seal is a digest of the DECLARATION:
  ``settings_authority.declaration()`` facts plus the loaded-library digest
  (the native ``.so`` set the python env ships — this closes the
  LD_PRELOAD/LD_LIBRARY_PATH substitution hole that env vars and a package's
  *own* metadata cannot see; each shipped lib's digest is derived from the
  wheel RECORD that installed it, and anything no RECORD covers is HASHED,
  which is exactly the preloaded-object case). Ambient mutation is
  structurally unable to move the digest — torch wheel defaults ride the
  ``toolchain`` key axis, and everything else in the codegen surface is either
  declared (sealed) or erased. ``seal_v`` versions the dict, so new sealed
  facts change digest VALUES only, never the axis set.
* :func:`assert_seal_unchanged` — the runtime TRIPWIRE, kept deliberately as
  read-back where the seal itself no longer is: boot stores the live read-back
  (posture, torch flags, dynamo facts, FULL portable inductor digest); every
  mint trace re-reads and REFUSES on drift, naming the fact and both values.
  Since boot verified read-back == declaration, any trip is also a declaration
  mismatch, so ambient mutation becomes a named refusal, never a silently
  different graph and never a different key. The per-call serving window is
  covered by dynamo's GlobalStateGuard plus the guard-miss doctrine.

The seal dict rides cell metadata verbatim (``artifact_metadata``) and is NOT
a key axis: the declaration digest and the loaded-libs digest fold into the
``toolchain`` axis instead (``compile_cache.toolchain_digest`` — "the compiler
as we configure it"), so a deliberate settings change re-keys through the axis
it honestly belongs to, while the boot verify and the pre-trace tripwire here
remain the GATES that make the fleet-wide single-declaration invariant true.
"""

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

# Versions the seal DICT. Adding, changing or deleting a sealed fact bumps
# THIS version only — never the key-axis set. A bump re-keys: every cell
# minted under the previous version stops matching and is re-minted once, per
# (family, lane, sm), so state the re-key cost when bumping it.
SEAL_VERSION = 4
SEAL_KEY = "env_seal"

# The behavior namespaces scrub_env ERASES wholesale. Known or unknown,
# hostile or informational: after the scrub, torch behavior is decided by
# CODE, never by whatever a base image or operator exported. Never an
# allowlist that REFUSES on an unrecognised var — erase-and-impose makes an
# unexpected var a deletion, never a fatal boot.
SCRUB_PREFIXES = (
    "TORCH",       # covers TORCH_*, TORCHINDUCTOR_*, TORCHDYNAMO_* ...
    "PYTORCH",     # incl. PYTORCH_CUDA_ALLOC_CONF and the build-info vars
    "TRITON",      # incl. TRITON_PTXAS_PATH (silently different cubins)
    "CUBLAS",      # workspace config alters cublas kernel splits
    "CUDNN",
    "NVIDIA_TF32",  # flips numerics under every torch flag
    "OMP_",        # thread counts enter cpp codegen decisions
    "MKL_",
    # Behavior namespaces a census of the installed torch/triton tree proved
    # are CONSULTED. Same doctrine: an ambient value is deleted, never
    # honored; what we need post-scrub is imposed
    # (settings_authority.DECLARED_ENV).
    "NCCL_",       # collective transport behavior (NVLS/P2P re-imposed by us)
    "ATEN_",       # ATEN_CPU_CAPABILITY flips CPU kernel dispatch
    "AOT_INDUCTOR",   # AOTI build/debug knobs (opt level, LTO, debug symbols)
    "AOTINDUCTOR",    # AOTI repro knobs
    "AOTI_",          # AOTI runtime knobs
    "AOT_PARTITIONER_DEBUG",
    "INDUCTOR_",   # inductor dump/provenance/test knobs
    "CUTLASS_", "CUTEDSL_",   # kernel-backend tuning
    "KMP_",        # Intel OpenMP runtime (same family as OMP_/MKL_)
    "CUDA_LAUNCH_BLOCKING",   # serializes every launch — behavior, not a path
    "CUDA_MODULE_LOADING",    # lazy/eager module load behavior
    "CUDA_PROFILE",
    "ENABLE_PERSISTENT_TMA_MATMUL",   # matmul kernel selection
    "ENABLE_TEMPLATE_TMA_STORE",
    "ENABLE_TMA_LOAD_FOR_TEMPLATE_EPILOGUE",
    "TENSORIFY_PYTHON_SCALARS",       # dynamo scalar handling
    "FAKE_ALLOW_META",                # functorch fake-tensor behavior
    "FX_PATCH_GETITEM",               # fx tracing behavior
    "PARTITIONER_MEMORY_BUDGET_PARETO",  # aot_autograd partitioner behavior
    "UNSAFE_SKIP_FSDP_MODULE_GUARDS",    # dynamo guard behavior
)


class EnvSealError(RuntimeError):
    """The live settings drifted from the boot/declared state, or the host-ISA
    clamp could not be imposed. Imposition failures raise
    ``settings_authority.SettingsImpositionError``."""


def scrub_env() -> List[str]:
    """ERASE every env var in the behavior namespaces. Logs and
    returns the erased names, sorted; NEVER raises — an unexpected var in a
    scrubbed namespace is deleted like any other, so a base-image export
    can neither refuse a boot nor change minted kernels. Load-bearing
    order: the entrypoint calls this BEFORE torch imports (many vars are
    read at import/CUDA-init); the SDK sets its own capture redirects
    (TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, TORCHINDUCTOR_AUTOGRAD_
    CACHE) AFTER the scrub. Plumbing (CUDA_VISIBLE_DEVICES, paths,
    credentials) is untouched."""
    erased = sorted(
        name for name in os.environ if name.startswith(SCRUB_PREFIXES))
    for name in erased:
        os.environ.pop(name, None)
    if erased:
        logger.info("env scrub (pgw#718): erased %d var(s): %s",
                    len(erased), ", ".join(erased))
    return erased


def effective_config() -> Dict[str, str]:
    """The LIVE values of the sealed config surface — the TRIPWIRE's config
    fact, never the seal's (the seal digests the declaration). The hash-seed
    facts record what this interpreter actually booted with;
    :func:`establish` refuses when they diverge from the declared
    ``PYTHONHASHSEED=0`` (imposition is the entrypoint's re-exec —
    ``settings_authority.ensure_interpreter_env``).

    A torchless worker has no matmul/cudnn surface to read back, so it reads
    the ABSENCE as a fact instead of crashing on the import."""
    base = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "hash_randomization": str(sys.flags.hash_randomization),
        # The host codegen target, read back live so a drifted clamp is named
        # legibly (beyond the opaque inductor digest below).
        **host_isa.effective(),
    }
    return {**settings_authority.torch_readback(), **base}


# Config entries torch MUTATES as a compile side effect — outputs, not knobs.
# `torch._inductor.aot_compile` writes machine facts (AOTI_CPU_ISA,
# AOTI_COMPUTE_CAPABILITY, ...) into the global `aot_inductor.metadata` and
# `save_config_portable()` includes it, so without this exclusion a process
# that has compiled digests differently from its own boot.
_PORTABLE_VOLATILE = ("aot_inductor.metadata",)


def inductor_config_digest() -> str:
    """Digest of torch's PORTABLE inductor config — the codegen surface a
    cell's kernels were minted under (machine-specific entries excluded by
    torch itself, torch's own compile-side-effect entries excluded here).
    ``"absent"`` on a torchless worker — a declared fact, so the seal digest
    stays meaningful for CPU cells."""
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


# Native libraries whose substitution changes compiled/served behavior:
# what /proc/self/maps says is ACTUALLY loaded, not what any package
# metadata claims — this closes the LD_PRELOAD/LD_LIBRARY_PATH hole.
_LIB_BASENAME_PREFIXES = (
    "libtorch", "libc10", "libcuda", "libcudart", "libcublas", "libcudnn",
    "libcufft", "libcusparse", "libcusolver", "libcupti", "libnvrtc",
    "libnvjitlink", "libtriton", "libnccl",
)

# Driver-side objects are NEVER identity: the manifest enumerates
# USERSPACE TOOLCHAIN libs only. The driver's userspace half (libcuda.so.*,
# libnvidia-*, libcudadebugger) is mounted from the HOST at pod start —
# it varies per machine and driver rollout, invisible to the image digest,
# and sealing it fractures cell keys per driver cohort (two libcuda patch
# levels split an L4 fleet and every worker kept self-minting). The driver
# stays a recorded-only metadata axis
# (`cuda_driver`); compiled kernels are driver-portable within a major.
# Note "libcuda" above still matches the image-shipped libcudart — the
# exclusion is by exact driver basenames, checked FIRST.
_DRIVER_LIB_BASENAME_PREFIXES = (
    "libcuda.so", "libcudadebugger", "libnvidia-", "libnvcuvid", "libnvoptix",
)


# Seam for tests: the loader map surface this process enumerates.
_MAPS_PATH = Path("/proc/self/maps")


@functools.lru_cache(maxsize=256)
def _lib_digest(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Where a library's identity digest comes from
# ---------------------------------------------------------------------------
# THREE sources, in this order, for the SAME value — the 16-hex prefix of the
# file's sha256. Which one served it changes the cost of a boot by 17 s and
# changes the seal by nothing:
#
# 1. The installing wheel's dist-info RECORD (:mod:`dist_records`):
#    identification stays CONTENT, derived from the declared manifest rather
#    than re-hashed. A KB-scale read covering, on the measured image, 36 of 36
#    shipped toolchain libraries.
# 2. The cross-process memo — the fallback STORE for whatever RECORD does not
#    cover, and what an entry-compile child reads so a 72-entry mint does not
#    re-pay the pass 72 times, K-wide, on the cores the compiles wanted (28 %
#    of per-entry compile_s, measured).
# 3. A full SHA-256 pass over the file — always correct, never skipped: a lib
#    covered by neither manifest above is HASHED, never assumed.
#
# Sources 1 and 2 are trusted under the same shape of guard (see
# :mod:`dist_records` for the exact statement of what that does and does not
# catch): the reader stats the file ITSELF and honours a claim only when the
# file it is looking at is the file the claim describes. Any mismatch, absence
# or unreadable manifest falls through to (3) for that file alone.
SEAL_LIB_MEMO_ENV = "GEN_WORKER_SEAL_LIB_MEMO"

_MEMO_V = 1

# Loaded once per process from SEAL_LIB_MEMO_ENV; {} when unset/unreadable.
_DISK_MEMO: Optional[Dict[str, str]] = None


def _memo_key(path: str, mtime_ns: int, size: int) -> str:
    # NUL-joined: a path can contain anything except NUL.
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
                memo = {}  # unreadable memo: full rehash, the safe path
        _DISK_MEMO = memo
    return _DISK_MEMO


class DigestSources(NamedTuple):
    """Where this process's library digests came from. Counted rather than
    derived: the interesting case is the PARTIAL one — a manifest covering
    most of the tree and leaving two files to hash — which a boolean hides."""

    record: int
    memo: int
    hashed: int


_SOURCES = DigestSources(0, 0, 0)


def digest_sources() -> DigestSources:
    """Per-source counts of the library-identity pass so far."""
    return _SOURCES


def _identity_digest(path: str, mtime_ns: int, size: int) -> str:
    """THE library-identity resolver: RECORD, then memo, then a full hash.
    Every caller (the identity manifest, the shipped-copy sets, the live
    substitution probe) goes through it, so one boot cannot mix a derived
    digest into one surface and a hashed digest into another."""
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
    """Persist this process's toolchain digests for short-lived children.

    This is the FALLBACK store: where RECORD covers a library the child
    derives the digest itself and never consults the memo, but a lib no wheel
    installed (a system object, a hand-patched ``.so``) would otherwise be
    re-hashed per child, so the parent banks what it paid.

    Cheap in a process that already sealed (the lru_cache is warm: the pass
    degenerates to stats); pays the full hash exactly once otherwise. The
    write is atomic (unique temp file + rename) so a reader never sees a torn
    file. Raises ``OSError`` on an unwritable destination — the CALLER decides
    whether that is worth a typed event; children fall back to the full
    rehash either way.

    The temp name is unique per WRITER, not derived from the destination, so
    the atomicity promise holds for whatever path arrives rather than
    depending on today's caller handing in a per-attempt path."""
    digests: Dict[str, str] = {}
    for _base, lib_paths in sorted(_toolchain_lib_paths().items()):
        for lib_path in lib_paths:
            try:
                st = os.stat(lib_path)
                digests[_memo_key(lib_path, st.st_mtime_ns, st.st_size)] = (
                    _identity_digest(lib_path, st.st_mtime_ns, st.st_size))
            except OSError:
                continue  # the child will record <unreadable> on its own stat
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
    """The loader-map probe as a TRI-STATE. ``readable`` says whether the map
    could be read AT ALL: an unreadable map yields no digests, and so does a
    process that mapped nothing relevant — the absence of a measurement is not
    a measurement of absence, and the two must not share a verdict
    (:func:`assert_seal_unchanged`)."""

    readable: bool
    digests: Tuple[Tuple[str, str], ...]


def loaded_library_probe() -> LoadedLibraries:
    """(basename, content digest) of every relevant native library the
    LOADER actually mapped into this process (``/proc/self/maps``), plus
    whether that map was readable at all. Deterministic: resolved real
    paths, sorted basenames. Off-Linux there is no maps surface, so the
    probe reports ``readable=False`` and nothing may be concluded from it.

    Resolved through :func:`_identity_digest` like the manifest it is compared
    against, so the comparison stays apples-to-apples and an LD_PRELOAD object
    — which no RECORD covers — is HASHED and therefore named."""
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
                continue  # host driver: recorded-only, never identity
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
    """The mapped-library digests alone. Callers that must distinguish an
    unreadable map from an empty one take :func:`loaded_library_probe`."""
    return loaded_library_probe().digests


# The IDENTITY manifest is enumerated from the python env ON DISK, never from
# /proc/self/maps. The mapped set is a function of LOAD PHASE (torch preloads
# cublas/cudnn at import; libtriton maps at first dynamo compile; libcuda at
# first CUDA call), so a frozen mapped-set snapshot freezes DIFFERENT sets in
# different consumers and cold-boot candidate keys can never match a published
# key. Disk enumeration is phase-independent by construction, and host driver
# objects can never appear in it (the driver is mounted from the host, not
# shipped in the python env). The maps-based probe above remains the LIVE
# integrity surface: assert_seal_unchanged compares what is actually mapped
# against this manifest and refuses, naming the library, when an
# LD_PRELOAD-style substitution diverges from the sealed disk content — or
# when the map could not be read, since a blind check is not a passing one.
_TOOLCHAIN_LIB_PACKAGES = ("torch", "triton", "nvidia")

# Test seam: when set, enumerate these directories instead of the resolved
# package roots.
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
    """{basename: [paths]} of every toolchain native lib the python env
    ships. Identity uses the FIRST path per basename (sorted-root order,
    deterministic); ``all_copies`` keeps every one — an env can ship the
    same basename twice with different bytes (cu126: triton/backends and
    nvidia/cuda_cupti both carry a ``libcupti.so.12``), and the live
    substitution check must not read the env's own second copy as an
    LD_PRELOAD."""
    paths: Dict[str, List[str]] = {}
    for root in _toolchain_lib_dirs():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.so*")):
            base = path.name
            if ".so" not in base or not base.startswith(_LIB_BASENAME_PREFIXES):
                continue
            if base.startswith(_DRIVER_LIB_BASENAME_PREFIXES):
                continue  # defense in depth: driver objects are never identity
            if path.is_symlink() or not path.is_file():
                continue  # digest real files once; alias links add nothing
            bucket = paths.setdefault(base, [])
            if all_copies or not bucket:
                bucket.append(str(path))
    return paths


def toolchain_library_digests() -> Tuple[Tuple[str, str], ...]:
    """(basename, content digest) of every userspace toolchain native lib
    the python env SHIPS — deterministic in the installed content,
    independent of what has been dlopened so far. Unreadable files record
    ``<unreadable>``. Enumeration and stat are always THIS process's own;
    only the per-file digest may come from a manifest (:func:`_identity_digest`
    — the installing wheel's RECORD, then the memo), and only when the file on
    disk is still the file that manifest describes."""
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
    """EVERY digest the env ships per basename — the live substitution
    check's reference set. Identity keeps the deterministic first copy; a
    mapped lib matching ANY shipped copy is the env's own file, not an
    LD_PRELOAD substitution."""
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


# The identity manifest is FROZEN at first computation. The disk content is
# already phase-independent, so the freeze is purely amortization — one
# identity pass per process — never a semantic phase pin.
_LIB_SNAPSHOT: Optional[Tuple[Tuple[str, str], ...]] = None

#: Seconds the last COLD identity pass took (telemetry only, read by
#: :func:`establish` into ``seal_libhash_s``). With RECORD coverage this
#: measures the manifest-and-stat pass; with none, the full SHA-256 pass —
#: which is the whole point of naming it.
_LAST_LIBHASH_S: float = 0.0


def frozen_library_digests() -> Tuple[Tuple[str, str], ...]:
    global _LIB_SNAPSHOT, _LAST_LIBHASH_S
    if _LIB_SNAPSHOT is None:
        t0 = time.monotonic()
        # THE library-identity phase. Reported as a boot phase, not just into
        # the seal dict: as a phase the hit and miss populations are two rows
        # of the same table and what a manifest saves is a subtraction.
        with boot_phases.span(boot_phases.PHASE_LIB_MEMO) as sp:
            _LIB_SNAPSHOT = toolchain_library_digests()
            src = digest_sources()
            dists, indexed = dist_records.coverage()
            # `hit`/`partial`/`miss`, never `refused` — all three are
            # successful outcomes of the same phase and the token is what a
            # hub-side count groups on. The token answers ONE question: did
            # this boot re-hash multi-GB of toolchain? The detail says which
            # manifest spared it, so a coverage regression is a number here
            # and not a slow boot nobody can explain.
            if not _LIB_SNAPSHOT:
                # No toolchain libraries enumerated at all (an env with no
                # torch/triton/nvidia packages). "nothing to hash" and "a
                # manifest served everything" are different answers and must
                # not share the token `hit`.
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


#: Declared knob overrides this process was established with — part of the
#: declaration, therefore part of the seal (a knob is keyed identity).
_ESTABLISHED_OVERRIDES: Optional[Dict[str, str]] = None


def loaded_libs_digest() -> str:
    """Combined 16-hex digest of the BOOT-frozen loaded-library snapshot:
    toolchain CONTENT the dist-info RECORDs cannot see — the
    LD_PRELOAD/LD_LIBRARY_PATH substitution hole. Rides the ``toolchain`` key
    axis and the seal dict; the per-library list rides metadata via
    ``compile_cache.artifact_metadata`` so a mismatch names the library."""
    libs_encoded = json.dumps(
        dict(frozen_library_digests()), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(libs_encoded).hexdigest()[:16]


def declaration_digest() -> str:
    """16-hex digest of the settings DECLARATION this process was
    established with (``settings_authority.declaration()`` + the declared
    knob overrides). This is the value that folds into the ``toolchain`` key
    axis ("the compiler as we configure it"), so a deliberate settings change
    re-keys through it. Deliberately excludes
    ``loaded_libs`` (a measured binaries fact with its own toolchain entry)
    and ``seal_v`` (the seal DICT's shape version, not a declaration
    fact)."""
    encoded = json.dumps(
        settings_authority.declaration(_ESTABLISHED_OVERRIDES),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def effective_seal() -> Dict[str, Any]:
    """The seal dict — a digest of the DECLARATION, recorded
    verbatim in cell metadata. Its settings facts come from
    ``settings_authority.declaration()``; ambient mutation cannot move them
    (it trips :func:`assert_seal_unchanged` instead). The one measured fact
    is ``loaded_libs`` (:func:`loaded_libs_digest`).

    The seal is NOT a key axis: its declaration and loaded-libs digests fold
    into the ``toolchain`` axis (``compile_cache.toolchain_digest``). This dict
    stays RECORDED on every artifact — the observable statement of the
    declaration a cell was minted under — and its digest stays on the published
    identity-axis map because the hub's ``ArtifactIdentity.env_seal_digest``
    requires it (a wire fact, like ``graph_contract``)."""
    return {
        "seal_v": SEAL_VERSION,
        **settings_authority.declaration(_ESTABLISHED_OVERRIDES),
        "loaded_libs": loaded_libs_digest(),
    }


def seal_digest(seal: Mapping[str, Any]) -> str:
    """The 16-hex digest of one seal dict — the artifact's ``env_seal`` wire
    fact (published identity-axis map + ``ArtifactIdentity``), not a key
    axis."""
    encoded = json.dumps(
        dict(seal), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def settings_readback() -> Dict[str, Any]:
    """The live settings state — the TRIPWIRE surface.

    Deliberately read-back where :func:`effective_seal` no longer is: the
    seal states the declaration; this states the process. The ``inductor``
    fact is the digest of the FULL portable config (wheel defaults included,
    torch-owned compile outputs excluded — ``_PORTABLE_VOLATILE``), so a
    mutation of an entry the declaration never names is still caught and
    refused by name, instead of silently forking the traced graph."""
    return {
        "posture": guard_closure.posture_snapshot(),
        "config": effective_config(),
        "dynamo": settings_authority.dynamo_readback(),
        "inductor": inductor_config_digest(),
    }


# The boot read-back: stored by establish() AFTER read-back was
# verified == declaration; every mint trace asserts the live state against
# it before tracing.
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
    """Point-of-use enforcement: the LIVE settings must still be
    the BOOT settings — and boot verified those against the declaration, so
    a trip is a declaration mismatch by transitivity. First call without an
    established boot read-back adopts the current state as boot
    (embedders/tests); any later drift refuses, naming the fact and both
    values — code mutating config/env behind our back becomes a named error,
    never a silently different graph, and NEVER a different key (the seal
    derives from the declaration and cannot follow the drift). The
    boot-frozen library snapshot is re-digested LIVE here: a substituted
    native lib (LD_PRELOAD-style, post-boot) is named even though the seal
    fact itself is frozen. Refuses too when that check could not RUN —
    unmeasurable is its own outcome, never folded into "unchanged"."""
    global _BOOT_READBACK
    live = settings_readback()
    if _BOOT_READBACK is None:
        _BOOT_READBACK = live
        return
    diffs = _seal_diff(_BOOT_READBACK, live)
    snapshot = dict(frozen_library_digests())
    # An env shipping no toolchain libs has nothing to substitute — the
    # torchless boot, not an unmeasured one.
    if snapshot:
        probe = loaded_library_probe()
        if not probe.readable:
            # UNMEASURABLE, not unchanged: the loader map is the only surface
            # that sees a substitution (the manifest is disk-derived and
            # structurally blind to LD_PRELOAD), so a map we cannot read
            # leaves every sealed library unchecked and must refuse.
            diffs.append(
                f"loader map {_MAPS_PATH} unreadable: {len(snapshot)} sealed "
                "libraries could not be checked for substitution")
        current = dict(probe.digests)
        shipped = _shipped_digest_sets()
        for base in sorted(snapshot):
            now = current.get(base)
            # Never mapped: nothing was loaded that could have been swapped.
            if now is None:
                continue
            if "<unreadable>" in (snapshot[base], now):
                # Two `<unreadable>` sides are equal without being a
                # comparison; naming that is the whole point of this row.
                diffs.append(
                    f"loaded lib {base}: content unverified (boot "
                    f"{snapshot[base]}, now {now}) — no comparison happened")
                continue
            if now == snapshot[base]:
                continue
            if now in shipped.get(base, ()):
                continue  # the env's own alternate copy, not a substitution
            diffs.append(
                f"loaded lib {base}: boot {snapshot[base]} != now {now} "
                "(native library substituted after boot)")
    if diffs:
        raise EnvSealError(
            "environment not verifiably unchanged since boot "
            f"({label or 'point-of-use'}): "
            + "; ".join(diffs))


#: What the LAST :func:`establish` call spent, per step. Telemetry only —
#: nothing reads it to decide anything and no digest depends on it. It exists
#: because ``establish()`` runs once per entry-compile CHILD, so on a 72-entry
#: mint its cost is multiplied by 72 inside the recorded ``compile_s``.
#: ``seal_libhash_s`` is the one that matters: hashing every toolchain ``.so``
#: the env ships measured 36 files / 3.96 GB / 10.6-17.3 s per PROCESS, and
#: deriving from the wheels' RECORDs turns it into a manifest read — this span
#: is how a coverage regression shows up.
LAST_ESTABLISH_SPANS: Dict[str, float] = {}


def establish(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The boot entry (entrypoint wiring): SCRUB the behavior namespaces,
    IMPOSE the declaration (env, torch flags + declared knobs, dynamo shape
    posture, host-ISA clamp, process posture), verify every read-back
    against it, store the boot read-back for the tripwire, return the seal.
    Never refuses on ambient env content — only on an imposition that does
    not take effect, an undeclared knob, or an interpreter that booted
    outside the declared env (``settings_authority.ensure_interpreter_env``
    is the imposition for that one)."""
    # THE envelope/toolchain/sm derivation phase. The span lives HERE, in the
    # function being measured, so every caller produces it: the entrypoint,
    # the mint child, and the in-process harness alike.
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
    # Re-impose OUR declared env after the scrub erased the whole namespace
    # (an ambient value is deleted, never honored), and refuse if the
    # interpreter itself booted outside the declaration (hash seed).
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
    # The library pass is timed where it runs (frozen_library_digests), not
    # inferred from `seal_effective_s`: under RECORD coverage the pass shrinks
    # to a manifest read while the rest of the seal (config read-back, inductor
    # digest) does not, and naming the wrong part "libhash" would hide exactly
    # the cost this span exists to expose. A split of `seal_effective_s`,
    # never a partition member.
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
