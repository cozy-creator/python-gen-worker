"""Execution-environment seal — ERASE AND IMPOSE (pgw#718/#719).

Paul's env contract: the worker OWNS its process environment. We do not
audit the world's env vars and refuse on surprises (the superseded #696
allowlist — it bit a 0.70.3 boot on an informational base-image var); we
ERASE the behavior namespaces wholesale and IMPOSE the canonical
configuration as code:

* :func:`scrub_env` — delete every var in the behavior namespaces
  (``TORCH*``/``PYTORCH*``/``TRITON*``/``CUBLAS*``/``CUDNN*``/
  ``NVIDIA_TF32*``/``OMP_*``/``MKL_*``), known or unknown; log the erased
  names; NEVER fail. Load-bearing order: the entrypoint calls it BEFORE
  torch imports (many vars are read at import/CUDA-init time). Plumbing
  (CUDA_VISIBLE_DEVICES, paths, credentials) is untouched.
* :data:`CANONICAL_CONFIG` + :func:`establish_config` — impose every
  behavior flag explicitly via torch APIs and verify the read-back. The
  canonical values ARE the ratified serving posture (pgw#654 TF32-on; see
  the table comment) — the point is that CODE decides them, never a
  library default and never an env var, and mint==serve by construction.
  The ONLY route to non-canonical behavior is a typed knob:
  ``establish_config(overrides=...)`` with keys validated against the
  canonical table — sealed, therefore keyed. One-way door: a scrubbed var
  that turns out to be needed becomes a knob, never an unscrub.
* :func:`effective_seal` — {seal_v, epoch, posture, config read-back,
  portable inductor digest, loaded-library digest (pgw#719: the native
  ``.so`` set actually mapped into the process — closes the
  LD_PRELOAD/LD_LIBRARY_PATH substitution hole that env vars and package
  RECORDs cannot see)}. After scrub+impose the seal is a pure function of
  (SDK build x declared knobs). Its :func:`seal_digest` is the ``env_seal``
  key axis; ``seal_v`` versions the dict, so new sealed facts change
  digest VALUES only, never the axis set.
* :func:`assert_seal_unchanged` — boot-vs-point-of-use (pgw#719): the boot
  seal is stored at :func:`establish`; every mint trace re-reads the
  effective state first and REFUSES on drift, naming the fact and both
  values (endpoint code mutating config/env behind our back becomes a
  named error, never a silently different graph). The per-call serving
  window is covered by dynamo's GlobalStateGuard + the pgw#680 guard-miss
  doctrine.

The seal dict rides cell metadata verbatim (``artifact_metadata``), so
``cell_key.from_artifact_metadata`` recomputes the axis from recorded facts
and a stamp can never disagree with the environment it summarizes.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import guard_closure, host_isa, torch_capability
import importlib.util

logger = logging.getLogger(__name__)

# v3 (pgw#718/#719 erase-and-impose): recorded-env facts left (scrubbed
# vars are constants by construction); + hash-seed facts + loaded-library
# digest. v2 added the operator `epoch` salt. v4 (pgw#745): host driver
# libs excluded from the loaded-lib manifest (gw#577: driver is never
# identity). v5 (pgw#749): the identity manifest is the python env's
# toolchain libs ON DISK — phase-independent — never the mapped set.
# Adding/changing sealed facts bumps THIS version only — never the
# key-axis set. v6 (pgw#754): host-ISA codegen clamp facts (cpp_march /
# cpp_simdlen) — deliberately retires every pre-clamp cell: they were
# compiled -march=native for their mint host's CPU and are not portable.
SEAL_VERSION = 6
SEAL_KEY = "env_seal"

# R2: the operator-settable generation salt. Bumping it disowns every cell
# minted under the previous epoch (their env_seal digests stop matching) —
# the recall lever for "a subtly broken image published cells" without a
# KEY_SCHEME bump. Set in the fleet env; default generation is "0".
EPOCH_ENV = "COZY_CELL_EPOCH"

# Behavior-affecting global flags: ONE canonical value each, set explicitly
# by establish_config() and read back effective. String-valued for JSON
# determinism.
#
# The canonical values ARE the ratified SERVING posture, not a preference:
# pgw#654 sets TF32 ON at executor bootstrap (bf16 compute path; TF32
# touches residual fp32 matmuls only), and inductor hashes the TF32 state
# (`cuda_matmul_settings`) into every inner FX key — so a mint sealed with
# TF32 off could never HIT in a pgw#654 serving process. The #719 drift
# check surfaced exactly that divergence live (every suite mint refused
# once an executor bootstrap ran); the seal's job is mint==serve
# consistency, so the table matches pgw#654. Note the 2.13 coupling:
# allow_tf32=True implies float32_matmul_precision "high".
CANONICAL_CONFIG: Dict[str, str] = {
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": "True",
    "cudnn_allow_tf32": "True",
    "cudnn_benchmark": "False",
}

# The behavior namespaces scrub_env ERASES wholesale (pgw#718). Known or
# unknown, hostile or informational: after the scrub, torch behavior is
# decided by CODE, never by whatever a base image or operator exported.
# This supersedes the #696 allowlist gate whose R7 widening refused the
# pytorch base image's informational PYTORCH_VERSION and killed EVERY
# fleet pod at boot (silent pod_exited before hello; sdxl 0.2.12 rollback,
# 2026-07-26) — erase-and-impose makes that failure class impossible: an
# unexpected var is deleted, never fatal.
SCRUB_PREFIXES = (
    "TORCH",       # covers TORCH_*, TORCHINDUCTOR_*, TORCHDYNAMO_* ...
    "PYTORCH",     # incl. PYTORCH_CUDA_ALLOC_CONF and the build-info vars
    "TRITON",      # incl. TRITON_PTXAS_PATH (silently different cubins)
    "CUBLAS",      # workspace config alters cublas kernel splits
    "CUDNN",
    "NVIDIA_TF32",  # flips numerics under every torch flag
    "OMP_",        # thread counts enter cpp codegen decisions
    "MKL_",
)


class EnvSealError(RuntimeError):
    """The environment could not be imposed (a canonical flag did not take
    effect, an unknown knob was declared) or drifted after boot."""


def scrub_env() -> List[str]:
    """ERASE every env var in the behavior namespaces (pgw#718). Logs and
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
    """The live values of every sealed config flag (read back, never
    assumed). Post-scrub these are a pure function of (SDK build x
    declared knobs) — no env var can reach them. The hash-seed facts
    record interpreter-level ordering entropy (pgw#719): canonical
    enforcement (PYTHONHASHSEED=0 pre-exec) is the entrypoint's wiring;
    until then the facts make a divergent seed VISIBLE in the key.

    pgw#788: a torchless worker has no matmul/cudnn surface to read back, so it
    seals the ABSENCE as a fact instead of crashing on the import."""
    torch = torch_capability.torch_or_none()
    if torch is None:
        return {
            "torch": torch_capability.ABSENT,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
            "hash_randomization": str(sys.flags.hash_randomization),
            **host_isa.effective(),
        }

    return {
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "hash_randomization": str(sys.flags.hash_randomization),
        # pgw#754: the host codegen target. Named here (beyond the opaque
        # inductor digest, which also covers cpp.march/cpp.simdlen) so a
        # cohort split is legible in the seal itself.
        **host_isa.effective(),
    }


def establish_config(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Impose the canonical table (plus DECLARED knob overrides) via torch
    APIs, then verify the read-back. ``overrides`` is the typed-knob
    surface (pgw#718): keys must exist in :data:`CANONICAL_CONFIG` — the
    only route to non-canonical behavior is a declared knob, which is
    sealed and therefore keyed. An unknown knob refuses, named.

    pgw#788: on a torchless worker there is nothing to impose. The knob names are
    still validated (that contract is torch-free) and the seal records
    ``torch: "absent"``. A DECLARED knob is a different matter: every canonical
    knob is a torch flag, so an endpoint that declares one in a torchless image
    is misconfigured, and honouring it silently would fork cell identity — so
    that refuses, named."""
    table = dict(CANONICAL_CONFIG)
    if overrides:
        unknown = sorted(set(overrides) - set(table))
        if unknown:
            raise EnvSealError(
                f"unknown config knob(s) {unknown!r}: not in the canonical "
                "table (env_seal.CANONICAL_CONFIG) — declare the knob "
                "there first (one-way door: knobs in, env vars never)")
        table.update({k: str(v) for k, v in overrides.items()})
    torch = torch_capability.torch_or_none()
    if torch is None:
        if overrides:
            raise EnvSealError(
                f"config knob(s) {sorted(overrides)!r} declared on a TORCHLESS "
                "worker: every canonical knob is a torch flag, so there is "
                "nothing to impose them on. Either ship torch in this image or "
                "drop the knob — honouring it silently would fork cell "
                "identity (pgw#788)")
        return effective_config()
    torch.set_float32_matmul_precision(table["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = (
        table["cuda_matmul_allow_tf32"] == "True")
    torch.backends.cudnn.allow_tf32 = (table["cudnn_allow_tf32"] == "True")
    torch.backends.cudnn.benchmark = (table["cudnn_benchmark"] == "True")
    effective = effective_config()
    diffs: List[str] = [
        f"{name}: imposed {want!r} != effective {effective.get(name)!r}"
        for name, want in table.items()
        if effective.get(name) != want
    ]
    if diffs:
        raise EnvSealError("config freeze failed: " + "; ".join(diffs))
    return effective


def inductor_config_digest() -> str:
    """Digest of torch's PORTABLE inductor config — the codegen surface a
    cell's kernels were minted under (machine-specific entries excluded by
    torch itself). ``"absent"`` on a torchless worker (pgw#788) — a declared
    fact, so the seal digest stays meaningful for CPU cells."""
    if not torch_capability.present():
        return torch_capability.ABSENT
    import torch._inductor.config as inductor_config

    portable = inductor_config.save_config_portable()
    encoded = json.dumps(
        portable, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


# Native libraries whose substitution changes compiled/served behavior:
# what /proc/self/maps says is ACTUALLY loaded, not what any package
# metadata claims (pgw#719 — closes the LD_PRELOAD/LD_LIBRARY_PATH hole).
_LIB_BASENAME_PREFIXES = (
    "libtorch", "libc10", "libcuda", "libcudart", "libcublas", "libcudnn",
    "libcufft", "libcusparse", "libcusolver", "libcupti", "libnvrtc",
    "libnvjitlink", "libtriton", "libnccl",
)

# Driver-side objects are NEVER identity (gw#577): the manifest enumerates
# USERSPACE TOOLCHAIN libs only. The driver's userspace half (libcuda.so.*,
# libnvidia-*, libcudadebugger) is mounted from the HOST at pod start —
# it varies per machine and driver rollout, invisible to the image digest,
# and sealing it fractures cell keys per driver cohort (pgw#745:
# libcuda.so.580.126.16 vs .580.159.04 split an L4 fleet; every worker
# kept self-minting). The driver stays a recorded-only metadata axis
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
# pgw#832: cross-process digest memo for short-lived workers
# ---------------------------------------------------------------------------
# The identity pass SHA-256s every toolchain .so the image ships (measured:
# 36 files, 3.96 GB, ~8 s). The lru_cache above amortizes it to once per
# PROCESS — which stopped being "once" the moment pgw#809's pool made the
# unit of parallelism a process that compiles one entry and exits: a
# 72-entry mint re-paid the pass 72 times, K-wide, on the cores the
# compiles wanted (28 % of per-entry compile_s, measured by pgw#830).
#
# The memo moves WHERE a digest comes from, never what it is. A parent that
# already paid the pass writes {(path, mtime_ns, size) -> digest} to a file
# (:func:`write_library_memo`); a child pointed at it via
# :data:`SEAL_LIB_MEMO_ENV` still enumerates the tree and stats every file
# ITSELF, and uses a memo digest only when its own (path, mtime_ns, size)
# matches an entry exactly — any mismatch, absence, or unreadable memo falls
# back to the full rehash of that file. So the seal is byte-identical to a
# full rehash in every detectable case, and the ONE undetectable case — a
# file rewritten with content of the same size and its mtime_ns restored —
# is exactly the case the in-process lru_cache (keyed on the same triple)
# has always trusted. The memo widens that existing trust boundary across
# processes; it does not create a new one.
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


def _lib_digest_memoized(path: str, mtime_ns: int, size: int) -> str:
    got = _disk_memo().get(_memo_key(path, mtime_ns, size))
    return got if got is not None else _lib_digest(path, mtime_ns, size)


def write_library_memo(path: Path) -> int:
    """Persist this process's toolchain digests for short-lived children.

    Cheap in a process that already sealed (the lru_cache is warm: the pass
    degenerates to stats); pays the full hash exactly once otherwise. The
    write is atomic (tmp + rename) so a reader never sees a torn file.
    Raises ``OSError`` on an unwritable destination — the CALLER decides
    whether that is worth a typed event; children fall back to the full
    rehash either way."""
    digests: Dict[str, str] = {}
    for _base, lib_path in sorted(_toolchain_lib_paths().items()):
        try:
            st = os.stat(lib_path)
            digests[_memo_key(lib_path, st.st_mtime_ns, st.st_size)] = (
                _lib_digest_memoized(lib_path, st.st_mtime_ns, st.st_size))
        except OSError:
            continue  # the child will record <unreadable> on its own stat
    encoded = json.dumps(
        {"memo_v": _MEMO_V, "digests": digests},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded)
    os.replace(tmp, path)
    return len(digests)


def loaded_library_digests() -> Tuple[Tuple[str, str], ...]:
    """(basename, content digest) of every relevant native library the
    LOADER actually mapped into this process (``/proc/self/maps``).
    Deterministic: resolved real paths, sorted basenames. Empty off-Linux
    (no maps surface — recorded as such)."""
    maps = _MAPS_PATH
    if not maps.is_file():
        return ()
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
        return ()
    out: Dict[str, str] = {}
    for base in sorted(paths):
        try:
            st = os.stat(paths[base])
            out[base] = _lib_digest(paths[base], st.st_mtime_ns, st.st_size)
        except OSError:
            out[base] = "<unreadable>"
    return tuple(sorted(out.items()))


# pgw#749: the IDENTITY manifest is enumerated from the python env ON DISK
# — never from /proc/self/maps. The mapped set is a function of LOAD PHASE
# (torch preloads cublas/cudnn at import; libtriton maps at first dynamo
# compile; libcuda at first CUDA call), so a "frozen at first computation"
# mapped-set snapshot froze DIFFERENT sets in different consumers: cold
# candidate computations sealed the import-time set while mints sealed the
# compile-warm set, and cold-boot candidate keys could NEVER match any
# published key (boot-attach adoption structurally dead — the th#1216
# requested_unresolvable evidence). Disk enumeration is phase-independent
# by construction, and host driver objects can never appear in it (the
# driver is mounted from the host, not shipped in the python env —
# gw#577/pgw#745 by construction). The maps-based probe above remains the
# LIVE integrity surface: assert_seal_unchanged compares what is actually
# mapped against this manifest and refuses, naming the library, when an
# LD_PRELOAD-style substitution diverges from the sealed disk content.
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


def _toolchain_lib_paths() -> Dict[str, str]:
    """{basename: path} of every toolchain native lib the python env ships.
    Duplicate basenames resolve by sorted path (deterministic)."""
    paths: Dict[str, str] = {}
    for root in _toolchain_lib_dirs():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.so*")):
            base = path.name
            if ".so" not in base or not base.startswith(_LIB_BASENAME_PREFIXES):
                continue
            if base.startswith(_DRIVER_LIB_BASENAME_PREFIXES):
                continue  # defense in depth; see pgw#745
            if path.is_symlink() or not path.is_file():
                continue  # digest real files once; alias links add nothing
            paths.setdefault(base, str(path))
    return paths


def toolchain_library_digests() -> Tuple[Tuple[str, str], ...]:
    """(basename, content digest) of every userspace toolchain native lib
    the python env SHIPS — deterministic in the installed content,
    independent of what has been dlopened so far. Unreadable files record
    ``<unreadable>``. Enumeration and stat are always THIS process's own;
    only the per-file digest may come from the pgw#832 memo, and only on an
    exact (path, mtime_ns, size) match."""
    out: Dict[str, str] = {}
    paths = _toolchain_lib_paths()
    for base in sorted(paths):
        try:
            st = os.stat(paths[base])
            out[base] = _lib_digest_memoized(
                paths[base], st.st_mtime_ns, st.st_size)
        except OSError:
            out[base] = "<unreadable>"
    return tuple(sorted(out.items()))


# The identity manifest is FROZEN at first computation: the disk content is
# already phase-independent (pgw#749), so the freeze is purely an
# amortization — one hashing pass per process (multi-GB cold, near-free when
# a pgw#832 memo serves the digests), never a semantic phase pin.
_LIB_SNAPSHOT: Optional[Tuple[Tuple[str, str], ...]] = None

#: Seconds the last COLD identity pass took (telemetry only, read by
#: :func:`establish` into ``seal_libhash_s``). With a memo this measures the
#: stat-and-lookup pass; without one, the full SHA-256 pass — which is the
#: whole point of naming it.
_LAST_LIBHASH_S: float = 0.0


def frozen_library_digests() -> Tuple[Tuple[str, str], ...]:
    global _LIB_SNAPSHOT, _LAST_LIBHASH_S
    if _LIB_SNAPSHOT is None:
        t0 = time.monotonic()
        _LIB_SNAPSHOT = toolchain_library_digests()
        _LAST_LIBHASH_S = round(time.monotonic() - t0, 3)
    return _LIB_SNAPSHOT


def effective_seal() -> Dict[str, Any]:
    """The live seal dict — recorded verbatim in cell metadata. The
    loaded-libs FACT is the combined digest of the BOOT-frozen library
    snapshot (identity); the per-library list rides metadata via
    ``compile_cache.artifact_metadata`` so a mismatch names the library."""
    libs_encoded = json.dumps(
        dict(frozen_library_digests()), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return {
        "seal_v": SEAL_VERSION,
        "epoch": os.environ.get(EPOCH_ENV, "0"),
        "posture": guard_closure.posture_snapshot(),
        "config": effective_config(),
        "inductor": inductor_config_digest(),
        "loaded_libs": hashlib.sha256(libs_encoded).hexdigest()[:16],
    }


def seal_digest(seal: Mapping[str, Any]) -> str:
    """The ``env_seal`` key-axis value for one seal dict."""
    encoded = json.dumps(
        dict(seal), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


# The boot seal (pgw#719): stored by establish(); every mint trace asserts
# the live state against it before tracing.
_BOOT_SEAL: Optional[Dict[str, Any]] = None


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
    """Point-of-use enforcement (pgw#719): the effective environment must
    still be the BOOT environment. First call without an established boot
    seal adopts the current state as boot (embedders/tests); any later
    drift refuses, naming the fact and both values — endpoint code
    mutating config/env behind our back becomes a named error, never a
    silently different graph. The boot-frozen library snapshot is
    re-digested LIVE here: a substituted native lib (LD_PRELOAD-style,
    post-boot) is named even though the seal fact itself is frozen."""
    global _BOOT_SEAL
    live = effective_seal()
    if _BOOT_SEAL is None:
        _BOOT_SEAL = live
        return
    diffs = _seal_diff(_BOOT_SEAL, live)
    snapshot = dict(frozen_library_digests())
    if snapshot:
        current = dict(loaded_library_digests())
        for base in sorted(snapshot):
            now = current.get(base)
            if now is not None and now != snapshot[base]:
                diffs.append(
                    f"loaded lib {base}: boot {snapshot[base]} != now {now} "
                    "(native library substituted after boot)")
    if diffs:
        raise EnvSealError(
            f"environment drifted since boot ({label or 'point-of-use'}): "
            + "; ".join(diffs))


#: pgw#830: what the LAST :func:`establish` call spent, per step. Telemetry
#: only — nothing reads it to decide anything, and no digest depends on it.
#: It exists because ``establish()`` is called once per pgw#809 entry-compile
#: CHILD, so on a 72-entry mint its cost is multiplied by 72 and lands inside
#: the recorded ``compile_s`` with no name. ``seal_libhash_s`` is the one that
#: matters: the identity manifest SHA-256s every toolchain ``.so`` the env
#: ships (measured off-pod: 36 files, 3.96 GB, ~8 s at 0.49 GB/s), and the
#: memo that makes it "once" is an lru_cache — per PROCESS, so a pool of
#: short-lived children re-pays it in full every time.
LAST_ESTABLISH_SPANS: Dict[str, float] = {}


def establish(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The boot entry (entrypoint wiring): SCRUB the behavior namespaces,
    IMPOSE the canonical config (+ declared knobs) and posture, store the
    boot seal, return it. Never refuses on env content — only on an
    imposition that does not take effect or an undeclared knob."""
    global _BOOT_SEAL

    spans: Dict[str, float] = {}
    marks = [time.monotonic()]

    def mark(name: str) -> None:
        marks.append(time.monotonic())
        spans[name] = round(marks[-1] - marks[-2], 3)

    scrub_env()
    mark("seal_scrub_s")
    establish_config(overrides)
    mark("seal_config_s")
    try:
        host_isa.impose()
    except host_isa.HostIsaError as exc:
        raise EnvSealError(str(exc)) from exc
    mark("seal_isa_s")
    guard_closure.establish_posture()
    mark("seal_posture_s")
    cold_libs = _LIB_SNAPSHOT is None
    _BOOT_SEAL = effective_seal()
    mark("seal_effective_s")
    # The library pass is timed where it runs (frozen_library_digests), not
    # inferred from `seal_effective_s`: with a pgw#832 memo the pass shrinks
    # to stats while the rest of the seal (config read-back, inductor digest)
    # does not, and naming the wrong part "libhash" would hide exactly the
    # cost this span exists to expose. Still a split of `seal_effective_s`,
    # never a partition member.
    spans["seal_libhash_s"] = _LAST_LIBHASH_S if cold_libs else 0.0
    LAST_ESTABLISH_SPANS.clear()
    LAST_ESTABLISH_SPANS.update(spans)
    return dict(_BOOT_SEAL)


__all__ = [
    "CANONICAL_CONFIG",
    "EPOCH_ENV",
    "LAST_ESTABLISH_SPANS",
    "EnvSealError",
    "SCRUB_PREFIXES",
    "SEAL_KEY",
    "SEAL_LIB_MEMO_ENV",
    "SEAL_VERSION",
    "assert_seal_unchanged",
    "effective_config",
    "effective_seal",
    "establish",
    "establish_config",
    "inductor_config_digest",
    "frozen_library_digests",
    "loaded_library_digests",
    "scrub_env",
    "seal_digest",
    "write_library_memo",
]
