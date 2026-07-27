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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import guard_closure

logger = logging.getLogger(__name__)

# v3 (pgw#718/#719 erase-and-impose): recorded-env facts left (scrubbed
# vars are constants by construction); + hash-seed facts + loaded-library
# digest. v2 added the operator `epoch` salt. Adding sealed facts bumps
# THIS version only — never the key-axis set.
SEAL_VERSION = 3
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
    until then the facts make a divergent seed VISIBLE in the key."""
    import torch

    return {
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "hash_randomization": str(sys.flags.hash_randomization),
    }


def establish_config(
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Impose the canonical table (plus DECLARED knob overrides) via torch
    APIs, then verify the read-back. ``overrides`` is the typed-knob
    surface (pgw#718): keys must exist in :data:`CANONICAL_CONFIG` — the
    only route to non-canonical behavior is a declared knob, which is
    sealed and therefore keyed. An unknown knob refuses, named."""
    import torch

    table = dict(CANONICAL_CONFIG)
    if overrides:
        unknown = sorted(set(overrides) - set(table))
        if unknown:
            raise EnvSealError(
                f"unknown config knob(s) {unknown!r}: not in the canonical "
                "table (env_seal.CANONICAL_CONFIG) — declare the knob "
                "there first (one-way door: knobs in, env vars never)")
        table.update({k: str(v) for k, v in overrides.items()})
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
    torch itself)."""
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


@functools.lru_cache(maxsize=256)
def _lib_digest(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def loaded_library_digests() -> Tuple[Tuple[str, str], ...]:
    """(basename, content digest) of every relevant native library the
    LOADER actually mapped into this process (``/proc/self/maps``).
    Deterministic: resolved real paths, sorted basenames. Empty off-Linux
    (no maps surface — recorded as such)."""
    maps = Path("/proc/self/maps")
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


# The lib snapshot is FROZEN at first computation (boot phase): torch
# lazily dlopens cudnn/cublas at first use, so a LIVE probe would make a
# post-compile mint seal differ from every boot-time consumer seal — the
# key must digest the same phase on both sides. Substitution of a
# boot-mapped lib is still caught at point-of-use (assert_seal_unchanged
# re-digests the SNAPSHOT set live); post-boot additions join the identity
# at the ck6 hashing slice (the graphs they alter hash differently).
_LIB_SNAPSHOT: Optional[Tuple[Tuple[str, str], ...]] = None


def frozen_library_digests() -> Tuple[Tuple[str, str], ...]:
    global _LIB_SNAPSHOT
    if _LIB_SNAPSHOT is None:
        _LIB_SNAPSHOT = loaded_library_digests()
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


def establish(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The boot entry (entrypoint wiring): SCRUB the behavior namespaces,
    IMPOSE the canonical config (+ declared knobs) and posture, store the
    boot seal, return it. Never refuses on env content — only on an
    imposition that does not take effect or an undeclared knob."""
    global _BOOT_SEAL
    scrub_env()
    establish_config(overrides)
    guard_closure.establish_posture()
    _BOOT_SEAL = effective_seal()
    return dict(_BOOT_SEAL)


__all__ = [
    "CANONICAL_CONFIG",
    "EPOCH_ENV",
    "EnvSealError",
    "SCRUB_PREFIXES",
    "SEAL_KEY",
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
]
