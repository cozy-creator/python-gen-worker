"""Execution-environment seal (pgw#696).

Compiled-cell identity depends on process environment facts no dynamo guard
row names: TF32 toggles, matmul precision, cudnn autotune, the inductor
config surface, and the pgw#695 process posture. This module freezes them:

* :data:`CANONICAL_CONFIG` is the ONE table of behavior-affecting flags;
  :func:`establish_config` sets every entry explicitly at boot and verifies
  the effective read-back (``cudnn.allow_tf32`` defaults TRUE on torch 2.13
  — a library default must never decide traced numerics).
* :func:`check_torch_env` allowlist-rejects unknown ``TORCH*`` env vars at
  boot, naming the variable: a stray ``TORCHINDUCTOR_*``/``TORCHDYNAMO_*``
  toggle silently changes minted kernels.
* :func:`effective_seal` snapshots posture + effective config + a digest of
  the portable inductor config into ONE versioned dict; its
  :func:`seal_digest` is the ``env_seal`` ck4 key axis. ``seal_v`` versions
  the dict itself, so adding a sealed fact later changes digest VALUES only
  — never the key-axis set: ck4 is the final planned scheme bump.

The seal dict rides cell metadata verbatim (``artifact_metadata``), so
``cell_key.from_artifact_metadata`` recomputes the axis from recorded facts
and a stamp can never disagree with the environment it summarizes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from . import guard_closure

# v2 (build-systems review R2/R7): + operator `epoch` salt (Bazel
# Action.salt / ccache HASH_PREFIX precedent — disowning a poisoned
# generation is one config change, never a scheme bump) and the widened
# recorded-env set. Adding sealed facts bumps THIS version only.
SEAL_VERSION = 2
SEAL_KEY = "env_seal"

# R2: the operator-settable generation salt. Bumping it disowns every cell
# minted under the previous epoch (their env_seal digests stop matching) —
# the recall lever for "a subtly broken image published cells" without a
# KEY_SCHEME bump. Set in the fleet env; default generation is "0".
EPOCH_ENV = "COZY_CELL_EPOCH"

# Behavior-affecting global flags: ONE canonical value each, set explicitly
# by establish_config() and read back effective. String-valued for JSON
# determinism.
CANONICAL_CONFIG: Dict[str, str] = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": "False",
    "cudnn_allow_tf32": "False",
    "cudnn_benchmark": "False",
}

# Value-semantic env recorded into the seal WITHOUT canonicalization (the
# pattern the build-systems review endorses — extend it, don't allowlist
# it): CUBLAS workspace alters cublas kernel splits; launch blocking and
# module loading change CUDA behavior; NVIDIA_TF32_OVERRIDE flips numerics
# under every torch flag; PYTHONHASHSEED can perturb codegen ordering.
# Path-shaped and host-shaped vars (LD_LIBRARY_PATH, OMP_NUM_THREADS, cache
# dirs) deliberately do NOT ride the seal — JAX's learned lesson: its own
# auto-set cache path was poisoning its own keys (PR notes in the review).
_RECORDED_ENV = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "NVIDIA_TF32_OVERRIDE",
    "PYTHONHASHSEED",
)

# Gated prefixes (R7): the behavior-affecting namespaces. The original
# gate matched only TORCH* — every PYTORCH_* var (including the LIVE
# allocator spelling PYTORCH_CUDA_ALLOC_CONF) evaded it, while the
# allowlist carried only the legacy TORCH_CUDA_ALLOC_CONF. TRITON_* is
# gated too: TRITON_PTXAS_PATH silently changes emitted cubins.
_GATED_PREFIXES = ("TORCH", "PYTORCH", "TRITON")

# Gated-namespace vars that may be present without compromising mint/serve
# determinism: storage locations, observability, and the SDK's own capture
# machinery. Everything else refuses boot, naming the variable.
ENV_ALLOWLIST = frozenset({
    "TORCH_HOME",
    "TORCH_LOGS",
    "TORCH_LOGS_FORMAT",
    "TORCH_LOGS_OUT",
    "TORCH_TRACE",
    "TORCH_EXTENSIONS_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_CUDA_ALLOC_CONF",       # allocator sizing (legacy spelling)
    "PYTORCH_CUDA_ALLOC_CONF",     # allocator sizing (live spelling)
    "PYTORCH_ALLOC_CONF",          # allocator sizing (2.13 preferred)
    "PYTORCH_NVML_BASED_CUDA_CHECK",  # probe-only: how availability is checked
    # Build-info constants stamped by the official pytorch/pytorch base
    # images (the fleet's serving base sets PYTORCH_VERSION=2.13.0). They
    # are informational, not toggles — but the R7 prefix widening made the
    # gate refuse them, which killed EVERY fleet pod at boot (silent
    # pod_exited before hello; sdxl 0.2.12 rollback, 2026-07-26).
    "PYTORCH_VERSION",
    "PYTORCH_BUILD_VERSION",
    "PYTORCH_BUILD_NUMBER",
    "TORCHINDUCTOR_CACHE_DIR",     # the SDK's own mint-capture redirect
    "TORCHINDUCTOR_AUTOGRAD_CACHE",  # set by the SDK's capture machinery
    "TRITON_CACHE_DIR",            # the SDK's own mint-capture redirect
})


class EnvSealError(RuntimeError):
    """The process environment cannot be sealed: an unknown gated-namespace
    variable is present, or a canonical flag did not take effect."""


def check_torch_env(environ: Optional[Mapping[str, str]] = None) -> None:
    """Refuse unknown ``TORCH*``/``PYTORCH*``/``TRITON*`` env vars, naming
    each one (pgw#696, widened by the build-systems review R7)."""
    env = os.environ if environ is None else environ
    unknown = sorted(
        name for name in env
        if name.startswith(_GATED_PREFIXES) and name not in ENV_ALLOWLIST
    )
    if unknown:
        raise EnvSealError(
            f"unknown gated environment variable(s) {unknown!r}: not in "
            "the pgw#696 allowlist (env_seal.ENV_ALLOWLIST) — a stray "
            "inductor/dynamo/triton toggle silently changes minted "
            "kernels; unset it or add it to the canonical table")


def effective_config() -> Dict[str, str]:
    """The live values of every sealed config flag (read back, never
    assumed)."""
    import torch

    config = {
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
    }
    for name in _RECORDED_ENV:
        config[name] = os.environ.get(name, "")
    return config


def establish_config() -> Dict[str, str]:
    """Set every canonical flag explicitly, then verify the read-back."""
    import torch

    torch.set_float32_matmul_precision(
        CANONICAL_CONFIG["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = (
        CANONICAL_CONFIG["cuda_matmul_allow_tf32"] == "True")
    torch.backends.cudnn.allow_tf32 = (
        CANONICAL_CONFIG["cudnn_allow_tf32"] == "True")
    torch.backends.cudnn.benchmark = (
        CANONICAL_CONFIG["cudnn_benchmark"] == "True")
    effective = effective_config()
    diffs: List[str] = [
        f"{name}: canonical {want!r} != effective {effective.get(name)!r}"
        for name, want in CANONICAL_CONFIG.items()
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


def effective_seal() -> Dict[str, Any]:
    """The live seal dict — recorded verbatim in cell metadata."""
    return {
        "seal_v": SEAL_VERSION,
        "epoch": os.environ.get(EPOCH_ENV, "0"),
        "posture": guard_closure.posture_snapshot(),
        "config": effective_config(),
        "inductor": inductor_config_digest(),
    }


def seal_digest(seal: Mapping[str, Any]) -> str:
    """The ``env_seal`` key-axis value for one seal dict."""
    encoded = json.dumps(
        dict(seal), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def establish() -> Dict[str, Any]:
    """The boot entry (executor wiring): refuse unknown TORCH* env, set the
    canonical config and posture, return the effective seal."""
    check_torch_env()
    establish_config()
    guard_closure.establish_posture()
    return effective_seal()


__all__ = [
    "CANONICAL_CONFIG",
    "ENV_ALLOWLIST",
    "EPOCH_ENV",
    "EnvSealError",
    "SEAL_KEY",
    "SEAL_VERSION",
    "check_torch_env",
    "effective_config",
    "effective_seal",
    "establish",
    "establish_config",
    "inductor_config_digest",
    "seal_digest",
]
