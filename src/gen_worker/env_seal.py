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

SEAL_VERSION = 1
SEAL_KEY = "env_seal"

# Behavior-affecting global flags: ONE canonical value each, set explicitly
# by establish_config() and read back effective. String-valued for JSON
# determinism.
CANONICAL_CONFIG: Dict[str, str] = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": "False",
    "cudnn_allow_tf32": "False",
    "cudnn_benchmark": "False",
}

# CUBLAS workspace config alters cublas kernel splits: recorded into the
# seal (value participates in identity) but not canonicalized — bitwise
# numerical determinism is an explicit pgw#694 non-goal.
_RECORDED_ENV = ("CUBLAS_WORKSPACE_CONFIG",)

# TORCH* env vars that may be present without compromising mint/serve
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
    "TORCH_CUDA_ALLOC_CONF",     # allocator sizing — never enters a graph
    "TORCHINDUCTOR_CACHE_DIR",   # the SDK's own mint-capture redirect
    "TORCHINDUCTOR_AUTOGRAD_CACHE",  # set by the SDK's capture machinery
})


class EnvSealError(RuntimeError):
    """The process environment cannot be sealed: an unknown ``TORCH*``
    variable is present, or a canonical flag did not take effect."""


def check_torch_env(environ: Optional[Mapping[str, str]] = None) -> None:
    """Refuse unknown ``TORCH*`` env vars, naming each one (pgw#696)."""
    env = os.environ if environ is None else environ
    unknown = sorted(
        name for name in env
        if name.startswith("TORCH") and name not in ENV_ALLOWLIST
    )
    if unknown:
        raise EnvSealError(
            f"unknown TORCH* environment variable(s) {unknown!r}: not in "
            "the pgw#696 allowlist (env_seal.ENV_ALLOWLIST) — a stray "
            "inductor/dynamo toggle silently changes minted kernels; unset "
            "it or add it to the canonical table")


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
