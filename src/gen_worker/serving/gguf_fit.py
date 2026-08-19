"""GGUF resolution and VRAM fit planning for the llama.cpp engine runtime.

Torch-free on purpose: a llama-server serve image carries no torch, and this
module is imported by :mod:`gen_worker.serving.engine_runtime` on that image.
The ``gguf`` reader is imported lazily so discovery stays cheap.

pgw#1421. This is the fit half of the deleted ``runtimes/llama.py``, carried
forward unchanged in substance. The STREAMING-CLIENT half of that module
(``chat_deltas``/``completion_deltas``) is deliberately NOT restored: both
fleet consumers roll their own ``httpx`` streaming against the engine's
OpenAI-compatible route, so restoring it would ship a surface with no caller.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

import msgspec

from ..api.errors import FatalError
from ..models.materialized_view import third_party_dir

logger = logging.getLogger(__name__)

_GiB = 1024**3
# Non-weight VRAM llama.cpp needs on top of layers + KV (compute graph,
# scratch buffers, CUDA pool). Conservative; sized for 7B-class contexts.
_OVERHEAD_GB = 1.0
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")

__all__ = [
    "GGUFInfo",
    "LlamaFitPlan",
    "free_vram_gb",
    "kv_bytes_per_token",
    "plan_fit",
    "plan_for",
    "read_gguf_info",
    "resolve_gguf",
]


# ---------------------------------------------------------------------------
# GGUF resolution: checkpoint tree -> the .gguf file
# ---------------------------------------------------------------------------


def _split_stem(path: Path) -> str:
    """Logical model stem: split shards of one model share a stem."""
    return _SPLIT_RE.sub(".gguf", path.name)


def resolve_gguf(source: Union[str, Path]) -> Path:
    """Resolve a checkpoint tree to the single ``.gguf`` file to serve.

    A ``.gguf`` file path passes through. A directory (``ctx.checkpoint_dir``)
    must contain exactly one logical GGUF model; split shards
    (``-00001-of-0000N.gguf``) count as one model and the first shard is
    returned (llama.cpp discovers the rest). Multiple distinct models fail
    closed — pin a flavor instead of guessing a quant.

    **The returned path is always a REAL file.** This is tier 3 of Paul's
    #1303 file-access ladder, narrowed by the 2026-08-19 ruling to exactly
    this class and PERMANENT there: everything downstream is llama.cpp — the
    ``llama-server -m`` argument, the ``gguf`` reader, and the ``st_size`` sum
    that sizes the fit. None of them can read our store, and a pod has no FUSE
    to give them file semantics.
    """
    p = Path(source)
    if p.suffix == ".gguf":
        return _real_shard_group(p)
    if not p.is_dir():
        raise FatalError(
            f"checkpoint {str(p)!r} is neither a .gguf file nor a directory"
        )
    ggufs = sorted(q for q in p.rglob("*.gguf") if q.is_file())
    if not ggufs:
        raise FatalError(f"no .gguf file found under checkpoint tree {str(p)!r}")
    stems = sorted({_split_stem(q) for q in ggufs})
    if len(stems) > 1:
        raise FatalError(
            f"checkpoint tree {str(p)!r} holds {len(stems)} distinct GGUF "
            f"models ({', '.join(stems)}); pin the flavor to exactly one quant"
        )
    return _real_shard_group(ggufs[0])


def _real_shard_group(gguf: Path) -> Path:
    """``gguf`` as a real file, with its shard siblings made real too.

    Materializes the CONTAINING DIRECTORY rather than the one file: llama.cpp
    opens ``-00002-of-0000N`` itself, having been told only about the first
    shard, so a per-file copy would hand it one real shard and N-1 stubs. On a
    tree that is not projected this returns ``gguf`` unchanged, so the
    non-projected paths (a bare download, a test fixture) are untouched.
    """
    return third_party_dir(
        gguf.parent, why="llama.cpp wants real GGUF files"
    ) / gguf.name


def _shard_group(gguf: Path) -> list[Path]:
    """All shards belonging to the same logical model as ``gguf``."""
    stem = _split_stem(gguf)
    siblings = [q for q in gguf.parent.glob("*.gguf") if _split_stem(q) == stem]
    return sorted(siblings) or [gguf]


# ---------------------------------------------------------------------------
# Header introspection (lazy `gguf` import)
# ---------------------------------------------------------------------------


class GGUFInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Fit-relevant facts from a GGUF header (weights never read)."""

    architecture: str = ""
    n_layers: int = 0
    n_ctx_train: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    size_bytes: int = 0


def _field_int(reader: Any, key: str) -> int:
    field = reader.get_field(key)
    if field is None:
        return 0
    try:
        return int(field.contents())
    except Exception:
        return 0


def read_gguf_info(path: Union[str, Path]) -> GGUFInfo:
    """Read fit metadata from a GGUF header. ``size_bytes`` sums all shards.

    ``size_bytes`` is what decides ``-ngl``, and it comes from ``st_size``.
    Over a projected tree a stub's ``st_size`` is ~128 bytes REGARDLESS of the
    model behind it, so the fit would conclude "everything fits" for a 30 GiB
    model and never degrade — silently, since nothing fails. The whole shard
    group is therefore made real first, and every stat below is taken on the
    real files.
    """
    try:
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover - gguf is a core dep
        raise FatalError(
            "the 'gguf' package is required for the llama.cpp engine runtime; "
            "install with 'pip install gen-worker'"
        ) from exc

    gguf_path = _real_shard_group(Path(path))
    reader = GGUFReader(str(gguf_path), "r")
    arch_field = reader.get_field("general.architecture")
    arch = str(arch_field.contents()) if arch_field is not None else ""
    n_head = _field_int(reader, f"{arch}.attention.head_count")
    n_head_kv = _field_int(reader, f"{arch}.attention.head_count_kv") or n_head
    return GGUFInfo(
        architecture=arch,
        n_layers=_field_int(reader, f"{arch}.block_count"),
        n_ctx_train=_field_int(reader, f"{arch}.context_length"),
        n_embd=_field_int(reader, f"{arch}.embedding_length"),
        n_head=n_head,
        n_head_kv=n_head_kv,
        size_bytes=sum(q.stat().st_size for q in _shard_group(gguf_path)),
    )


# ---------------------------------------------------------------------------
# Fit planning: free VRAM -> n_gpu_layers / n_ctx. Never raises.
# ---------------------------------------------------------------------------


class LlamaFitPlan(msgspec.Struct, frozen=True, kw_only=True):
    n_gpu_layers: int = 0
    n_ctx: int = 0
    degraded: bool = False
    reason: str = ""


def kv_bytes_per_token(info: GGUFInfo, *, bytes_per_elem: float = 2.0) -> int:
    """K+V cache bytes per context token (f16 default; q8_0 ~= 1.06)."""
    if not (info.n_layers and info.n_head and info.n_head_kv and info.n_embd):
        return 0
    head_dim = info.n_embd / info.n_head
    return int(info.n_layers * info.n_head_kv * head_dim * 2 * bytes_per_elem)


def plan_fit(
    info: GGUFInfo,
    *,
    free_vram_gb: float,
    n_ctx: Optional[int] = None,
    kv_bytes_per_elem: float = 2.0,
    overhead_gb: float = _OVERHEAD_GB,
) -> LlamaFitPlan:
    """Size ``-ngl`` / ``-c`` to the VRAM budget.

    Full offload when everything fits; otherwise as many layers as fit with
    their share of the KV cache (the rest runs on CPU). Floor is 0 GPU layers
    — a plan is always returned, never an exception (degrade-never-OOM: a
    tight card serves slowly and says so, it does not refuse).
    """
    ctx = int(n_ctx or info.n_ctx_train or 4096)
    if info.n_ctx_train:
        ctx = min(ctx, info.n_ctx_train)
    total_layers = info.n_layers + 1  # +1: output/embedding layer
    budget = free_vram_gb * _GiB - overhead_gb * _GiB
    if budget <= 0 or not info.n_layers or not info.size_bytes:
        why = "no VRAM budget" if budget <= 0 else "unknown model geometry"
        return LlamaFitPlan(
            n_gpu_layers=0, n_ctx=ctx, degraded=free_vram_gb > 0,
            reason=f"{why}; running all layers on CPU",
        )
    kv_total = kv_bytes_per_token(info, bytes_per_elem=kv_bytes_per_elem) * ctx
    if budget >= info.size_bytes + kv_total:
        return LlamaFitPlan(
            n_gpu_layers=total_layers, n_ctx=ctx, degraded=False,
            reason=(
                f"full offload: {total_layers} layers + "
                f"{kv_total // _GiB}GiB KV fit"
            ),
        )
    layer_bytes = info.size_bytes / total_layers
    kv_per_layer = kv_total / info.n_layers if info.n_layers else 0
    n = int(budget // (layer_bytes + kv_per_layer))
    n = max(0, min(n, total_layers))
    return LlamaFitPlan(
        n_gpu_layers=n, n_ctx=ctx, degraded=True,
        reason=(
            f"partial offload: {n}/{total_layers} layers fit in "
            f"{free_vram_gb:.1f}GiB free VRAM (ctx={ctx})"
        ),
    )


def free_vram_gb() -> float:
    """Free VRAM on device 0. Torch-free fallback: nvidia-smi (llama-server
    serve images carry no torch)."""
    try:
        from ..models.memory import get_available_vram_gb

        via_torch = get_available_vram_gb()
    except Exception:
        via_torch = 0.0
    if via_torch > 0:
        return via_torch
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        return float(out[0]) / 1024.0 if out else 0.0
    except Exception:
        return 0.0


def plan_for(
    gguf_path: Union[str, Path],
    *,
    vram_budget_gb: Optional[float] = None,
    n_ctx: Optional[int] = None,
) -> Optional[LlamaFitPlan]:
    """Best-effort plan for a checkpoint: ``None`` when the header is
    unreadable (the caller then falls back to llama.cpp's own defaults rather
    than failing the boot)."""
    try:
        info = read_gguf_info(gguf_path)
    except Exception as exc:
        logger.debug("gguf header unreadable for %s: %s", gguf_path, exc)
        return None
    budget = free_vram_gb() if vram_budget_gb is None else vram_budget_gb
    plan = plan_fit(info, free_vram_gb=budget, n_ctx=n_ctx)
    logger.info("llama fit plan for %s: %s", Path(gguf_path).name, plan.reason)
    return plan
