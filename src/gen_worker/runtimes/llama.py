"""llama.cpp serving runtime (gw#402).

GGUF checkpoint resolution, VRAM fit planning (n_gpu_layers / context from
the free-VRAM budget — degraded mode is fewer GPU layers, never a crash),
and OpenAI-compatible streaming helpers for ``llama-server`` endpoints.

Torch-free: works in llama-server-only serve images. The ``gguf`` reader
(core dep) is imported lazily so discovery stays cheap.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import msgspec

from ..api.errors import FatalError
from ..api.streaming import IncrementalTokenDelta, TokenUsage

logger = logging.getLogger(__name__)

_GiB = 1024**3
# Non-weight VRAM llama.cpp needs on top of layers + KV (compute graph,
# scratch buffers, CUDA pool). Conservative; sized for 7B-class contexts.
_OVERHEAD_GB = 1.0
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")

__all__ = [
    "GGUFInfo",
    "LlamaFitPlan",
    "chat_deltas",
    "completion_deltas",
    "free_vram_gb",
    "plan_fit",
    "read_gguf_info",
    "resolve_gguf",
]


# ---------------------------------------------------------------------------
# GGUF resolution: snapshot dir (Hub()-injected str/Path) -> the .gguf file
# ---------------------------------------------------------------------------


def _split_stem(path: Path) -> str:
    """Logical model stem: split shards of one model share a stem."""
    return _SPLIT_RE.sub(".gguf", path.name)


def resolve_gguf(source: Union[str, Path]) -> Path:
    """Resolve a model binding path to the single ``.gguf`` file to serve.

    A ``.gguf`` file path passes through. A directory (the usual Hub()
    snapshot dir) must contain exactly one logical GGUF model; split shards
    (``-00001-of-0000N.gguf``) count as one model and the first shard is
    returned (llama.cpp discovers the rest). Multiple distinct models fail
    closed — pin a flavor instead of guessing a quant.
    """
    p = Path(source)
    if p.suffix == ".gguf":
        return p
    if not p.is_dir():
        raise FatalError(f"model binding {str(p)!r} is neither a .gguf file nor a directory")
    ggufs = sorted(q for q in p.rglob("*.gguf") if q.is_file())
    if not ggufs:
        raise FatalError(f"no .gguf file found under snapshot dir {str(p)!r}")
    stems = sorted({_split_stem(q) for q in ggufs})
    if len(stems) > 1:
        raise FatalError(
            f"snapshot dir {str(p)!r} holds {len(stems)} distinct GGUF models "
            f"({', '.join(stems)}); pin the flavor to exactly one quant"
        )
    return ggufs[0]


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
    """Read fit metadata from a GGUF header. ``size_bytes`` sums all shards."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover - gguf is a core dep
        raise FatalError(
            "the 'gguf' package is required for the llama.cpp runtime; "
            "install with 'pip install gen-worker'"
        ) from exc

    gguf_path = Path(path)
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
    """K+V cache bytes per context token (f16 default; q8_0 ≈ 1.06)."""
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
    their share of the KV cache (the rest runs on CPU). Floor is 0 GPU
    layers — a plan is always returned, never an exception (fit-ladder
    contract: degraded, not dead).
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
            reason=f"full offload: {total_layers} layers + {kv_total // _GiB}GiB KV fit",
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
    from ..models.memory import get_available_vram_gb

    via_torch = get_available_vram_gb()
    if via_torch > 0:
        return via_torch
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
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
    """Best-effort plan for a checkpoint: None when the header is unreadable
    (caller falls back to llama.cpp defaults rather than failing the boot)."""
    try:
        info = read_gguf_info(gguf_path)
    except Exception as exc:
        logger.debug("gguf header unreadable for %s: %s", gguf_path, exc)
        return None
    budget = free_vram_gb() if vram_budget_gb is None else vram_budget_gb
    plan = plan_fit(info, free_vram_gb=budget, n_ctx=n_ctx)
    logger.info("llama fit plan for %s: %s", Path(gguf_path).name, plan.reason)
    return plan


# ---------------------------------------------------------------------------
# Streaming client (sync generators; the executor pumps sync gens in a
# thread). Yields IncrementalTokenDelta per token group, then one TokenUsage
# terminal signal for billing.
# ---------------------------------------------------------------------------


def _base_url(server: Any) -> str:
    url = getattr(server, "base_url", server)
    if not isinstance(url, str) or not url.startswith("http"):
        raise FatalError(f"expected a ServerHandle or base URL, got {server!r}")
    return url.rstrip("/")


def _delta_text(choice: dict[str, Any]) -> str:
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _stream(
    url: str,
    payload: dict[str, Any],
    cancelled: Optional[Callable[[], bool]],
    idle_timeout_s: float,
) -> Iterator[Union[IncrementalTokenDelta, TokenUsage]]:
    import requests

    prompt_tokens = completion_tokens = 0
    tokens_per_second = 0.0
    started = time.monotonic()
    with requests.post(
        url, json=payload, stream=True, timeout=(10.0, idle_timeout_s)
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if cancelled is not None and cancelled():
                return
            line = (raw or "").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("malformed llama-server stream chunk ignored: %.200r", data)
                continue
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
            timings = chunk.get("timings")
            if isinstance(timings, dict):
                tokens_per_second = float(timings.get("predicted_per_second") or 0.0)
                prompt_tokens = int(timings.get("prompt_n") or prompt_tokens)
                completion_tokens = int(timings.get("predicted_n") or completion_tokens)
            for choice in chunk.get("choices") or []:
                if isinstance(choice, dict):
                    text = _delta_text(choice)
                    if text:
                        yield IncrementalTokenDelta(text=text)
    if not tokens_per_second and completion_tokens:
        elapsed = max(time.monotonic() - started, 1e-6)
        tokens_per_second = completion_tokens / elapsed
    yield TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=round(tokens_per_second, 2),
    )


def chat_deltas(
    server: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[Sequence[str]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    idle_timeout_s: float = 300.0,
    extra: Optional[dict[str, Any]] = None,
) -> Iterator[Union[IncrementalTokenDelta, TokenUsage]]:
    """Stream a chat completion from llama-server as typed deltas + usage."""
    payload: dict[str, Any] = {
        "messages": list(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if stop:
        payload["stop"] = list(stop)
    if extra:
        payload.update(extra)
    yield from _stream(
        f"{_base_url(server)}/v1/chat/completions", payload, cancelled, idle_timeout_s
    )


def completion_deltas(
    server: Any,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[Sequence[str]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    idle_timeout_s: float = 300.0,
    extra: Optional[dict[str, Any]] = None,
) -> Iterator[Union[IncrementalTokenDelta, TokenUsage]]:
    """Stream a raw text completion from llama-server as typed deltas + usage."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if stop:
        payload["stop"] = list(stop)
    if extra:
        payload.update(extra)
    yield from _stream(
        f"{_base_url(server)}/v1/completions", payload, cancelled, idle_timeout_s
    )
