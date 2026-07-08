"""Per-request LoRA adapter overlays (gw#393 BYOM).

``RunJob.models[].loras`` reaches the executor, which materializes each
adapter snapshot via the normal ``ensure_local`` path, parses + validates the
state dict (digest-keyed RAM LRU so repeat requests skip disk + parse), then
applies the adapters UNFUSED via ``load_lora_weights`` + ``set_adapters``
around the handler call and unloads them after — the base pipeline is
restored between requests (anima's hot-toggle pattern).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from ..api.errors import RefCompatibilitySurprise, ValidationError

logger = logging.getLogger(__name__)

_GiB = 1024**3

# Worker-side sanity bounds. Weight bounds mirror the hub's `_models` gate;
# the rest guard a worker that receives a job from a buggy/bypassed hub.
MAX_LORAS_PER_REQUEST = 8
MAX_LORA_FILE_BYTES = 2 * _GiB
LORA_WEIGHT_BOUND = 4.0
ADAPTER_CACHE_MAX_BYTES = 1 * _GiB

# LoRA-shaped keys only: kohya (`…lora_down.weight` / `…lora_up.weight` /
# `….alpha`), peft (`…lora_A.weight` / `…lora_B.weight`, DoRA magnitude), and
# diffusers attn-processor (`…lora.down.weight`) conventions. Anything else in
# an adapter file is key stuffing and is rejected before touching the model.
_LORA_KEY_RE = re.compile(
    r"(?:"
    r"\.(?:lora[._])?(?:down|up)\.weight"
    r"|\.lora_[AB](?:\.[\w-]+)?\.weight"
    r"|\.alpha"
    r"|\.dora_scale"
    r"|\.lora_magnitude_vector(?:\.[\w-]+)?(?:\.weight)?"
    r")$"
)


@runtime_checkable
class LoraCapablePipeline(Protocol):
    def load_lora_weights(self, *args: Any, **kwargs: Any) -> None: ...
    def set_adapters(self, *args: Any, **kwargs: Any) -> None: ...
    def unload_lora_weights(self) -> None: ...


@dataclass
class PreparedAdapter:
    """One overlay, materialized and parsed, ready for GPU application."""

    slot: str
    ref: str
    name: str  # unique-per-request diffusers adapter_name
    weight: float
    state_dict: Dict[str, Any]
    from_cache: bool = False
    ensure_ms: int = 0  # snapshot materialization (0 when already on disk)
    parse_ms: int = 0   # safetensors read + validation (0 on cache hit)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_overlay_weight(weight: float, *, ref: str = "") -> float:
    """Mirror the hub's [-4, 4] weight gate. 0.0 (proto3 unset) means 1.0."""
    w = float(weight)
    if not math.isfinite(w) or abs(w) > LORA_WEIGHT_BOUND:
        raise ValidationError(
            f"lora weight {w} out of bounds [-{LORA_WEIGHT_BOUND:g}, "
            f"{LORA_WEIGHT_BOUND:g}] (ref={ref})"
        )
    return w if w != 0.0 else 1.0


def validate_lora_keys(keys: Iterable[str], *, ref: str = "") -> None:
    bad = [k for k in keys if not _LORA_KEY_RE.search(k)]
    if bad:
        raise RefCompatibilitySurprise(
            f"adapter contains {len(bad)} non-LoRA key(s) "
            f"(e.g. {', '.join(sorted(bad)[:3])}) — only "
            "lora_down/up, lora_A/B, and alpha-shaped keys are accepted",
            ref=ref,
            axis="state_dict",
        )


def find_adapter_file(snapshot_path: Path, *, ref: str = "") -> Path:
    """The adapter payload inside a materialized snapshot: its (largest)
    ``.safetensors`` file. Safetensors-only — no pickle formats, ever."""
    p = Path(snapshot_path)
    if p.is_file():
        if p.suffix != ".safetensors":
            raise RefCompatibilitySurprise(
                f"adapter file is not safetensors: {p.name}",
                ref=ref, axis="component_missing",
            )
        return p
    files = sorted(p.rglob("*.safetensors"), key=lambda f: f.stat().st_size, reverse=True)
    if not files:
        raise RefCompatibilitySurprise(
            "adapter snapshot contains no .safetensors file",
            ref=ref, axis="component_missing",
        )
    return files[0]


def load_adapter_state_dict(path: Path, *, ref: str = "") -> Dict[str, Any]:
    """Parse + validate one adapter file. Injects missing kohya ``alpha``
    keys (alpha = rank) so diffusers doesn't error."""
    from safetensors.torch import load_file as load_safetensors
    import torch

    size = Path(path).stat().st_size
    if size > MAX_LORA_FILE_BYTES:
        raise ValidationError(
            f"lora adapter too large: {size} bytes (max {MAX_LORA_FILE_BYTES}) (ref={ref})"
        )
    try:
        state_dict = load_safetensors(str(path))
    except Exception as exc:
        raise RefCompatibilitySurprise(
            f"unreadable adapter safetensors: {exc}", ref=ref, axis="state_dict"
        ) from exc
    validate_lora_keys(state_dict.keys(), ref=ref)
    for key in list(state_dict.keys()):
        if key.endswith("lora_down.weight"):
            alpha_key = key[: -len("lora_down.weight")] + "alpha"
            if alpha_key not in state_dict:
                state_dict[alpha_key] = torch.tensor(float(state_dict[key].shape[0]))
    return state_dict


# ---------------------------------------------------------------------------
# Digest-keyed RAM cache of parsed state dicts
# ---------------------------------------------------------------------------


def state_dict_bytes(state_dict: Dict[str, Any]) -> int:
    total = 0
    for v in state_dict.values():
        n = getattr(v, "nbytes", 0)
        total += int(n or 0)
    return total


class AdapterCache:
    """LRU of parsed adapter state dicts keyed by ``ref@digest`` (RAM tier).

    LoRAs are small; a modest byte cap lets repeat requests skip disk + parse
    without competing with base-component residency. Thread-safe."""

    def __init__(self, max_bytes: int = ADAPTER_CACHE_MAX_BYTES) -> None:
        self._max = int(max_bytes)
        self._entries: "OrderedDict[str, tuple[Dict[str, Any], int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            hit = self._entries.get(key)
            if hit is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return hit[0]

    def put(self, key: str, state_dict: Dict[str, Any]) -> None:
        size = state_dict_bytes(state_dict)
        if size > self._max:
            return
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = (state_dict, size)
            self._bytes += size
            while self._bytes > self._max and len(self._entries) > 1:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Apply / unload (GPU side; run off the event loop)
# ---------------------------------------------------------------------------


def apply_adapters(
    pipeline: Any, adapters: Sequence[PreparedAdapter], request_id: str = ""
) -> None:
    """Load *adapters* onto *pipeline* as named UNFUSED adapters and activate
    them with their weights. Any failure rolls the pipeline back clean."""
    if not isinstance(pipeline, LoraCapablePipeline):
        raise ValidationError(
            "model slot does not support LoRA adapters "
            "(pipeline lacks load_lora_weights/set_adapters/unload_lora_weights)"
        )
    loaded: List[str] = []
    t0 = time.monotonic()
    try:
        for a in adapters:
            try:
                # Shallow copy: diffusers' conversion utilities consume the
                # dict they're given; the cached entry must stay intact.
                pipeline.load_lora_weights(dict(a.state_dict), adapter_name=a.name)
            except (ValidationError, RefCompatibilitySurprise):
                raise
            except Exception as exc:
                raise RefCompatibilitySurprise(
                    f"adapter failed to load onto base pipeline: {exc}",
                    ref=a.ref, axis="pipeline_load",
                ) from exc
            loaded.append(a.name)
        load_ms = int((time.monotonic() - t0) * 1000)
        t1 = time.monotonic()
        pipeline.set_adapters(
            [a.name for a in adapters], adapter_weights=[a.weight for a in adapters]
        )
        set_ms = int((time.monotonic() - t1) * 1000)
        logger.info(
            "[request_id=%s] lora adapters active: %s (load_ms=%d set_ms=%d)",
            request_id,
            "; ".join(
                f"{a.ref}@{a.weight:g} [{'cache' if a.from_cache else 'cold'}"
                f" ensure_ms={a.ensure_ms} parse_ms={a.parse_ms}]"
                for a in adapters
            ),
            load_ms, set_ms,
        )
    except BaseException:
        if loaded:
            unload_adapters(pipeline, request_id=request_id)
        raise


def unload_adapters(pipeline: Any, request_id: str = "") -> int:
    """Remove every request-scoped adapter from *pipeline* (guaranteed-clean
    teardown; never raises). Returns unload wall ms, -1 on failure."""
    t0 = time.monotonic()
    try:
        pipeline.unload_lora_weights()
    except Exception:
        logger.warning(
            "[request_id=%s] unload_lora_weights failed; pipeline may have stale adapters",
            request_id, exc_info=True,
        )
        return -1
    ms = int((time.monotonic() - t0) * 1000)
    logger.info("[request_id=%s] lora adapters unloaded (unload_ms=%d)", request_id, ms)
    return ms
