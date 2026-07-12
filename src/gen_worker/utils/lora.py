"""Per-request LoRA adapter overlays with adapter residency (gw#393 + gw#399).

``RunJob.models[].loras`` reaches the executor, which materializes each
adapter snapshot via the normal ``ensure_local`` path and parses + validates
the state dict (digest-keyed RAM LRU so repeat requests skip disk + parse).

Adapters stay ATTACHED to the resident pipeline (``load_lora_weights`` is the
expensive step — ~1s at SDXL scale, measured); each request only toggles the
ACTIVE SET: ``set_adapters(named list)`` + ``enable_lora`` on the way in
(~50ms), ``disable_lora`` on every exit path (~25ms). Nothing is ever active
unless the current request named it — zero-leakage by explicit activation.
Attached-but-inactive adapters are LRU-evicted under count/byte caps and
dropped when the pipeline demotes out of VRAM (re-attached lazily from the
AdapterCache on next use).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, runtime_checkable

from ..api.errors import RefCompatibilitySurprise, ValidationError

logger = logging.getLogger(__name__)

_GiB = 1024**3

# Worker-side sanity bounds. Weight bounds mirror the hub's `_models` gate;
# the rest guard a worker that receives a job from a buggy/bypassed hub.
MAX_LORAS_PER_REQUEST = 8
MAX_LORA_FILE_BYTES = 2 * _GiB
LORA_WEIGHT_BOUND = 4.0
ADAPTER_CACHE_MAX_BYTES = 1 * _GiB

# Residency caps for adapters left attached to a pipeline between requests.
# No deployment has ever overridden these (pgw#514 dead-config sweep found
# zero producers for the env vars that used to back them as Settings
# fields), so they're fixed constants.
MAX_ATTACHED_ADAPTERS = 8
MAX_ATTACHED_ADAPTER_BYTES = 2 * _GiB

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


def adapter_name(cache_key: str) -> str:
    """Stable diffusers adapter_name for one ``ref@digest`` — identical across
    requests so a repeat request reuses the already-attached adapter."""
    return "gw-" + hashlib.sha1(cache_key.encode()).hexdigest()[:12]


@dataclass
class PreparedAdapter:
    """One overlay, materialized and parsed, ready for GPU application."""

    slot: str
    ref: str
    cache_key: str  # ref@digest — the AdapterCache / attachment identity
    name: str       # stable diffusers adapter_name (adapter_name(cache_key))
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
# Adapter residency: attachments persist on the pipeline; requests toggle the
# active set (GPU side; run off the event loop)
# ---------------------------------------------------------------------------


@dataclass
class _PipeAttachments:
    """Adapters currently attached to one pipeline object."""

    pipe_id: int  # id(pipe) — detects object replacement after reload
    attached: "OrderedDict[str, tuple[str, int]]" = field(
        default_factory=OrderedDict)  # cache_key -> (adapter_name, bytes)
    active: bool = False  # an activation may be live (crash-leak guard)

    @property
    def total_bytes(self) -> int:
        return sum(b for _, b in self.attached.values())


class AdapterResidency:
    """Per-pipeline attachment registry (gw#399), keyed by model ref.

    ``activate`` attaches missing adapters (load_lora_weights — the ~1s step,
    paid once per adapter per pipeline), toggles the active set, and LRU-evicts
    attached-but-inactive adapters over the count/byte caps. ``deactivate``
    disables all adapters (never raises). Thread-safe; all pipeline calls run
    on the caller's (worker) thread."""

    def __init__(
        self,
        max_attached: int = MAX_ATTACHED_ADAPTERS,
        max_attached_bytes: int = MAX_ATTACHED_ADAPTER_BYTES,
    ) -> None:
        self._max = max(1, int(max_attached))
        self._max_bytes = int(max_attached_bytes)
        self._pipes: Dict[str, _PipeAttachments] = {}
        self._lock = threading.RLock()

    def _state(self, ref: str, pipe: Any) -> _PipeAttachments:
        st = self._pipes.get(ref)
        if st is None or st.pipe_id != id(pipe):
            # New or replaced pipeline object: prior attachments died with it.
            st = _PipeAttachments(pipe_id=id(pipe))
            self._pipes[ref] = st
        return st

    def activate(
        self, ref: str, pipe: Any, adapters: Sequence[PreparedAdapter],
        request_id: str = "",
    ) -> None:
        """Make exactly *adapters* the pipeline's active set. Attach failure
        rolls back to a fully-deactivated pipeline."""
        if not isinstance(pipe, LoraCapablePipeline):
            raise ValidationError(
                "model slot does not support LoRA adapters "
                "(pipeline lacks load_lora_weights/set_adapters/unload_lora_weights)"
            )
        with self._lock:
            st = self._state(ref, pipe)
            try:
                load_ms = 0
                attached_now: List[str] = []
                for a in adapters:
                    if a.cache_key in st.attached:
                        st.attached.move_to_end(a.cache_key)
                        continue
                    t0 = time.monotonic()
                    try:
                        # Shallow copy: diffusers' conversion utilities consume
                        # the dict; the cached entry must stay intact.
                        pipe.load_lora_weights(dict(a.state_dict), adapter_name=a.name)
                    except (ValidationError, RefCompatibilitySurprise):
                        raise
                    except Exception as exc:
                        raise RefCompatibilitySurprise(
                            f"adapter failed to load onto base pipeline: {exc}",
                            ref=a.ref, axis="pipeline_load",
                        ) from exc
                    load_ms += int((time.monotonic() - t0) * 1000)
                    st.attached[a.cache_key] = (a.name, state_dict_bytes(a.state_dict))
                    attached_now.append(a.name)
                t1 = time.monotonic()
                pipe.set_adapters(
                    [a.name for a in adapters],
                    adapter_weights=[a.weight for a in adapters],
                )
                # disable_lora (deactivate) flips a peft-level disable flag
                # that set_adapters alone does NOT clear — always re-enable.
                if hasattr(pipe, "enable_lora"):
                    pipe.enable_lora()
                set_ms = int((time.monotonic() - t1) * 1000)
                st.active = True
                self._evict_over_caps(st, pipe, keep={a.cache_key for a in adapters})
                logger.info(
                    "[request_id=%s] lora adapters active: %s (load_ms=%d set_ms=%d "
                    "attached=%d attached_bytes=%d)",
                    request_id,
                    "; ".join(
                        f"{a.ref}@{a.weight:g} "
                        f"[{'resident' if a.name not in attached_now else 'attach'}"
                        f" {'cache' if a.from_cache else 'cold'}"
                        f" ensure_ms={a.ensure_ms} parse_ms={a.parse_ms}]"
                        for a in adapters
                    ),
                    load_ms, set_ms, len(st.attached), st.total_bytes,
                )
            except BaseException:
                self.deactivate(ref, pipe, request_id=request_id)
                raise

    def deactivate(self, ref: str, pipe: Any, request_id: str = "") -> None:
        """Nothing active after this call (attachments stay). Never raises."""
        with self._lock:
            st = self._pipes.get(ref)
            if st is None:
                return
            if st.pipe_id != id(pipe):
                self._pipes.pop(ref, None)  # pipeline was replaced; state is stale
                return
            t0 = time.monotonic()
            try:
                if hasattr(pipe, "disable_lora"):
                    pipe.disable_lora()
                else:
                    pipe.unload_lora_weights()
                    st.attached.clear()
                st.active = False
                logger.info(
                    "[request_id=%s] lora adapters deactivated (disable_ms=%d attached=%d)",
                    request_id, int((time.monotonic() - t0) * 1000), len(st.attached),
                )
            except Exception:
                logger.warning(
                    "[request_id=%s] lora deactivate failed; pipeline may have "
                    "active adapters", request_id, exc_info=True,
                )

    def needs_deactivation(self, ref: str) -> bool:
        """Cheap guard for bare requests: True only when a previous request's
        activation may still be live on this ref's pipeline."""
        with self._lock:
            st = self._pipes.get(ref)
            return bool(st and st.active)

    def detach(self, ref: str, pipe: Any) -> None:
        """Drop every attachment from the pipeline (demotion out of VRAM);
        the AdapterCache re-attaches lazily on next use. Never raises."""
        with self._lock:
            st = self._pipes.pop(ref, None)
            if st is None or not st.attached or st.pipe_id != id(pipe):
                return
            try:
                pipe.unload_lora_weights()
                logger.info(
                    "lora attachments dropped on demote: ref=%s adapters=%d",
                    ref, len(st.attached),
                )
            except Exception:
                logger.warning("lora detach on demote failed for %s", ref, exc_info=True)

    def _evict_over_caps(self, st: _PipeAttachments, pipe: Any, keep: Set[str]) -> None:
        while len(st.attached) > self._max or (
            st.total_bytes > self._max_bytes and len(st.attached) > 1
        ):
            victim = next((k for k in st.attached if k not in keep), None)
            if victim is None:
                return
            name, _ = st.attached.pop(victim)
            try:
                pipe.delete_adapters(name)
                logger.info("lora attachment evicted (LRU): %s", victim)
            except Exception:
                logger.warning("lora eviction failed for %s", victim, exc_info=True)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                ref: {"adapters": len(st.attached), "bytes": st.total_bytes,
                      "active": st.active}
                for ref, st in self._pipes.items()
            }
