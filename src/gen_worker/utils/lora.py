"""Per-request LoRA adapter overlays with adapter residency.

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

from .. import activity as activity_mod
from ..component_vocab import denoiser_components, text_encoder_components
from ..api.errors import RefCompatibilitySurprise, ValidationError
from dataclasses import replace
from ..models import lora_lifted, w8a8_lora
from ..models.projection import logical_size
from ..models.tensor_source import load_state_dict

logger = logging.getLogger(__name__)

_GiB = 1024**3

# Worker-side sanity bounds. Weight bounds mirror the hub's `_models` gate;
# the rest guard a worker that receives a job from a buggy/bypassed hub.
MAX_LORAS_PER_REQUEST = 8
MAX_LORA_FILE_BYTES = 2 * _GiB
LORA_WEIGHT_BOUND = 4.0
ADAPTER_CACHE_MAX_BYTES = 1 * _GiB

# Residency caps for adapters left attached to a pipeline between requests.
# No deployment has ever overridden these, so they're fixed constants.
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


# Normalized+split adapter halves, keyed by (pipeline class, denoiser-config
# fingerprint, cache_key). The cached den_sd OBJECT is reused across requests
# so the branch layer's staging cache (keyed on id(sd)) hits — repeat swaps
# skip key-mapping AND the CPU flatten. Split tensors alias the AdapterCache
# entries (converters rename keys, they don't copy data), but the cap stays
# small so this cache can never pin many evicted adapters by itself.
# Guarded by its own lock: activate() calls this BEFORE taking the residency
# lock, and multi-slot workers activate from several threads.
_SPLIT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SPLIT_CACHE_MAX = 8
_SPLIT_CACHE_LOCK = threading.Lock()
# Component prefix -> pipe attribute, for both normalized and kohya-flat key
# grammars. Kohya te1/te2 numbering follows diffusers' convention.
# The kohya-flat te aliases are sd-scripts' own grammar, not our vocabulary,
# so they stay literal; they map ONTO vocabulary names.
_KOHYA_TE_ALIASES = (
    ("lora_te3_", "text_encoder_3"),
    ("lora_te2_", "text_encoder_2"),
    ("lora_te1_", "text_encoder"),
    ("lora_te_", "text_encoder"),
)


def te_prefix_to_component() -> tuple[tuple[str, str], ...]:
    """Component prefix -> pipe attribute, longest first so ``text_encoder_2``
    wins over ``text_encoder``. Read at call time.

    Shared with :mod:`gen_worker.models.lora_fold`, which needs the same table
    to route an adapter's text-encoder half — one alias table, not two."""
    dotted = tuple(
        (c, c) for c in sorted(text_encoder_components(), key=len, reverse=True)
    )
    return dotted + _KOHYA_TE_ALIASES


# Config fields that DISCRIMINATE denoiser structure. It must cover DiTs as
# well as UNets: a UNet-only set fingerprints every transformer to the same
# empty value, so two structurally different DiTs would share one
# normalized-split cache entry. The fingerprint takes whatever the module
# actually declares, and says so when it declares nothing.
_FINGERPRINT_FIELDS = (
    # UNet-shaped
    "down_block_types", "up_block_types", "block_out_channels",
    "layers_per_block", "cross_attention_dim", "attention_head_dim",
    "addition_embed_type", "transformer_layers_per_block",
    # DiT-shaped
    "num_layers", "num_attention_heads", "attention_head_dim",
    "num_single_layers", "in_channels", "out_channels", "joint_attention_dim",
    "pooled_projection_dim", "patch_size", "axes_dims_rope", "caption_channels",
)


def _denoiser_fingerprint(pipe: Any) -> str:
    """Kohya/SGM normalization consults the denoiser's config — two
    checkpoints sharing a pipeline class but differing in block layout must
    not share a normalized-split cache entry.

    A fingerprint that resolves to nothing is NOT a cache key — it is a
    collision — so a module whose config declares none of these fields falls
    back to its class name plus its parameter shape signature rather than to
    an empty string.
    """
    for name in denoiser_components():
        module = getattr(pipe, name, None)
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        present = [
            f"{key}={getattr(cfg, key)!s}"
            for key in dict.fromkeys(_FINGERPRINT_FIELDS)
            if getattr(cfg, key, None) is not None
        ]
        if present:
            return f"{type(module).__name__}|" + "|".join(present)
        # Nothing recognized: never return a value that every module shares.
        return f"{type(module).__name__}|{_shape_signature(module)}"
    return ""


def _shape_signature(module: Any) -> str:
    """A structural last resort: how many parameters, and their shapes.
    Distinguishes two same-class denoisers whose configs are unreadable."""
    try:
        shapes = [tuple(p.shape) for _n, p in module.named_parameters()]
    except Exception:  # noqa: BLE001 — a fingerprint must not raise
        return "unfingerprintable"
    if not shapes:
        return "no-parameters"
    return f"params={len(shapes)}|first={shapes[0]}|last={shapes[-1]}"


def _split_adapters(
    pipe: Any, adapters: Sequence["PreparedAdapter"], components: Sequence[str],
) -> tuple[List["PreparedAdapter"], Dict[str, List[tuple]]]:
    """(peft-path adapters with denoiser keys stripped, branch set BY
    COMPONENT) for a branch-capable pipeline. Each adapter is first
    normalized through the pipeline class's own ``lora_state_dict`` converter
    (zero drift with the boot-time path), then split: denoiser keys ride the
    additive branch, the rest (text-encoder halves) keep peft. Branch entries
    are (state_dict, weight, ref) — the
    models.w8a8_lora.apply_branch_adapter_set contract.

    The denoiser half is ROUTED to the component its keys name, so a
    dual-expert MoE lands each half of a distillation on ITS expert. The
    routed slices are cached with the split (not rebuilt per request) —
    the branch staging cache keys on ``id(sd)``, and a fresh dict every
    request would re-pay the ~700ms CPU flatten."""

    fp = _denoiser_fingerprint(pipe)
    peft: List[PreparedAdapter] = []
    branch: Dict[str, List[tuple]] = {}
    for a in adapters:
        key = (type(pipe).__qualname__, fp, a.cache_key)
        with _SPLIT_CACHE_LOCK:
            cached = _SPLIT_CACHE.get(key)
            if cached is not None:
                _SPLIT_CACHE.move_to_end(key)
        if cached is None:
            sd = w8a8_lora.normalize_adapter_state_dict(
                pipe, a.state_dict, ref=a.ref
            )
            den, rest = w8a8_lora.split_state_dict(sd)
            if den:
                # RAW keys decide routability on a multi-expert pipeline — the
                # converter above rewrites every non-diffusers key onto the
                # high-noise prefix whatever expert it came from. Only adapters
                # that HAVE a denoiser half owe a declaration; a
                # text-encoder-only overlay is unaffected.
                w8a8_lora.require_component_declaration(
                    components, a.state_dict, ref=a.ref)
            cached = (
                w8a8_lora.route_denoiser_keys(den, components, ref=a.ref),
                rest,
            )
            with _SPLIT_CACHE_LOCK:
                # A racing thread may have inserted the same key — keep the
                # FIRST entry so every caller shares one den_sd object (the
                # branch staging cache keys on its id).
                cached = _SPLIT_CACHE.setdefault(key, cached)
                while len(_SPLIT_CACHE) > _SPLIT_CACHE_MAX:
                    _SPLIT_CACHE.popitem(last=False)
        routed, rest = cached
        for comp, den_sd in routed.items():
            if den_sd:
                branch.setdefault(comp, []).append((den_sd, a.weight, a.ref))
        if rest:
            peft.append(replace(a, state_dict=rest))
    return peft, branch


def _reject_te_keys_on_cast_te(
    pipe: Any, adapters: Sequence["PreparedAdapter"],
) -> None:
    """Typed refusal for text-encoder adapter halves TARGETING a cast TE
    (the ``fp8+te`` lane): peft module wrapping breaks under the
    block-window/layerwise cast — fail loud, never fight the hooks. Keys
    targeting an UNCAST encoder in a mixed setup stay on the peft path."""
    def component_of(key: str) -> str:
        for prefix, comp in te_prefix_to_component():
            if key.startswith(prefix):
                return comp
        return ""

    for a in adapters:
        for k in a.state_dict:
            comp = component_of(k)
            if not comp:
                continue
            te = getattr(pipe, comp, None)
            if te is not None and getattr(te, "_cozy_fp8_storage_applied", False):
                raise RefCompatibilitySurprise(
                    f"adapter targets {comp} (e.g. {k}) but this pipeline "
                    f"serves {comp} fp8-cast — text-encoder adapters are "
                    "unsupported on the fp8+te lane",
                    ref=a.ref, axis="state_dict",
                )


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
    # pgw#1330: the LOGICAL size, not the inode's. Under a projected tree
    # every candidate is a ~128 B stub, so `st_size` makes them all tie and
    # "the largest adapter" silently becomes "whichever sorted first".
    files = sorted(p.rglob("*.safetensors"), key=logical_size, reverse=True)
    if not files:
        raise RefCompatibilitySurprise(
            "adapter snapshot contains no .safetensors file",
            ref=ref, axis="component_missing",
        )
    return files[0]


# Matrix halves of one low-rank pair, all three key grammars: kohya
# (`…lora_down/up.weight`), diffusers attn-processor (`…lora.down/up.weight`),
# peft (`…lora_A/B[.<adapter>].weight`). Group 1 = module prefix, group 2/3 =
# which half.
_LORA_PAIR_RE = re.compile(
    r"^(.*?)\.(?:lora[._])?(down|up)\.weight$"
    r"|^(.*?)\.lora_([AB])(?:\.[\w-]+)?\.weight$"
)


def _reject_zero_delta(state_dict: Dict[str, Any], *, ref: str = "") -> None:
    """The attach-but-invisible rule, ONE implementation: the
    adapter's low-rank product must be provably nonzero, or attaching it
    silently serves the bare base model (e.g. undistilled output labeled
    turbo). Delta per module is ``up @ down`` — a pair with EITHER half
    all-zero contributes nothing, and alpha keys are nonzero by construction
    so they can never vouch. Accept as soon as one pair has both halves
    nonzero; refuse typed otherwise."""
    pairs: Dict[str, Dict[str, bool]] = {}
    for key, t in state_dict.items():
        m = _LORA_PAIR_RE.match(key)
        if m is None or not hasattr(t, "is_floating_point"):
            continue
        prefix = m.group(1) if m.group(1) is not None else m.group(3)
        half = m.group(2) if m.group(2) is not None else m.group(4)
        half = "down" if half in ("down", "A") else "up"
        entry = pairs.setdefault(prefix, {})
        if entry.get(half):
            continue
        try:
            nonzero = bool((t != 0).any())
        except Exception:  # exotic dtype without eq — cannot vouch either way
            continue
        entry[half] = nonzero
        if entry.get("down") and entry.get("up"):
            return
    raise RefCompatibilitySurprise(
        "adapter carries NO visible delta (no lora down/up pair with both "
        "halves nonzero) — attaching it would be invisible (th#1036); "
        "refusing",
        ref=ref,
        axis="state_dict",
    )


def load_adapter_state_dict(path: Path, *, ref: str = "") -> Dict[str, Any]:
    """Parse + validate one adapter file. Injects missing kohya ``alpha``
    keys (alpha = rank) so diffusers doesn't error. Zero-delta extractions
    are refused here — every consumer (executor overlays, BYO per-request
    loras, endpoint code) inherits the guard from this one seam."""
    import torch

    # The DECLARED size: a stub whose model is 40 GiB must still trip the cap.
    size = logical_size(path)
    if size > MAX_LORA_FILE_BYTES:
        raise ValidationError(
            f"lora adapter too large: {size} bytes (max {MAX_LORA_FILE_BYTES}) (ref={ref})"
        )
    try:
        state_dict = load_state_dict(
            path, why="the adapter's every low-rank matrix comes from this read"
        )
    except Exception as exc:
        raise RefCompatibilitySurprise(
            f"unreadable adapter safetensors: {exc}", ref=ref, axis="state_dict"
        ) from exc
    validate_lora_keys(state_dict.keys(), ref=ref)
    _reject_zero_delta(state_dict, ref=ref)
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
    """Per-pipeline attachment registry, keyed by model ref.

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
        rolls back to a fully-deactivated pipeline.

        Branch-capable pipelines: denoiser keys go to the additive
        side-branch (peft can't target Fp8ScaledLinear and fights the
        layerwise-cast hooks); text-encoder keys keep the peft path below.
        Plain-bf16 pipelines whose adapter cannot map onto branch-capable
        Linears (e.g. conv-targeting LoCon) fall back to the whole-adapter
        peft path when the pipeline supports it — capability preserved,
        branch primary.

        A pipeline's denoiser is a SET. Each adapter half is routed
        to the expert its keys name and the set is applied atomically; the
        peft fallback is refused outright on a multi-expert pipeline
        (diffusers expresses the second expert as a ``load_lora_weights(...,
        load_into_transformer_2=True)`` kwarg the peft path cannot reach, so
        falling back there would re-create the silent mis-landing)."""

        all_adapters = list(adapters)
        targets = w8a8_lora.branch_targets(pipe)
        branch_set: Dict[str, List[tuple]] = {}
        # Normalization + routing run OFF the residency lock (they are pure
        # CPU work under the split cache's own lock), but a refusal here
        # still owes the caller a deactivated pipeline — a rejected request
        # must never leave the previous request's branch live.
        try:
            if targets:
                adapters, branch_set = _split_adapters(
                    pipe, adapters, tuple(targets))
                _reject_te_keys_on_cast_te(pipe, adapters)
            if adapters and not isinstance(pipe, LoraCapablePipeline):
                raise ValidationError(
                    "model slot does not support LoRA adapters "
                    "(pipeline lacks load_lora_weights/set_adapters/"
                    "unload_lora_weights)"
                )
        except BaseException:
            self.deactivate(ref, pipe, request_id=request_id)
            raise
        with self._lock:
            st = self._state(ref, pipe)
            try:
                # A lifted binding exists only when the AOT arm installed it —
                # its presence IS the routing fact. An armed exported artifact
                # reads the adapter from the lifted flat pair (call inputs); a
                # buffer copy would be invisible to it, so every attach/clear
                # on a lifted denoiser writes through the binding views instead.
                lifted = any(
                    lora_lifted.lifted_binding(m) is not None
                    for m in targets.values()) if targets else False
                if targets and branch_set and lifted:
                    # Same apply/scale-fold/refusal code as the buffer path
                    # (it IS that path, through views), canonical placement,
                    # no resize — the traced bucket is the floor, and a set
                    # that needs more refuses instead of recompiling.
                    lora_lifted.swap_lifted_execution_lane_set(
                        pipe, branch_set, request_id=request_id)
                    w8a8_lora.stamp_execution_lane(pipe, targets)
                elif targets and branch_set:
                    # Compiled pipelines keep canonical placement and ONE
                    # traced bucket (a resize would mean a recompile at swap
                    # time — never allowed in prod); eager pipelines use
                    # sparse placement (branch kernels only where covered).
                    compiled = getattr(pipe, "_cozy_compile", None) is not None
                    sole = next(iter(targets.values())) if len(targets) == 1 else None
                    try:
                        w8a8_lora.apply_branch_adapter_set(
                            pipe, branch_set,
                            allow_resize=not compiled, uniform=compiled,
                            request_id=request_id,
                        )
                    except RefCompatibilitySurprise:
                        if (sole is not None
                                and w8a8_lora.branch_execution_lane(sole) == ""
                                and not compiled
                                and isinstance(pipe, LoraCapablePipeline)):
                            logger.info(
                                "[request_id=%s] adapter set does not map onto "
                                "branch Linears; falling back to the peft path "
                                "(plain lane)", request_id,
                            )
                            w8a8_lora.clear_branch_execution_lanes(pipe)
                            w8a8_lora.stamp_execution_lane(pipe, targets)
                            adapters = all_adapters
                        else:
                            raise
                    else:
                        w8a8_lora.stamp_execution_lane(pipe, targets)
                elif targets:
                    # Adapter set has no denoiser half: make sure a previous
                    # request's branches are off. Lifted denoisers clear
                    # through their binding (zero-B lands in the flat pair
                    # the artifact reads).
                    if lifted:
                        for model in targets.values():
                            binding = lora_lifted.lifted_binding(model)
                            if binding is not None:
                                binding.clear()
                            else:
                                w8a8_lora.clear_branch_adapters(model)
                    else:
                        w8a8_lora.clear_branch_execution_lanes(pipe)
                    w8a8_lora.stamp_execution_lane(pipe, targets)
                if targets and not adapters:
                    # No peft half — make sure a previous request's peft
                    # adapters are off, then we're done. Only touch the peft
                    # surface when THIS registry attached something there:
                    # diffusers' disable_lora raises on peft-less images (some
                    # serving images ship no peft), and a branch-only pipeline
                    # never needs it.
                    if st.attached and hasattr(pipe, "disable_lora"):
                        pipe.disable_lora()
                    st.active = True
                    return
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
                targets = w8a8_lora.branch_targets(pipe)
                if targets:
                    # A lifted denoiser deactivates through its binding —
                    # zero-B must land in the flat pair the armed artifact
                    # reads, not in the orphaned canonical buffers.
                    for model in targets.values():
                        binding = lora_lifted.lifted_binding(model)
                        if binding is not None:
                            binding.clear()
                        else:
                            w8a8_lora.clear_branch_adapters(model)
                    w8a8_lora.stamp_execution_lane(pipe, targets)
            except Exception as exc:
                logger.warning(
                    "[request_id=%s] lora branch clear failed", request_id,
                    exc_info=True,
                )
                # Adapter deltas may still be live in the branch buffers — the
                # NEXT tenant's request can render with THIS request's LoRA.
                # Serving-correctness, never log-only.
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_HYGIENE,
                    f"ref={ref} request={request_id}: branch clear failed; "
                    f"adapter deltas may persist into later requests: "
                    f"{type(exc).__name__}: {exc}",
                    phase="branch_clear_failed",
                )
            try:
                # Peft-surface teardown only when peft attachments exist —
                # diffusers raises "PEFT backend is required" otherwise on
                # peft-less serving images (branch-only lanes).
                if not st.attached:
                    pass
                elif hasattr(pipe, "disable_lora"):
                    pipe.disable_lora()
                elif hasattr(pipe, "unload_lora_weights"):
                    pipe.unload_lora_weights()
                    st.attached.clear()
                st.active = False
                logger.info(
                    "[request_id=%s] lora adapters deactivated (disable_ms=%d attached=%d)",
                    request_id, int((time.monotonic() - t0) * 1000), len(st.attached),
                )
            except Exception as exc:
                logger.warning(
                    "[request_id=%s] lora deactivate failed; pipeline may have "
                    "active adapters", request_id, exc_info=True,
                )
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_HYGIENE,
                    f"ref={ref} request={request_id} "
                    f"attached={len(st.attached)}: peft deactivate failed; "
                    f"pipeline may serve later requests with active "
                    f"adapters: {type(exc).__name__}: {exc}",
                    phase="deactivate_failed",
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
            try:
                from ..models import w8a8_lora

                targets = w8a8_lora.branch_targets(pipe)
                # Bucket guard: never-lora pipelines skip the module walk
                # entirely — this runs on EVERY demote.
                if targets and w8a8_lora.pipeline_branch_bucket(pipe):
                    w8a8_lora.disable_branch_execution_lanes(pipe)
                    w8a8_lora.stamp_execution_lane(pipe, targets)
            except Exception as exc:
                logger.warning("lora branch drop on demote failed for %s",
                               ref, exc_info=True)
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_HYGIENE,
                    f"ref={ref}: branch-lane drop on demote failed; adapter "
                    f"deltas may survive the demote: "
                    f"{type(exc).__name__}: {exc}",
                    phase="detach_failed",
                )
            if st is None or not st.attached or st.pipe_id != id(pipe):
                return
            try:
                pipe.unload_lora_weights()
                logger.info(
                    "lora attachments dropped on demote: ref=%s adapters=%d",
                    ref, len(st.attached),
                )
            except Exception as exc:
                logger.warning("lora detach on demote failed for %s", ref, exc_info=True)
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_HYGIENE,
                    f"ref={ref} adapters={len(st.attached)}: peft unload on "
                    f"demote failed; attachments persist on the demoted "
                    f"pipeline: {type(exc).__name__}: {exc}",
                    phase="detach_failed",
                )

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
            except Exception as exc:
                logger.warning("lora eviction failed for %s", victim, exc_info=True)
                # The attachment is dropped from bookkeeping but its tensors
                # stay on the pipeline — repeated failures creep VRAM.
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_HYGIENE,
                    f"adapter={victim}: LRU eviction failed; adapter tensors "
                    f"remain on the pipeline (VRAM creep): "
                    f"{type(exc).__name__}: {exc}",
                    phase="evict_failed",
                )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                ref: {"adapters": len(st.attached), "bytes": st.total_bytes,
                      "active": st.active}
                for ref, st in self._pipes.items()
            }
