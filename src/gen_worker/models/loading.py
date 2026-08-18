"""Load-time helpers endpoints (and the executor's typed injection) use around
``from_pretrained``: dtype mapping, on-disk variant detection, and quant-config
synthesis. There is no PipelineLoader — callers own ``from_pretrained``.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import activity as activity_mod
from ..component_vocab import (
    denoiser_components,
    text_encoder_components,
    weight_components,
)
from ..families.facts import component_dtype_for_class
import importlib
import importlib.util
import inspect
import os
import struct

from ..capability import HostRamCapacityError, InsufficientHostRamError
from . import disk_gc, load_progress
from .materialized_view import third_party_dir
from .file_layout import (
    MULTI_FILE,
    SINGLE_FILE,
    is_single_file_snapshot,
    observed_file_layout,
)
from .tensor_layout_contract import (
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_HF_FP8_BLOCKWISE,
    CONTRACT_NUNCHAKU_V1,
    CONTRACT_PLAIN_BF16,
    ELEMENT_BF16,
    ELEMENT_FP16,
    ELEMENT_FP32,
    KEYS_DIFFUSERS_SPLIT_QKV,
    KEYS_TRANSFORMERS_SPLIT_QKV,
    SCALE_NONE,
    DecodeDimensions,
    implements_contract,
    unregistered_decode_path,
)
from .fp8_storage import restructure_fp8_storage
from .rung import touches_host_ram
from .memory import (
    flush_memory,
    get_available_vram_gb,
    meta_tensors,
    probe_host_ram,
)
from .hf_fp8_blockwise import detect_hf_fp8_blockwise, load_hf_fp8_blockwise
from .safetensors_header import header_len_ok, read_header
from .svdq import detect_svdq_artifact, load_svdq_pipeline
from ..hostfacts import cuda_ready
from .w4a4 import (
    detect_w4a4_artifact,
    load_w4a4_denoiser,
    load_w4a4_pipeline,
    load_w4a4_root_pipeline,
)
from .w8a8 import (
    detect_w8a8_artifact,
    load_w8a8_denoiser,
    load_w8a8_pipeline,
    load_w8a8_root_pipeline,
)

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


def get_torch_dtype(dtype_str: Optional[str]) -> Any:
    """Map a dtype string to a torch dtype. Empty/None -> bfloat16 (the
    de-facto inference default). UNKNOWN strings raise instead of silently
    loading as bf16 — quantized checkpoints (fp8/int4/...) don't take a
    ``torch_dtype`` and must not be mislabeled."""
    import torch

    if not dtype_str:
        return torch.bfloat16
    name = _DTYPE_MAP.get(dtype_str.strip().lower())
    if name is None:
        raise ValueError(
            f"unknown torch dtype string {dtype_str!r}; expected one of "
            f"{sorted(set(_DTYPE_MAP))}"
        )
    return getattr(torch, name)


def detect_diffusers_variant(model_path: Path) -> Optional[str]:
    """Detect a diffusers ``variant=`` value from files on disk (e.g.
    ``unet/diffusion_pytorch_model.fp16.safetensors`` -> ``"fp16"``)."""
    candidates = ("bf16", "fp8", "fp16")
    try:
        for p in Path(model_path).rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if not name.endswith((".safetensors", ".json")):
                continue
            for v in candidates:
                if f".{v}." in name:
                    return v
    except OSError:
        return None
    return None


_SAFETENSORS_DTYPE_NAMES = {
    "BF16": "bf16", "F16": "fp16", "F32": "fp32", "F8_E4M3": "fp8",
}


def safetensors_file_valid(path: Path) -> bool:
    """Cheap structural integrity check for one ``.safetensors`` file: the
    header must parse and the file must contain every declared tensor byte.
    Catches truncation (interrupted writes) without hashing; zero-page
    corruption inside tensor data needs the digest check instead."""

    try:
        p = Path(path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            (n,) = struct.unpack("<Q", raw)
            if not header_len_ok(n) or 8 + n > size:
                return False
            header = json.loads(f.read(n))
        if not isinstance(header, dict):
            return False
        data_end = 0
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict) or "data_offsets" not in value:
                return False
            data_end = max(data_end, int(value["data_offsets"][1]))
        return size >= 8 + n + data_end
    except (OSError, ValueError, KeyError, TypeError):
        return False


def detect_on_disk_dtype(model_path: Path) -> str:
    """Majority weight dtype across the snapshot's safetensors headers
    ("bf16" / "fp16" / "fp32" / "fp8", "" when undetectable). Hub bindings
    carry no dtype and mirrored repos use unsuffixed filenames, so without
    this a bf16 snapshot silently loads via diffusers' fp32 default — 2x the
    VRAM. "fp8" marks an fp8-E4M3-stored flavor whose storage precision must
    be preserved (see :func:`apply_fp8_storage`).

    **On a projected tree the answer comes from the MANIFEST, and a tree whose
    manifest cannot be recovered is a refusal, never a ""** (pgw#1308). A
    TFSSTUB1 stub carries a body digest and a size and no dtype at all, and
    its first eight bytes are ASCII magic that ``header_len_ok`` correctly
    rejects — so on master every weight file was skipped by a ``continue``,
    ``counts`` stayed empty, and this returned "", i.e. exactly the silent
    fp32 default the docstring above was written to prevent. The stub failed
    loudly at the parse site as designed; this caller read the loud failure as
    "no information". A guarantee about how a read fails constrains nothing
    about what the caller concludes, so the branch has to live here."""

    counts: Dict[str, int] = {}
    try:
        paths = sorted(Path(model_path).rglob("*.safetensors"))
    except OSError:
        return ""
    for p in paths:
        for value in read_header(p, why=_DTYPE_WHY).values():
            if isinstance(value, dict) and "dtype" in value:
                counts[str(value["dtype"])] = counts.get(str(value["dtype"]), 0) + 1
    if not counts:
        return ""
    top = max(counts, key=lambda k: counts[k])
    return _SAFETENSORS_DTYPE_NAMES.get(top, "")


_DTYPE_WHY = (
    "the weight dtype decides whether this model loads in its stored precision "
    "or via diffusers' fp32 default, at 2x the VRAM"
)


def read_on_disk_quant_config(model_path: Path) -> bool:
    """True when model_index.json / component config.json on disk carries a
    ``quantization_config`` block (diffusers auto-picks it up)."""
    model_path = Path(model_path)
    candidates: List[Path] = []
    if model_path.is_dir():
        for rel in ("model_index.json", "config.json"):
            p = model_path / rel
            if p.exists():
                candidates.append(p)
        for sub in weight_components():
            cfg = model_path / sub / "config.json"
            if cfg.exists():
                candidates.append(cfg)
    for p in candidates:
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("quantization_config"):
            return True
    return False


def synthesize_quantization_config(attrs: Optional[Dict[str, str]]) -> Optional[Any]:
    """Build a BitsAndBytesConfig from resolved checkpoint attrs when the
    on-disk config doesn't already carry one. Returns None when the attrs
    don't indicate a library that needs a synthesized config."""
    if not attrs:
        return None
    lib = str(attrs.get("quant_library") or "").strip().lower()
    if lib != "bitsandbytes":
        return None
    recipe = str(attrs.get("quant_recipe") or "").strip().lower()
    scheme = recipe.split(":", 1)[-1] if ":" in recipe else recipe
    if scheme not in ("nf4", "fp4"):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        logger.warning("bnb quant detected but BitsAndBytesConfig unavailable: %s", exc)
        return None
    compute_dtype_name = str(attrs.get("quant_compute_dtype") or "bfloat16").strip().lower()
    compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
    double_quant = str(attrs.get("quant_double_quant") or "true").strip().lower() in ("1", "true", "yes")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=scheme,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
    )


# Pipeline components fp8 storage applies to: the denoiser dominates VRAM and
# tolerates fp8-E4M3 weight rounding; text encoders / VAE stay at compute
# precision (QUANTIZATION-POLICY.md component policy).
#
# These read the vocabulary at CALL time, never at import: an endpoint's
# declare_components() may run after this module is imported, and a
# module-level tuple would freeze the pre-declaration vocabulary.
_fp8_storage_components = denoiser_components
# The "+te" rung (component fit-ladder rung 2): the pipeline's text encoders.
_fp8_text_encoder_components = text_encoder_components

class _Fp8WeightWindow:
    """Weight-only fp8 storage for one transformer block: the block's
    Linear/conv WEIGHTS live in fp8 at rest; a forward-pre hook upcasts them
    to the compute dtype for the whole block forward and a forward hook
    recasts after. Everything else in the block (norms, embeddings, biases,
    raw parameters) never leaves compute precision.

    Block-window (not per-leaf-layer) granularity is required for transformers
    models, which — unlike diffusers denoisers — read weight dtype and touch
    weights OUTSIDE the owning leaf's forward (Gemma3's embed-scale multiply,
    T5's ``T5DenseActDense`` casting activations to ``self.wo.weight.dtype``),
    so a leaf-hooked ``wo`` still poisons the stream with fp8. Inside a block
    window every dtype read sees the compute dtype. Transient cost: one block
    resident at compute dtype."""

    def __init__(self, params: List[Any], storage: Any, compute: Any) -> None:
        self._params = params
        self._storage = storage
        self._compute = compute

    def install(self, block: Any) -> None:
        for p in self._params:
            p.data = p.data.to(self._storage)
        block.register_forward_pre_hook(self._pre)
        block.register_forward_hook(self._post)

    def _pre(self, module: Any, args: Any) -> None:
        for p in self._params:
            p.data = p.data.to(self._compute)

    def _post(self, module: Any, args: Any, output: Any) -> None:
        for p in self._params:
            p.data = p.data.to(self._storage)


def _fp8_block_windows(mod: Any) -> List[tuple[str, Any, List[Any]]]:
    """(name, block, castable params) per repeated transformer block: the
    children of top-level ``nn.ModuleList`` containers (``model.layers``,
    ``encoder.block``, vision-tower layers, ...). Castable = Linear/conv
    weights not shared with any module outside a block (tied lm_head /
    embedding weights stay at compute dtype). Parameters outside blocks —
    embeddings, final norms, poolers, lm_head — are never cast."""
    import torch.nn as nn

    castable_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    list_names = [n for n, m in mod.named_modules() if isinstance(m, nn.ModuleList)]
    top = [n for n in list_names
           if not any(n != o and n.startswith(o + ".") for o in list_names)]

    blocks: List[tuple[str, Any]] = []
    seen_blocks: set[int] = set()
    for name in top:
        ml = mod.get_submodule(name)
        for i, block in enumerate(ml):
            if id(block) in seen_blocks:  # ALBERT-style shared blocks
                continue
            seen_blocks.add(id(block))
            blocks.append((f"{name}.{i}", block))

    # Any parameter reachable outside the blocks must keep compute dtype: a
    # weight cast through a block but read elsewhere breaks.
    block_param_owners: Dict[int, int] = {}
    for _, block in blocks:
        for p in block.parameters():
            block_param_owners[id(p)] = block_param_owners.get(id(p), 0) + 1
    outside: set[int] = set()
    for p in mod.parameters():
        if id(p) not in block_param_owners:
            outside.add(id(p))
    for name, m in mod.named_modules():
        in_block = any(name == bn or name.startswith(bn + ".") for bn, _ in blocks)
        if not in_block:
            outside.update(id(p) for p in m.parameters(recurse=False))

    windows: List[tuple[str, Any, List[Any]]] = []
    for name, block in blocks:
        params: List[Any] = []
        seen: set[int] = set()
        for m in block.modules():
            if not isinstance(m, castable_types):
                continue
            w = getattr(m, "weight", None)
            if w is None or id(w) in seen or id(w) in outside:
                continue
            if not w.is_floating_point():
                continue
            seen.add(id(w))
            params.append(w)
        if params:
            windows.append((name, block, params))
    return windows


def _apply_transformers_fp8(mod: Any, storage: Any, compute_dtype: Any) -> None:
    """Weight-only fp8 storage for a transformers module (Gemma3/T5/CLIP-class
    text encoders) via per-block :class:`_Fp8WeightWindow` hooks. Falls back
    to a single whole-module window when no repeated blocks are found."""
    windows = _fp8_block_windows(mod)
    if not windows:
        # No ModuleList blocks: one whole-module window (correct, but the
        # transient upcast is the full module).
        all_windows = _fp8_block_windows_whole(mod)
        if not all_windows:
            raise ValueError("no fp8-castable weights found")
        windows = all_windows
    total = 0
    for _name, block, params in windows:
        _Fp8WeightWindow(params, storage, compute_dtype).install(block)
        total += len(params)
    logger.info("fp8 weight windows installed: %d blocks, %d weights",
                len(windows), total)


def _fp8_block_windows_whole(mod: Any) -> List[tuple[str, Any, List[Any]]]:
    import torch.nn as nn

    castable_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    params: List[Any] = []
    seen: set[int] = set()
    shared_out: set[int] = set()
    for m in mod.modules():
        if not isinstance(m, castable_types):
            shared_out.update(id(p) for p in m.parameters(recurse=False))
    for m in mod.modules():
        if not isinstance(m, castable_types):
            continue
        w = getattr(m, "weight", None)
        if w is None or id(w) in seen or id(w) in shared_out:
            continue
        if not w.is_floating_point():
            continue
        seen.add(id(w))
        params.append(w)
    if not params:
        return []
    return [(type(mod).__name__, mod, params)]


def apply_fp8_storage(obj: Any, *, compute_dtype: Any = None,
                      text_encoders: bool = False,
                      components: Optional[tuple[str, ...]] = None) -> bool:
    """fp8-E4M3 weight storage with per-layer upcast to ``compute_dtype`` on a
    pipeline's denoiser — or on ``obj`` itself when it is a bare module.
    Diffusers denoisers are RESTRUCTURED into fp8 storage modules
    (:mod:`gen_worker.models.fp8_storage`: upcast at the use site inside
    forward); transformers text encoders keep the :class:`_Fp8WeightWindow`
    block hooks — they read weight dtype OUTSIDE the owning leaf's forward,
    which resident fp8 would poison, and they are not on any compiled path.
    ``text_encoders=True`` (the ``storage_dtype="fp8+te"`` rung) extends the
    cast to the pipeline's text encoders via the transformers-aware path.
    ``components`` overrides the target component names entirely: the w8a8
    lane casts ONLY the text encoders — its denoiser holds fp8 scaled-mm
    modules that cast hooks must never touch.

    This is the universal VRAM-fit mechanism: fp8 bytes resident, bf16/fp16
    compute, no fp8 silicon required. Also the consumption path for stored
    fp8 flavors — their storage precision is preserved instead of being
    upcast into 2x the VRAM. Returns True when any module was converted;
    failures degrade to full-precision serving with a warning."""
    try:
        import torch
    except ImportError:
        logger.warning("storage_dtype=fp8 ignored: torch not installed")
        return False
    storage = getattr(torch, "float8_e4m3fn", None)
    if storage is None:
        logger.warning("storage_dtype=fp8 ignored: torch lacks float8_e4m3fn")
        return False
    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    if components is None:
        components = _fp8_storage_components()
        if text_encoders:
            components += _fp8_text_encoder_components()
    targets: List[tuple[str, Any]] = []
    for name in components:
        mod = getattr(obj, name, None)
        if mod is not None and hasattr(mod, "parameters"):
            targets.append((name, mod))
    if not targets and hasattr(obj, "parameters"):
        targets.append((type(obj).__name__, obj))

    applied = False
    for name, mod in targets:
        if getattr(mod, "_cozy_fp8_storage_applied", False):
            # Idempotence: a content-shared module injected into a
            # sibling lane is already armed; double hooks would double-cast.
            applied = True
            continue
        try:
            if callable(getattr(mod, "enable_layerwise_casting", None)):
                # diffusers ModelMixin. Same semantics and coverage set as
                # ``enable_layerwise_casting`` (the model's own skip patterns
                # included), expressed as module STRUCTURE instead of a
                # forward-boundary mutation: the hook form is compile-hostile
                # and torch.export refuses it; the structural form is 14.8%
                # faster under dynamo and exports clean.
                if not restructure_fp8_storage(mod, storage_dtype=storage,
                                               compute_dtype=compute_dtype):
                    raise ValueError("no fp8-castable leaves found")
            else:
                _apply_transformers_fp8(mod, storage, compute_dtype)
            mod._cozy_fp8_storage_applied = True
            applied = True
            logger.info("fp8 storage enabled on %s (compute %s)", name, compute_dtype)
        except Exception as exc:
            logger.warning("fp8 storage failed on %s (%s); serving at full precision",
                           name, exc)
            # A PARTIAL failure still returns applied=True and reads as
            # success, so report it: this component alone now holds ~2x its
            # budgeted VRAM at full precision.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"component={name} obj={type(obj).__name__}: fp8 storage "
                f"cast failed; this component serves at full precision "
                f"(over its budgeted VRAM): {type(exc).__name__}: {exc}",
                phase="fp8_cast_failed",
            )
    return applied


class _BlockOffloadWindow:
    """Degraded-mode rung 2: one transformer block's weights REST in
    host RAM (pinned when possible) and stream to the execution device only
    for that block's forward. The pre-hook is PREPENDED so the H2D copy runs
    before any fp8 upcast window on the same block — composed order:
    host fp8 bytes -> device fp8 -> device compute dtype. The post-hook
    rebinds ``.data`` to the pristine host copy (weights are read-only at
    inference; no copy-back), so whatever dtype games other hooks played in
    between are discarded. ``always_call`` keeps the rebind on exceptions —
    a mid-block CUDA OOM must not leave the window resident."""

    def __init__(self, params: List[Any], hosts: List[Any], device: Any) -> None:
        self._params = params
        self._hosts = hosts
        self._device = device

    def install(self, block: Any) -> None:
        block.register_forward_pre_hook(self._pre, prepend=True)
        block.register_forward_hook(self._post, always_call=True)

    def _pre(self, module: Any, args: Any) -> None:
        for p, h in zip(self._params, self._hosts):
            p.data = h.to(self._device, non_blocking=True)

    def _post(self, module: Any, args: Any, output: Any = None) -> None:
        for p, h in zip(self._params, self._hosts):
            p.data = h


def apply_block_window_offload(
    obj: Any,
    *,
    components: tuple[str, ...] | None = None,
    device: Any = None,
) -> bool:
    """Park a module's per-block weights in pinned host RAM and stream each
    block to ``device`` for its forward only — the block windows in reverse
    (degraded-mode rung 2). Quality-preserving but slow (whole-model PCIe
    traffic per forward); a guaranteed-completion last resort for
    VRAM-constrained cards, never a production serving mode.

    ``obj`` is a pipeline (named ``components`` are offloaded) or a bare
    module. Parameters outside the discovered block windows — embeddings,
    final norms, projections — are moved TO ``device`` (they must be
    resident; the outside-a-block dtype hazard applies to device placement
    too). Composes with fp8 storage windows: fp8 bytes stream over PCIe (half
    the traffic), upcast happens on-device.

    Returns True when any module was armed. Idempotent per module."""
    # Default resolved at call time, not in the signature: a def-time default
    # would freeze the vocabulary before declare_components() ever runs.
    if components is None:
        components = denoiser_components()
    try:
        import torch
    except ImportError:
        logger.warning("block-window offload ignored: torch not installed")
        return False
    if device is None:
        if not cuda_ready():
            logger.warning("block-window offload ignored: no CUDA device")
            return False
        device = "cuda"

    targets: List[tuple[str, Any]] = []
    for name in components:
        mod = getattr(obj, name, None)
        if mod is not None and hasattr(mod, "parameters"):
            targets.append((name, mod))
    if not targets and hasattr(obj, "parameters"):
        targets.append((type(obj).__name__, obj))

    applied = False
    for name, mod in targets:
        if getattr(mod, "_cozy_block_offload_applied", False):
            applied = True
            continue
        windows = _fp8_block_windows(mod) or _fp8_block_windows_whole(mod)
        if not windows:
            logger.warning("block-window offload: no weight windows in %s", name)
            continue
        pin = True
        parked_ids: set[int] = set()
        parked_bytes = 0
        for _bname, block, params in windows:
            hosts: List[Any] = []
            for p in params:
                host = None
                if pin:
                    try:
                        host = torch.empty_like(p.data, device="cpu", pin_memory=True)
                    except RuntimeError as exc:
                        pin = False
                        logger.warning(
                            "block-window offload: pinned host alloc failed (%s); "
                            "falling back to pageable staging (slower)", exc,
                        )
                if host is None:
                    host = torch.empty_like(p.data, device="cpu")
                host.copy_(p.data)
                p.data = host
                hosts.append(host)
                parked_ids.add(id(p))
                parked_bytes += host.numel() * host.element_size()
            _BlockOffloadWindow(params, hosts, device).install(block)
        # Everything OUTSIDE the windows must be resident on the device.
        for p in mod.parameters():
            if id(p) not in parked_ids and p.device != torch.device(device):
                p.data = p.data.to(device)
        for b in mod.buffers():
            # `parked_ids` guards buffers too: the storage lanes hold their
            # weights as BUFFERS (w8a8's scaled weights, the fp8 storage
            # leaves), so without this the just-parked block weights are
            # pulled straight back onto the device and the rung saves nothing.
            if id(b) not in parked_ids and b.device != torch.device(device):
                b.data = b.data.to(device)
        mod._cozy_block_offload_applied = True
        applied = True
        logger.warning(
            "DEGRADED_MODE=engaged model=%s phase=load rung=resident->block_offload: "
            "%d blocks / %.1f GiB parked in %s host RAM, streaming per forward",
            name, len(windows), parked_bytes / float(1 << 30),
            "pinned" if pin else "pageable",
        )
        # Structurally reported, not just logged: every forward on this
        # component now streams its weights over PCIe from host RAM, the
        # biggest per-request latency change the loader can make. `pinned` vs
        # `pageable` rides the detail — they differ by roughly 2x on a
        # transfer that now sits in the critical path.
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"component={name} obj={type(obj).__name__}: block-window offload "
            f"ENGAGED — {len(windows)} block(s) / "
            f"{parked_bytes / float(1 << 30):.1f} GiB rest in "
            f"{'pinned' if pin else 'pageable'} host RAM and stream to the "
            f"device per forward; every request on this component pays that "
            f"transfer",
            phase="block_offload_engaged",
        )
    return applied


def block_offload_active(obj: Any) -> bool:
    """True when :func:`apply_block_window_offload` armed ``obj`` or any of
    its standard components."""
    if getattr(obj, "_cozy_block_offload_applied", False):
        return True
    return any(
        getattr(getattr(obj, name, None), "_cozy_block_offload_applied", False)
        for name in denoiser_components()
    )


def require_decodable(contract: str, path: Any, *, component: str = "") -> None:
    """Refuse, typed, before handing bytes to a decoder this IMAGE does not
    declare (pgw#1245).

    Three questions, all answered from HEADERS and directory shape, all before
    any tensor is read: is this contract in the image's decode-set, is the
    artifact's on-disk SHAPE one a decoder of that contract opens, and is its
    tensor-KEY convention one that decoder ingests. The third is not the
    first: `plain.bf16@1` minimax-native weights (fused `blocks.N.attn.qkv_proj`)
    and `plain.bf16@1` diffusers weights (split `to_q/to_k/to_v`) are the same
    contract, the same file topology and one key in common, and the diffusers
    class cannot read the native tree at all. A DENOISER whose convention
    matches nothing registered refuses too — unknown is never a hopeful pass
    where a model class is chosen from the architecture.

    The declared decode-set is th#1938's third intersection and the hub answers
    it ahead of time — but the worker is where the bytes actually arrive, so it
    is where an image that lost a decoder, or was offered the other
    repackaging, must say so by name instead of dying five libraries away as
    `Cannot detect the model type`.

    Imported lazily: `discovery` walks `models` to derive the set, and a
    module-level import here would make that walk import its own caller.
    """
    from ..discovery.decode_set import require_decodable as _require
    from .key_topology import classify_snapshot

    tree = Path(path) / component if component else Path(path)
    _require(
        contract,
        where=str(path),
        keys=classify_snapshot(Path(path), component),
        layout=observed_file_layout(tree),
    )


def detect_gguf_snapshot(path: Path) -> Optional[tuple[Path, str]]:
    """Return the GGUF denoiser and qtype in a composed diffusers snapshot."""
    p = Path(path)
    if not p.is_dir() or not (p / "model_index.json").exists():
        return None
    # cycle: gguf_local imports loading at module top
    from .gguf_local import gguf_qtype, read_marker

    marker = read_marker(p)
    if marker:
        gguf = p / str(marker.get("gguf_path") or "")
        qtype = str(marker.get("qtype") or "")
        if gguf.is_file() and qtype:
            return gguf, qtype
    ggufs = sorted(x for x in p.rglob("*.gguf") if x.is_file())
    if len(ggufs) != 1:
        return None
    tail = ggufs[0].stem.replace(".", "-").rsplit("-", 1)[-1].lower()
    qtype = gguf_qtype("gguf-" + tail)
    return (ggufs[0], qtype) if qtype else None


@unregistered_decode_path(
    reason="gguf.native@1 is a TOPOLOGY contract; no QUANT contract names the "
           "k-quant block encodings this reads, and diffusers' "
           "GGUFQuantizationConfig decodes them inside its own loader. So the "
           "bytes are decodable here and unnameable in the decode-set until "
           "the platform registers a descriptor for them.",
)
def load_gguf_pipeline(
    cls: Any,
    path: Path,
    gguf_file: Path,
    *,
    components: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load a GGUF denoiser into the remaining components' base tree."""

    import torch
    from diffusers import GGUFQuantizationConfig

    path = Path(path)
    index = json.loads((path / "model_index.json").read_text("utf-8"))
    component = next(
        (
            name
            for name in _fp8_storage_components()
            if isinstance(index.get(name), list) and len(index[name]) == 2
        ),
        None,
    )
    if component is None:
        raise ValueError(
            f"GGUF composition in {path} has no transformer/unet component"
        )
    module_name, class_name = index[component]
    denoiser_cls = getattr(importlib.import_module(str(module_name)), str(class_name))
    compute = torch.bfloat16
    denoiser = denoiser_cls.from_single_file(
        str(third_party_dir(gguf_file, why="GGUF from_single_file wants a real file")),
        config=str(third_party_dir(path / component, why="GGUF config dir")),
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute),
        torch_dtype=compute,
    )
    kwargs = dict(components or {})
    kwargs[component] = denoiser
    pipe = cls.from_pretrained(
        str(third_party_dir(path, why="gguf-denoiser sibling parts from_pretrained")),
        torch_dtype=compute, **kwargs)
    for name in _fp8_text_encoder_components():
        text_encoder = getattr(pipe, name, None)
        if text_encoder is not None and hasattr(text_encoder, "parameters"):
            apply_fp8_storage(text_encoder, compute_dtype=compute)
    return pipe


def _single_file_checkpoint(path: Path) -> Optional[Path]:
    """A snapshot that is one loose checkpoint rather than a pretrained layout:
    the path itself when it's a ``.safetensors`` file, or the directory's sole
    ``*.safetensors`` when no ``model_index.json``/``config.json`` exists
    (e.g. Illustrious-XL, civitai checkpoints).

    Mirrors reshard oversize safetensors into byte-offset shards + an
    ``*.safetensors.index.json`` (the HF shard convention), so a big
    single-file checkpoint arrives as N shards. Those are reassembled once
    into the original file (mmap-backed, ~disk-copy cost) and cached in the
    snapshot dir — ``from_single_file`` only takes one file.

    The SHAPE test is :func:`file_layout.is_single_file_snapshot`, shared with
    the load-side layout observation so the routing decision and the declared
    axis cannot disagree about what "single-file" means."""
    if not is_single_file_snapshot(path):
        return None
    if path.is_file():
        return path
    singles = sorted(p for p in path.glob("*.safetensors") if p.is_file())
    if len(singles) == 1:
        return singles[0]
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if len(indexes) == 1 and singles:
        try:
            return _merge_sharded_checkpoint(path, indexes[0])
        except Exception:
            logger.exception("failed to reassemble sharded single-file checkpoint in %s", path)
            return None
    return None


def _merge_sharded_checkpoint(snapshot_dir: Path, index_path: Path) -> Path:
    """Reassemble ``<name>.safetensors`` from its HF-convention shards at the
    BYTE level (8-byte header length + JSON header + raw buffer): rebuild one
    combined header with rebased offsets, then stream-copy each tensor's byte
    range. No torch/safetensors dependency, no RAM spike. Idempotent: the
    merged file is cached next to the shards."""

    merged = snapshot_dir / index_path.name[: -len(".index.json")]
    if merged.exists():
        if safetensors_file_valid(merged):
            return merged
        # A kill mid-writeback persists a truncated merged file; without this
        # revalidation it would be trusted forever.
        logger.warning(
            "cached merged checkpoint %s is structurally invalid (truncated?); re-merging",
            merged.name,
        )
        merged.unlink(missing_ok=True)
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise ValueError(f"empty weight_map in {index_path}")

    entries: List[tuple[str, dict, Path, int, int]] = []  # name, info, shard, start, end
    for shard in shard_names:
        shard_path = snapshot_dir / shard
        with open(shard_path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if not header_len_ok(n):
                raise ValueError(
                    f"safetensors: implausible header_length={n} in {shard}")
            header = json.loads(f.read(n))
        data_start = 8 + n
        header.pop("__metadata__", None)
        for name, info in header.items():
            s, e = info["data_offsets"]
            entries.append((name, info, shard_path, data_start + s, data_start + e))

    out_header: Dict[str, Any] = {}
    offset = 0
    for name, info, _, start, end in entries:
        size = end - start
        out_header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")

    tmp = merged.with_name(merged.name + ".__merge__")
    chunk = 8 << 20
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for _, _, shard_path, start, end in entries:
            with open(shard_path, "rb") as src:
                src.seek(start)
                remaining = end - start
                while remaining > 0:
                    buf = src.read(min(chunk, remaining))
                    if not buf:
                        raise ValueError(f"short read in {shard_path}")
                    out.write(buf)
                    remaining -= len(buf)
        out.flush()

        os.fsync(out.fileno())  # durable before rename
    tmp.rename(merged)
    logger.info("reassembled sharded single-file checkpoint: %s (%d shards, %d tensors, %d bytes)",
                merged.name, len(shard_names), len(entries), offset)
    return merged


# --- fp8 download stays the fp8 storage lane ----------------------
# NEVER REBUILD the "voluntary upgrade" that upcast an fp8 download to plain
# bf16-resident weights when free VRAM allowed. The serving lane must be
# deterministic per (release x declared config), never a function of the
# individual card's free VRAM: such a probe makes `lane` a GPU-dependent axis
# of the compiled graph key, so a card with a small VRAM surplus over its same-SKU peers
# falls into a lane nothing mints for and serves eager for life. The tax it
# dodged is only +1.9% for the structural storage lane.
# Involuntary transitions stay: the fit-ladder rung below (can't-fit fp8)
# and the w8a8/w4a4 dequant-on-unsupported-host lanes are declared rungs, not
# probe outcomes.

# The pipeline's weight lane, part of the compile-cache graph key:
# "" = plain resident weights (incl. the involuntary w8a8/w4a4 dequant
# lanes), "fp8-hooks" = fp8 weights resident with a per-layer upcast (traced
# INTO the FX graphs). The "fp8-hooks" spelling is the WIRE value — tensorhub
# maps it to `w8a16` and compiled graphs key on it — so it must stay byte-identical.
_WEIGHT_LANE_ATTR = "_cozy_weight_lane"

#: EVERY base lane a loader can leave on ``_WEIGHT_LANE_ATTR``.
#:
#: THE single source of this vocabulary.
#: ``tests/test_speculative_execution_lane_completeness_pgw918.py`` parses every
#: assignment site under ``gen_worker/models`` and fails if a loader stamps a
#: lane this tuple does not name.
#:
#: ``"bf16-resident"`` is deliberately absent: :func:`pipeline_weight_lane`
#: folds it to ``""`` (it traces identically to plain bf16), so it is never a
#: distinct compiled graph-identity lane. Bucketed LoRA lanes
#: (``w8a8_lora.lora_execution_lane``) are these bases with a rank suffix and are
#: decomposed by ``compile_cache.execution_lane_bucket``, so the BASE set is complete.
STAMPABLE_BASE_EXECUTION_LANES: Tuple[str, ...] = (
    "",              # loading.py (plain/bf16-resident, folded)
    "fp8-hooks",     # loading.py fp8 storage cast
    "w8a8",          # models/w8a8.py
    "w4a4",          # models/w4a4.py
    "svdq-native",   # models/svdq_native.py
)


def pipeline_weight_lane(pipeline: Any) -> str:
    execution_lane = str(getattr(pipeline, _WEIGHT_LANE_ATTR, "") or "")
    if execution_lane == "bf16-resident":
        return ""  # traces identically to plain bf16
    if execution_lane:
        return execution_lane
    for name in _fp8_storage_components():
        if getattr(getattr(pipeline, name, None), "_cozy_fp8_storage_applied", False):
            return "fp8-hooks"
    if getattr(pipeline, "_cozy_fp8_storage_applied", False):
        return "fp8-hooks"
    return ""


# --- The runtime fit rung -------------------------------------------------
# Fit ladder: bf16 -> fp8 flavor -> nvfp4 (Blackwell) -> runtime fp8-E4M3
# storage -> CPU offload. When even the downloaded flavor cannot fit free
# VRAM, the load path tries fp8-E4M3 weight storage (apply_fp8_storage: fp8
# bytes resident, bf16 compute — quality ~= a stored #fp8 flavor) and then
# hands the slot to the offload ladder. Nothing QUANTIZES here: the bnb-nf4
# emergency rung was deleted in pgw#1206 D (Paul: "We shouldn't be doing
# runtime quants; if we're really memory-constrained then we should be
# fetching the quant we need"). Armed on CUDA hosts (gw#420: fitting is the
# runtime's job, not a flag).
# th#1867: there is no fp8-storage fit FACTOR. The rung is entered from a
# measured load, never from a factor times a declared card size.
_EMERGENCY_MARGIN_GB = 2.0


def runtime_fp8_storage_supported() -> bool:
    """The runtime fp8-E4M3 storage rung needs a CUDA host and a torch that
    ships the float8_e4m3fn dtype — no fp8 silicon required (per-layer upcast
    compute; see apply_fp8_storage)."""
    try:
        import torch

        return bool(cuda_ready()) and hasattr(torch, "float8_e4m3fn")
    except ImportError:
        return False


def model_index_components(path: str | Path) -> set:
    """Component names the snapshot's model_index.json declares. Empty set
    when there is no readable model_index.json (single-file checkpoints,
    transformers layouts)."""
    try:
        with open(Path(path) / "model_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
        return {k for k in index if not k.startswith("_")}
    except Exception:
        return set()


def model_index_component_classes(path: str | Path) -> Dict[str, str]:
    """``{component: class name}`` the snapshot's ``model_index.json`` declares.

    The authoritative component-class vocabulary at LOAD time: a
    fine-tune may substitute a class, and the bytes on disk decide. Empty when
    there is no readable ``model_index.json`` (single-file checkpoints,
    transformers layouts)."""
    out: Dict[str, str] = {}
    try:
        with open(Path(path) / "model_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return out
    if not isinstance(index, dict):
        return out
    for key, entry in index.items():
        if str(key).startswith("_"):
            continue
        if (isinstance(entry, (list, tuple)) and len(entry) == 2
                and all(isinstance(e, str) and e for e in entry)):
            out[str(key)] = entry[1]
    return out


def component_load_dtypes(
    pipeline_cls: Any, path: str | Path,
) -> Dict[str, Any]:
    """``{component: ComponentDtype}`` this composition's parts require at LOAD
    time — the snapshot's own ``model_index.json`` classes first, the
    pipeline class's ``__init__`` annotations as the fallback.

    Empty for every uniform composition, which is the common case: the caller
    then keeps its single scalar ``torch_dtype`` and nothing changes."""
    # CYCLE: api.tree reaches back into loading (FP8_STORAGE_FIT_FACTOR).
    from ..api.tree import component_dtypes

    cls = pipeline_cls if isinstance(pipeline_cls, type) else None
    return dict(component_dtypes(
        cls, model_index_classes=model_index_component_classes(path),
    ))


def model_index_entry(path: str | Path, component: str) -> Optional[tuple]:
    """``(library, class_name)`` the tree's model_index.json declares for
    ``component``, or None when absent/unreadable."""
    try:
        with open(Path(path) / "model_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
        entry = index.get(component)
        if (isinstance(entry, (list, tuple)) and len(entry) == 2
                and all(isinstance(e, str) and e for e in entry)):
            return (entry[0], entry[1])
    except Exception:
        pass
    return None


#: Compute dtype of the quantized-artifact lanes when the binding declares
#: none — MUST equal the ``compute_dtype or torch.bfloat16`` default in
#: load_w8a8_pipeline / load_w4a4_pipeline / the svdq lane (test-guarded).
QUANT_EXECUTION_LANE_COMPUTE_DEFAULT = "bf16"


def composition_compute_dtype(base_path: str | Path, dtype: str = "") -> str:
    """The compute dtype the COMPOSED pipeline will run at: the base binding's
    declared dtype when present, else the dtype the base tree's LOAD LANE
    actually computes at. ``""`` = unknown (an fp32-defaulting composition).

    Lane selection mirrors :func:`load_from_pretrained`: a quantized-artifact
    tree (svdq / w8a8 / w4a4) computes at the lane's bf16 default regardless
    of the tree's MAJORITY on-disk dtype — a produced w8a8 flavor quantizes
    only the repeated-block Linears and passes every other tensor through at
    SOURCE precision, so a fine-tune mirrored from an fp16 upstream sniffs
    majority-fp16 while its pipeline loads ``torch_dtype=bf16``. Sniffing the
    majority there yields an fp16 component override inside a bf16
    composition, and every forward dies on the dtype mismatch."""
    if dtype:
        return dtype
    base = Path(base_path)
    if (
        detect_svdq_artifact(base) is not None
        or detect_w8a8_artifact(base) is not None
        or detect_w4a4_artifact(base) is not None
    ):
        return QUANT_EXECUTION_LANE_COMPUTE_DEFAULT
    sniffed = detect_on_disk_dtype(base)
    if sniffed == "fp8":
        # Scale-free fp8 storage flavors compute at the bf16 default;
        # per-layer storage precision is restored separately
        # (apply_fp8_storage).
        return "bf16"
    if sniffed in ("bf16", "fp16"):
        return sniffed
    return ""


class MixedComputeDtypeError(RuntimeError):
    """A composed pipeline presents more than one COMPUTE dtype to its GEMMs.

    torch's matmul/conv kernels take no dtype opinion from the module — they
    raise mid-forward, with a message that names neither the tensor nor the
    component::

        RuntimeError: mat1 and mat2 must have the same dtype, but got
        BFloat16 and Half      (an nn.Linear WITH bias: addmm)
        RuntimeError: Input type (c10::BFloat16) and bias type (c10::Half)
        should be the same     (a conv)

    This error is raised at LOAD instead, naming the component, the parameter
    path and both dtypes.
    """


#: Dtypes a GEMM can actually compute in. Storage-only dtypes (fp8/fp4/int)
#: are excluded on purpose: the w8a8/svdq/nvfp4 lanes and diffusers' layerwise
#: casting hold weights at those precisions BY DESIGN and upcast per forward.
_COMPUTE_DTYPE_NAMES = ("float16", "bfloat16", "float32", "float64")
#: The pair that cannot interoperate and never legitimately coexists in one
#: composition. fp32 is the DECLARED widening axis and is reported
#: but never fatal — widening is a precision decision, not a dtype collision.
_INCOMPATIBLE_COMPUTE = ("float16", "bfloat16")


def _gemm_param_dtypes(module: Any) -> Dict[str, str]:
    """``{parameter path: dtype name}`` for every GEMM input under ``module``
    — Linear/conv weights and biases, the tensors torch actually refuses to
    mix, plus every quantized leaf's DECLARED ``compute_dtype``. Norms and
    embeddings are excluded: they carry their own (legitimately wider)
    precision and never meet a weight in one kernel.

    An isinstance selector alone is BLIND to the quantized lanes: all five
    quantized leaves (``_Fp8ScaledLinear``, ``_W4A4Linear``, ``_SvdqLinear``,
    ``_SvdqFusedLinear``, ``_AwqPackedLinear``) subclass ``nn.Module``
    directly and would read as ``{}``. Their ``self.compute_dtype`` is what
    every one of their forwards computes in (and what its bias must match),
    so it counts as a GEMM input dtype whether or not a bias tensor exists.
    """
    import torch.nn as nn

    gemm_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    out: Dict[str, str] = {}
    for name, leaf in module.named_modules():
        declared = getattr(leaf, "compute_dtype", None)
        # An embedding declares one on the fp8-storage lane
        # (`restructure_fp8_storage` stamps every covered leaf, embeddings
        # included). It stays excluded — the exclusion is about the kernel it
        # feeds, not about how it stores its rows.
        if isinstance(leaf, nn.Embedding):
            continue
        if not isinstance(leaf, gemm_types) and declared is None:
            continue
        for attr in ("weight", "bias"):
            t = getattr(leaf, attr, None)
            dt = getattr(t, "dtype", None)
            if dt is None:
                continue
            dt_name = str(dt).rsplit(".", 1)[-1]
            if dt_name in _COMPUTE_DTYPE_NAMES:
                out[f"{name}.{attr}" if name else attr] = dt_name
        # Storage dtypes stay uncounted: a leaf declaring an fp8
        # `compute_dtype` fails the membership test as its fp8 weight does.
        dec_name = str(declared).rsplit(".", 1)[-1] if declared is not None else ""
        if dec_name in _COMPUTE_DTYPE_NAMES:
            out[f"{name}.compute_dtype" if name else "compute_dtype"] = dec_name
    return out


def assert_uniform_compute_dtype(
    obj: Any, expected: str = "", *, label: str = "",
) -> None:
    """Refuse a MIXED-precision composition at LOAD.

    Checks every GEMM input of every component: an fp16 weight and a bf16
    weight in one composition means some forward will die on a dtype the
    kernel cannot reconcile, and torch's message names nothing. Two verdicts
    are fatal:

    1. a single component that is internally fp16 AND bf16;
    2. any component whose compute dtype is fp16/bf16 but is not ``expected``
       (the composition's own compute dtype) — the cross-composition aliasing
       shape: a content-keyed shared component loaded by another pick's record
       at ITS dtype and injected here unconverted.

    fp32 parts are legal (wider components are declared deliberately) and
    storage dtypes are legal (fp8/fp4 lanes upcast per forward), so neither
    is counted. Introspection failures never fail a load — only a proven
    collision does.
    """
    try:
        comps = getattr(obj, "components", None)
        if isinstance(comps, dict) and comps:
            parts = [(str(n), m) for n, m in comps.items()
                     if hasattr(m, "named_modules")]
        elif hasattr(obj, "named_modules"):
            parts = [(type(obj).__name__, obj)]
        else:
            return
        seen: Dict[str, Dict[str, str]] = {}
        for name, module in parts:
            dtypes = _gemm_param_dtypes(module)
            if dtypes:
                seen[name] = dtypes
    except Exception:  # introspection is best-effort; never fail a load on it
        logger.debug("compute-dtype invariant could not inspect %r", label,
                     exc_info=True)
        return

    what = label or type(obj).__name__
    for name, dtypes in seen.items():
        present = {d for d in dtypes.values() if d in _INCOMPATIBLE_COMPUTE}
        if len(present) > 1:
            offenders = sorted(
                f"{p}={d}" for p, d in dtypes.items()
                if d in _INCOMPATIBLE_COMPUTE
            )
            raise MixedComputeDtypeError(
                f"{what}: component {name!r} is internally mixed-precision "
                f"({'/'.join(sorted(present))}) — some forward will die on "
                f"`mat1 and mat2 must have the same dtype`. Offending GEMM "
                f"inputs: {offenders[:6]}"
            )
    if expected not in ("fp16", "bf16", "float16", "bfloat16"):
        return
    want = "float16" if expected in ("fp16", "float16") else "bfloat16"
    for name, dtypes in seen.items():
        wrong = sorted(
            f"{p}={d}" for p, d in dtypes.items()
            if d in _INCOMPATIBLE_COMPUTE and d != want
        )
        if wrong:
            raise MixedComputeDtypeError(
                f"{what}: component {name!r} loaded at "
                f"{wrong[0].rsplit('=', 1)[-1]} inside a {want} composition — "
                f"a foreign-precision component in a composed pipeline is the "
                f"pgw#683 warmup/serve fatal. Offending GEMM inputs: "
                f"{wrong[:6]}"
            )


def _component_dtype_map(
    cls: Any, path: str | Path, scalar_dtype: Any,
) -> Optional[Dict[str, Any]]:
    """diffusers' per-component ``torch_dtype`` map for this composition, or
    None when every part loads at the composition's own dtype.

    Shape is diffusers': ``{"default": <compute dtype>, "<part>": <wider
    dtype>}``. A part is included only when its declared load dtype DIFFERS
    from the composition default — a fact that agrees with the default is a
    no-op and stays out of the kwargs so nothing changes for uniform trees.
    Returns None on a torch-less host (the loader fails on its own terms) and
    on loaders with no component vocabulary at all.
    """
    facts = component_load_dtypes(cls, path)
    if not facts:
        return None
    try:
        default = scalar_dtype if scalar_dtype is not None else get_torch_dtype("bf16")
    except ImportError:
        return None
    out: Dict[str, Any] = {}
    for part, fact in facts.items():
        try:
            wanted = get_torch_dtype(fact.dtype)
        except ImportError:
            return None
        if wanted is default:
            continue
        out[part] = wanted
        logger.info(
            "COMPONENT_DTYPE model=%s: loading %r at %s (composition default "
            "%s) — %s", path, part, fact.dtype, default, fact.reason,
        )
    if not out:
        return None
    out["default"] = default
    return out


#: Load dtype a component asks for, keyed by what its OWN safetensors headers
#: store. fp8 is a STORAGE fact — the artifact carries its own
#: quantization config and bf16 is the compute dtype over it, which is why it
#: maps to :data:`QUANT_EXECUTION_LANE_COMPUTE_DEFAULT` rather than to itself.
_CHECKPOINT_LOAD_DTYPE = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp8": QUANT_EXECUTION_LANE_COMPUTE_DEFAULT,
}


def checkpoint_load_dtype(source: str | Path) -> str:
    """The dtype ONE component tree's own bytes ask to be loaded at, or ``""``
    when its headers say nothing.

    Read per COMPONENT, never per snapshot: a majority vote over a whole
    mixed-precision tree upcasts every narrow component when the vote lands
    wide and truncates every wide one when it lands narrow — and a tree-wide
    vote falling outside the map hands the load to diffusers' fp32 default
    (4 bytes/param)."""
    return _CHECKPOINT_LOAD_DTYPE.get(detect_on_disk_dtype(Path(source)), "")


def _declared_component_dtype(name: str, declared: Any) -> Any:
    """The dtype the CALLER declared for ``name`` — diffusers' own
    ``{"default": ..., "<part>": ...}`` routing, or a scalar that governs
    every component. None when nothing was declared for it."""
    if isinstance(declared, dict):
        if name in declared:
            return declared[name]
        return declared.get("default")
    return declared


def _modular_declared_dtypes(
    cls: Any, path: str | Path, scalar_dtype: Any,
) -> Any:
    """Everything the modular lane DECLARES about component dtypes: the
    binding's own dtype when it has one, plus pgw#667's per-part facts.

    ``None`` when nothing is declared — the hydration loop then reads each
    component's checkpoint. With a declared composition dtype this
    is exactly :func:`_component_dtype_map`; without one the facts stand
    alone, because a ``"default"`` key would put every unlisted component
    back under a guess."""
    if scalar_dtype is not None:
        return _component_dtype_map(cls, path, scalar_dtype) or scalar_dtype
    out: Dict[str, Any] = {}
    for part, fact in component_load_dtypes(cls, path).items():
        try:
            out[part] = get_torch_dtype(fact.dtype)
        except ImportError:
            return None
        logger.info(
            "COMPONENT_DTYPE model=%s: loading %r at %s (no composition "
            "dtype declared; every other component loads at its own "
            "checkpoint dtype) — %s", path, part, fact.dtype, fact.reason,
        )
    return out or None


def _hydration_dtype(name: str, declared: Any, source: str | Path) -> Any:
    """Load dtype for ONE modular component: what the caller declared, else
    what the component's own checkpoint stores, else None (nothing is known,
    so diffusers keeps its own default and the load stays as honest as the
    bytes allow)."""
    wanted = _declared_component_dtype(name, declared)
    if wanted is not None:
        return wanted
    token = checkpoint_load_dtype(source)
    if not token:
        return None
    try:
        return get_torch_dtype(token)
    except ImportError:
        return None  # torch-less host: the loader fails on its own terms


class ComponentExecutionLaneUnsupported(RuntimeError):
    """This flavor has no component-level loader, so no honest one exists.

    svdq and gguf materialize their denoiser INSIDE the pipeline build (a
    nunchaku file / a single gguf checkpoint the pipeline class assembles).
    Handing back a plain ``from_pretrained`` module for those would not be
    what serving runs, so the component path refuses by name instead."""


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True when ``fn`` can take ``name=`` (declared or via ``**kwargs``).

    Deliberately NOT the ``except TypeError: retry without it`` idiom, which
    catches ANY construction-time TypeError — including one raised deep inside
    diffusers' quantization-config reconstruction — and retries a path that
    fails identically, hiding the real cause. Introspection failure means
    "pass it": the call then fails naming itself."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def load_component(
    tree: str | Path, component: str, *, dtype: str = "",
) -> Any:
    """THE production loader for ONE named pipeline component.

    Every caller that needs a single component — the rotation preloader, the
    swap benchmark — goes through here, so what they load is by construction
    what serving loads. ``tree`` is the composition: it names the module
    class (model_index.json), decides the compute dtype, and holds the bytes.

    Quantized artifacts take their OWN lane, exactly as
    :func:`load_from_pretrained` routes a whole pipeline: a w8a8/w4a4
    denoiser is materialized by its artifact loader. A modelopt-produced
    tree carries a ``quantization_config`` block diffusers reconstructs into
    ``NVIDIAModelOptConfig``, whose constructor requires a ``quant_type``
    the block does not supply — so a bare ``from_pretrained`` on the
    denoiser dies at config reconstruction on every flavor the fleet
    actually serves. Lanes with no component-level loader raise
    :class:`ComponentLaneUnsupported`.

    dtype resolution: the base binding's declared dtype wins; otherwise the
    component inherits the BASE COMPOSITION's compute dtype
    (:func:`composition_compute_dtype`); the weights' own on-disk dtype is
    only the last resort. Hub-resolved bindings carry no dtype, so falling
    back to the override's on-disk dtype loads e.g. an fp32-stored VAE into a
    bf16 pipeline and setup dies on the first latent. Blocking; callers on an
    event loop run it off-thread."""

    base = Path(tree)
    root = base
    entry = model_index_entry(base, component)
    if entry is None:
        raise ValueError(
            f"component {component!r} is not in the base composition "
            f"(model_index components: "
            f"{sorted(model_index_components(base))})"
        )
    library, class_name = entry
    module = importlib.import_module(library)
    cls = getattr(module, class_name, None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise ValueError(
            f"{library}.{class_name} declares no from_pretrained loader"
        )
    src = root / component
    if not src.is_dir():
        src = root

    # a component with a declared load-dtype fact keeps it when it is
    # SUBSTITUTED too — the fact is a property of the component class, and the
    # substituted tree's part must be resident at the same precision the base
    # part required or the composition is silently degraded.
    fact = component_dtype_for_class(class_name)
    wanted = (
        (fact.dtype if fact is not None else "")
        or composition_compute_dtype(base, dtype)
        or detect_on_disk_dtype(src)
    )
    torch_dtype: Any = None
    if wanted in ("bf16", "fp16", "bfloat16", "float16", "fp32", "float32"):
        try:
            torch_dtype = get_torch_dtype(wanted)
        except ImportError:
            pass  # torch-less environment: loader fails on its own terms

    contracted = contract_loaded_component(
        root, component, cls=cls, compute_dtype=torch_dtype, src=src)
    if contracted is not None:
        return contracted

    kwargs: Dict[str, Any] = {}
    if torch_dtype is not None and _accepts_kwarg(
            cls.from_pretrained, "torch_dtype"):
        kwargs["torch_dtype"] = torch_dtype
    return cls.from_pretrained(
        str(third_party_dir(src, why="contract_loaded_component from_pretrained")),
        **kwargs)


def contract_loaded_component(
    root: Path, component: str, *, cls: Any, compute_dtype: Any = None,
    src: Optional[Path] = None,
) -> Optional[Any]:
    """THE contract dispatch: one component's tree -> its registered loader.

    ``None`` means this tree declares no layout contract and the caller's own
    generic load is correct. A module means the tree's layout was recognised
    and the SDK's declaring loader built it. A raise means the layout IS
    recognised and has no component-level loader — a typed refusal, never a
    fall-through.

    ONE dispatch, deliberately shared by the non-modular and modular entry
    points rather than duplicated: the same artifact bytes reach both, and two
    copies that agree today drift into a modular hydration that falls through
    to plain ``from_pretrained`` and dies inside ``DiffusersAutoQuantizer`` on
    the tree's own ``quantization_config``.

    The silent fall-through is the defect. A contract-bearing tree that is
    loaded generically does not fail loudly — it produces a module whose
    numerics cannot serve, or an exception from three libraries away that
    names none of this.
    """
    weights = Path(root)
    where = Path(src) if src is not None else weights

    def _covers(artifact_component: str) -> bool:
        """The artifact's weight set IS this component: a diffusers tree
        names it, a bare override tree (root layout) has nothing else in
        it."""
        return artifact_component == component or (
            not artifact_component and where == weights
        )

    w8a8_art = detect_w8a8_artifact(weights)
    if w8a8_art is not None and _covers(w8a8_art.component):
        require_decodable(
            CONTRACT_COZY_FP8_ROWWISE, weights, component=w8a8_art.component)
        return load_w8a8_denoiser(
            weights, w8a8_art, compute_dtype=compute_dtype, cls=cls)
    w4a4_art = detect_w4a4_artifact(weights)
    if w4a4_art is not None and _covers(w4a4_art.component):
        return load_w4a4_denoiser(
            weights, w4a4_art, compute_dtype=compute_dtype, cls=cls)
    svdq_art = detect_svdq_artifact(weights)
    if svdq_art is not None and _covers(svdq_art.component):
        raise ComponentExecutionLaneUnsupported(
            f"component {component!r} of {weights} is an "
            f"svdq-{svdq_art.precision} artifact ({svdq_art.file.name}): its "
            f"denoiser is built by the svdq engine during the PIPELINE load, "
            f"so there is no component-level production loader to borrow"
        )
    if component in denoiser_components() and detect_gguf_snapshot(weights):
        raise ComponentExecutionLaneUnsupported(
            f"component {component!r} of {weights} is a GGUF denoiser: it is "
            f"dequantized by the pipeline's own gguf loader, so there is no "
            f"component-level production loader to borrow"
        )
    # hf.fp8-blockwise@1: a transformers component tree (the conditioner /
    # text encoder position), so it is detected on the component dir and
    # after the denoiser lanes above. Without this arm the tree loads
    # GENERICALLY — transformers reads `quantization_config` out of
    # config.json by itself, so nothing fails and the image's declared
    # decoder for this contract never runs. Measured (pgw#1253): the arm the
    # decode-set declares was reached by nothing, so a transposed or rowwise
    # scale grid was accepted here instead of refused by name, and the
    # resident-vs-upcast lane was transformers' default rather than this
    # image's declaration.
    blockwise = detect_hf_fp8_blockwise(where)
    if blockwise is not None:
        require_decodable(
            CONTRACT_HF_FP8_BLOCKWISE, weights,
            component=component if where != weights else "")
        return load_hf_fp8_blockwise(
            where, cls=cls, dtype=compute_dtype, tree=blockwise)
    return None


class ModularHydrationError(RuntimeError):
    """A ModularPipeline slot could not be hydrated from the LOCAL tree
. Typed so the failure is a refusal at load — never a silent
    shell handed to ``setup()``, and never a fetch from the repo id the
    snapshot's index happens to name."""


def _pipeline_component_names(cls: Any) -> Optional[set]:
    """Component names the pipeline CLASS will actually construct, by
    diffusers' OWN rule (``_get_signature_keys``: required ``__init__``
    parameters plus the declared optional components), falling back to the
    ``__init__`` signature for classes that predate it.

    None when the class cannot be introspected (``**kwargs`` catch-alls), in
    which case the index is judged in full. Judging the index alone would
    refuse a load diffusers would have completed: a component the index names
    and the signature does not is one ``from_pretrained`` never touches."""
    getter = getattr(cls, "_get_signature_keys", None)
    if callable(getter):
        try:
            expected, _optional = getter(cls)
            return set(expected)
        except Exception:  # noqa: BLE001 — any introspection gap => judge all
            pass
    init = getattr(cls, "__init__", None)
    if init is None:
        return None
    try:
        params = inspect.signature(init).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    names = {
        name for name, p in params.items()
        if name != "self"
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return names or None


def is_modular_pipeline_class(cls: Any) -> bool:
    """Duck-typed: a modular pipeline class exposes ``load_components``
    (weights hydrate AFTER construction) — ``DiffusionPipeline`` does not."""
    return (
        isinstance(cls, type)
        and callable(getattr(cls, "from_pretrained", None))
        and callable(getattr(cls, "load_components", None))
    )


def _is_modular_pipeline(obj: Any) -> bool:
    return (
        hasattr(obj, "_component_specs")
        and callable(getattr(obj, "load_components", None))
    )


def _local_component_dir(base: Path, spec: Any, name: str) -> Optional[Path]:
    """The LOCAL snapshot dir for a component spec: the spec's subfolder
    under the snapshot root (falling back to the component name). None when
    the snapshot has no such dir at all."""
    for sub in (str(getattr(spec, "subfolder", "") or ""), name):
        if not sub:
            continue
        cand = base / sub
        if cand.is_dir():
            return cand
    return None


def _weightless_model_dir(src: Path) -> bool:
    """True for a config-only weight-bearing dir — the deliberate-partition
    shape (a pin by FILE SET keeps ``config.json`` for the unselected
    partition): the component is EXCLUDED from this slot, not missing."""
    if not (src / "config.json").is_file():
        return False  # tokenizer/processor/scheduler dirs: no model config
    return next(src.rglob("*.safetensors"), None) is None


_GIB = 1024 ** 3
# Mirrors residency/staging's host-RAM floor policy.
_STAGING_FLOOR_GB = 8.0
_STAGING_FLOOR_FRACTION = 0.2


def _staging_floor_bytes(total_bytes: int) -> int:
    if total_bytes <= 0:
        return int(_STAGING_FLOOR_GB * _GIB)
    return int(min(_STAGING_FLOOR_GB * _GIB,
                   max(_GIB, total_bytes * _STAGING_FLOOR_FRACTION)))


def _admit_component_staging(component: str, nbytes: int) -> None:
    """Admit ONE component's staging against the cgroup budget.

    ``probe_host_ram`` speaks cgroup (v1 and v2) and credits clean reclaimable
    page cache, so the just-fetched tree's own cache never blocks its own
    load. A component that cannot fit an EMPTY host is the structural verdict;
    one that cannot fit right now is the transient one. Both carry the
    measured numbers. An unreadable probe fails open.

    The estimate can be wrong (an upcast, a quant unpack, allocator overhead),
    and when it is the load does not fail — it crawls in direct reclaim until
    the kernel kills it. So a MEASURED verdict outranks this arithmetic: a
    process the load dial has caught re-reading its own set instead of staging
    it admits nothing further, whatever the numbers below would have said."""
    if nbytes <= 0:
        return
    ram = probe_host_ram()
    total = int(ram.total_gb * _GIB)
    avail = int(ram.available_gb * _GIB)
    thrash = load_progress.thrash_verdict()
    if thrash:
        raise HostRamCapacityError(
            f"modular component {component!r} after a measured re-read "
            f"crawl ({thrash})",
            incoming_bytes=int(nbytes), floor_bytes=0,
            required_bytes=int(nbytes), available_before_bytes=avail,
            available_after_bytes=avail, total_bytes=total,
        )
    if total <= 0:
        return
    floor = _staging_floor_bytes(total)
    required = int(nbytes) + floor
    label = f"modular component {component!r}"
    cls = (HostRamCapacityError if required > total
           else InsufficientHostRamError if required > avail else None)
    if cls is not None:
        raise cls(
            label, incoming_bytes=int(nbytes), floor_bytes=floor,
            required_bytes=required, available_before_bytes=avail,
            available_after_bytes=avail, total_bytes=total,
        )


def modular_staging_units(base: Path) -> Dict[str, int]:
    """Bytes per INDEPENDENTLY STAGED unit of a modular snapshot.

    :func:`hydrate_modular_pipeline` loads one component at a time from that
    component's own source dir, so the unit of host-RAM staging is a
    component dir — never the tree. This reads the same sources the hydration
    loop will: each ``modular_model_index.json`` entry's subfolder under
    ``base`` (falling back to the component name).

    Empty when the index is absent or unreadable: callers treat that as "no
    per-component knowledge" and keep whole-tree accounting."""
    index = Path(base) / "modular_model_index.json"
    try:
        entries = json.loads(index.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    units: Dict[str, int] = {}
    for name, entry in entries.items():
        if str(name).startswith("_"):
            continue
        sub = ""
        if isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[2], dict):
            sub = str(entry[2].get("subfolder") or "")
        src = None
        for cand in (sub, str(name)):
            if cand and (Path(base) / cand).is_dir():
                src = Path(base) / cand
                break
        if src is None:
            continue
        units[str(name)] = disk_gc.tree_bytes(src)
    return units


# The card must hold the tree with room to spare before hydration is allowed
# to place components as they land. THREAT (§4.24): free VRAM is read once,
# before the first component loads, and another tenant can take some of it
# before the last one is placed — a `.to(device)` that OOMs mid-hydration
# leaves a half-placed pipeline. This is the same 2 GB the placement ladder
# holds back (`memory._DEFAULT_SAFETY_MARGIN_GB`); it is not a fudge factor
# for estimate error, since these bytes are measured on disk, not estimated.
_STREAMED_HYDRATION_VRAM_MARGIN_GB = 2.0


@dataclass(frozen=True)
class StreamedHydrationPlan:
    """Whether a modular slot may hydrate component-by-component ONTO THE
    DEVICE, so host RAM never holds more than one component at a time."""

    engaged: bool
    reason: str
    tree_bytes: int
    largest_unit_bytes: int
    unit_count: int
    host_total_bytes: int
    device_free_bytes: int
    #: The rung the pipeline will be placed on. An offload rung
    #: keeps the weights in host RAM, so it never takes the discount.
    placement_mode: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "engaged": self.engaged, "reason": self.reason,
            "tree_bytes": self.tree_bytes,
            "largest_unit_bytes": self.largest_unit_bytes,
            "unit_count": self.unit_count,
            "host_total_bytes": self.host_total_bytes,
            "device_free_bytes": self.device_free_bytes,
            "placement_mode": self.placement_mode,
        }

    def summary(self) -> str:
        return (
            f"tree={self.tree_bytes / _GIB:.1f}GiB "
            f"largest_component={self.largest_unit_bytes / _GIB:.1f}GiB "
            f"host_total={self.host_total_bytes / _GIB:.1f}GiB "
            f"device_free={self.device_free_bytes / _GIB:.1f}GiB "
            + (f"rung={self.placement_mode} " if self.placement_mode else "")
            + f"({self.reason})"
        )


def plan_streamed_hydration(
    base: Path,
    *,
    device_free_bytes: Optional[int] = None,
    placement_mode: str = "",
) -> StreamedHydrationPlan:
    """Decide whether this modular slot stages PER COMPONENT ONTO THE DEVICE
    instead of staging its whole tree in host RAM first.

    THREAT (§4.25): a tree the CARD holds but the HOST does not would
    otherwise be refused structurally at boot, and no pod size fixes it —
    host RAM can bind tighter than VRAM purely because staging is
    all-or-nothing while the load is already component-sequential.

    THE OBSERVABLES, all measured rather than estimated:

    1. the whole tree does NOT fit host RAM (bytes on disk vs
       :func:`probe_host_ram`'s cgroup-aware total plus the staging floor) —
       otherwise nothing is wrong and the whole-tree path stands;
    2. the LARGEST single component DOES fit host RAM — otherwise the
       structural refusal is honest and must survive (no amount of
       sequencing places a component that cannot be staged at all);
    3. free VRAM holds the whole tree with :data:`
       _STREAMED_HYDRATION_VRAM_MARGIN_GB` to spare — the components have to
       go somewhere, and on this path that somewhere is the card.

    Engaged only on all three. Every other answer keeps today's behaviour,
    including the refusal, so this can only turn a refusal into a boot."""
    units = modular_staging_units(Path(base))
    if device_free_bytes is None:
        device_free_bytes = int(get_available_vram_gb() * _GIB)
    return decide_streamed_hydration(
        tree_bytes=sum(units.values()),
        largest_unit_bytes=max(units.values(), default=0),
        unit_count=len(units),
        host_total_bytes=int(probe_host_ram().total_gb * _GIB),
        device_free_bytes=int(device_free_bytes),
        placement_mode=placement_mode,
    )


def decide_streamed_hydration(
    *,
    tree_bytes: int,
    largest_unit_bytes: int,
    unit_count: int,
    host_total_bytes: int,
    device_free_bytes: int,
    placement_mode: str = "",
) -> StreamedHydrationPlan:
    """:func:`plan_streamed_hydration`'s decision, separated from its
    measurements so the rule can be read — and tested — at the byte counts
    that produced the issue rather than at whatever this host happens to
    have.

    ``placement_mode`` is the rung the pipeline will be PLACED on. A rung
    that keeps weights in host RAM (any CPU-offload rung, including the
    sticky floor an OOM degrade learned) cannot take this discount at all —
    see the refusal below."""
    tree = int(tree_bytes)
    largest = int(largest_unit_bytes)
    host_total = int(host_total_bytes)
    plan = functools.partial(
        StreamedHydrationPlan, tree_bytes=tree, largest_unit_bytes=largest,
        unit_count=int(unit_count), host_total_bytes=host_total,
        device_free_bytes=int(device_free_bytes),
        placement_mode=str(placement_mode or ""),
    )
    if touches_host_ram(placement_mode):
        # The discount is admissible ONLY because each component leaves the
        # host for the card. An offload rung puts it back — the weights live
        # on the host by definition — so the honest requirement is the whole
        # tree; charging one component here admits a re-stage into a cgroup
        # that cannot hold it, and the OOM kill is certain from minute zero.
        return plan(
            engaged=False,
            reason=f"placement rung {placement_mode!r} keeps the weights in "
                   f"host RAM: an offloaded pipeline is charged its whole "
                   f"tree, never one component")
    if unit_count <= 0 or largest <= 0:
        return plan(engaged=False, reason="no per-component staging units")
    if host_total <= 0:
        return plan(engaged=False, reason="host RAM total unreadable")
    floor = _staging_floor_bytes(host_total)
    if tree + floor <= host_total:
        return plan(engaged=False, reason="the whole tree fits host RAM")
    if largest + floor > host_total:
        return plan(
            engaged=False,
            reason="the largest component alone exceeds host RAM")
    need = tree + int(_STREAMED_HYDRATION_VRAM_MARGIN_GB * _GIB)
    if int(device_free_bytes) < need:
        return plan(
            engaged=False, reason="the device does not hold the whole tree")
    return plan(
        engaged=True,
        reason="tree exceeds host RAM, largest component fits, device holds "
               "the tree")


def _place_and_release(pipe: Any, name: str, device: str) -> None:
    """Move one just-hydrated component to ``device`` and drop the host copy.

    This is what makes the per-component staging loop a per-component HIGH
    WATER MARK rather than just an ordering: without it every hydrated
    component stays in host RAM until placement, so the tree is resident on
    the host by the last component either way."""
    comp = getattr(pipe, name, None)
    to = getattr(comp, "to", None)
    if comp is None or not callable(to):
        return
    try:
        to(device)
    except Exception as exc:
        raise ModularHydrationError(
            f"per-component staging could not place component {name!r} on "
            f"{device!r}: {type(exc).__name__}: {exc}. The tree does not fit "
            f"host RAM, so there is no whole-tree path to fall back to — the "
            f"device has to hold it."
        ) from exc
    del comp, to
    flush_memory()


def _contract_component_for_spec(
    spec: Any, name: str, src: Path,
) -> Optional[Any]:
    """A modular component's contract loader, or ``None`` for a plain tree.

    The class comes from the spec (`ComponentSpec.type_hint`) because a modular
    pipeline has no `model_index.json` to name it — which is precisely why
    :func:`load_component` cannot simply be called here and why the DISPATCH,
    not the whole loader, is what the two paths share.

    A spec with no class at all is left to diffusers rather than guessed at: a
    contract loader needs a class to build, and inventing one would be a second
    source of truth for what a component IS.

    The class is NOT required to expose ``from_pretrained``. The contract
    loaders build through ``load_config`` + ``from_config``, so demanding the
    generic entry point here would refuse exactly the classes this path exists
    to serve.
    """
    cls = getattr(spec, "type_hint", None)
    if cls is None or not isinstance(cls, type):
        return None
    return contract_loaded_component(src, name, cls=cls, src=src)


def hydrate_modular_pipeline(
    pipe: Any,
    path: str | Path,
    *,
    torch_dtype: Any = None,
    preloaded: Optional[Dict[str, Any]] = None,
    place_device: str = "",
) -> Dict[str, str]:
    """Hydrate a freshly constructed ``ModularPipeline`` from the LOCAL
    snapshot tree.

    ``ModularPipeline.__init__`` registers every ``from_pretrained``
    component as ``None`` and copies each spec's
    ``pretrained_model_name_or_path`` verbatim out of the snapshot's index —
    which on a mirrored repo names the UPSTREAM repo id. This function is the
    hydration guard: every spec is re-pointed at the local tree BEFORE
    ``load_components`` runs, so neither this load nor any later endpoint-side
    ``pipe.load_components()`` can reach huggingface.co.

    - base components load from ``<snapshot>/<subfolder>``;
    - a config-only weight-bearing dir is the unselected partition: SKIPPED,
      its spec neutralized so nothing can ever fetch it;
    - a component the index names but the snapshot does not carry refuses
      typed (:class:`ModularHydrationError`) — never a fetch;
    - ``preloaded`` shared modules are registered via ``update_components``
      instead of the ``from_pretrained`` kwarg ``ModularPipeline.__init__``
      silently discards.

    Returns ``{component: source_path}`` for everything hydrated. The result
    is verified: ``load_components`` swallows load errors into a logger
    warning, so every requested component is re-checked non-``None`` and a
    miss raises typed with the captured diffusers log text.

    ``torch_dtype`` is what the caller DECLARED — a scalar governing every
    component, or diffusers' ``{"default": ..., "<part>": ...}`` map. What it
    does not name loads at that component's OWN checkpoint dtype
    (:func:`checkpoint_load_dtype`), never at a snapshot-wide majority and
    never at diffusers' fp32 default. ``_keep_in_fp32_modules``
    stays diffusers' business: naming a dtype is what lets it act at all.

    ``place_device`` moves each component onto that device as it
    lands and drops the host copy, so the host-RAM high-water mark is ONE
    component instead of the tree. Set it from
    :func:`plan_streamed_hydration`, never by hand: it is admissible only
    when the card holds the whole tree, and a mid-hydration placement
    failure is a typed refusal with no whole-tree path left to fall back
    to."""
    base = Path(path)
    specs = dict(getattr(pipe, "_component_specs", None) or {})
    pre = dict(preloaded or {})

    sources: Dict[str, str] = {}
    skipped: List[str] = []
    for name, spec in specs.items():
        if getattr(spec, "default_creation_method", "") != "from_pretrained":
            continue
        if name in pre:
            continue
        src = _local_component_dir(base, spec, name)
        if src is None:
            raise ModularHydrationError(
                f"component {name!r} is named by the snapshot's index "
                f"(pretrained_model_name_or_path="
                f"{getattr(spec, 'pretrained_model_name_or_path', None)!r}) "
                f"but the local tree {base} has no {name!r} dir — refusing "
                f"rather than fetching from the index's repo id")
        if _weightless_model_dir(src):
            skipped.append(name)
            continue
        # A third-party `ComponentSpec.load` reads this path itself, so it
        # gets real files. THE #1303 CENSUS MISSED THIS SITE: it counted
        # literal `from_pretrained(dir)` calls, and modular hydration hands
        # the directory to diffusers through a spec field instead.
        sources[name] = str(
            third_party_dir(src, why=f"ComponentSpec.load({name!r})"))

    # Re-point the SPECS first: a later bare `pipe.load_components()` (e.g.
    # endpoint-side) must be equally incapable of reaching the index's repo
    # id. Skipped partitions get None — load_components ignores a spec with
    # no path, and spec.load() refuses one outright.
    for name in sources:
        spec = specs[name]
        spec.pretrained_model_name_or_path = sources[name]
        spec.subfolder = ""
    for name in skipped:
        specs[name].pretrained_model_name_or_path = None
    for name in pre:
        # A preloaded shared module loads from no path at all;
        # update_components below replaces its spec, but neutralize first so
        # the guard can never read its stale upstream id as a fetch source.
        if name in specs:
            specs[name].pretrained_model_name_or_path = None

    for name, spec in specs.items():
        if getattr(spec, "default_creation_method", "") != "from_pretrained":
            continue
        p = getattr(spec, "pretrained_model_name_or_path", None)
        if p is not None and not Path(str(p)).exists():
            try:
                listing = sorted(q.name for q in base.iterdir())
            except OSError:
                listing = ["<unreadable>"]
            raise ModularHydrationError(
                f"component {name!r} spec still names a non-local source "
                f"{p!r} after re-pointing — refusing rather than fetching "
                f"(base tree {base} holds: {listing})")

    names = sorted(sources)
    dtypes: Dict[str, str] = {}
    if names:
        # load_components swallows per-component load failures into a
        # diffusers logger warning and registers nothing — capture that text
        # so the typed refusal below can carry the actual cause.
        records: List[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        dlog = logging.getLogger("diffusers")
        handler = _Capture(level=logging.WARNING)
        dlog.addHandler(handler)
        try:
            # One component at a time, admission-checked against the CGROUP
            # budget before it stages, page cache chilled after it lands:
            # staged anon and the tree's own read cache share one cgroup
            # limit, so sequencing + chilling is what keeps the high-water at
            # anon + ONE component instead of anon + the whole tree.
            for n in names:
                comp_src = Path(sources[n])
                comp_bytes = disk_gc.tree_bytes(comp_src)
                load_progress.set_phase(f"hydrate:{n}", comp_bytes)
                _admit_component_staging(n, comp_bytes)
                kwargs: Dict[str, Any] = {
                    "pretrained_model_name_or_path": {n: sources[n]},
                    "subfolder": {n: ""},
                }
                # This component's OWN checkpoint dtype when the
                # caller declared none. Sniffed here rather than by the
                # caller because this is the only place that knows each
                # component's actual source dir — an override tree is a
                # different artifact from the base dir it replaces.
                dt = _hydration_dtype(n, torch_dtype, comp_src)
                if dt is not None:
                    kwargs["torch_dtype"] = dt
                    dtypes[n] = str(dt).removeprefix("torch.")
                # THE CONTRACT DISPATCH, on the modular path too.
                # `load_components` is diffusers' own hydration and reaches
                # plain `from_pretrained`, which never consults the SDK's
                # registered contract loaders — a produced fp8-rowwise tree
                # would die inside `DiffusersAutoQuantizer` on its own
                # `quantization_config`. The contract-loaded module is
                # injected through `update_components`, the same seam a
                # preloaded shared module uses.
                built = _contract_component_for_spec(specs[n], n, comp_src)
                if built is not None:
                    pipe.update_components(**{n: built})
                else:
                    pipe.load_components(names=[n], **kwargs)
                # place it now and drop the host copy, so the next
                # component's admission above sees the host RAM this one
                # gave back rather than the tree accumulating behind it.
                if place_device:
                    _place_and_release(pipe, n, place_device)
                disk_gc.reclaim_file_cache(comp_src)
        finally:
            dlog.removeHandler(handler)
        missing = [n for n in names if getattr(pipe, n, None) is None]
        if missing:
            detail = "\n".join(r for r in records if missing[0] in r) or \
                "\n".join(records[-3:])
            raise ModularHydrationError(
                f"modular hydration failed for component(s) {missing} of "
                f"{type(pipe).__name__} from local source(s) "
                f"{ {n: sources[n] for n in missing} }: {detail}")

    if pre:
        pipe.update_components(**pre)
        for name in pre:
            if getattr(pipe, name, None) is None:
                raise ModularHydrationError(
                    f"preloaded component {name!r} did not register on "
                    f"{type(pipe).__name__}")
            sources[name] = "<preloaded shared module>"

    try:
        pipe._cozy_modular_hydration = dict(sources)
    except Exception:  # noqa: BLE001
        pass
    detail = " ".join(f"{n}<-{sources[n]}" for n in sorted(sources))
    # The dtype each component actually loaded at belongs in the hub-visible
    # record, not only in a log line: an fp32 upcast is invisible otherwise.
    dtype_detail = " ".join(f"{n}={dtypes[n]}" for n in sorted(dtypes))
    logger.info("modular hydration (%s): %s; dtypes: %s; skipped partitions: "
                "%s%s",
                type(pipe).__name__, detail, dtype_detail or "loader default",
                skipped or "none",
                f"; staged per component onto {place_device}" if place_device
                else "")
    activity_mod.emit_event(
        activity_mod.KIND_MODULAR_HYDRATION,
        f"pipeline={type(pipe).__name__} base={base} {detail}"
        + (f" dtypes=[{dtype_detail}]" if dtype_detail else "")
        + (f" skipped={','.join(skipped)}" if skipped else "")
        + (f" place_device={place_device}" if place_device else ""),
        phase="hydrated",
    )
    return sources


def _safetensors_data_bytes(p: Path) -> int:

    header = read_header(
        p,
        why="tensor bytes per component size the VRAM plan; a stub read as "
            "zero bytes plans a model that is not there",
    )
    total = 0
    for value in header.values():
        if isinstance(value, dict) and "data_offsets" in value:
            s, e = value["data_offsets"]
            total += int(e) - int(s)
    return total


def snapshot_component_weight_bytes(model_path: Path) -> Dict[str, int]:
    """Tensor bytes per top-level component dir (header-declared data ranges;
    no tensor reads). Root-level files book under ``""``. Empty dict when
    undetectable."""
    out: Dict[str, int] = {}
    root = Path(model_path)
    try:
        for p in sorted(root.rglob("*.safetensors")):
            rel = p.relative_to(root)
            comp = rel.parts[0] if len(rel.parts) > 1 else ""
            out[comp] = out.get(comp, 0) + _safetensors_data_bytes(p)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in out.items() if v > 0}


# th#1867 deleted `specialized_weight_layout` with pgw#1117's envelope
# precondition. The lane detectors it wrapped live in the loader itself.

def _adaptive_fit_rung(
    cls: Any, path: Path, *, fp8_planned: bool, compute_dtype: Any = None
) -> tuple[str, Optional[Any]]:
    """Serve-time fit rung at load: when the snapshot's estimated resident
    bytes exceed free VRAM, engage runtime fp8-E4M3 weight storage if it makes
    the difference (denoiser weights ~halve). Otherwise the rung is SKIPPED
    and the OFFLOAD ladder carries the slot — a lower-precision serve is never
    manufactured here.

    Returns ``(mode, config)``: ``("", None)`` fits as planned, or no rung
    helps and offload takes it; ``("fp8", None)`` engage fp8 storage. The
    config slot stays in the signature because the caller's contract is
    "a rung may hand me a quantization_config" — an AOT artifact read off disk
    still does, through ``synthesize_quantization_config``.
    """
    if not runtime_fp8_storage_supported():
        return "", None

    free_gb = get_available_vram_gb()
    if free_gb <= 0:
        return "", None
    comp_bytes = snapshot_component_weight_bytes(path)
    total = sum(comp_bytes.values())
    if total <= 0:
        return "", None
    budget = max(0.0, free_gb - _EMERGENCY_MARGIN_GB) * float(1 << 30)

    named = model_index_components(path) or set(comp_bytes)
    denoisers = [c for c in _fp8_storage_components()
                 if c in named and comp_bytes.get(c, 0) > 0]
    denoiser_bytes = sum(comp_bytes[c] for c in denoisers)

    on_disk = detect_on_disk_dtype(path)
    resident = float(total)
    if fp8_planned and on_disk != "fp8":
        resident -= 0.5 * denoiser_bytes  # fp8 storage halves the denoiser
    total_gb = total / float(1 << 30)
    if resident <= budget:
        return "", None
    # fp8-storage rung: only for un-quantized bf16/fp16 snapshots (an already
    # quantized flavor can't be halved again) when the halved estimate fits.
    if not fp8_planned and on_disk in ("bf16", "fp16") and denoisers \
            and total - 0.5 * denoiser_bytes <= budget:
        logger.warning(
            "fp8-E4M3 emergency weight storage engaged for %s (%.1f GB "
            "weights, %.1f GB free) — near-native quality; a stored #fp8 "
            "flavor of this model would serve natively here.",
            path, total_gb, free_gb,
        )
        return "fp8", None
    logger.warning(
        "no runtime fit rung applies to %s (%.1f GB weights, %.1f GB free): "
        "serving at stored precision and letting the OFFLOAD ladder carry it "
        "— a smaller-precision serve is an AOT artifact to fetch, never one "
        "to manufacture here",
        path, total_gb, free_gb,
    )
    return "", None


def _load_modular_pipeline(
    cls: Any,
    path: str,
    *,
    dtype: str = "",
    storage_dtype: str = "",
    components: Optional[Dict[str, Any]] = None,
    placement_mode: str = "",
) -> Any:
    """The modular lane of :func:`load_from_pretrained`:
    ``cls.from_pretrained(path)`` builds a SHELL (every weight-bearing
    component ``None``, specs naming the index's repo id verbatim), then
    :func:`hydrate_modular_pipeline` re-points every spec at the local tree
    and loads the weights. The quant/storage rungs (svdq/w8a8/w4a4/gguf,
    fp8 storage, the adaptive fit ladder) are pipeline-lane mechanisms and do
    not apply here v1 — a quantized modular component arrives as its own
    self-describing component tree (e.g. a transformers FineGrainedFP8
    artifact) and loads natively through its spec."""
    if storage_dtype:
        logger.warning(
            "storage_dtype=%s ignored on the modular lane (component "
            "precision is a per-component artifact fact)", storage_dtype)
    # DECLARED dtypes only — deliberately NOT a snapshot-wide majority sniff,
    # which truncates a wide component when the vote falls narrow (an fp32 VAE
    # loading bf16) and, when the vote falls outside the sniff's vocabulary,
    # produces NO dtype at all and lets diffusers' fp32 default upcast the
    # whole tree. Undeclared components load at their OWN checkpoint dtype,
    # decided inside the hydration loop from each component's real source dir.
    scalar_dtype: Any = None
    if dtype:
        try:
            scalar_dtype = get_torch_dtype(dtype)
        except ImportError:
            pass  # torch-less environment: loaders fail on their own terms
    torch_dtype: Any = _modular_declared_dtypes(cls, path, scalar_dtype)
    # A tree the card holds but the host does not stages ONE COMPONENT AT A
    # TIME straight onto the device. Decided here, from the
    # same measurements the executor's admission gate reads, so the two
    # cannot disagree about which shape the load takes; if free VRAM moved
    # between them the per-component gate inside hydration still refuses
    # with the measured numbers rather than thrashing the host.
    plan = plan_streamed_hydration(Path(path), placement_mode=placement_mode)
    if plan.engaged:
        logger.info(
            "modular slot stages per component onto the device: %s",
            plan.summary())
    pipe = cls.from_pretrained(
        third_party_dir(path, why="modular pipeline shell from_pretrained"))
    if not _is_modular_pipeline(pipe):
        raise ModularHydrationError(
            f"{getattr(cls, '__name__', cls)}.from_pretrained returned "
            f"{type(pipe).__name__} without _component_specs/load_components;"
            f" cannot hydrate")
    hydrate_modular_pipeline(
        pipe, Path(path), torch_dtype=torch_dtype, preloaded=components,
        place_device="cuda" if plan.engaged else "",
    )
    unmaterialized = meta_tensors(pipe)
    if unmaterialized:
        raise RuntimeError(
            f"{type(pipe).__name__} load left {len(unmaterialized)} "
            f"unmaterialized meta tensors (e.g. {unmaterialized[:3]})"
        )
    return pipe


@implements_contract(
    contract=CONTRACT_PLAIN_BF16,
    serves=("bf16-w16a16", "fp8-w8a16"),
    composes_lora=True,
    decodes=DecodeDimensions(
        elements=(ELEMENT_BF16, ELEMENT_FP16, ELEMENT_FP32),
        # `none` is the DECLARATION, not an omission: dense weights carry no
        # scale tensors, and a tree that does carry them belongs to a
        # quantized contract whose own decoder reads them.
        scales=(SCALE_NONE,),
        # `from_pretrained` addresses tensors by the class's own parameter
        # names. `native.fused-qkv@1` is registered and DECLARED BY NOTHING
        # here: a minimax-native tree offered to this loader is refused by
        # name rather than dying as an md5 miss inside a detection helper.
        key_topologies=(KEYS_DIFFUSERS_SPLIT_QKV, KEYS_TRANSFORMERS_SPLIT_QKV),
        # BOTH, and each by its own entry point: a component-directory tree
        # goes to `from_pretrained`, and `_single_file_checkpoint` routes a
        # loose checkpoint to `from_single_file` (:2583).
        file_layouts=(MULTI_FILE, SINGLE_FILE),
        bakes=(),
    ),
    why="the dense-weights path: plain bf16 bytes are read as stored "
        "(bf16-w16a16), and `storage_dtype=fp8` restructures the SAME bytes "
        "into fp8 storage with per-layer upcast (fp8-w8a16). Both are "
        "adapter-branch-capable (gw#558).",
)
def load_from_pretrained(
    cls: Any,
    path: str | Path,
    *,
    dtype: str = "",
    attrs: Optional[Dict[str, str]] = None,
    storage_dtype: str = "",
    components: Optional[Dict[str, Any]] = None,
    ref: str = "",
    placement_mode: str = "",
) -> Any:
    """``cls.from_pretrained(path)`` with the standard trimmings: torch dtype
    from the binding's dtype string, on-disk variant detection, quant-library
    preload, and quant-config synthesis; single-file checkpoints route through
    ``cls.from_single_file``. ``storage_dtype="fp8"`` (or an fp8-stored
    snapshot) keeps denoiser weights in fp8 storage with per-layer upcast to
    the compute dtype; ``"fp8+te"`` extends that to the pipeline's text
    encoders (transformers-aware). When the snapshot cannot fit free VRAM as
    stored, the adaptive fit rung engages runtime fp8-E4M3 storage (automatic
    on CUDA hosts); below that, the offload ladder carries it.
    ``components`` are PRELOADED module objects (content-keyed shared
    components) forwarded to ``from_pretrained`` — diffusers skips loading
    those from disk and wires the given objects in. Used by the executor to
    satisfy pipeline-typed ``setup()`` annotations; endpoints may also call
    it. A modular pipeline class (one exposing ``load_components``) takes its
    own lane — construct, re-point every component spec at the LOCAL tree,
    hydrate.
    ``placement_mode`` is the rung the worker will place
    this pipeline on: an offload rung keeps the weights in host RAM, so the
    modular lane must not stage them onto the card and must not take the
    per-component host-RAM discount."""
    path = str(path)
    if is_modular_pipeline_class(cls):
        return _load_modular_pipeline(
            cls, path, dtype=dtype, storage_dtype=storage_dtype,
            components=components, placement_mode=placement_mode,
        )
    # SVDQuant 4-bit flavors: self-describing snapshots take the svdq lane —
    # our native decoder wired into the standard pipeline (pgw#1298 deleted
    # the nunchaku engine; "nunchaku" survives as the FORMAT's name). Detection
    # precedes every other rung; failures are typed (SvdqInt4Unsupported /
    # SvdqHardwareError / SvdqNativeError), never a mid-denoise crash.

    svdq_art = detect_svdq_artifact(Path(path))
    if svdq_art is not None and callable(getattr(cls, "from_pretrained", None)):
        require_decodable(CONTRACT_NUNCHAKU_V1, path)
        if components:
            logger.warning("preloaded components ignored on the svdq lane")
        return load_svdq_pipeline(cls, Path(path), svdq_art)
    # W8A8 fp8-GEMM flavors: fp8 weights WITH scales take the
    # scaled-mm lane (fp8 resident, no per-layer cast); hosts without usable
    # scaled_mm dequant once to bf16-resident. Precedes the storage-cast
    # rungs — a scale-free fp8 tree never detects here.

    w8a8_art = detect_w8a8_artifact(Path(path))
    if w8a8_art is not None and callable(getattr(cls, "from_pretrained", None)):
        require_decodable(CONTRACT_COZY_FP8_ROWWISE, path)
        compute = None
        if dtype:
            try:
                compute = get_torch_dtype(dtype)
            except ImportError:
                pass
        if not w8a8_art.component:
            # Root layout: the pipeline class's own loader
            # constructs; the worker swaps post-construction.
            if components:
                logger.warning(
                    "preloaded components ignored on the root w8a8 lane")
            if storage_dtype == "fp8+te":
                logger.warning(
                    "storage_dtype=fp8+te ignored on the root w8a8 lane "
                    "(no component identity)")
            return load_w8a8_root_pipeline(
                cls, Path(path), w8a8_art, compute_dtype=compute)
        return load_w8a8_pipeline(
            cls, Path(path), w8a8_art, compute_dtype=compute,
            components=components,
            fp8_text_encoders=storage_dtype == "fp8+te",
        )
    # W4A4 nvfp4 flavors: packed fp4 weights WITH two-level scales
    # take the blockwise fp4 scaled_mm lane on Blackwell (sm_100+); other
    # qualifying hosts dequant once to bf16-resident. Disjoint from w8a8
    # detection (uint8 vs e4m3 weights) and from scale-free trees.

    w4a4_art = detect_w4a4_artifact(Path(path))
    if w4a4_art is not None and callable(getattr(cls, "from_pretrained", None)):
        compute = None
        if dtype:
            try:
                compute = get_torch_dtype(dtype)
            except ImportError:
                pass
        if storage_dtype == "fp8+te":
            logger.warning(
                "storage_dtype=fp8+te ignored on the w4a4 lane (gw#540 v1: "
                "denoiser-only quantization)")
        if not w4a4_art.component:
            if components:
                logger.warning(
                    "preloaded components ignored on the root w4a4 lane")
            return load_w4a4_root_pipeline(
                cls, Path(path), w4a4_art, compute_dtype=compute)
        return load_w4a4_pipeline(
            cls, Path(path), w4a4_art, compute_dtype=compute,
            components=components,
        )
    gguf_art = detect_gguf_snapshot(Path(path))
    if gguf_art is not None and callable(getattr(cls, "from_pretrained", None)):
        gguf_file, qtype = gguf_art
        pipe = load_gguf_pipeline(
            cls, Path(path), gguf_file, components=components,
        )
        try:
            pipe._cozy_gguf_quant = qtype
        except Exception:
            pass
        return pipe
    # The generic arm IS the plain.bf16@1 decoder. An image whose decoder
    # modules failed to import declares nothing, and "nothing" must refuse by
    # name here rather than produce a pipeline nobody declared.
    require_decodable(CONTRACT_PLAIN_BF16, path)
    kwargs: Dict[str, Any] = {}
    if components:
        kwargs.update(components)
    if dtype:
        try:
            kwargs["torch_dtype"] = get_torch_dtype(dtype)
        except ImportError:
            # torch-less environment (unit tests / CPU tools) — loaders that
            # actually need torch will fail on their own terms.
            pass
    variant = detect_diffusers_variant(Path(path))
    if variant in ("bf16", "fp16"):
        kwargs["variant"] = variant
    sniffed = detect_on_disk_dtype(Path(path))
    if "torch_dtype" not in kwargs:
        # Bindings without an explicit dtype (Hub mirrors): honor the weights'
        # own precision instead of diffusers' fp32 default. fp8-stored flavors
        # load at the compute default (bf16) and get their storage precision
        # restored by apply_fp8_storage below.
        if sniffed in ("bf16", "fp16", "fp8"):
            try:
                kwargs["torch_dtype"] = get_torch_dtype(
                    "bf16" if sniffed == "fp8" else sniffed
                )
            except ImportError:
                # torch-less environment (unit tests / CPU tools) — loaders
                # that actually need torch will fail on their own terms.
                pass
    # A declared fp8 storage lane is SERVED as fp8 storage; only the
    # involuntary fit-ladder rungs below may move the lane, and only downward.
    fp8_storage = storage_dtype in ("fp8", "fp8+te") or sniffed == "fp8"
    fp8_text_encoders = storage_dtype == "fp8+te"
    adaptive_rung = ""  # load-time rung engagement, stamped on the pipe
    if not read_on_disk_quant_config(Path(path)):
        qc = synthesize_quantization_config(attrs)
        if qc is None:
            mode, eqc = _adaptive_fit_rung(
                cls, Path(path), fp8_planned=fp8_storage,
                compute_dtype=kwargs.get("torch_dtype"),
            )
            assert eqc is None  # the fp8 rung carries no quantization_config
            if mode == "fp8":
                fp8_storage = True  # runtime fp8-E4M3 storage rung
                adaptive_rung = "fp8"
        if qc is not None:
            kwargs["quantization_config"] = qc
    # The composition's ONE compute dtype, captured before the per-component
    # map can replace the kwarg with a dict.
    scalar_dtype = kwargs.get("torch_dtype")
    single = _single_file_checkpoint(Path(path))
    if single is not None and callable(getattr(cls, "from_single_file", None)):
        kwargs.pop("variant", None)
        pipe = cls.from_single_file(
            str(third_party_dir(single, why="from_single_file wants a real file")),
            **kwargs)
    else:
        # A part whose dtype opinion is WIDER than the composition's compute
        # dtype must come off disk that way — upcasting a bf16-loaded
        # component afterwards recovers no precision, it only hides the
        # truncation. diffusers takes a per-component dtype MAP (a "default"
        # key plus per-part overrides), so the widening happens inside the one
        # from_pretrained instead of a hand-assembled sibling load.
        per_component = _component_dtype_map(cls, path, scalar_dtype)
        if per_component:
            kwargs["torch_dtype"] = per_component
        try:
            pipe = cls.from_pretrained(
                third_party_dir(path, why="pipeline from_pretrained"), **kwargs)
        except (TypeError, ValueError):
            # Not every loader takes variant=/quantization_config= (transformers
            # models, single-file components); retry with the bare essentials.
            # A dict torch_dtype is in the same class — a loader that predates
            # per-component dtypes must not lose the load entirely, so it
            # collapses back to the composition's single compute dtype.
            kwargs.pop("variant", None)
            kwargs.pop("quantization_config", None)
            if isinstance(kwargs.get("torch_dtype"), dict):
                if scalar_dtype is None:
                    kwargs.pop("torch_dtype", None)
                else:
                    kwargs["torch_dtype"] = scalar_dtype
                logger.warning(
                    "COMPONENT_DTYPE model=%s: %s.from_pretrained rejected a "
                    "per-component torch_dtype map; retrying at the "
                    "composition's single compute dtype (widened parts will "
                    "load truncated)", path, getattr(cls, "__name__", cls),
                )
            pipe = cls.from_pretrained(
                third_party_dir(path, why="pipeline from_pretrained"), **kwargs)

    unmaterialized = meta_tensors(pipe)
    if unmaterialized:
        raise RuntimeError(
            f"{type(pipe).__name__} load left {len(unmaterialized)} "
            f"unmaterialized meta tensors (e.g. {unmaterialized[:3]})"
        )
    if fp8_storage and "quantization_config" not in kwargs:
        # The SCALAR compute dtype — `kwargs["torch_dtype"]` may be the
        # per-component MAP by now, and a dict reaching
        # `enable_layerwise_casting(compute_dtype=...)` either explodes into
        # "serving at full precision" or arms windows that upcast to a
        # non-dtype. The cast window's compute dtype is a composition-level
        # fact, so the composition default belongs here.
        applied = apply_fp8_storage(pipe, compute_dtype=scalar_dtype,
                                    text_encoders=fp8_text_encoders)
        # A cast that silently no-ops (denoiser-less pipeline) must surface as
        # a structural degradation upstream, not vanish into a log line.
        try:
            pipe._cozy_fp8_storage_requested = True
            pipe._cozy_fp8_storage_ok = bool(applied)
            if applied:
                setattr(pipe, _WEIGHT_LANE_ATTR, "fp8-hooks")
        except Exception:
            pass
    if adaptive_rung:
        # An emergency rung must never engage silently: the executor
        # reconciles this stamp into ServePlan.ran / FnDegraded.
        try:
            pipe._cozy_adaptive_rung = adaptive_rung
        except Exception:
            pass
    return pipe


__all__ = [
    "get_torch_dtype",
    "detect_diffusers_variant",
    "detect_on_disk_dtype",
    "detect_gguf_snapshot",
    "safetensors_file_valid",
    "read_on_disk_quant_config",
    "synthesize_quantization_config",
    "apply_fp8_storage",
    "assert_uniform_compute_dtype",
    "MixedComputeDtypeError",
    "composition_compute_dtype",
    "QUANT_EXECUTION_LANE_COMPUTE_DEFAULT",
    "apply_block_window_offload",
    "block_offload_active",
    "pipeline_weight_lane",
    "runtime_fp8_storage_supported",
    "component_load_dtypes",
    "model_index_components",
    "model_index_component_classes",
    "snapshot_component_weight_bytes",
    "load_from_pretrained",
    "is_modular_pipeline_class",
    "hydrate_modular_pipeline",
    "modular_staging_units",
    "decide_streamed_hydration",
    "plan_streamed_hydration",
    "StreamedHydrationPlan",
    "ModularHydrationError",
    "load_gguf_pipeline",
]
