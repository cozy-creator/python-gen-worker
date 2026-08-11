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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import activity as activity_mod
from ..component_vocab import (
    denoiser_components,
    text_encoder_components,
    weight_components,
)
from ..families.facts import component_dtype_for_class
from .ladder import EMERGENCY_NF4_VRAM_FACTOR, NF4_WEIGHT_BYTES_FACTOR
import importlib
import importlib.util
import inspect
import os
import struct
import sys

from ..capability import HostRamCapacityError, InsufficientHostRamError
from . import disk_gc, load_progress
from .tensor_layout_contract import CONTRACT_PLAIN_BF16, implements_contract
from .fp8_storage import restructure_fp8_storage
from .memory import (
    flush_memory,
    get_available_vram_gb,
    keeps_weights_in_host_ram,
    meta_tensors,
    probe_host_ram,
)
from .safetensors_header import header_len_ok
from .svdq import detect_svdq_artifact, load_svdq_pipeline
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
    loading as bf16 (#358) — quantized checkpoints (fp8/int4/...) don't take a
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
    Catches truncation (pod-churn-interrupted writes, gw#408) without hashing;
    zero-page corruption inside tensor data needs the digest check instead."""

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
    be preserved (see :func:`apply_fp8_storage`)."""

    counts: Dict[str, int] = {}
    try:
        for p in sorted(Path(model_path).rglob("*.safetensors")):
            with open(p, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    continue
                (n,) = struct.unpack("<Q", raw)
                if not header_len_ok(n):
                    continue
                header = json.loads(f.read(n))
            for value in header.values():
                if isinstance(value, dict) and "dtype" in value:
                    counts[str(value["dtype"])] = counts.get(str(value["dtype"]), 0) + 1
    except (OSError, ValueError):
        return ""
    if not counts:
        return ""
    top = max(counts, key=lambda k: counts[k])
    return _SAFETENSORS_DTYPE_NAMES.get(top, "")


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
# precision (quality-safe default, QUANTIZATION-POLICY.md component policy).
#
# These read the vocabulary at CALL time, never at import: an endpoint's
# declare_components() runs at endpoint-module import, which may be after this
# module is imported. A module-level tuple would freeze the pre-declaration
# vocabulary and silently skip the declared components (pgw#740 B5).
_fp8_storage_components = denoiser_components
# The "+te" rung (component fit-ladder rung 2): the pipeline's text encoders.
_fp8_text_encoder_components = text_encoder_components

#: pgw#824: the emergency nf4 rung was engaged and landed on ZERO modules. A
#: rung OUTCOME, not the absence of one — the pipeline serves full precision on
#: a host whose free VRAM was already below the stored-precision footprint,
#: which is the worst outcome the ladder can produce. Distinct from "nf4" (it
#: landed) and from "" (no rung was needed), because placement must be able to
#: tell those three apart. Consumed by ``models.provision``.
RUNG_NF4_UNLANDED = "nf4-unlanded"

class _Fp8WeightWindow:
    """Weight-only fp8 storage for one transformer block: the block's
    Linear/conv WEIGHTS live in fp8 at rest; a forward-pre hook upcasts them
    to the compute dtype for the whole block forward and a forward hook
    recasts after. Everything else in the block (norms, embeddings, biases,
    raw parameters) never leaves compute precision.

    Block-window (not per-leaf-layer) granularity is what makes this safe for
    transformers models, which — unlike diffusers denoisers — read weight
    dtype and touch weights OUTSIDE the owning leaf's forward (gw#460):
    Gemma3's embed-scale multiply runs on the embedding output, and T5's
    ``T5DenseActDense`` casts ACTIVATIONS to ``self.wo.weight.dtype`` before
    calling ``wo``, so a leaf-hooked (diffusers-style) ``wo`` still poisons
    the stream with fp8. Inside a block window every dtype read sees the
    compute dtype. Transient cost: one block resident at compute dtype."""

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

    # Any parameter reachable outside the blocks must keep compute dtype —
    # a weight cast through a block but read elsewhere is the gw#460 break.
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
    pipeline's denoiser — or on ``obj`` itself when it is a bare module
    (th#546 two-format policy). Diffusers denoisers are RESTRUCTURED into fp8
    storage modules (pgw#727, :mod:`gen_worker.models.fp8_storage`: upcast at
    the use site inside forward); transformers text encoders keep the
    :class:`_Fp8WeightWindow` block hooks — they read weight dtype OUTSIDE the
    owning leaf's forward (gw#460), which resident fp8 would poison, and they
    are not on any compiled path.
    ``text_encoders=True`` (the ``storage_dtype="fp8+te"`` rung) extends the
    cast to the pipeline's text encoders via the transformers-aware path.
    ``components`` overrides the target component names entirely (gw#557:
    the w8a8 lane casts ONLY the text encoders — its denoiser holds fp8
    scaled-mm modules that cast hooks must never touch).

    This is the universal VRAM-fit mechanism: fp8 bytes resident, bf16/fp16
    compute, no fp8 silicon required. Also the consumption path for stored
    ``#fp8`` flavors — their storage precision is preserved instead of being
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
            # Idempotence (gw#479): a content-shared module injected into a
            # sibling lane is already armed; double hooks would double-cast.
            applied = True
            continue
        try:
            if callable(getattr(mod, "enable_layerwise_casting", None)):
                # diffusers ModelMixin. pgw#727: same semantics as
                # ``enable_layerwise_casting`` (and the SAME coverage set —
                # the model's own skip patterns included), expressed as module
                # STRUCTURE instead of a forward-boundary mutation. The hook
                # form is compile-hostile (0.2% dynamo regression for a 38.9s
                # mint) and torch.export refuses it; the structural form is
                # 14.8% faster under dynamo and exports clean.
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
            # pgw#760: the all-components failure is structurally reported
            # (th#737 cast_dropped), but a PARTIAL failure returns
            # applied=True and reads as success — this component alone now
            # holds ~2x its budgeted VRAM at full precision.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"component={name} obj={type(obj).__name__}: fp8 storage "
                f"cast failed; this component serves at full precision "
                f"(over its budgeted VRAM): {type(exc).__name__}: {exc}",
                phase="fp8_cast_failed",
            )
    return applied


class _BlockOffloadWindow:
    """Degraded-mode rung 2 (ie#468): one transformer block's weights REST in
    host RAM (pinned when possible) and stream to the execution device only
    for that block's forward. The pre-hook is PREPENDED so the H2D copy runs
    before any fp8 upcast window (gw#460) on the same block — composed order:
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
    block to ``device`` for its forward only — the gw#460 block windows in
    reverse (degraded-mode rung 2, ie#468). Quality-preserving but slow
    (whole-model PCIe traffic per forward); a guaranteed-completion last
    resort for VRAM-constrained cards, never a production serving mode.

    ``obj`` is a pipeline (named ``components`` are offloaded) or a bare
    module. Parameters outside the discovered block windows — embeddings,
    final norms, projections — are moved TO ``device`` (they must be
    resident; the gw#460 outside-a-block dtype/device hazard applies to
    device placement too). Composes with fp8 storage windows: fp8 bytes
    stream over PCIe (half the traffic), upcast happens on-device.

    Returns True when any module was armed. Idempotent per module."""
    # Default resolved at call time, not in the signature: a default argument
    # is evaluated at def time, which would freeze the vocabulary before an
    # endpoint's declare_components() ever runs (pgw#740 B5).
    if components is None:
        components = denoiser_components()
    try:
        import torch
    except ImportError:
        logger.warning("block-window offload ignored: torch not installed")
        return False
    if device is None:
        if not torch.cuda.is_available():
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
            # weights as BUFFERS (w8a8's scaled weights, and pgw#727's fp8
            # storage leaves), so without this the just-parked block weights
            # are pulled straight back onto the device and the rung silently
            # saves nothing.
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
        # pgw#824: the SIBLING the pgw#760 apply_fp8_storage fix missed. This
        # is the same class of fact and a larger one: every forward on this
        # component now streams its weights over PCIe from host RAM, which is
        # the single biggest per-request latency change the loader can make,
        # and it was a `logger.warning` on a pod with no stdout. `pinned` vs
        # `pageable` rides the detail because the two differ by roughly 2x on
        # the transfer that now sits in the critical path.
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
        str(gguf_file),
        config=str(path / component),
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute),
        torch_dtype=compute,
    )
    kwargs = dict(components or {})
    kwargs[component] = denoiser
    pipe = cls.from_pretrained(str(path), torch_dtype=compute, **kwargs)
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
    ``*.safetensors.index.json`` (the HF shard convention — NOT an "R2
    single-PUT cap"; no such cap exists here, uploads are multipart and the
    hub grants 64 GiB/file), so a big single-file checkpoint arrives as N
    shards. Those are reassembled once into the
    original file (mmap-backed, ~disk-copy cost) and cached in the snapshot
    dir — ``from_single_file`` only takes one file."""
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None
    if (path / "model_index.json").exists() or (path / "config.json").exists():
        return None
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
        # A pod kill mid-writeback can persist a truncated merged file that
        # was then trusted forever — every load fataled with "Unable to load
        # weights from checkpoint file" until manual delete (gw#408).
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

        os.fsync(out.fileno())  # durable before rename (gw#408)
    tmp.rename(merged)
    logger.info("reassembled sharded single-file checkpoint: %s (%d shards, %d tensors, %d bytes)",
                merged.name, len(shard_names), len(entries), offset)
    return merged


# --- fp8 download stays the fp8 storage lane (pgw#772) ----------------------
# The gw#534 "rung 2" voluntary upgrade (fp8 download upcast ONCE to plain
# bf16-resident weights whenever the snapshot fit free VRAM with headroom,
# `bf16_resident_fits` / BF16_RESIDENT_MARGIN_GB) is REMOVED, ruled by Paul on
# pgw#772. The serving lane is deterministic per (release x declared config)
# — never a function of the individual card's free VRAM. The probe made
# `lane` the only GPU-dependent axis of the cell key: a 4090's ~1.5 GiB
# VRAM surplus over an L4 (same release/image/sm_89) flipped it to base lane
# "", a lane NOTHING mints for, so the better card missed all 144 published
# checkpoints INCLUDING its own same-SKU cell and served eager for life
# (th#1198 CP-D, −21% request-level AOT win forfeited). The tax the upgrade
# dodged is +1.9% for the structural storage lane (pgw#727 re-measure; the
# +44-73% figure that justified it measured the retired HOOK form), so it
# traded ~2x weight VRAM for ~1.9% latency AND identity determinism.
# Involuntary transitions stay: the fit-ladder rungs below (can't-fit fp8/nf4)
# and the w8a8/w4a4 dequant-on-unsupported-host lanes are declared rungs, not
# probe outcomes.

# The pipeline's weight lane, part of the compile-cache graph key (gw#534):
# "" = plain resident weights (incl. the involuntary w8a8/w4a4 dequant
# lanes), "fp8-hooks" =
# fp8 weights resident with a per-layer upcast (traced INTO the FX graphs).
# The "fp8-hooks" spelling is the WIRE value — tensorhub maps it to `w8a16`
# and cells key on it — and it is kept byte-identical across the pgw#727
# restructure (hooks -> module structure) on purpose. The restructure DOES
# change the traced graph, and that shows up where it should: the module
# types and hook counts in `compile_cache.execution_contract`, i.e. new cell
# keys, no cross-lane adoption.
_WEIGHT_LANE_ATTR = "_cozy_weight_lane"

#: EVERY base lane a loader can leave on ``_WEIGHT_LANE_ATTR`` (pgw#918).
#:
#: THE single source of this vocabulary. It is authored here, next to the
#: attribute itself, because this is where the assignments live — and
#: ``tests/test_speculative_execution_lane_completeness_pgw918.py`` parses every
#: assignment site under ``gen_worker/models`` and fails if a loader stamps a
#: lane this tuple does not name. An authored list nothing checks is what
#: ie#546 cost 9 pods, and what pgw#918 found still open for two more lanes.
#:
#: ``"bf16-resident"`` is deliberately absent: :func:`pipeline_weight_lane`
#: folds it to ``""`` (it traces identically to plain bf16), so it is never a
#: distinct cell-identity lane. Bucketed LoRA lanes
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


# --- Runtime fit rungs (th#546 emergency lane + th#683 fp8 storage) --------
# Fit ladder: bf16 -> #fp8 flavor -> #nvfp4 (Blackwell) -> runtime fp8-E4M3
# storage -> EMERGENCY nf4 -> CPU offload. When even the downloaded flavor
# cannot fit free VRAM, the load path first tries fp8-E4M3 weight storage
# (apply_fp8_storage: fp8 bytes resident, bf16 compute — quality ~= a stored
# #fp8 flavor), then runtime-quantizes the denoiser to bnb nf4. Always armed
# on CUDA hosts (gw#420: fitting is the runtime's job, not a flag); the
# platform never reaches it because its scheduler places by declared
# Resources.
# Coarse whole-model resident factor after nf4-quantizing the denoiser
# (denoiser ~4x smaller; encoders/VAE stay at compute dtype). Single-sourced
# from the shared ladder spec (ladder.EMERGENCY_NF4_VRAM_FACTOR) so the
# runtime rung and the Go/Py ladder never drift.
EMERGENCY_FIT_FACTOR = EMERGENCY_NF4_VRAM_FACTOR
# Resident factor after fp8-E4M3 storage of the denoiser, expressed against
# the declared CARD SIZE (resources.vram_gb — includes activation/framework
# headroom over raw weights). The ONE fp8 fit factor (pgw#515 deleted the
# duplicate ladder walk and its weight-bytes-based 0.75 estimate).
FP8_STORAGE_FIT_FACTOR = 0.55
_EMERGENCY_MARGIN_GB = 2.0


def emergency_quant_enabled() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def runtime_fp8_storage_supported() -> bool:
    """The runtime fp8-E4M3 storage rung needs a CUDA host and a torch that
    ships the float8_e4m3fn dtype — no fp8 silicon required (per-layer upcast
    compute; see apply_fp8_storage)."""
    try:
        import torch

        return bool(torch.cuda.is_available()) and hasattr(torch, "float8_e4m3fn")
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

    The authoritative component-class vocabulary at LOAD time (pgw#667): a
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
    time (pgw#667) — the snapshot's own ``model_index.json`` classes first, the
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
    """The compute dtype the COMPOSED pipeline will run at (pgw#647 gap #2):
    the base binding's declared dtype when present, else the dtype the base
    tree's LOAD LANE actually computes at. ``""`` = unknown (an
    fp32-defaulting composition).

    Lane selection mirrors :func:`load_from_pretrained` (pgw#675): a
    quantized-artifact tree (svdq / w8a8 / w4a4) computes at the lane's bf16
    default regardless of the tree's MAJORITY on-disk dtype — a produced
    ``#fp8-w8a8`` flavor quantizes only the repeated-block Linears and passes
    every other tensor through at SOURCE precision, so a fine-tune mirrored
    from an fp16 upstream sniffs majority-fp16 while its pipeline loads
    ``torch_dtype=bf16``. The old majority sniff loaded a component override
    fp16 into that bf16 composition and every warm/serve forward died with
    ``Input type (c10::BFloat16) and bias type (c10::Half)`` (ie#546 sdxl
    finale, 3/3 workers)."""
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

    pgw#683's invariant. torch's matmul/conv kernels take no dtype opinion
    from the module — they raise mid-forward, with a message that names
    neither the tensor nor the component::

        RuntimeError: mat1 and mat2 must have the same dtype, but got
        BFloat16 and Half      (an nn.Linear WITH bias: addmm)
        RuntimeError: Input type (c10::BFloat16) and bias type (c10::Half)
        should be the same     (a conv)

    Live, that message arrived at ``self_mint_compile phase=warmup_forward``
    warm unit 4/18 on an L4 and cost `generate` on a prod release, with
    nothing in it to attribute the fault to a component, a ref or a load path.
    This error is raised at LOAD instead, naming the component, the parameter
    path and both dtypes.
    """


#: Dtypes a GEMM can actually compute in. Storage-only dtypes (fp8/fp4/int)
#: are excluded on purpose: the w8a8/svdq/nvfp4 lanes and diffusers' layerwise
#: casting hold weights at those precisions BY DESIGN and upcast per forward.
_COMPUTE_DTYPE_NAMES = ("float16", "bfloat16", "float32", "float64")
#: The pair that cannot interoperate and never legitimately coexists in one
#: composition. fp32 is the DECLARED widening axis (pgw#667) and is reported
#: but never fatal — widening is a precision decision, not a dtype collision.
_INCOMPATIBLE_COMPUTE = ("float16", "bfloat16")


def _gemm_param_dtypes(module: Any) -> Dict[str, str]:
    """``{parameter path: dtype name}`` for every GEMM input under ``module``
    — Linear/conv weights and biases, the tensors torch actually refuses to
    mix, plus every quantized leaf's DECLARED ``compute_dtype``. Norms and
    embeddings are excluded: they carry their own (legitimately wider)
    precision and never meet a weight in one kernel.

    pgw#1020: the isinstance selector alone is BLIND to the quantized lanes.
    All five quantized leaves (``_Fp8ScaledLinear``, ``_W4A4Linear``,
    ``_SvdqLinear``, ``_SvdqFusedLinear``, ``_AwqPackedLinear``) subclass
    ``nn.Module`` directly, so a w8a8 fp16 denoiser inside a bf16 composition
    — the exact cross-composition aliasing shape pgw#683 exists to refuse —
    read as ``{}`` here and PASSED the guard. Their upcast target is a fact
    they state: ``self.compute_dtype`` is what every one of their forwards
    computes in (and what its bias must match), so it is a GEMM input dtype
    whether or not a bias tensor exists to carry it.
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
        # Storage dtypes stay uncounted here too: a leaf declaring an fp8
        # `compute_dtype` fails the membership test exactly as its fp8 weight
        # does, so pgw#683's carve-out is unchanged.
        dec_name = str(declared).rsplit(".", 1)[-1] if declared is not None else ""
        if dec_name in _COMPUTE_DTYPE_NAMES:
            out[f"{name}.compute_dtype" if name else "compute_dtype"] = dec_name
    return out


def assert_uniform_compute_dtype(
    obj: Any, expected: str = "", *, label: str = "",
) -> None:
    """Refuse a MIXED-precision composition at LOAD (pgw#683).

    Checks every GEMM input of every component: an fp16 weight and a bf16
    weight in one composition means some forward will die on a dtype the
    kernel cannot reconcile, and torch's message names nothing. Two verdicts
    are fatal:

    1. a single component that is internally fp16 AND bf16;
    2. any component whose compute dtype is fp16/bf16 but is not ``expected``
       (the composition's own compute dtype) — the cross-composition aliasing
       shape: a content-keyed shared component loaded by another pick's record
       at ITS dtype and injected here unconverted.

    fp32 parts are legal (pgw#667 declares wider components deliberately) and
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
    None when every part loads at the composition's own dtype (pgw#667).

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
#: store (pgw#1071). fp8 is a STORAGE fact — the artifact carries its own
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
    when its headers say nothing (pgw#1071).

    Read per COMPONENT, never per snapshot: a majority vote over a whole
    mixed-precision tree upcasts every narrow component when the vote lands
    wide and truncates every wide one when it lands narrow. ie#615 measured
    both halves on minimax-h3 — a 66.28 GB bf16 DiT hydrating at 74.9 GiB
    (4 bytes/param) because the tree-wide vote fell outside the map and
    diffusers' fp32 default took over."""
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
    component's checkpoint (pgw#1071). With a declared composition dtype this
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
    what serving runs, so the component path refuses by name instead
    (pgw#689: a benchmark that measures something other than the serve path
    is worse than one that refuses)."""


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True when ``fn`` can take ``name=`` (declared or via ``**kwargs``).

    Replaces the ``except TypeError: retry without it`` idiom, which caught
    ANY construction-time TypeError — including one raised deep inside
    diffusers' quantization-config reconstruction — and retried a path that
    failed identically, so the real cause never surfaced (pgw#689 defect 2).
    Introspection failure means "pass it": the call then fails naming
    itself."""
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
    tree: str | Path, component: str, *,
    dtype: str = "", weights_tree: str | Path | None = None,
) -> Any:
    """THE production loader for ONE named pipeline component.

    Every caller that needs a single component — the executor's pgw#617
    substitution, the pgw#674 rotation preloader, the swap benchmark — goes
    through here, so what they load is by construction what serving loads.

    ``tree`` is the BASE composition: it names the module class
    (model_index.json) and decides the compute dtype. ``weights_tree`` is
    where the bytes live (default: ``tree``); an override binding points it
    at its own snapshot, whose ``<component>/`` subtree is used when present,
    else its root.

    Quantized artifacts take their OWN lane, exactly as
    :func:`load_from_pretrained` routes a whole pipeline: a w8a8/w4a4
    denoiser is materialized by its artifact loader. A modelopt-produced
    tree carries a ``quantization_config`` block diffusers reconstructs into
    ``NVIDIAModelOptConfig``, whose constructor requires a ``quant_type``
    the block does not supply — so a bare ``from_pretrained`` on the
    denoiser dies at config reconstruction on every flavor the fleet
    actually serves (pgw#689 defect 1). Lanes with no component-level
    loader raise :class:`ComponentLaneUnsupported`.

    dtype resolution (pgw#647 gap #2): the base binding's declared dtype
    wins; otherwise the component inherits the BASE COMPOSITION's compute
    dtype (:func:`composition_compute_dtype`); the weights' own on-disk
    dtype is only the last resort. Hub-resolved bindings carry no dtype, so
    the old override-on-disk fallback loaded e.g. the fp32-stored fp16-fix
    VAE into a bf16 pipeline and setup died on the first latent (ie#546
    canary, 2/2 pods). Blocking; callers on an event loop run it
    off-thread."""

    base = Path(tree)
    root = Path(weights_tree) if weights_tree is not None else base
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

    # pgw#667: a component with a declared load-dtype fact keeps it when it is
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

    def _covers(artifact_component: str) -> bool:
        """The artifact's weight set IS this component: a diffusers tree
        names it, a bare override tree (root layout) has nothing else in
        it."""
        return artifact_component == component or (
            not artifact_component and src == root
        )

    w8a8_art = detect_w8a8_artifact(root)
    if w8a8_art is not None and _covers(w8a8_art.component):
        return load_w8a8_denoiser(
            root, w8a8_art, compute_dtype=torch_dtype, cls=cls)
    w4a4_art = detect_w4a4_artifact(root)
    if w4a4_art is not None and _covers(w4a4_art.component):
        return load_w4a4_denoiser(
            root, w4a4_art, compute_dtype=torch_dtype, cls=cls)
    svdq_art = detect_svdq_artifact(root)
    if svdq_art is not None and _covers(svdq_art.component):
        raise ComponentExecutionLaneUnsupported(
            f"component {component!r} of {root} is an svdq-{svdq_art.precision} "
            f"artifact ({svdq_art.file.name}): its denoiser is built by the "
            f"svdq engine during the PIPELINE load, so there is no "
            f"component-level production loader to borrow"
        )
    if component in denoiser_components() and detect_gguf_snapshot(root):
        raise ComponentExecutionLaneUnsupported(
            f"component {component!r} of {root} is a GGUF denoiser: it is "
            f"dequantized by the pipeline's own gguf loader, so there is no "
            f"component-level production loader to borrow"
        )

    kwargs: Dict[str, Any] = {}
    if torch_dtype is not None and _accepts_kwarg(
            cls.from_pretrained, "torch_dtype"):
        kwargs["torch_dtype"] = torch_dtype
    return cls.from_pretrained(str(src), **kwargs)


def load_component_override(
    base_path: str | Path, component: str, override_path: str | Path,
    *, dtype: str = "",
) -> Any:
    """Load one named pipeline component from an OVERRIDE snapshot tree
    (pgw#617 hierarchical bindings) — :func:`load_component` with the
    weights pointed at the override."""
    return load_component(
        base_path, component, dtype=dtype, weights_tree=override_path)


class ModularHydrationError(RuntimeError):
    """A ModularPipeline slot could not be hydrated from the LOCAL tree
    (pgw#1036). Typed so the failure is a refusal at load — never a silent
    shell handed to ``setup()``, and never a fetch from the repo id the
    snapshot's index happens to name."""


class ComponentSubstitutionError(RuntimeError):
    """A non-modular diffusers composition names a component the local tree
    does not carry and the dispatch injected nothing for it (pgw#1048).

    DETERMINISTIC: the tree is already materialized and the injected set is
    already known, so nothing about a retry can change the answer — a refetch
    cannot widen a manifest the hub narrowed. Callers classify it terminal for
    the dispatched identity rather than retryable.

    ``missing``/``expected``/``injected``/``tree`` carry the comparison the
    raw ``OSError: Error no file named config.json found in directory <root>``
    does not: what the composition wanted, what the tree had, what arrived."""

    def __init__(
        self,
        message: str,
        *,
        tree: str = "",
        missing: Sequence[str] = (),
        expected: Sequence[str] = (),
        injected: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.tree = tree
        self.missing = tuple(missing)
        self.expected = tuple(expected)
        self.injected = tuple(injected)


def _component_dir_present(root: Path, component: str) -> bool:
    """True when the tree carries a non-empty ``<root>/<component>/`` dir.

    Deliberately not a config-name check: schedulers, tokenizers, processors
    and models each name their config differently, and a layout we do not
    model must not be refused. An ABSENT (or empty) dir is the narrowing this
    guards — the shape th#1711 produces when it withholds a component's files
    from the outbound snapshot."""
    src = root / component
    if not src.is_dir():
        return False
    return next(src.iterdir(), None) is not None


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


def assert_composition_satisfiable(
    cls: Any,
    path: str | Path,
    *,
    components: Optional[Dict[str, Any]] = None,
    ref: str = "",
) -> None:
    """Refuse a diffusers-layout load whose composition cannot be satisfied.

    ``model_index.json`` names the parts; each must arrive either from its own
    dir in the local tree or through the pgw#617 ``components=`` injection the
    dispatched binding's overrides derive. When one does neither, diffusers
    raises ``OSError: Error no file named config.json found in directory
    <snapshot root>`` — naming neither the component nor the cause — and the
    caller retries a condition no retry can fix (pgw#1047: a pod burned 9
    minutes on it before the hub reaped it).

    Skipped, because the composition is not this tree's to satisfy: layouts
    with no readable ``model_index.json`` (single-file checkpoints,
    transformers trees, root-layout quantized artifacts — all of which
    detect only in the absence of an index), the svdq lane (it swaps a
    nunchaku denoiser in and ignores ``components=`` outright), and gguf
    snapshots (the denoiser is a loose ``.gguf``, not a component dir).
    Also skipped per-component: anything the index names that the pipeline
    class's own signature does not (see :func:`_pipeline_component_names`).
    The modular lane never reaches here — it refuses in
    :func:`hydrate_modular_pipeline` with ``ModularHydrationError``."""
    root = Path(path)
    expected = model_index_component_classes(root)
    if not expected:
        return
    declared = _pipeline_component_names(cls)
    if declared is not None:
        expected = {k: v for k, v in expected.items() if k in declared}
        if not expected:
            return
    injected = tuple(sorted(components or ()))
    missing = tuple(
        name for name in sorted(expected)
        if name not in (components or {})
        and not _component_dir_present(root, name)
    )
    if not missing:
        return
    # Only on the refusal path: both probes walk the tree, and the happy path
    # must not pay for a lane it is not on.
    if detect_svdq_artifact(root) is not None:
        return
    if detect_gguf_snapshot(root) is not None:
        return
    detail = (
        f"pipeline={getattr(cls, '__name__', cls)} "
        f"ref={ref or '<none>'} tree={root} "
        f"missing={','.join(missing)} "
        f"expected={','.join(sorted(expected))} "
        f"injected={','.join(injected) or '<nothing>'}"
    )
    activity_mod.emit_event(
        activity_mod.KIND_COMPONENT_MISS, detail, phase="refused")
    raise ComponentSubstitutionError(
        f"base tree {root} carries no {missing[0]!r}/ and the dispatch "
        f"injected " + (f"only {list(injected)}" if injected else "nothing")
        + f" for it (composition {getattr(cls, '__name__', cls)} names "
        f"{sorted(expected)}; missing {list(missing)}). A narrowed snapshot "
        f"with no matching component override is the th#1711/th#1715 shape — "
        f"deterministic, so this load is refused rather than retried.",
        tree=str(root), missing=missing,
        expected=tuple(sorted(expected)), injected=injected,
    )


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


def _resolve_override_tree(root: Path, component: str) -> Path:
    """Override-tree layout convention (same as :func:`load_component`): the
    tree's ``<component>/`` subtree when present, else its root."""
    sub = root / component
    return sub if sub.is_dir() else root


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
    shape (ie#613 pins by FILE SET and keeps ``config.json`` for the
    unselected partition, e.g. H3's ``transformer_ref/``): the component is
    EXCLUDED from this slot, not missing."""
    if not (src / "config.json").is_file():
        return False  # tokenizer/processor/scheduler dirs: no model config
    return next(src.rglob("*.safetensors"), None) is None


_GIB = 1024 ** 3
# Mirrors residency/staging's host-RAM floor policy (gw#407).
_STAGING_FLOOR_GB = 8.0
_STAGING_FLOOR_FRACTION = 0.2


def _staging_floor_bytes(total_bytes: int) -> int:
    if total_bytes <= 0:
        return int(_STAGING_FLOOR_GB * _GIB)
    return int(min(_STAGING_FLOOR_GB * _GIB,
                   max(_GIB, total_bytes * _STAGING_FLOOR_FRACTION)))


def _admit_component_staging(component: str, nbytes: int) -> None:
    """pgw#1041: admit ONE component's staging against the cgroup budget.

    ``probe_host_ram`` already speaks cgroup (v1 and v2) and credits clean
    reclaimable page cache (pgw#752), so the just-fetched tree's own cache
    never blocks its own load. A component that cannot fit an EMPTY host is
    the structural pgw#752 verdict; one that cannot fit right now is the
    transient one. Both carry the measured numbers. An unreadable probe
    fails open — no worse than the unchecked load it replaces.

    pgw#1063: the estimate can be wrong (an upcast, a quant unpack, an
    allocator's own overhead), and when it is the load does not fail — it
    crawls in direct reclaim until the kernel kills it. So a MEASURED
    verdict outranks this arithmetic: a process the load dial has caught
    re-reading its own set instead of staging it admits nothing further,
    structurally, whatever the numbers below would have said."""
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


def modular_staging_units(
    base: Path,
    component_trees: Optional[Mapping[str, str]] = None,
) -> Dict[str, int]:
    """Bytes per INDEPENDENTLY STAGED unit of a modular snapshot.

    :func:`hydrate_modular_pipeline` loads one component at a time from that
    component's own source dir, so the unit of host-RAM staging is a
    component dir — never the tree. This reads the same sources the hydration
    loop will: each ``modular_model_index.json`` entry's subfolder under
    ``base`` (falling back to the component name), and each override tree in
    ``component_trees`` in place of the base dir it replaces.

    Empty when the index is absent or unreadable: callers treat that as "no
    per-component knowledge" and keep whole-tree accounting."""
    index = Path(base) / "modular_model_index.json"
    try:
        entries = json.loads(index.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    trees = dict(component_trees or {})
    units: Dict[str, int] = {}
    for name, entry in entries.items():
        if str(name).startswith("_") or name in trees:
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
    for name, tree in trees.items():
        root = Path(tree)
        if not root.is_dir():
            continue
        units[f"{name} (override)"] = disk_gc.tree_bytes(
            _resolve_override_tree(root, str(name)))
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
    #: The rung the pipeline will be placed on (pgw#1063). An offload rung
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
    component_trees: Optional[Mapping[str, str]] = None,
    device_free_bytes: Optional[int] = None,
    placement_mode: str = "",
) -> StreamedHydrationPlan:
    """pgw#1026: decide whether this modular slot stages PER COMPONENT ONTO
    THE DEVICE instead of staging its whole tree in host RAM first.

    THREAT (§4.25): a tree the CARD holds but the HOST does not is refused
    structurally at boot and no pod size fixes it — measured on ie#615's H3
    bring-up, 134.1 GiB tree + the 8 GiB staging floor against 116.4 GiB of
    host RAM on a 1x H100-80 pod, `HostRamCapacityError`. Host RAM binds
    ~26 GiB tighter than VRAM there purely because staging is all-or-nothing
    while the load is already component-sequential (pgw#1041).

    THE OBSERVABLES, all measured rather than estimated:

    1. the whole tree does NOT fit host RAM (bytes on disk vs
       :func:`probe_host_ram`'s cgroup-aware total plus the gw#407 floor) —
       otherwise nothing is wrong and the whole-tree path stands;
    2. the LARGEST single component DOES fit host RAM — otherwise the
       structural refusal is honest and must survive (no amount of
       sequencing places a component that cannot be staged at all);
    3. free VRAM holds the whole tree with :data:`
       _STREAMED_HYDRATION_VRAM_MARGIN_GB` to spare — the components have to
       go somewhere, and on this path that somewhere is the card.

    Engaged only on all three. Every other answer keeps today's behaviour,
    including the refusal, so this can only turn a refusal into a boot."""
    units = modular_staging_units(Path(base), component_trees)
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
    if keeps_weights_in_host_ram(placement_mode):
        # pgw#1063: the discount is admissible ONLY because each component
        # leaves the host for the card. An offload rung puts it back — the
        # weights live on the host by definition — so the honest requirement
        # is the whole tree, and charging one component here is what admitted
        # ie#615's 105 GB re-stage into a cgroup that could not hold it (37
        # minutes of direct-reclaim crawl, 1.578 TB read for a 105 GB set,
        # then an OOM kill that was arithmetically certain at minute zero).
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


def hydrate_modular_pipeline(
    pipe: Any,
    path: str | Path,
    *,
    torch_dtype: Any = None,
    component_trees: Optional[Dict[str, str]] = None,
    preloaded: Optional[Dict[str, Any]] = None,
    place_device: str = "",
) -> Dict[str, str]:
    """Hydrate a freshly constructed ``ModularPipeline`` from the LOCAL
    snapshot tree (pgw#1036).

    ``ModularPipeline.__init__`` registers every ``from_pretrained``
    component as ``None`` and copies each spec's
    ``pretrained_model_name_or_path`` verbatim out of the snapshot's index —
    which on a mirrored repo names the UPSTREAM repo id. This function is the
    hydration guard: every spec is re-pointed at the local tree BEFORE
    ``load_components`` runs, so neither this load nor any later endpoint-side
    ``pipe.load_components()`` can reach huggingface.co.

    - base components load from ``<snapshot>/<subfolder>``;
    - ``component_trees`` (th#980/pgw#617 overrides) re-route a component to
      its OWN materialized tree (``<tree>/<component>/`` or the tree root);
    - a config-only weight-bearing dir is the unselected partition: SKIPPED,
      its spec neutralized so nothing can ever fetch it;
    - a component the index names but the snapshot does not carry refuses
      typed (:class:`ModularHydrationError`) — never a fetch;
    - ``preloaded`` modules (gw#479 shared components) are registered via
      ``update_components`` instead of the ``from_pretrained`` kwarg
      ``ModularPipeline.__init__`` silently discards.

    Returns ``{component: source_path}`` for everything hydrated. The result
    is verified: ``load_components`` swallows load errors into a logger
    warning, so every requested component is re-checked non-``None`` and a
    miss raises typed with the captured diffusers log text.

    ``torch_dtype`` is what the caller DECLARED — a scalar governing every
    component, or diffusers' ``{"default": ..., "<part>": ...}`` map. What it
    does not name loads at that component's OWN checkpoint dtype
    (:func:`checkpoint_load_dtype`), never at a snapshot-wide majority and
    never at diffusers' fp32 default (pgw#1071). ``_keep_in_fp32_modules``
    stays diffusers' business: naming a dtype is what lets it act at all.

    ``place_device`` (pgw#1026) moves each component onto that device as it
    lands and drops the host copy, so the host-RAM high-water mark is ONE
    component instead of the tree. Set it from
    :func:`plan_streamed_hydration`, never by hand: it is admissible only
    when the card holds the whole tree, and a mid-hydration placement
    failure is a typed refusal with no whole-tree path left to fall back
    to."""
    base = Path(path)
    specs = dict(getattr(pipe, "_component_specs", None) or {})
    trees = dict(component_trees or {})
    pre = dict(preloaded or {})

    unknown = sorted(set(trees) - set(specs))
    if unknown:
        raise ModularHydrationError(
            f"component override {unknown[0]!r} is not a component of "
            f"{type(pipe).__name__} (specs: {sorted(specs)})")
    both = sorted(set(trees) & set(pre))
    if both:
        raise ModularHydrationError(
            f"component {both[0]!r} arrived as BOTH an override tree and a "
            f"preloaded module; one delivery mechanism per component")

    sources: Dict[str, str] = {}
    skipped: List[str] = []
    for name, spec in specs.items():
        if getattr(spec, "default_creation_method", "") != "from_pretrained":
            continue
        if name in pre:
            continue
        if name in trees:
            root = Path(trees[name])
            if not root.is_dir():
                raise ModularHydrationError(
                    f"override tree for component {name!r} does not exist: "
                    f"{root}")
            sources[name] = str(_resolve_override_tree(root, name))
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
        sources[name] = str(src)

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
        # A preloaded (gw#479 shared) module loads from no path at all;
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
            # pgw#1041 (pgw#1026's minimal per-stage form on this lane): one
            # component at a time, admission-checked against the CGROUP
            # budget before it stages, page cache chilled after it lands.
            # ie#615 attempt 4 measured the whole-tree form running the
            # cgroup AT its ceiling (max_usage == limit, 250.0/251.0 GB):
            # staged anon plus the tree's own read cache share one limit, so
            # sequencing + chilling is what keeps the high-water at
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
                # pgw#1071: this component's OWN checkpoint dtype when the
                # caller declared none. Sniffed here rather than by the
                # caller because this is the only place that knows each
                # component's actual source dir — an override tree is a
                # different artifact from the base dir it replaces.
                dt = _hydration_dtype(n, torch_dtype, comp_src)
                if dt is not None:
                    kwargs["torch_dtype"] = dt
                    dtypes[n] = str(dt).removeprefix("torch.")
                pipe.load_components(names=[n], **kwargs)
                # pgw#1026: place it now and drop the host copy, so the next
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
    # pgw#1071: the dtype each component actually loaded at is the evidence
    # the fp32-upcast wall was invisible for — it belongs in the hub-visible
    # record, not only in a log line.
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

    with open(p, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return 0
        (n,) = struct.unpack("<Q", raw)
        if not header_len_ok(n):
            return 0
        header = json.loads(f.read(n))
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


def specialized_weight_layout(model_path: str | Path) -> str:
    """The non-plain lane :func:`load_from_pretrained` would take for this
    snapshot (``"quantized"``/``"svdq"``/``"w8a8"``/``"w4a4"``/``"gguf"``), or
    ``""`` for the plain dense-safetensors path.

    pgw#1117 asks exactly one question of it: is this tree's resident size
    computable from safetensors headers? On every lane named here it is not —
    packed 4-bit weights, fp8 GEMM scale tables and GGUF blocks all have a
    header story that differs from their in-memory story — so the envelope
    precondition abstains instead of guessing. Same detectors, same ORDER as
    the loader, so the answer cannot drift from the lane actually taken."""
    p = Path(model_path)
    if read_on_disk_quant_config(p):
        return "quantized"
    if detect_svdq_artifact(p) is not None:
        return "svdq"
    if detect_w8a8_artifact(p) is not None:
        return "w8a8"
    if detect_w4a4_artifact(p) is not None:
        return "w4a4"
    if detect_gguf_snapshot(p) is not None:
        return "gguf"
    return ""


def bitsandbytes_available() -> bool:
    """Importability gate for the bnb-nf4 rung (gw#469): the quant config
    constructs fine without bitsandbytes and the load then dies deep in
    ``validate_environment`` (PackageNotFoundError -> setup_failed). An
    unavailable rung must be SKIPPED, never attempted."""

    if "bitsandbytes" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("bitsandbytes") is not None
    except (ImportError, ValueError):
        return False


def emergency_quantization_config(
    cls: Any,
    *,
    components: Optional[List[str]] = None,
    compute_dtype: Any = None,
) -> Optional[Any]:
    """bnb-nf4 config for the emergency rung, scoped to ``components`` (the
    snapshot's REAL denoiser/text-encoder names — gw#521: a config naming
    absent components is silently ignored by diffusers, so the caller derives
    the list from the tree and this function refuses an empty one). None
    (with a warning) when the stack can't do it — the offload ladder then
    carries it."""
    if not bitsandbytes_available():
        logger.warning(
            "emergency nf4 unavailable (bitsandbytes not installed in this "
            "image); skipping the quantized rung — the offload ladder carries it"
        )
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        import diffusers
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError as exc:
        logger.warning("emergency nf4 unavailable (%s); falling to offload", exc)
        return None
    kwargs: Dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": compute_dtype or torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
    if isinstance(cls, type) and issubclass(cls, diffusers.DiffusionPipeline):
        targets = list(_fp8_storage_components()) if components is None else list(components)
        if not targets:
            logger.warning(
                "emergency nf4 skipped: no quantizable component in the "
                "snapshot (denoiser-less tree); the offload ladder carries it"
            )
            return None
        try:
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs=kwargs,
                components_to_quantize=targets,
            )
        except ValueError as exc:
            # diffusers validates the bnb config signature against BOTH
            # libraries; a diffusers/transformers skew raises here — skip the
            # rung instead of killing the load.
            logger.warning("emergency nf4 unavailable (%s); falling to offload", exc)
            return None
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig

    return BitsAndBytesConfig(**kwargs)


def _bnb_quantized_components(pipe: Any, targets: List[str]) -> List[str]:
    """The subset of ``targets`` whose modules actually hold bnb 4-bit layers
    after the load — the gw#521 no-op detector (diffusers silently ignores
    config components absent from the pipeline)."""
    landed: List[str] = []
    for name in targets:
        mod = getattr(pipe, name, None)
        if mod is None or not hasattr(mod, "modules"):
            continue
        try:
            for m in mod.modules():
                if type(m).__name__ in ("Linear4bit", "LinearNF4", "LinearFP4"):
                    landed.append(name)
                    break
        except Exception:  # noqa: BLE001
            continue
    return landed


def _adaptive_fit_rung(
    cls: Any, path: Path, *, fp8_planned: bool, compute_dtype: Any = None
) -> tuple[str, Optional[Any]]:
    """Serve-time fit ladder at load (th#683 P3): when the snapshot's
    estimated resident bytes (after any planned fp8 storage) exceed free
    VRAM, engage the cheapest-quality-loss runtime lever that FITS:
    fp8-E4M3 storage first (denoiser weights ~halve, quality ~= a stored
    #fp8 flavor), then the nf4 emergency rung — denoiser first, text
    encoders joining only when the denoiser alone isn't enough. Targets are
    the snapshot's REAL component names (gw#521: a config naming absent
    components is silently ignored by diffusers — the rung must never be a
    hard-coded archetype guess). When even nf4 cannot fit, the rung is
    SKIPPED (full-precision weights preserved; the offload ladder carries
    it) instead of paying the quality cost for nothing.

    Returns ``(mode, config)``: ``("", None)`` fits as planned (or no rung
    helps); ``("fp8", None)`` engage fp8 storage; ``("nf4", qc)``
    emergency-quantize."""
    if not emergency_quant_enabled():
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
    encoders = [c for c in _fp8_text_encoder_components()
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
            and runtime_fp8_storage_supported() \
            and total - 0.5 * denoiser_bytes <= budget:
        logger.warning(
            "fp8-E4M3 emergency weight storage engaged for %s (%.1f GB "
            "weights, %.1f GB free) — near-native quality; a stored #fp8 "
            "flavor of this model would serve natively here.",
            path, total_gb, free_gb,
        )
        return "fp8", None
    if not denoisers:
        logger.warning(
            "emergency nf4 skipped for %s: no denoiser component in the "
            "snapshot (components: %s); the offload ladder carries it",
            path, sorted(named),
        )
        return "", None
    # nf4 rung: denoiser first; text encoders join only when needed.
    targets = list(denoisers)
    est = total - denoiser_bytes * (1.0 - NF4_WEIGHT_BYTES_FACTOR)
    if est > budget and encoders:
        targets += encoders
        est -= sum(comp_bytes[c] for c in encoders) * (1.0 - NF4_WEIGHT_BYTES_FACTOR)
    if est > budget:
        logger.warning(
            "emergency nf4 skipped for %s: even 4-bit weights (~%.1f GB) "
            "exceed the %.1f GB budget; keeping full precision — the "
            "offload ladder carries it",
            path, est / float(1 << 30), budget / float(1 << 30),
        )
        return "", None
    qc = emergency_quantization_config(
        cls, components=targets, compute_dtype=compute_dtype)
    if qc is not None:
        logger.warning(
            "EMERGENCY 4-bit quantization engaged for %s (components %s; "
            "%.1f GB weights, %.1f GB free) — quality below platform "
            "standards; a larger card or Blackwell SKU would serve stored "
            "flavors instead.",
            path, targets, total_gb, free_gb,
        )
    return "nf4", qc


def _load_modular_pipeline(
    cls: Any,
    path: str,
    *,
    dtype: str = "",
    storage_dtype: str = "",
    components: Optional[Dict[str, Any]] = None,
    component_trees: Optional[Dict[str, str]] = None,
    placement_mode: str = "",
) -> Any:
    """The modular lane of :func:`load_from_pretrained` (pgw#1036):
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
    # pgw#1071: DECLARED dtypes only. The snapshot-wide dtype sniff that used
    # to stand in for a declaration was a majority vote over every safetensors
    # header in the tree — it truncated a wide component when the vote fell
    # narrow (an fp32 VAE loading bf16) and, when the vote fell outside the
    # sniff's own vocabulary, produced NO dtype at all and let diffusers'
    # fp32 default upcast the whole tree (ie#615: a 66.28 GB bf16 DiT
    # hydrating at 74.9 GiB, 4 bytes/param, OOM on an 80 GB card and ~130 GB
    # of host staging anon). Undeclared components now load at their OWN
    # checkpoint dtype, decided inside the hydration loop from each
    # component's real source dir.
    scalar_dtype: Any = None
    if dtype:
        try:
            scalar_dtype = get_torch_dtype(dtype)
        except ImportError:
            pass  # torch-less environment: loaders fail on their own terms
    torch_dtype: Any = _modular_declared_dtypes(cls, path, scalar_dtype)
    # pgw#1026: a tree the card holds but the host does not stages ONE
    # COMPONENT AT A TIME straight onto the device. Decided here, from the
    # same measurements the executor's admission gate reads, so the two
    # cannot disagree about which shape the load takes; if free VRAM moved
    # between them the per-component gate inside hydration still refuses
    # with the measured numbers rather than thrashing the host.
    plan = plan_streamed_hydration(
        Path(path), component_trees=component_trees,
        placement_mode=placement_mode)
    if plan.engaged:
        logger.info(
            "modular slot stages per component onto the device: %s",
            plan.summary())
    pipe = cls.from_pretrained(path)
    if not _is_modular_pipeline(pipe):
        raise ModularHydrationError(
            f"{getattr(cls, '__name__', cls)}.from_pretrained returned "
            f"{type(pipe).__name__} without _component_specs/load_components;"
            f" cannot hydrate")
    hydrate_modular_pipeline(
        pipe, Path(path), torch_dtype=torch_dtype,
        component_trees=component_trees, preloaded=components,
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
    component_trees: Optional[Dict[str, str]] = None,
    declared_vram_gb: float = 0.0,
    ref: str = "",
    placement_mode: str = "",
) -> Any:
    """``cls.from_pretrained(path)`` with the standard trimmings: torch dtype
    from the binding's dtype string, on-disk variant detection, quant-library
    preload, and quant-config synthesis; single-file checkpoints route through
    ``cls.from_single_file``. ``storage_dtype="fp8"`` (or an fp8-stored
    snapshot) keeps denoiser weights in fp8 storage with per-layer upcast to
    the compute dtype; ``"fp8+te"`` extends that to the pipeline's text
    encoders (transformers-aware, gw#460). When the snapshot cannot fit free
    VRAM as stored, the adaptive fit ladder engages runtime fp8-E4M3 storage
    first, then the emergency nf4 rung (automatic on CUDA hosts).
    ``components`` are PRELOADED module objects (content-keyed shared
    components, gw#479) forwarded to ``from_pretrained`` — diffusers skips
    loading those from disk and wires the given objects in. Used by the
    executor to satisfy pipeline-typed ``setup()`` annotations; endpoints may
    also call it. A modular pipeline class (pgw#1036: exposes
    ``load_components``) takes its own lane — construct, re-point every
    component spec at the LOCAL tree, hydrate; ``component_trees`` routes
    th#980/pgw#617 component overrides to their own materialized trees on
    that lane (the ``components=`` kwarg is what ``ModularPipeline.__init__``
    silently discards). ``placement_mode`` is the rung the worker will place
    this pipeline on: an offload rung keeps the weights in host RAM, so the
    modular lane must not stage them onto the card and must not take the
    per-component host-RAM discount (pgw#1063)."""
    path = str(path)
    if is_modular_pipeline_class(cls):
        return _load_modular_pipeline(
            cls, path, dtype=dtype, storage_dtype=storage_dtype,
            components=components, component_trees=component_trees,
            placement_mode=placement_mode,
        )
    if component_trees:
        raise ModularHydrationError(
            f"component_trees is the MODULAR delivery mechanism and "
            f"{getattr(cls, '__name__', cls)} is not a modular pipeline "
            f"class; non-modular overrides ride components= (pgw#617)")
    # pgw#1048: the composition the index names must be satisfiable from the
    # tree plus the injection BEFORE any lane touches from_pretrained. A
    # component that is in neither is a deterministic miss, and every lane
    # below reports it as the same nameless OSError against the snapshot root.
    assert_composition_satisfiable(cls, path, components=components, ref=ref)
    # SVDQuant/nunchaku 4-bit flavors (gw#415): self-describing snapshots take
    # the svdq lane — a nunchaku transformer swapped into the standard
    # pipeline. Detection precedes every other rung; failures are typed
    # (SvdqStackError / SvdqHardwareError / SvdqSnapshotError), never a
    # mid-denoise crash.

    svdq_art = detect_svdq_artifact(Path(path))
    if svdq_art is not None and callable(getattr(cls, "from_pretrained", None)):
        if components:
            logger.warning("preloaded components ignored on the svdq lane")
        return load_svdq_pipeline(cls, Path(path), svdq_art)
    # W8A8 fp8-GEMM flavors (gw#534): fp8 weights WITH scales take the
    # scaled-mm lane (fp8 resident, no per-layer cast); hosts without usable
    # scaled_mm dequant once to bf16-resident. Precedes the storage-cast
    # rungs — a scale-free fp8 tree never detects here.

    w8a8_art = detect_w8a8_artifact(Path(path))
    if w8a8_art is not None and callable(getattr(cls, "from_pretrained", None)):
        compute = None
        if dtype:
            try:
                compute = get_torch_dtype(dtype)
            except ImportError:
                pass
        if not w8a8_art.component:
            # Root layout (gw#562): the pipeline class's own loader
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
    # W4A4 nvfp4 flavors (gw#540): packed fp4 weights WITH two-level scales
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
    # pgw#772: a declared fp8 storage lane is SERVED as fp8 storage — the
    # voluntary free-VRAM bf16-resident upgrade is removed (see the lane
    # tombstone above BF16_RESIDENT's old site). Only the involuntary
    # fit-ladder rungs below may move the lane, and only downward.
    fp8_storage = storage_dtype in ("fp8", "fp8+te") or sniffed == "fp8"
    fp8_text_encoders = storage_dtype == "fp8+te"
    adaptive_rung = ""  # gw#491: load-time rung engagement, stamped on the pipe
    if not read_on_disk_quant_config(Path(path)):
        qc = synthesize_quantization_config(attrs)
        if qc is None:
            mode, eqc = _adaptive_fit_rung(
                cls, Path(path), fp8_planned=fp8_storage,
                compute_dtype=kwargs.get("torch_dtype"),
            )
            if mode == "fp8":
                fp8_storage = True  # runtime fp8-E4M3 storage rung (th#683)
                adaptive_rung = "fp8"
            elif eqc is not None:
                qc = eqc
                fp8_storage = False  # nf4 supersedes the fp8 rung
                adaptive_rung = "nf4"
        if qc is not None:
            kwargs["quantization_config"] = qc
    # The composition's ONE compute dtype, captured before pgw#667's
    # per-component map can replace the kwarg with a dict (pgw#683).
    scalar_dtype = kwargs.get("torch_dtype")
    single = _single_file_checkpoint(Path(path))
    if single is not None and callable(getattr(cls, "from_single_file", None)):
        kwargs.pop("variant", None)
        pipe = cls.from_single_file(str(single), **kwargs)
    else:
        # pgw#667: a part whose dtype opinion is WIDER than the composition's
        # compute dtype must come off disk that way — upcasting a bf16-loaded
        # component afterwards recovers no precision, it only hides the
        # truncation. diffusers takes a per-component dtype MAP (a "default"
        # key plus per-part overrides), so the widening happens inside the one
        # from_pretrained instead of a hand-assembled sibling load.
        per_component = _component_dtype_map(cls, path, scalar_dtype)
        if per_component:
            kwargs["torch_dtype"] = per_component
        try:
            pipe = cls.from_pretrained(path, **kwargs)
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
            pipe = cls.from_pretrained(path, **kwargs)

    unmaterialized = meta_tensors(pipe)
    if unmaterialized:
        raise RuntimeError(
            f"{type(pipe).__name__} load left {len(unmaterialized)} "
            f"unmaterialized meta tensors (e.g. {unmaterialized[:3]})"
        )
    if fp8_storage and "quantization_config" not in kwargs:
        # pgw#683: the SCALAR compute dtype — `kwargs["torch_dtype"]` may be
        # pgw#667's per-component MAP by now, and a dict reaching
        # `enable_layerwise_casting(compute_dtype=...)` either explodes into
        # "serving at full precision" or arms windows that upcast to a
        # non-dtype. The cast window's compute dtype is a composition-level
        # fact, so it is the composition default that belongs here.
        applied = apply_fp8_storage(pipe, compute_dtype=scalar_dtype,
                                    text_encoders=fp8_text_encoders)
        # th#737: make the outcome observable — a cast that silently no-ops
        # (denoiser-less pipeline) must surface as a structural degradation
        # upstream, not vanish into a log line.
        try:
            pipe._cozy_fp8_storage_requested = True
            pipe._cozy_fp8_storage_ok = bool(applied)
            if applied:
                setattr(pipe, _WEIGHT_LANE_ATTR, "fp8-hooks")
        except Exception:
            pass
    if adaptive_rung == "nf4":
        # gw#521: verify the quant actually LANDED — a config whose component
        # names miss the pipeline is silently ignored by diffusers, and a
        # full-precision pipeline stamped "nf4" lies to placement and billing.
        targets = list(getattr(
            kwargs.get("quantization_config"), "components_to_quantize", None) or [])
        if targets and not _bnb_quantized_components(pipe, targets):
            logger.error(
                "EMERGENCY nf4 did NOT land on %s (targets %s, pipeline %s) — "
                "serving full precision; the offload ladder carries it",
                path, targets, type(pipe).__name__,
            )
            # pgw#824: this was `adaptive_rung = ""`, which made the failure
            # SELF-SUPPRESSING — the `if adaptive_rung:` stamp below is the
            # very mechanism that reports rung outcomes to placement, and
            # clearing the variable is exactly what switches it off. So the
            # worst rung outcome on the ladder (serving FULL PRECISION over
            # the budgeted VRAM, on a host that was already too tight for
            # stored precision) was the only one that reported nothing at all,
            # while every sibling rung reported itself.
            #
            # A distinct token instead: `provision` routes it to
            # SlotLoad.rung/rung_detail, so it reaches placement through the
            # SAME ServePlan/FnDegraded path as every other rung
            # (`_record_adaptive_rung`) rather than through a log line no
            # hub-spawned pod can expose.
            adaptive_rung = RUNG_NF4_UNLANDED
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"model={path} pipeline={type(pipe).__name__} "
                f"targets={targets}: the emergency nf4 rung was engaged "
                f"because free VRAM was below the stored-precision footprint, "
                f"and it landed on ZERO modules (the config's component names "
                f"miss this pipeline). Serving FULL PRECISION over the "
                f"budgeted VRAM; only the offload ladder carries it now",
                phase="nf4_rung_did_not_land",
            )
    if adaptive_rung:
        # gw#491: a silently-engaged emergency rung is the th#736 bug class —
        # the executor reconciles this stamp into ServePlan.ran / FnDegraded.
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
    "emergency_quant_enabled",
    "bitsandbytes_available",
    "runtime_fp8_storage_supported",
    "emergency_quantization_config",
    "component_load_dtypes",
    "model_index_components",
    "model_index_component_classes",
    "snapshot_component_weight_bytes",
    "specialized_weight_layout",
    "load_from_pretrained",
    "is_modular_pipeline_class",
    "hydrate_modular_pipeline",
    "modular_staging_units",
    "decide_streamed_hydration",
    "plan_streamed_hydration",
    "StreamedHydrationPlan",
    "ModularHydrationError",
    "ComponentSubstitutionError",
    "assert_composition_satisfiable",
    "load_gguf_pipeline",
]
