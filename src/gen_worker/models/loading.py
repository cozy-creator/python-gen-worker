"""Load-time helpers endpoints (and the executor's typed injection) use around
``from_pretrained``: dtype mapping, on-disk variant detection, and quant-config
synthesis. There is no PipelineLoader — callers own ``from_pretrained``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from .artifact_contract import CONTRACT_PLAIN_BF16, implements_contract
from .fp8_storage import restructure_fp8_storage
from .memory import get_available_vram_gb, meta_tensors
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
    mix. Norms/embeddings are excluded: they carry their own (legitimately
    wider) precision and never meet a weight in one kernel."""
    import torch.nn as nn

    gemm_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    out: Dict[str, str] = {}
    for name, leaf in module.named_modules():
        if not isinstance(leaf, gemm_types):
            continue
        for attr in ("weight", "bias"):
            t = getattr(leaf, attr, None)
            dt = getattr(t, "dtype", None)
            if dt is None:
                continue
            dt_name = str(dt).rsplit(".", 1)[-1]
            if dt_name in _COMPUTE_DTYPE_NAMES:
                out[f"{name}.{attr}" if name else attr] = dt_name
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


def hydrate_modular_pipeline(
    pipe: Any,
    path: str | Path,
    *,
    torch_dtype: Any = None,
    component_trees: Optional[Dict[str, str]] = None,
    preloaded: Optional[Dict[str, Any]] = None,
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
    miss raises typed with the captured diffusers log text."""
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
    if names:
        kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": {n: sources[n] for n in names},
            "subfolder": {n: "" for n in names},
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
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
            pipe.load_components(names=names, **kwargs)
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
    logger.info("modular hydration (%s): %s; skipped partitions: %s",
                type(pipe).__name__, detail, skipped or "none")
    activity_mod.emit_event(
        activity_mod.KIND_MODULAR_HYDRATION,
        f"pipeline={type(pipe).__name__} base={base} {detail}"
        + (f" skipped={','.join(skipped)}" if skipped else ""),
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
    scalar_dtype: Any = None
    wanted = dtype or {
        "bf16": "bf16", "fp16": "fp16", "fp8": "bf16",
    }.get(detect_on_disk_dtype(Path(path)), "")
    if wanted:
        try:
            scalar_dtype = get_torch_dtype(wanted)
        except ImportError:
            pass  # torch-less environment: loaders fail on their own terms
    torch_dtype: Any = scalar_dtype
    per_component = _component_dtype_map(cls, path, scalar_dtype)
    if per_component:
        # Same {"default": ..., part: ...} shape load_components' dict
        # routing understands (pgw#667 widened parts included).
        torch_dtype = per_component
    pipe = cls.from_pretrained(path)
    if not _is_modular_pipeline(pipe):
        raise ModularHydrationError(
            f"{getattr(cls, '__name__', cls)}.from_pretrained returned "
            f"{type(pipe).__name__} without _component_specs/load_components;"
            f" cannot hydrate")
    hydrate_modular_pipeline(
        pipe, Path(path), torch_dtype=torch_dtype,
        component_trees=component_trees, preloaded=components,
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
    silently discards)."""
    path = str(path)
    if is_modular_pipeline_class(cls):
        return _load_modular_pipeline(
            cls, path, dtype=dtype, storage_dtype=storage_dtype,
            components=components, component_trees=component_trees,
        )
    if component_trees:
        raise ModularHydrationError(
            f"component_trees is the MODULAR delivery mechanism and "
            f"{getattr(cls, '__name__', cls)} is not a modular pipeline "
            f"class; non-modular overrides ride components= (pgw#617)")
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
    "load_from_pretrained",
    "is_modular_pipeline_class",
    "hydrate_modular_pipeline",
    "ModularHydrationError",
    "load_gguf_pipeline",
]
