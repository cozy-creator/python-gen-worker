"""Load-time helpers endpoints (and the executor's typed injection) use around
``from_pretrained``: dtype mapping, on-disk variant detection, and quant-config
synthesis. There is no PipelineLoader — callers own ``from_pretrained``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ladder import EMERGENCY_NF4_VRAM_FACTOR

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
_MAX_SAFETENSORS_HEADER_BYTES = 100 << 20


def safetensors_file_valid(path: Path) -> bool:
    """Cheap structural integrity check for one ``.safetensors`` file: the
    header must parse and the file must contain every declared tensor byte.
    Catches truncation (pod-churn-interrupted writes, gw#408) without hashing;
    zero-page corruption inside tensor data needs the digest check instead."""
    import struct

    try:
        p = Path(path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or n > _MAX_SAFETENSORS_HEADER_BYTES or 8 + n > size:
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
    import struct

    counts: Dict[str, int] = {}
    try:
        for p in sorted(Path(model_path).rglob("*.safetensors")):
            with open(p, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    continue
                (n,) = struct.unpack("<Q", raw)
                if n <= 0 or n > _MAX_SAFETENSORS_HEADER_BYTES:
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
        for sub in ("transformer", "unet", "text_encoder", "text_encoder_2", "vae"):
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
_FP8_STORAGE_COMPONENTS: tuple[str, ...] = ("transformer", "unet")
# The "+te" rung (component fit-ladder rung 2): the pipeline's text encoders.
_FP8_TEXT_ENCODER_COMPONENTS: tuple[str, ...] = (
    "text_encoder", "text_encoder_2", "text_encoder_3",
)

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
                      text_encoders: bool = False) -> bool:
    """fp8-E4M3 weight storage with per-layer upcast to ``compute_dtype``
    (diffusers layerwise casting) on a pipeline's denoiser — or on ``obj``
    itself when it is a bare module (th#546 two-format policy).
    ``text_encoders=True`` (the ``storage_dtype="fp8+te"`` rung) extends the
    cast to the pipeline's text encoders via the transformers-aware path.

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

    components = _FP8_STORAGE_COMPONENTS
    if text_encoders:
        components += _FP8_TEXT_ENCODER_COMPONENTS
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
            fn = getattr(mod, "enable_layerwise_casting", None)
            if callable(fn):
                # diffusers ModelMixin — honors the model's own skip patterns.
                fn(storage_dtype=storage, compute_dtype=compute_dtype)
            else:
                _apply_transformers_fp8(mod, storage, compute_dtype)
            mod._cozy_fp8_storage_applied = True
            applied = True
            logger.info("fp8 storage enabled on %s (compute %s)", name, compute_dtype)
        except Exception as exc:
            logger.warning("fp8 storage failed on %s (%s); serving at full precision",
                           name, exc)
    return applied


def _single_file_checkpoint(path: Path) -> Optional[Path]:
    """A snapshot that is one loose checkpoint rather than a pretrained layout:
    the path itself when it's a ``.safetensors`` file, or the directory's sole
    ``*.safetensors`` when no ``model_index.json``/``config.json`` exists
    (e.g. Illustrious-XL, civitai checkpoints).

    Mirrors reshard any >5GB safetensors into byte-offset shards + an
    ``*.safetensors.index.json`` (R2 single-PUT cap), so a big single-file
    checkpoint arrives as N shards. Those are reassembled once into the
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
    import struct

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
        import os

        os.fsync(out.fileno())  # durable before rename (gw#408)
    tmp.rename(merged)
    logger.info("reassembled sharded single-file checkpoint: %s (%d shards, %d tensors, %d bytes)",
                merged.name, len(shard_names), len(entries), offset)
    return merged


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
# the declared CARD SIZE (resources.vram_gb) — a DIFFERENT base than the
# ladder's CAST_FP8_VRAM_FACTOR (0.75), which is against raw WEIGHT BYTES
# (base_size_gb). They estimate the same physical fp8-resident VRAM but the
# denominators differ (card size includes activation/framework headroom over
# raw weights), so the numbers legitimately differ and must NOT be equated.
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


def snapshot_weight_bytes(model_path: Path) -> int:
    """Total tensor bytes across the snapshot's safetensors (header-declared
    data ranges; no tensor reads). 0 when undetectable."""
    import struct

    total = 0
    try:
        for p in sorted(Path(model_path).rglob("*.safetensors")):
            with open(p, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    continue
                (n,) = struct.unpack("<Q", raw)
                if n <= 0 or n > _MAX_SAFETENSORS_HEADER_BYTES:
                    continue
                header = json.loads(f.read(n))
            for value in header.values():
                if isinstance(value, dict) and "data_offsets" in value:
                    s, e = value["data_offsets"]
                    total += int(e) - int(s)
    except (OSError, ValueError):
        return 0
    return total


def bitsandbytes_available() -> bool:
    """Importability gate for the bnb-nf4 rung (gw#469): the quant config
    constructs fine without bitsandbytes and the load then dies deep in
    ``validate_environment`` (PackageNotFoundError -> setup_failed). An
    unavailable rung must be SKIPPED, never attempted."""
    import importlib.util
    import sys

    if "bitsandbytes" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("bitsandbytes") is not None
    except (ImportError, ValueError):
        return False


def emergency_quantization_config(cls: Any) -> Optional[Any]:
    """Denoiser-scoped bnb-nf4 config for the emergency rung. None (with a
    warning) when the stack can't do it — the offload ladder then carries it."""
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
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
    if isinstance(cls, type) and issubclass(cls, diffusers.DiffusionPipeline):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs=kwargs,
            components_to_quantize=list(_FP8_STORAGE_COMPONENTS),
        )
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig

    return BitsAndBytesConfig(**kwargs)


def _adaptive_fit_rung(
    cls: Any, path: Path, *, fp8_planned: bool
) -> tuple[str, Optional[Any]]:
    """Serve-time fit ladder at load (th#683 P3): when the snapshot's
    estimated resident bytes (after any planned fp8 storage) exceed free
    VRAM, engage the cheapest-quality-loss runtime lever that fits:
    fp8-E4M3 storage first (bf16/fp16 weights ~halve, quality ~= a stored
    #fp8 flavor), then the nf4 emergency rung. Returns ``(mode, config)``:
    ``("", None)`` fits as planned; ``("fp8", None)`` engage fp8 storage;
    ``("nf4", qc)`` emergency-quantize (qc None when the stack can't — the
    offload ladder then carries it)."""
    if not emergency_quant_enabled():
        return "", None
    from .memory import get_available_vram_gb

    free_gb = get_available_vram_gb()
    if free_gb <= 0:
        return "", None
    disk = snapshot_weight_bytes(path)
    if disk <= 0:
        return "", None
    resident_gb = disk / float(1 << 30)
    if fp8_planned and detect_on_disk_dtype(path) != "fp8":
        resident_gb *= 0.5  # bf16/fp16 snapshot halved by fp8 storage
    budget_gb = max(0.0, free_gb - _EMERGENCY_MARGIN_GB)
    if resident_gb <= budget_gb:
        return "", None
    # fp8-storage rung: only for un-quantized bf16/fp16 snapshots (an already
    # quantized flavor can't be halved again) when the halved estimate fits.
    if not fp8_planned and detect_on_disk_dtype(path) in ("bf16", "fp16") \
            and runtime_fp8_storage_supported() \
            and resident_gb * 0.5 <= budget_gb:
        logger.warning(
            "fp8-E4M3 emergency weight storage engaged for %s (%.1f GB "
            "weights, %.1f GB free) — near-native quality; a stored #fp8 "
            "flavor of this model would serve natively here.",
            path, resident_gb, free_gb,
        )
        return "fp8", None
    qc = emergency_quantization_config(cls)
    if qc is not None:
        logger.warning(
            "EMERGENCY 4-bit quantization engaged for %s (%.1f GB weights, "
            "%.1f GB free) — quality below platform standards; a larger card "
            "or Blackwell SKU would serve stored flavors instead.",
            path, resident_gb, free_gb,
        )
    return "nf4", qc


def load_from_pretrained(
    cls: Any,
    path: str | Path,
    *,
    dtype: str = "",
    attrs: Optional[Dict[str, str]] = None,
    storage_dtype: str = "",
    components: Optional[Dict[str, Any]] = None,
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
    also call it."""
    path = str(path)
    # SVDQuant/nunchaku 4-bit flavors (gw#415): self-describing snapshots take
    # the svdq lane — a nunchaku transformer swapped into the standard
    # pipeline. Detection precedes every other rung; failures are typed
    # (SvdqStackError / SvdqHardwareError / SvdqSnapshotError), never a
    # mid-denoise crash.
    from .svdq import detect_svdq_artifact, load_svdq_pipeline

    svdq_art = detect_svdq_artifact(Path(path))
    if svdq_art is not None and callable(getattr(cls, "from_pretrained", None)):
        if components:
            logger.warning("preloaded components ignored on the svdq lane")
        return load_svdq_pipeline(cls, Path(path), svdq_art)
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
    fp8_storage = storage_dtype in ("fp8", "fp8+te") or sniffed == "fp8"
    fp8_text_encoders = storage_dtype == "fp8+te"
    adaptive_rung = ""  # gw#491: load-time rung engagement, stamped on the pipe
    if not read_on_disk_quant_config(Path(path)):
        qc = synthesize_quantization_config(attrs)
        if qc is None:
            mode, eqc = _adaptive_fit_rung(cls, Path(path), fp8_planned=fp8_storage)
            if mode == "fp8":
                fp8_storage = True  # runtime fp8-E4M3 storage rung (th#683)
                adaptive_rung = "fp8"
            elif eqc is not None:
                qc = eqc
                fp8_storage = False  # nf4 supersedes the fp8 rung
                adaptive_rung = "nf4"
        if qc is not None:
            kwargs["quantization_config"] = qc
    single = _single_file_checkpoint(Path(path))
    if single is not None and callable(getattr(cls, "from_single_file", None)):
        kwargs.pop("variant", None)
        pipe = cls.from_single_file(str(single), **kwargs)
    else:
        try:
            pipe = cls.from_pretrained(path, **kwargs)
        except (TypeError, ValueError):
            # Not every loader takes variant=/quantization_config= (transformers
            # models, single-file components); retry with the bare essentials.
            kwargs.pop("variant", None)
            kwargs.pop("quantization_config", None)
            pipe = cls.from_pretrained(path, **kwargs)
    if fp8_storage and "quantization_config" not in kwargs:
        applied = apply_fp8_storage(pipe, compute_dtype=kwargs.get("torch_dtype"),
                                    text_encoders=fp8_text_encoders)
        # th#737: make the outcome observable — a cast that silently no-ops
        # (denoiser-less pipeline) must surface as a structural degradation
        # upstream, not vanish into a log line.
        try:
            pipe._cozy_fp8_storage_requested = True
            pipe._cozy_fp8_storage_ok = bool(applied)
        except Exception:
            pass
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
    "safetensors_file_valid",
    "read_on_disk_quant_config",
    "synthesize_quantization_config",
    "apply_fp8_storage",
    "emergency_quant_enabled",
    "bitsandbytes_available",
    "runtime_fp8_storage_supported",
    "emergency_quantization_config",
    "snapshot_weight_bytes",
    "load_from_pretrained",
]
