"""Load-time helpers endpoints (and the executor's typed injection) use around
``from_pretrained``: dtype mapping, on-disk variant detection, and quant-config
synthesis. There is no PipelineLoader — callers own ``from_pretrained``.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    candidates = ("bf16", "fp8", "fp16", "int8", "int4")
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


_SAFETENSORS_DTYPE_NAMES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
_MAX_SAFETENSORS_HEADER_BYTES = 100 << 20


def detect_on_disk_dtype(model_path: Path) -> str:
    """Majority weight dtype across the snapshot's safetensors headers
    ("bf16" / "fp16" / "fp32", "" when undetectable). Hub bindings carry no
    dtype and mirrored repos use unsuffixed filenames, so without this a bf16
    snapshot silently loads via diffusers' fp32 default — 2x the VRAM."""
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


# quant-library -> import-side-effect hooks. torchao registers its tensor
# subclasses on import; loading a torchao-quantized state_dict before that
# import fails with ATen/dispatcher errors.
_QUANT_LIBRARY_IMPORT_HOOKS: Dict[str, str] = {
    "torchao": "torchao",
}


def ensure_quant_library_imported(attrs: Optional[Dict[str, str]]) -> None:
    """Best-effort preload of the quant library whose tensor subclasses must be
    registered before the weights are touched. No-op when not applicable."""
    if not attrs:
        return
    lib = str(attrs.get("quant_library") or "").strip().lower()
    mod = _QUANT_LIBRARY_IMPORT_HOOKS.get(lib)
    if not mod:
        return
    try:
        importlib.import_module(mod)
        logger.info("pre-imported %s for tensor-subclass registration", mod)
    except ImportError as exc:
        logger.warning("failed to pre-import %s: %s", mod, exc)


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
    don't indicate a library that needs a synthesized config (torchao restores
    via import side-effect; see :func:`ensure_quant_library_imported`)."""
    if not attrs:
        return None
    lib = str(attrs.get("quant_library") or "").strip().lower()
    if lib != "bitsandbytes":
        return None
    recipe = str(attrs.get("quant_recipe") or "").strip().lower()
    scheme = recipe.split(":", 1)[-1] if ":" in recipe else recipe
    if scheme not in ("nf4", "fp4", "int8"):
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
    if scheme in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=scheme,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


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
        return merged
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
    tmp.rename(merged)
    logger.info("reassembled sharded single-file checkpoint: %s (%d shards, %d tensors, %d bytes)",
                merged.name, len(shard_names), len(entries), offset)
    return merged


def load_from_pretrained(
    cls: Any,
    path: str | Path,
    *,
    dtype: str = "",
    attrs: Optional[Dict[str, str]] = None,
) -> Any:
    """``cls.from_pretrained(path)`` with the standard trimmings: torch dtype
    from the binding's dtype string, on-disk variant detection, quant-library
    preload, and quant-config synthesis; single-file checkpoints route through
    ``cls.from_single_file``. Used by the executor to satisfy pipeline-typed
    ``setup()`` annotations; endpoints may also call it."""
    path = str(path)
    ensure_quant_library_imported(attrs)
    kwargs: Dict[str, Any] = {}
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
    if "torch_dtype" not in kwargs:
        # Bindings without an explicit dtype (Hub mirrors): honor the weights'
        # own precision instead of diffusers' fp32 default.
        sniffed = detect_on_disk_dtype(Path(path))
        if sniffed in ("bf16", "fp16"):
            try:
                kwargs["torch_dtype"] = get_torch_dtype(sniffed)
            except ImportError:
                # torch-less environment (unit tests / CPU tools) — loaders
                # that actually need torch will fail on their own terms.
                pass
    if not read_on_disk_quant_config(Path(path)):
        qc = synthesize_quantization_config(attrs)
        if qc is not None:
            kwargs["quantization_config"] = qc
    single = _single_file_checkpoint(Path(path))
    if single is not None and callable(getattr(cls, "from_single_file", None)):
        kwargs.pop("variant", None)
        return cls.from_single_file(str(single), **kwargs)
    try:
        return cls.from_pretrained(path, **kwargs)
    except (TypeError, ValueError):
        # Not every loader takes variant=/quantization_config= (transformers
        # models, single-file components); retry with the bare essentials.
        kwargs.pop("variant", None)
        kwargs.pop("quantization_config", None)
        return cls.from_pretrained(path, **kwargs)


__all__ = [
    "get_torch_dtype",
    "detect_diffusers_variant",
    "detect_on_disk_dtype",
    "ensure_quant_library_imported",
    "read_on_disk_quant_config",
    "synthesize_quantization_config",
    "load_from_pretrained",
]
