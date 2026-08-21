"""Load-time helpers endpoints (and the executor's typed injection) use around ``from_pretrained``: dtype mapping, on-disk variant detection, and quant-config synthesis."""

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
from .file_layout import is_single_file_snapshot
from .tensor_layout_contract import (
    implements_quant_rule,
    unregistered_decode_path,
)

#: The ratified v2 quant rules this module's arms decode, by their tensorfs
#: `spec/v2/rules/` handles. Named once so a lane refusal and the decorator
#: below spell one handle, never two.
RULE_PLAIN_BF16 = "plain.bf16@1"
RULE_COZY_FP8_ROWWISE = "cozy.fp8-rowwise@1"
RULE_HF_FP8_BLOCKWISE = "hf.fp8-blockwise@1"
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
    """Map a dtype string to a torch dtype."""
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
    """Detect a diffusers ``variant=`` value from files on disk (e.g."""
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
    """Cheap structural integrity check for one ``.safetensors`` file: the header must parse and the file must contain every declared tensor byte."""

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
    """Majority weight dtype across the snapshot's safetensors headers ("bf16" / "fp16" / "fp32" / "fp8", "" when undetectable)."""

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
    """True when model_index.json / component config.json on disk carries a ``quantization_config`` block (diffusers auto-picks it up)."""
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
    """Build a BitsAndBytesConfig from resolved checkpoint attrs when the on-disk config doesn't already carry one."""
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


_fp8_storage_components = denoiser_components
_fp8_text_encoder_components = text_encoder_components

class _Fp8WeightWindow:

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
            if id(block) in seen_blocks:
                continue
            seen_blocks.add(id(block))
            blocks.append((f"{name}.{i}", block))

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
    windows = _fp8_block_windows(mod)
    if not windows:
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
    """fp8-E4M3 weight storage with per-layer upcast to ``compute_dtype`` on a pipeline's denoiser — or on ``obj`` itself when it is a bare module."""
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
            applied = True
            continue
        try:
            if callable(getattr(mod, "enable_layerwise_casting", None)):
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
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"component={name} obj={type(obj).__name__}: fp8 storage "
                f"cast failed; this component serves at full precision "
                f"(over its budgeted VRAM): {type(exc).__name__}: {exc}",
                phase="fp8_cast_failed",
            )
    return applied


# This keeps its public name because it has a production consumer OUTSIDE this repo (serverless-endpoints/ltx-video-2.3 main.py) — "no call site in this repo" is NOT "no consumer", and deleting it breaks that endpoint at import. It is now a thin adapter onto the one streaming mechanism (models.stream_residency) at budget zero: park everything, stream per forward.


def apply_block_window_offload(
    obj: Any,
    *,
    components: tuple[str, ...] | None = None,
    device: Any = None,
) -> bool:
    """Park a pipeline's weights in pinned host RAM and stream them per forward (degraded-mode rung 2)."""
    if components is None:
        components = denoiser_components()
    try:
        from .stream_residency import StreamedResidency
    except Exception as exc:  # noqa: BLE001 — torch-less host
        logger.warning("block-window offload ignored: %s", exc)
        return False
    if device is None:
        if not cuda_ready():
            logger.warning("block-window offload ignored: no CUDA device")
            return False
        device = "cuda"

    targets: List[tuple[str, Any]] = []
    for name in components:
        mod = getattr(obj, name, None)
        if mod is not None and hasattr(mod, "named_modules"):
            targets.append((name, mod))
    if not targets and hasattr(obj, "named_modules"):
        targets.append((type(obj).__name__, obj))

    applied = False
    for name, mod in targets:
        if getattr(mod, "_cozy_block_offload_applied", False):
            applied = True
            continue
        residency = StreamedResidency([(name, mod)], device=device, budget_bytes=0)
        plan = residency.engage()
        if not plan.streamed:
            logger.warning("block-window offload: no streamable leaves in %s", name)
            continue
        try:
            mod._cozy_block_offload_applied = True
        except Exception:  # noqa: BLE001
            pass
        applied = True
        logger.warning(
            "DEGRADED_MODE=engaged model=%s phase=load rung=resident->"
            "block_offload: %d leaves / %.1f GiB parked in host RAM, streaming "
            "per forward",
            name, len(plan.streamed), plan.streamed_bytes / float(1 << 30),
        )
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"component={name} obj={type(obj).__name__}: block-window offload "
            f"ENGAGED — {len(plan.streamed)} leaf module(s) / "
            f"{plan.streamed_bytes / float(1 << 30):.1f} GiB rest in host RAM "
            f"and stream to the device per forward; every request on this "
            f"component pays that transfer",
            phase="block_offload_engaged",
        )
    return applied


def require_decodable(rule: str, path: Any, *, component: str = "") -> None:
    """Refuse, typed, before handing bytes to a decoder this IMAGE does not
    declare (pgw#1245).

    Two questions, both answered from HEADERS and directory shape, both before
    any tensor is read: is this QUANT RULE in the image's decode-set, and was
    the tree's tensor-KEY convention classifiable at all. The second is not the
    first: `plain.bf16@1` minimax-native weights (fused `blocks.N.attn.qkv_proj`)
    and `plain.bf16@1` diffusers weights (split `to_q/to_k/to_v`) are the same
    rule with one key in common, and the diffusers class cannot read the native
    tree at all. A DENOISER whose convention matches nothing this image
    recognizes refuses — unknown is never a hopeful pass where a model class is
    chosen from the architecture.

    The per-decoder FILE-LAYOUT axis is gone with the v1 vocabulary (pgw#1621),
    so the on-disk SHAPE is no longer checked here; `observed_file_layout` is
    still what a publish-side classifier reads, and nothing in this image
    intersects on it any more.

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

    _require(
        rule,
        where=str(path),
        keys=classify_snapshot(Path(path), component),
    )


def detect_gguf_snapshot(path: Path) -> Optional[tuple[Path, str]]:
    """Return the GGUF denoiser and qtype in a composed diffusers snapshot."""
    p = Path(path)
    if not p.is_dir() or not (p / "model_index.json").exists():
        return None
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
    reason="gguf.native@1 is a LOADER-SHAPE topology handle; no ratified v2 "
           "QUANT RULE names the ggml block encodings this decodes "
           "(`models/gguf_dequant.py`) — the eight rules are three plain "
           "dtypes, three fp8 packagings and two nvfp4 packagings, and a "
           "k-quant block is none of them. So the bytes are decodable here and "
           "unnameable in the decode-set until a `spec/v2/rules/` document "
           "describes them (per block quant, with its own capability floor — a "
           "GGUF rule states an honest 0) and is re-vendored. That authoring "
           "is the platform-side half of pgw#1498's storage ruling and is what "
           "closes this exemption.",
)
def load_gguf_pipeline(
    cls: Any,
    path: Path,
    gguf_file: Path,
    *,
    components: Optional[Dict[str, Any]] = None,
    source: Optional[Any] = None,
) -> Any:
    """Load a GGUF denoiser into the remaining components' base tree."""

    import torch

    from .gguf_diffusers import SingleFileGguf, build_denoiser

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
    denoiser = build_denoiser(
        denoiser_cls,
        third_party_dir(path / component, why="GGUF config dir"),
        source or SingleFileGguf(
            third_party_dir(gguf_file, why="the community .gguf edge reads a real file")),
        compute_dtype=compute,
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
    try:
        setattr(pipe, _WEIGHT_LANE_ATTR, EXECUTION_LANE_GGUF)
    except Exception:  # noqa: BLE001 — diffusers __setattr__ registers components
        logger.warning("could not stamp the gguf weight lane on %s",
                       type(pipe).__name__)
    return pipe


def _single_file_checkpoint(path: Path) -> Optional[Path]:
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

    merged = snapshot_dir / index_path.name[: -len(".index.json")]
    if merged.exists():
        if safetensors_file_valid(merged):
            return merged
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

    entries: List[tuple[str, dict, Path, int, int]] = []
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

        os.fsync(out.fileno())
    tmp.rename(merged)
    logger.info("reassembled sharded single-file checkpoint: %s (%d shards, %d tensors, %d bytes)",
                merged.name, len(shard_names), len(entries), offset)
    return merged


# NEVER REBUILD the "voluntary upgrade" that upcast an fp8 download to bf16-resident when free VRAM allowed: the serving lane must be deterministic per (release x declared config), never a function of the individual card's free VRAM — such a probe makes `lane` a GPU-dependent axis of the compiled graph key, so a card with a small VRAM surplus over its same-SKU peers falls into a lane nothing mints for and serves eager for life. Involuntary transitions (the fit-ladder rung, the w8a8/w4a4 dequant-on-unsupported-host lanes) are declared rungs and stay.

# The pipeline's weight lane, part of the compile-cache graph key: "" = plain resident weights (incl. the involuntary dequant lanes), "fp8-hooks" = fp8 weights resident with a per-layer upcast traced INTO the FX graphs. "fp8-hooks" is the WIRE value — tensorhub maps it to w8a16 and compiled graphs key on it — so it must stay byte-identical.
_WEIGHT_LANE_ATTR = "_cozy_weight_lane"

EXECUTION_LANE_GGUF = "gguf"

STAMPABLE_BASE_EXECUTION_LANES: Tuple[str, ...] = (
    "",
    "fp8-hooks",
    "gguf",
    "w8a8",
    "w4a4",
    "svdq-native",
)


def pipeline_weight_lane(pipeline: Any) -> str:
    execution_lane = str(getattr(pipeline, _WEIGHT_LANE_ATTR, "") or "")
    if execution_lane == "bf16-resident":
        return ""
    if execution_lane:
        return execution_lane
    for name in _fp8_storage_components():
        if getattr(getattr(pipeline, name, None), "_cozy_fp8_storage_applied", False):
            return "fp8-hooks"
    if getattr(pipeline, "_cozy_fp8_storage_applied", False):
        return "fp8-hooks"
    return ""


_EMERGENCY_MARGIN_GB = 2.0


def runtime_fp8_storage_supported() -> bool:
    """The runtime fp8-E4M3 storage rung needs a CUDA host and a torch that ships the float8_e4m3fn dtype — no fp8 silicon required (per-layer upcast compute; see apply_fp8_storage)."""
    try:
        import torch

        return bool(cuda_ready()) and hasattr(torch, "float8_e4m3fn")
    except ImportError:
        return False


def model_index_components(path: str | Path) -> set:
    """Component names the snapshot's model_index.json declares."""
    try:
        with open(Path(path) / "model_index.json", "r", encoding="utf-8") as f:
            index = json.load(f)
        return {k for k in index if not k.startswith("_")}
    except Exception:
        return set()


def model_index_component_classes(path: str | Path) -> Dict[str, str]:
    """``{component: class name}`` the snapshot's ``model_index.json`` declares."""
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
    """``{component: ComponentDtype}`` this composition's parts require at LOAD time — the snapshot's own ``model_index.json`` classes first, the pipeline class's ``__init__`` annotations as the fallback."""
    from ..families.facts import component_classes, component_dtypes_for_classes

    classes: Dict[str, str] = {}
    if isinstance(pipeline_cls, type):
        classes.update(component_classes(pipeline_cls))
    for part, class_name in model_index_component_classes(path).items():
        if str(class_name or "").strip():
            classes[str(part)] = str(class_name).strip()
    return dict(component_dtypes_for_classes(classes))


def model_index_entry(path: str | Path, component: str) -> Optional[tuple]:
    """``(library, class_name)`` the tree's model_index.json declares for ``component``, or None when absent/unreadable."""
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


QUANT_EXECUTION_LANE_COMPUTE_DEFAULT = "bf16"


def composition_compute_dtype(base_path: str | Path, dtype: str = "") -> str:
    """The compute dtype the COMPOSED pipeline will run at: the base binding's declared dtype when present, else the dtype the base tree's LOAD LANE actually computes at."""
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
        return "bf16"
    if sniffed in ("bf16", "fp16"):
        return sniffed
    return ""


class ProjectedTreeNotEagerlyLoadable(RuntimeError):
    """ProjectedTreeNotEagerlyLoadable."""


class MixedComputeDtypeError(RuntimeError):
    """A composed pipeline presents more than one COMPUTE dtype to its GEMMs."""


_COMPUTE_DTYPE_NAMES = ("float16", "bfloat16", "float32", "float64")
_INCOMPATIBLE_COMPUTE = ("float16", "bfloat16")


def _gemm_param_dtypes(module: Any) -> Dict[str, str]:
    import torch.nn as nn

    gemm_types = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    out: Dict[str, str] = {}
    for name, leaf in module.named_modules():
        declared = getattr(leaf, "compute_dtype", None)
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
        dec_name = str(declared).rsplit(".", 1)[-1] if declared is not None else ""
        if dec_name in _COMPUTE_DTYPE_NAMES:
            out[f"{name}.compute_dtype" if name else "compute_dtype"] = dec_name
    return out


def assert_uniform_compute_dtype(
    obj: Any, expected: str = "", *, label: str = "",
) -> None:
    """Refuse a MIXED-precision composition at LOAD."""
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
    except Exception:
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


_CHECKPOINT_LOAD_DTYPE = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp8": QUANT_EXECUTION_LANE_COMPUTE_DEFAULT,
}


def checkpoint_load_dtype(source: str | Path) -> str:
    """The dtype ONE component tree's own bytes ask to be loaded at, or ``""`` when its headers say nothing."""
    return _CHECKPOINT_LOAD_DTYPE.get(detect_on_disk_dtype(Path(source)), "")


def _declared_component_dtype(name: str, declared: Any) -> Any:
    if isinstance(declared, dict):
        if name in declared:
            return declared[name]
        return declared.get("default")
    return declared


def _modular_declared_dtypes(
    cls: Any, path: str | Path, scalar_dtype: Any,
) -> Any:
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
    wanted = _declared_component_dtype(name, declared)
    if wanted is not None:
        return wanted
    token = checkpoint_load_dtype(source)
    if not token:
        return None
    try:
        return get_torch_dtype(token)
    except ImportError:
        return None


class ComponentExecutionLaneUnsupported(RuntimeError):
    """This flavor has no component-level loader, so no honest one exists."""


def _accepts_kwarg(fn: Any, name: str) -> bool:
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
    """THE production loader for ONE named pipeline component."""

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
            pass

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
    """THE contract dispatch: one component's tree -> its registered loader."""
    weights = Path(root)
    where = Path(src) if src is not None else weights

    def _covers(artifact_component: str) -> bool:
        return artifact_component == component or (
            not artifact_component and where == weights
        )

    w8a8_art = detect_w8a8_artifact(weights)
    if w8a8_art is not None and _covers(w8a8_art.component):
        require_decodable(
            RULE_COZY_FP8_ROWWISE, weights, component=w8a8_art.component)
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
            f"built from its config and filled with block bytes by the "
            f"pipeline's own gguf loader, so there is no component-level "
            f"production loader to borrow"
        )
    blockwise = detect_hf_fp8_blockwise(where)
    if blockwise is not None:
        require_decodable(
            RULE_HF_FP8_BLOCKWISE, weights,
            component=component if where != weights else "")
        return load_hf_fp8_blockwise(
            where, cls=cls, dtype=compute_dtype, tree=blockwise)
    return None


class ModularHydrationError(RuntimeError):
    """A ModularPipeline slot could not be hydrated from the LOCAL tree ."""


def _pipeline_component_names(cls: Any) -> Optional[set]:
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
    """Duck-typed: a modular pipeline class exposes ``load_components`` (weights hydrate AFTER construction) — ``DiffusionPipeline`` does not."""
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
    for sub in (str(getattr(spec, "subfolder", "") or ""), name):
        if not sub:
            continue
        cand = base / sub
        if cand.is_dir():
            return cand
    return None


def _weightless_model_dir(src: Path) -> bool:
    if not (src / "config.json").is_file():
        return False
    return next(src.rglob("*.safetensors"), None) is None


_GIB = 1024 ** 3
_STAGING_FLOOR_GB = 8.0
_STAGING_FLOOR_FRACTION = 0.2


def _staging_floor_bytes(total_bytes: int) -> int:
    if total_bytes <= 0:
        return int(_STAGING_FLOOR_GB * _GIB)
    return int(min(_STAGING_FLOOR_GB * _GIB,
                   max(_GIB, total_bytes * _STAGING_FLOOR_FRACTION)))


def _admit_component_staging(component: str, nbytes: int) -> None:
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
    """Bytes per INDEPENDENTLY STAGED unit of a modular snapshot."""
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


_STREAMED_HYDRATION_VRAM_MARGIN_GB = 2.0


@dataclass(frozen=True)
class StreamedHydrationPlan:
    """Whether a modular slot may hydrate component-by-component ONTO THE DEVICE, so host RAM never holds more than one component at a time."""

    engaged: bool
    reason: str
    tree_bytes: int
    largest_unit_bytes: int
    unit_count: int
    host_total_bytes: int
    device_free_bytes: int
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
    """Decide whether this modular slot stages PER COMPONENT ONTO THE DEVICE instead of staging its whole tree in host RAM first."""
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
    """:func:`plan_streamed_hydration`'s decision, separated from its measurements so the rule can be read — and tested — at the byte counts that produced the issue rather than at whatever this host happe..."""
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
    """Hydrate a freshly constructed ``ModularPipeline`` from the LOCAL snapshot tree."""
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
        sources[name] = str(
            third_party_dir(src, why=f"ComponentSpec.load({name!r})"))

    for name in sources:
        spec = specs[name]
        spec.pretrained_model_name_or_path = sources[name]
        spec.subfolder = ""
    for name in skipped:
        specs[name].pretrained_model_name_or_path = None
    for name in pre:
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
        records: List[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        dlog = logging.getLogger("diffusers")
        handler = _Capture(level=logging.WARNING)
        dlog.addHandler(handler)
        try:
            for n in names:
                comp_src = Path(sources[n])
                comp_bytes = disk_gc.tree_bytes(comp_src)
                load_progress.set_phase(f"hydrate:{n}", comp_bytes)
                _admit_component_staging(n, comp_bytes)
                kwargs: Dict[str, Any] = {
                    "pretrained_model_name_or_path": {n: sources[n]},
                    "subfolder": {n: ""},
                }
                dt = _hydration_dtype(n, torch_dtype, comp_src)
                if dt is not None:
                    kwargs["torch_dtype"] = dt
                    dtypes[n] = str(dt).removeprefix("torch.")
                built = _contract_component_for_spec(specs[n], n, comp_src)
                if built is not None:
                    pipe.update_components(**{n: built})
                else:
                    pipe.load_components(names=[n], **kwargs)
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
    """Tensor bytes per top-level component dir (header-declared data ranges; no tensor reads)."""
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


def _adaptive_fit_rung(
    cls: Any, path: Path, *, fp8_planned: bool, compute_dtype: Any = None
) -> tuple[str, Optional[Any]]:
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
        resident -= 0.5 * denoiser_bytes
    total_gb = total / float(1 << 30)
    if resident <= budget:
        return "", None
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
    if storage_dtype:
        logger.warning(
            "storage_dtype=%s ignored on the modular lane (component "
            "precision is a per-component artifact fact)", storage_dtype)
    scalar_dtype: Any = None
    if dtype:
        try:
            scalar_dtype = get_torch_dtype(dtype)
        except ImportError:
            pass
    torch_dtype: Any = _modular_declared_dtypes(cls, path, scalar_dtype)
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


# ONE RULE PER ELEMENT TYPE, and this arm declares only bf16. `plain.bf16@1`,
# `plain.f16@1` and `plain.f32@1` are three ratified rules with three digests
# and three capability floors (80 / 70 / 0) — under v1 they were one handle
# with an `elements=` side axis listing all three, which is precisely the
# shape the v2 cut removes. The loader still READS an f16 or f32 tree (it
# honours the weights' own precision below); what it can no longer do is claim
# all three under one handle. Declaring the other two is a separate, checkable
# statement — `serves=` and the floor differ — not a comma in a tuple.
#
# `scales=(none,)` is likewise the handle now: a plain rule transforms nothing,
# and a tree carrying scale tensors is a quantized rule whose own decoder reads
# them. The `key_topologies=` axis is the v2 TOPOLOGY half of a lane stamp and
# is no longer a decoder's to declare — see the report note on what that costs.
@implements_quant_rule(
    rule=RULE_PLAIN_BF16,
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
    ref: str = "",
    placement_mode: str = "",
) -> Any:
    """``cls.from_pretrained(path)`` with the standard trimmings: torch dtype from the binding's dtype string, on-disk variant detection, quant-library preload, and quant-config synthesis; single-file che..."""
    path = str(path)
    from .projection import stub_at_any

    if stub_at_any(Path(path)):
        raise ProjectedTreeNotEagerlyLoadable(
            f"load_from_pretrained({getattr(cls, '__name__', cls)}, {path}): "
            f"this is a PROJECTED tensorfs tree — its tensor containers are "
            f"~128 B TFSSTUB1 pointer stubs whose bytes live in the CAS. The "
            f"eager loader would materialize a full second copy of the tree "
            f"to read them (tier 3 of the pgw#1303 ladder), which the "
            f"2026-08-19 no-fill ruling leaves to external binaries only. "
            f"Load through `ctx.load(<PipelineClass>)`, which binds the "
            f"pgw#1380 streaming engine and walks the chunk store straight to "
            f"VRAM with no tensor file written or read."
        )
    if is_modular_pipeline_class(cls):
        return _load_modular_pipeline(
            cls, path, dtype=dtype, storage_dtype=storage_dtype,
            components=components, placement_mode=placement_mode,
        )

    svdq_art = detect_svdq_artifact(Path(path))
    if svdq_art is not None and callable(getattr(cls, "from_pretrained", None)):
        # NO `require_decodable` ON THIS ARM (pgw#1621), and it is a gap rather
        # than a decision: the check takes a RULE HANDLE, and no ratified v2
        # rule names the svdq packaging `models/svdq_layout.py` decodes (see
        # its `@unregistered_decode_path`). The v1 handle this passed —
        # `nunchaku.v1@1` — is deleted, and passing it now would refuse the
        # whole lane, since an unregistered path contributes no decode-set
        # entry. Ratifying the rule restores the check as one line.
        if components:
            logger.warning("preloaded components ignored on the svdq lane")
        return load_svdq_pipeline(cls, Path(path), svdq_art)

    w8a8_art = detect_w8a8_artifact(Path(path))
    if w8a8_art is not None and callable(getattr(cls, "from_pretrained", None)):
        require_decodable(RULE_COZY_FP8_ROWWISE, path)
        compute = None
        if dtype:
            try:
                compute = get_torch_dtype(dtype)
            except ImportError:
                pass
        if not w8a8_art.component:
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
    require_decodable(RULE_PLAIN_BF16, path)
    kwargs: Dict[str, Any] = {}
    if components:
        kwargs.update(components)
    if dtype:
        try:
            kwargs["torch_dtype"] = get_torch_dtype(dtype)
        except ImportError:
            pass
    variant = detect_diffusers_variant(Path(path))
    if variant in ("bf16", "fp16"):
        kwargs["variant"] = variant
    sniffed = detect_on_disk_dtype(Path(path))
    if "torch_dtype" not in kwargs:
        if sniffed in ("bf16", "fp16", "fp8"):
            try:
                kwargs["torch_dtype"] = get_torch_dtype(
                    "bf16" if sniffed == "fp8" else sniffed
                )
            except ImportError:
                pass
    fp8_storage = storage_dtype in ("fp8", "fp8+te") or sniffed == "fp8"
    fp8_text_encoders = storage_dtype == "fp8+te"
    adaptive_rung = ""
    if not read_on_disk_quant_config(Path(path)):
        qc = synthesize_quantization_config(attrs)
        if qc is None:
            mode, eqc = _adaptive_fit_rung(
                cls, Path(path), fp8_planned=fp8_storage,
                compute_dtype=kwargs.get("torch_dtype"),
            )
            assert eqc is None
            if mode == "fp8":
                fp8_storage = True
                adaptive_rung = "fp8"
        if qc is not None:
            kwargs["quantization_config"] = qc
    scalar_dtype = kwargs.get("torch_dtype")
    single = _single_file_checkpoint(Path(path))
    if single is not None and callable(getattr(cls, "from_single_file", None)):
        kwargs.pop("variant", None)
        pipe = cls.from_single_file(
            str(third_party_dir(single, why="from_single_file wants a real file")),
            **kwargs)
    else:
        per_component = _component_dtype_map(cls, path, scalar_dtype)
        if per_component:
            kwargs["torch_dtype"] = per_component
        try:
            pipe = cls.from_pretrained(
                third_party_dir(path, why="pipeline from_pretrained"), **kwargs)
        except (TypeError, ValueError):
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
        applied = apply_fp8_storage(pipe, compute_dtype=scalar_dtype,
                                    text_encoders=fp8_text_encoders)
        try:
            pipe._cozy_fp8_storage_requested = True
            pipe._cozy_fp8_storage_ok = bool(applied)
            if applied:
                setattr(pipe, _WEIGHT_LANE_ATTR, "fp8-hooks")
        except Exception:
            pass
    if adaptive_rung:
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
    "apply_block_window_offload",
    "apply_fp8_storage",
    "assert_uniform_compute_dtype",
    "MixedComputeDtypeError",
    "ProjectedTreeNotEagerlyLoadable",
    "composition_compute_dtype",
    "QUANT_EXECUTION_LANE_COMPUTE_DEFAULT",
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
