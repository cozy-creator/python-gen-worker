"""``hf.fp8-blockwise@1`` — the consumer half of th#1803's layout gate.

The artifact side of the tensor-layout contract shipped in tensorhub
(`internal/tensorlayout/seed_fp8_blockwise_th1803.go`, transcribed from the
real header bytes of `tensorhub/minimax-h3`'s fp8 conditioner). Nothing
DECLARED it, so the hub correctly refused every rebind onto that artifact:
no decoder in any image said it could read those bytes. This module is the
declaration, and the loader that makes it true.

**The layout, stated once.** fp8 e4m3 weights `[out, in]`, each carrying a
`weight_scale_inv` F32 twin shaped `[out/128, in/128]` — one scale per
128x128 block, applied as a MULTIPLIER (`w.float() * scale`, DeepSeek-V3 /
transformers `FineGrainedFP8` convention; the `_inv` suffix names the
quantizer's divisor, not the dequant direction). Layers outside
`modules_to_not_convert` are detected by dtype + scale presence, never by a
name list.

**Why this cannot be read as ``cozy.fp8-rowwise@1``.** That contract is also
"fp8 e4m3, dynamic activations" — same element type, same activation scheme,
one English name. Its scale leaf is `weight_scale`, shaped `[out]`, per ROW.
A decoder that reads a blockwise `[out/128, in/128]` grid as a per-row vector
broadcasts one block's scale across a whole row: no exception, no shape
error at the point that matters, just plausible-looking numbers that are
wrong. That is why the two are separate handles and why this loader REFUSES
a rowwise tree instead of adapting to it.

The two are not inter-convertible either (DESIGN-RULINGS §1.33 ladder):
re-blocking rowwise scales into a 128x128 grid is a RE-QUANTIZATION with new
numerics, so the pair is PRODUCIBLE-not-CONVERTIBLE — a priced production job
from a named higher-precision source, never a silent load-time repack.

**No quantization happens here** (Paul, 2026-08-11): quantization is done
ahead of time by a conversion endpoint and served as an artifact. This module
only READS a pre-quantized tree. There is deliberately no
quantize-if-missing fallback — that path is what th#1803 deleted.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgspec

from .safetensors_header import header_len_ok
from .tensor_layout_contract import CONTRACT_HF_FP8_BLOCKWISE, implements_contract

# The registry's declared reference dequant for this contract. Named here so
# the SDK's transform and tensorhub's descriptor point at ONE spec rather than
# two implementations that agree by luck.
REFERENCE_DEQUANT = "hf.fp8_blockwise.dequant@1"

QUANT_METHOD = "fp8"
SCALE_LEAF = "weight_scale_inv"
ROWWISE_SCALE_LEAF = "weight_scale"
_FP8_DTYPES = ("F8_E4M3",)
_SCALE_DTYPES = ("F32", "F8_E8M0", "F8_E8M0FNU", "U8")


class HfFp8BlockwiseError(RuntimeError):
    """Typed failure of the blockwise fp8 loader."""


class HfFp8BlockwiseLayoutError(HfFp8BlockwiseError):
    """The tree is not in ``hf.fp8-blockwise@1``.

    Raised in preference to loading anything: a layout refusal is the whole
    point of the contract, and the hub's gate exists so this is normally
    unreachable at serve time.
    """


class BlockwiseUnit(msgspec.Struct, frozen=True, kw_only=True):
    """One quantized Linear: its fp8 weight and the block-scale grid."""

    module: str
    out_features: int
    in_features: int
    scale_rows: int
    scale_cols: int

    @property
    def block(self) -> Tuple[int, int]:
        return (
            math.ceil(self.out_features / self.scale_rows),
            math.ceil(self.in_features / self.scale_cols),
        )


class HfFp8BlockwiseTree(msgspec.Struct, frozen=True, kw_only=True):
    """A verified ``hf.fp8-blockwise@1`` component tree."""

    root: Path
    component: str
    block_size: Tuple[int, int]
    activation_scheme: str
    scale_fmt: str
    modules_to_not_convert: Tuple[str, ...]
    units: Tuple[BlockwiseUnit, ...]
    files: Tuple[Path, ...]

    @property
    def path(self) -> Path:
        return self.root / self.component if self.component else self.root


def _read_header(path: Path) -> Dict[str, Any]:
    """Safetensors header only — bytes read are the declared length, capped
    by §4.24's bound. No tensor data is touched."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            (n,) = struct.unpack("<Q", raw)
            if not header_len_ok(n):
                return {}
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return {}
    return header if isinstance(header, dict) else {}


def _weight_files(d: Path) -> Tuple[Path, ...]:
    sharded: set[str] = set()
    for idx in sorted(d.glob("*.safetensors.index.json")):
        try:
            weight_map = json.loads(idx.read_text("utf-8")).get("weight_map") or {}
        except (OSError, ValueError):
            continue
        sharded.update(str(v) for v in weight_map.values())
    files = [d / s for s in sorted(sharded) if (d / s).is_file()]
    files += [p for p in sorted(d.glob("*.safetensors"))
              if p.is_file() and p.name not in sharded]
    return tuple(dict.fromkeys(files))


def _quant_config(path: Path) -> Dict[str, Any]:
    cfg_path = path / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path} is unreadable: {exc}") from exc
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path} declares no quantization_config — this tree is not "
            f"{CONTRACT_HF_FP8_BLOCKWISE}. A dense tree is plain.bf16@1; bind "
            "that, or bind an artifact a conversion endpoint produced in this "
            "layout.")
    method = str(qc.get("quant_method", "")).lower()
    if method != QUANT_METHOD:
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: quant_method={method!r}, want {QUANT_METHOD!r} for "
            f"{CONTRACT_HF_FP8_BLOCKWISE}")
    return qc


def _declared_block(qc: Dict[str, Any], cfg_path: Path) -> Tuple[int, int]:
    raw = qc.get("weight_block_size")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: weight_block_size={raw!r} is not a [block_m, "
            f"block_n] pair — a per-tensor or per-row fp8 tree is NOT "
            f"{CONTRACT_HF_FP8_BLOCKWISE}")
    try:
        bm, bn = int(raw[0]), int(raw[1])
    except (TypeError, ValueError) as exc:
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: weight_block_size={raw!r} is not integral") from exc
    if bm <= 0 or bn <= 0:
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: weight_block_size={raw!r} must be positive")
    return bm, bn


def inspect_hf_fp8_blockwise(
    root: Path, *, component: str = "",
) -> HfFp8BlockwiseTree:
    """Verify a tree against ``hf.fp8-blockwise@1`` from headers alone.

    The same tier-1 evidence tensorhub's `tensorlayout.Identify` reads at
    publish, checked again by the consumer: the declaration on the release
    row says the image CAN decode this layout, and this says the bytes in
    front of it really are that layout. Disagreement is a typed refusal —
    never a best-effort load.
    """
    root = Path(root)
    path = root / component if component else root
    qc = _quant_config(path)
    block = _declared_block(qc, path / "config.json")
    scale_fmt = str(qc.get("scale_fmt", "float"))
    activation_scheme = str(qc.get("activation_scheme", "dynamic"))
    skip = tuple(str(m) for m in (qc.get("modules_to_not_convert") or []))

    files = _weight_files(path)
    if not files:
        raise HfFp8BlockwiseLayoutError(f"{path} holds no safetensors shards")

    dtypes: Dict[str, str] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    for f in files:
        for name, info in _read_header(f).items():
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            dtypes[name] = str(info["dtype"])
            shapes[name] = tuple(int(d) for d in info.get("shape") or ())

    fp8_weights = sorted(
        k for k, dt in dtypes.items()
        if k.endswith(".weight") and dt in _FP8_DTYPES)
    if not fp8_weights:
        raise HfFp8BlockwiseLayoutError(
            f"{path}: config declares fp8 but no F8_E4M3 weight is present — "
            "the config and the bytes disagree, which is a refusal, never a "
            "fallback (te#148 rule)")

    units: list[BlockwiseUnit] = []
    for wkey in fp8_weights:
        module = wkey[: -len(".weight")]
        skey = f"{module}.{SCALE_LEAF}"
        if skey not in dtypes:
            rowwise = f"{module}.{ROWWISE_SCALE_LEAF}"
            if rowwise in dtypes:
                raise HfFp8BlockwiseLayoutError(
                    f"{path}: {module} carries {ROWWISE_SCALE_LEAF} "
                    f"{shapes.get(rowwise)}, not {SCALE_LEAF} — this tree is "
                    "cozy.fp8-rowwise@1, a DIFFERENT tensor-layout contract "
                    "with the same element type and the same activation "
                    "scheme. Reading a per-row multiplier as a 128x128 "
                    "reciprocal grid (or the reverse) broadcasts one scale "
                    "over the wrong span and yields plausible, wrong numbers "
                    "rather than an error. The two are not convertible: "
                    "re-blocking is a re-quantization (PRODUCIBLE, "
                    "DESIGN-RULINGS §1.33), so produce the artifact in this "
                    "layout on a conversion endpoint and bind that.")
            raise HfFp8BlockwiseLayoutError(
                f"{path}: {wkey} is F8_E4M3 with no {skey} — an fp8 weight "
                "with no scale is undecodable, not a dense weight")
        sdt = dtypes[skey]
        if sdt not in _SCALE_DTYPES:
            raise HfFp8BlockwiseLayoutError(
                f"{path}: {skey} has dtype {sdt}, want one of "
                f"{', '.join(_SCALE_DTYPES)}")
        wshape, sshape = shapes.get(wkey, ()), shapes.get(skey, ())
        if len(wshape) != 2 or len(sshape) != 2:
            raise HfFp8BlockwiseLayoutError(
                f"{path}: {module} weight {wshape} / scale {sshape} are not "
                "both rank 2 — a rank-1 scale is the rowwise contract")
        out_f, in_f = wshape
        want = (math.ceil(out_f / block[0]), math.ceil(in_f / block[1]))
        if tuple(sshape) != want:
            raise HfFp8BlockwiseLayoutError(
                f"{path}: {skey} is {tuple(sshape)}, want {want} for weight "
                f"{wshape} at block {block} — a transposed or mis-blocked "
                "scale grid decodes silently wrong")
        units.append(BlockwiseUnit(
            module=module, out_features=out_f, in_features=in_f,
            scale_rows=sshape[0], scale_cols=sshape[1]))

    return HfFp8BlockwiseTree(
        root=root, component=component, block_size=block,
        activation_scheme=activation_scheme, scale_fmt=scale_fmt,
        modules_to_not_convert=skip, units=tuple(units), files=files)


def dequantize_block_scaled(weight: Any, scale: Any, *, out_dtype: Any = None) -> Any:
    """``hf.fp8_blockwise.dequant@1`` — the contract's reference dequant.

    ``w[i, j] * scale[i // block_m, j // block_n]``, with the block size
    DERIVED from the scale grid rather than the config: one checkpoint may
    mix granularities, and the grid is the fact. Bit-identical to
    transformers' `_dequantize_one` for F32 scales; the exactness that makes
    a §1.33 conversion provable is measured against this function.
    """
    import torch

    q = weight.to(torch.float32)
    if q.ndim != 2:
        raise HfFp8BlockwiseError(f"weight must be rank 2, got {tuple(q.shape)}")
    rows, cols = q.shape
    s = scale
    if s.ndim != 2:
        raise HfFp8BlockwiseError(f"scale must be rank 2, got {tuple(s.shape)}")
    srows, scols = s.shape
    block_m, block_n = math.ceil(rows / srows), math.ceil(cols / scols)
    if s.dtype == torch.uint8:  # ue8m0 exponents stored as bytes
        s = (s.to(torch.float32) - 127.0).exp2()
    else:
        s = s.to(torch.float32)
    expanded = s.repeat_interleave(block_m, dim=0).repeat_interleave(block_n, dim=1)
    out = q * expanded[:rows, :cols]
    return out.to(out_dtype or torch.bfloat16)


def _hf_model_class(path: Path, cls: Any) -> Any:
    if cls is not None:
        return cls
    from transformers import AutoModel

    return AutoModel


@implements_contract(
    contract=CONTRACT_HF_FP8_BLOCKWISE,
    serves=("fp8-w8a8-dynamic", "fp8-w8a16"),
    composes_lora=False,
    why="th#1803: transformers' FineGrainedFP8 reads this layout natively — "
        "resident fp8 weights with a 128x128 block scale grid, dynamic "
        "per-token activation scales through the triton/DeepGEMM blockwise "
        "GEMM (fp8-w8a8-dynamic), or upcast-ahead to the compute dtype at "
        "load (fp8-w8a16). No adapter branch exists in FP8Linear, so this "
        "decoder does not compose runtime LoRAs.",
)
def load_hf_fp8_blockwise(
    root: Path,
    *,
    component: str = "",
    cls: Any = None,
    dtype: Any = None,
    device_map: Any = None,
    resident: bool = True,
    tree: Optional[HfFp8BlockwiseTree] = None,
) -> Any:
    """Load a pre-quantized ``hf.fp8-blockwise@1`` component.

    ``resident=True`` keeps the fp8 weights and runs the blockwise GEMM
    (`fp8-w8a8-dynamic`, CUDA); ``resident=False`` upcasts ahead to
    ``dtype`` at load (`fp8-w8a16`), which is the portable arm and the only
    one that runs on CPU.

    ``cls`` pins the model class; the default resolves through
    ``transformers.AutoModel``, which is what the component's own
    ``config.json`` already names.
    """
    import torch
    from transformers import FineGrainedFP8Config

    verified = tree or inspect_hf_fp8_blockwise(root, component=component)
    path = verified.path
    compute = dtype or torch.bfloat16

    quant = FineGrainedFP8Config(
        activation_scheme=verified.activation_scheme,
        weight_block_size=verified.block_size,
        modules_to_not_convert=list(verified.modules_to_not_convert) or None,
        scale_fmt=verified.scale_fmt,
        dequantize=not resident,
    )
    model_cls = _hf_model_class(path, cls)
    kwargs: Dict[str, Any] = {"dtype": compute, "quantization_config": quant}
    if device_map is not None:
        kwargs["device_map"] = device_map
    return model_cls.from_pretrained(str(path), **kwargs)


__all__ = [
    "BlockwiseUnit",
    "CONTRACT_HF_FP8_BLOCKWISE",
    "HfFp8BlockwiseError",
    "HfFp8BlockwiseLayoutError",
    "HfFp8BlockwiseTree",
    "REFERENCE_DEQUANT",
    "dequantize_block_scaled",
    "inspect_hf_fp8_blockwise",
    "load_hf_fp8_blockwise",
]
