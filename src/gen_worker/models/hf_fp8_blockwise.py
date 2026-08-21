"""hf.fp8-blockwise@1 consumer half of the layout gate. The layout: fp8 e4m3 weights [out, in], each with a weight_scale_inv F32 twin shaped [out/128, in/128] — one scale per 128x128 block, applied as a MULTIPLIER (DeepSeek-V3 / transformers FineGrainedFP8 convention; `_inv` names the quantizer's divisor, not the dequant direction). NOT readable as cozy.fp8-rowwise@1: that contract's scale leaf is weight_scale, shaped [out], per ROW — reading a blockwise grid as a row vector broadcasts one block's scale across a whole row with no error, just plausible wrong numbers — so this loader REFUSES a rowwise tree, and the pair is PRODUCIBLE-not-CONVERTIBLE (re-blocking is a re-quantization). No quantization happens here; this only reads a pre-quantized tree."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgspec

from .materialized_view import third_party_dir
from .safetensors_header import read_header
from .tensor_layout_contract import implements_quant_rule

#: The ratified v2 quant rule these bytes are (`spec/v2/rules/`). Its
#: conventions — `element: fp8_e4m3`, `scale: block_128x128`, `scale_dtype:
#: F32`, `scale_leaf: weight_scale_inv` — are exactly the facts this module's
#: prose states, and they are IN THE RULE'S DIGEST, which is what makes the
#: rowwise conflation above a refusal rather than a silent mis-scale.
QUANT_RULE = "hf.fp8-blockwise@1"

REFERENCE_DEQUANT = "hf.fp8_blockwise.dequant@1"

QUANT_METHOD = "fp8"
SCALE_LEAF = "weight_scale_inv"
ROWWISE_SCALE_LEAF = "weight_scale"
_FP8_DTYPES = ("F8_E4M3",)
_SCALE_DTYPES = ("F32", "F8_E8M0", "F8_E8M0FNU", "U8")


class HfFp8BlockwiseError(RuntimeError):
    """Typed failure of the blockwise fp8 loader."""


class HfFp8BlockwiseLayoutError(HfFp8BlockwiseError):
    """The tree is not in ``hf.fp8-blockwise@1``."""


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

    return read_header(
        path,
        why="an fp8-blockwise artifact whose dtypes and shapes go unseen is "
            "routed to the plain bf16 lane and loads as the wrong model",
    )


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
            f"{QUANT_RULE}. A dense tree is plain.bf16@1; bind "
            "that, or bind an artifact a conversion endpoint produced in this "
            "layout.")
    method = str(qc.get("quant_method", "")).lower()
    if method != QUANT_METHOD:
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: quant_method={method!r}, want {QUANT_METHOD!r} for "
            f"{QUANT_RULE}")
    return qc


def _declared_block(qc: Dict[str, Any], cfg_path: Path) -> Tuple[int, int]:
    raw = qc.get("weight_block_size")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise HfFp8BlockwiseLayoutError(
            f"{cfg_path}: weight_block_size={raw!r} is not a [block_m, "
            f"block_n] pair — a per-tensor or per-row fp8 tree is NOT "
            f"{QUANT_RULE}")
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
    """Verify a tree against ``hf.fp8-blockwise@1`` from headers alone."""
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


def declares_quant_rule(quantization_config: Any) -> bool:
    """Does this ``quantization_config`` DECLARE ``hf.fp8-blockwise@1``?

    CONFIG ONLY — no header is read and no file beside ``config.json`` is
    touched, which is what makes it usable from the meta skeleton
    (pgw#1638): ``ctx.load`` builds from configs and never opens a tensor
    container to find out what a class wants. The header verification stays
    in :func:`inspect_hf_fp8_blockwise`, where a tree with bytes is present.

    Takes the mapping out of ``config.json`` or the typed
    ``FineGrainedFP8Config``; ``quant_method`` may be a plain string or
    transformers' str-Enum, so the value is unwrapped before comparing.
    """
    qc = quantization_config
    if qc is None:
        return False
    if isinstance(qc, dict):
        read = qc.get
    else:
        def read(key: str, default: Any = None) -> Any:
            return getattr(qc, key, default)
    method = read("quant_method", "")
    method = getattr(method, "value", method)
    if str(method or "").lower() != QUANT_METHOD:
        return False
    raw = read("weight_block_size", None)
    return isinstance(raw, (list, tuple)) and len(raw) == 2


def detect_hf_fp8_blockwise(path: Path) -> Optional[HfFp8BlockwiseTree]:
    """The verified ``hf.fp8-blockwise@1`` tree at ``path``, or ``None``."""
    p = Path(path)
    try:
        cfg = json.loads((p / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    qc = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    if not isinstance(qc, dict) or not declares_quant_rule(qc):
        return None
    return inspect_hf_fp8_blockwise(p)


def dequantize_block_scaled(weight: Any, scale: Any, *, out_dtype: Any = None) -> Any:
    """``hf.fp8_blockwise.dequant@1`` — the contract's reference dequant."""
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
    if s.dtype == torch.uint8:
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


# The 128x128 grid, the `weight_scale_inv` leaf and the dynamic activation
# scheme are no longer declared beside the handle: they ARE
# `hf.fp8-blockwise@1`, written into its conventions and its digest. That is
# what makes the rowwise conflation this module refuses (`inspect_...` verifies
# the grid) inexpressible rather than merely guarded — there is no side axis
# left on which a decoder could claim "rowwise too" under this handle.
@implements_quant_rule(
    rule=QUANT_RULE,
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
    """Load a pre-quantized ``hf.fp8-blockwise@1`` component."""
    import torch
    from transformers import FineGrainedFP8Config

    from ..discovery.decode_set import require_decodable

    require_decodable(QUANT_RULE, where=str(root))
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
    return model_cls.from_pretrained(
        str(third_party_dir(path, why=f"{model_cls} fp8-blockwise from_pretrained")),
        **kwargs,
    )


__all__ = [
    "BlockwiseUnit",
    "HfFp8BlockwiseError",
    "HfFp8BlockwiseLayoutError",
    "HfFp8BlockwiseTree",
    "QUANT_RULE",
    "REFERENCE_DEQUANT",
    "declares_quant_rule",
    "dequantize_block_scaled",
    "detect_hf_fp8_blockwise",
    "inspect_hf_fp8_blockwise",
    "load_hf_fp8_blockwise",
]
