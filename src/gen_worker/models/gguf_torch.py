"""GGML-quantized weights served by the ORDINARY torch path."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import gguf_dequant
from .fp8_storage import BASE_ATTR, structural_base

logger = logging.getLogger(__name__)

LEAF_MARKER = "_cozy_gguf_leaf"

SPEC_ATTR = "gguf_specs"

ADAPTER_ATTR = "gguf_adapters"

MATERIALIZED_ATTR = "gguf_materialized"

_PUNNED: Dict[Tuple[Any, str], type] = {}


@dataclass(frozen=True)
class QuantSpec:
    """What a leaf needs to decode one tensor: the GGML type and the shape the flat block stream expands to."""

    qtype: int
    shape: Any

    def __post_init__(self) -> None:
        if not gguf_dequant.is_supported(self.qtype):
            raise NotImplementedError(
                f"gguf-torch: {gguf_dequant.qtype_name(self.qtype)} has no "
                "vectorized decode")


@dataclass(frozen=True)
class QuantizedTensor:
    """Block bytes plus what they decode to — the unit that crosses into this module, whether it came from a ``.gguf`` file or from the CAS."""

    blocks: Any
    spec: QuantSpec

    @property
    def qtype(self) -> int:
        return self.spec.qtype

    @property
    def shape(self) -> Any:
        return self.spec.shape


@dataclass(frozen=True)
class LoraPatch:
    """One low-rank adapter branch, applied to the DEQUANTIZED weight."""

    down: Any
    up: Any
    scale: float = 1.0

    def delta(self, like: Any) -> Any:
        up = self.up.to(device=like.device, dtype=like.dtype)
        down = self.down.to(device=like.device, dtype=like.dtype)
        product = up.flatten(1) @ down.flatten(1)
        return product.reshape(like.shape) * self.scale


def _materialize(leaf: Any, name: str, dtype: Any) -> Any:
    tensor = getattr(leaf, name, None)
    if tensor is None:
        return None

    spec = getattr(leaf, SPEC_ATTR, {}).get(name)
    if spec is None:
        weight = tensor.to(dtype)
    else:
        weight = gguf_dequant.dequantize(
            tensor, spec.qtype, spec.shape,
            dtype=getattr(leaf, "dequant_dtype", None) or dtype)

    patches = getattr(leaf, ADAPTER_ATTR, {}).get(name, ())
    if patches:
        weight = weight.to(dtype)
        for patch in patches:
            weight = weight + patch.delta(weight)
    return weight.to(dtype)


def _compute_dtype(leaf: Any, x: Any) -> Any:
    import torch

    if x is not None and x.is_floating_point():
        return x.dtype
    return getattr(leaf, "compute_dtype", None) or torch.get_default_dtype()


@functools.lru_cache(maxsize=1)
def _forwards() -> Dict[str, Any]:
    import torch.nn.functional as F

    def linear_forward(self: Any, x: Any) -> Any:
        dtype = _compute_dtype(self, x)
        return F.linear(x, _materialize(self, "weight", dtype),
                        _materialize(self, "bias", dtype))

    def conv_forward(self: Any, x: Any) -> Any:
        dtype = _compute_dtype(self, x)
        return self._conv_forward(x, _materialize(self, "weight", dtype),
                                  _materialize(self, "bias", dtype))

    def conv_transpose_forward(self: Any, x: Any,
                               output_size: Optional[Any] = None) -> Any:
        if self.padding_mode != "zeros":
            raise ValueError(
                "gguf-torch: only `zeros` padding is supported for transposed "
                f"convolution (got {self.padding_mode!r})")
        dims = len(self.kernel_size)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, dims,
            self.dilation)
        fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[dims - 1]
        dtype = _compute_dtype(self, x)
        return fn(x, _materialize(self, "weight", dtype),
                  _materialize(self, "bias", dtype), self.stride, self.padding,
                  output_padding, self.groups, self.dilation)

    def embedding_forward(self: Any, x: Any) -> Any:
        dtype = _compute_dtype(self, x)
        return F.embedding(x, _materialize(self, "weight", dtype),
                           self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)

    return {
        "linear": linear_forward,
        "conv": conv_forward,
        "conv_transpose": conv_transpose_forward,
        "embedding": embedding_forward,
    }


def _kind(module: Any) -> str:
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "conv"
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d,
                           nn.ConvTranspose3d)):
        return "conv_transpose"
    if isinstance(module, nn.Embedding):
        return "embedding"
    return ""


def punned_class(base: Any, kind: str) -> type:
    """The GGML-storage twin of ``base``: same class, same attributes, only ``forward`` replaced by the decode-at-use-site form."""
    cached = _PUNNED.get((base, kind))
    if cached is not None:
        return cached
    built = type(
        f"Gguf{base.__name__}", (base,),
        {
            "forward": _forwards()[kind],
            LEAF_MARKER: True,
            BASE_ATTR: base,
            "__doc__": (
                f"{base.__name__} holding GGML block bytes, decoded to "
                "``compute_dtype`` at the use site inside forward (pgw#1498)."
            ),
        },
    )
    return _PUNNED.setdefault((base, kind), built)


def is_gguf_leaf(module: Any) -> bool:
    return bool(getattr(type(module), LEAF_MARKER, False))


def gguf_leaves(model: Any) -> Dict[str, Any]:
    """``path -> leaf`` for every punned leaf under ``model``."""
    return {n: m for n, m in model.named_modules() if is_gguf_leaf(m)}


def install_quantized_weights(
    model: Any,
    tensors: Mapping[str, Any],
    *,
    compute_dtype: Any,
    device: Any = None,
    dequant_dtype: Any = None,
) -> List[str]:
    """Install ``tensors`` into ``model``, punning every leaf that gets blocks."""
    import torch

    if device is not None:
        device = torch.device(device)

    punned: List[str] = []
    missing: List[str] = []
    for key in sorted(tensors):
        value = tensors[key]
        path, _, attr = key.rpartition(".")
        try:
            leaf = model.get_submodule(path) if path else model
        except AttributeError:
            missing.append(key)
            continue

        if isinstance(value, QuantizedTensor):
            if not is_gguf_leaf(leaf):
                kind = _kind(leaf)
                if not kind:
                    raise ValueError(
                        f"gguf-torch: no decode-at-use-site forward for {path!r} "
                        f"({type(leaf).__name__}); it cannot hold block bytes")
                if "forward" in leaf.__dict__:
                    raise ValueError(
                        f"gguf-torch: {path!r} already carries an instance "
                        "forward (hooks or a LoRA wrap must be applied AFTER)")
                leaf.__class__ = punned_class(type(leaf), kind)
                setattr(leaf, SPEC_ATTR, {})
                setattr(leaf, ADAPTER_ATTR, {})
                setattr(leaf, MATERIALIZED_ATTR, {})
                punned.append(path)
            getattr(leaf, SPEC_ATTR)[attr] = value.spec
            _install_buffer(leaf, attr, _detached(value.blocks, device))
        else:
            dense = _detached(value, device)
            if dense.is_floating_point():
                dense = dense.to(compute_dtype)
            _install_parameter(leaf, attr, dense)

        leaf.compute_dtype = compute_dtype
        leaf.dequant_dtype = dequant_dtype

    if missing:
        raise KeyError(
            f"gguf-torch: {len(missing)} tensor(s) name no module on the model "
            f"(first: {missing[0]!r})")

    logger.info("gguf-torch: %d leaves hold block bytes (compute %s)",
                len(punned), compute_dtype)
    return sorted(punned)


def _detached(tensor: Any, device: Any) -> Any:
    if device is None:
        return tensor.detach().clone()
    return tensor.detach().to(device=device, copy=True)


def _install_parameter(leaf: Any, name: str, tensor: Any) -> None:
    import torch

    param = leaf._parameters.get(name)
    if param is not None and param.device.type != "meta":
        param.data = tensor
        param.requires_grad_(False)
        return
    leaf._parameters.pop(name, None)
    leaf._buffers.pop(name, None)
    leaf.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def _install_buffer(leaf: Any, name: str, tensor: Any) -> None:
    param = leaf._parameters.get(name)
    if param is not None and param.device.type == "meta":
        param = None
    if param is not None and param.data.data_ptr() != tensor.data_ptr():
        param.requires_grad_(False)
        param.data = tensor
    leaf._parameters.pop(name, None)
    leaf._buffers.pop(name, None)
    leaf.register_buffer(name, tensor, persistent=True)


def dequant_ahead(model: Any, *, surplus_bytes: float, dtype: Any) -> List[str]:
    """Decode as many weights ONCE at load as ``surplus_bytes`` pays for."""
    import torch

    itemsize = torch.empty((), dtype=dtype).element_size()
    candidates: List[Tuple[int, int, str, Any, str]] = []
    for path, leaf in gguf_leaves(model).items():
        for name, spec in getattr(leaf, SPEC_ATTR, {}).items():
            blocks = getattr(leaf, name)
            dense = _elements(spec.shape) * itemsize
            price = dense - blocks.numel() * blocks.element_size()
            candidates.append((dense, price, path, leaf, name))
    candidates.sort(key=lambda c: -c[0])

    spent = 0
    done: List[str] = []
    for dense, price, path, leaf, name in candidates:
        if spent + price > surplus_bytes:
            continue
        materialize(leaf, name, dtype=dtype)
        spent += price
        done.append(f"{path}.{name}")

    logger.info("gguf-torch: dequant-ahead materialized %d/%d weights for "
                "%d bytes of a %s surplus", len(done), len(candidates), spent,
                surplus_bytes)
    return done


def materialize(leaf: Any, name: str, *, dtype: Any) -> None:
    """Decode one weight once and drop its blocks."""
    specs = getattr(leaf, SPEC_ATTR, {})
    spec = specs.get(name)
    if spec is None:
        return
    dense = gguf_dequant.dequantize(getattr(leaf, name), spec.qtype, spec.shape,
                                    dtype=dtype)
    del specs[name]
    getattr(leaf, MATERIALIZED_ATTR)[name] = spec
    _install_parameter(leaf, name, dense.contiguous())


def peak_transient_bytes(model: Any, *, dtype: Any) -> int:
    """The largest weight still decoded per forward — the headroom a fit plan must reserve on top of the resident bytes."""
    import torch

    itemsize = torch.empty((), dtype=dtype).element_size()
    return max(
        (_elements(spec.shape) * itemsize
         for leaf in gguf_leaves(model).values()
         for spec in getattr(leaf, SPEC_ATTR, {}).values()),
        default=0)


def _elements(shape: Any) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def quantized_bytes(model: Any) -> int:
    """Resident bytes of block storage — the number the small-card ladder cares about."""
    total = 0
    for leaf in gguf_leaves(model).values():
        for name in getattr(leaf, SPEC_ATTR, {}):
            tensor = getattr(leaf, name, None)
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def attach_lora(leaf: Any, patches: Iterable[LoraPatch], *,
                name: str = "weight") -> None:
    """Attach adapter branches to one leaf, applied AFTER the decode."""
    if not is_gguf_leaf(leaf):
        raise ValueError("gguf-torch: attach_lora expects a punned leaf")
    if name not in getattr(leaf, SPEC_ATTR, {}) and \
            name not in getattr(leaf, MATERIALIZED_ATTR, {}):
        raise ValueError(
            f"gguf-torch: {name!r} on this leaf never held block bytes — "
            "apply the adapter through the ordinary LoRA path")
    getattr(leaf, ADAPTER_ATTR)[name] = tuple(patches)


def detach_lora(model: Any) -> int:
    """Drop every attached adapter."""
    dropped = 0
    for leaf in gguf_leaves(model).values():
        adapters = getattr(leaf, ADAPTER_ATTR, None)
        if adapters:
            dropped += len(adapters)
            adapters.clear()
    return dropped


def quantized_tensors_from_views(views: Mapping[str, Any], *,
                                 pin_memory: bool = False) -> Dict[str, Any]:
    """The SERVED path: :class:`tensorfs.tensors.TensorView` -> installable values."""
    import gguf
    import torch

    out: Dict[str, Any] = {}
    for key, view in views.items():
        shape = torch.Size(tuple(int(d) for d in reversed(view.shape)))
        blocks = torch.empty(int(view.nbytes), dtype=torch.uint8,
                             pin_memory=pin_memory)
        view.readinto(memoryview(blocks.numpy()))
        qtype = int(gguf.GGMLQuantizationType[str(view.dtype)])
        if qtype in gguf_dequant.passthrough_qtypes():
            dense = blocks.view(torch.float32 if view.dtype == "F32" else torch.float16)
            out[key] = dense.view(shape)
        else:
            out[key] = QuantizedTensor(blocks, QuantSpec(qtype, shape))
    return out


@dataclass
class GgufRead:
    """What one ``.gguf`` file yields: tensors keyed by their name in the file, plus the container's own metadata."""

    tensors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    architecture: str = ""


def read_gguf(path: Any, *, prefix: str = "") -> GgufRead:
    """Read a ``.gguf`` container into :class:`QuantizedTensor` values."""
    import warnings

    import gguf
    import torch

    reader = gguf.GGUFReader(str(path))
    out = GgufRead(architecture=_field_str(reader, "general.architecture") or "")

    for tensor in reader.tensors:
        name = tensor.name
        if prefix:
            if not name.startswith(prefix):
                continue
            name = name[len(prefix):]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            blocks = torch.from_numpy(tensor.data)

        shape = _orig_shape(reader, tensor.name) or torch.Size(
            tuple(int(v) for v in reversed(tensor.shape)))
        qtype = int(tensor.tensor_type)

        if qtype in gguf_dequant.passthrough_qtypes():
            out.tensors[name] = blocks.view(shape)
        else:
            out.tensors[name] = QuantizedTensor(blocks, QuantSpec(qtype, shape))

    counts: Dict[str, int] = {}
    for value in out.tensors.values():
        key = (gguf_dequant.qtype_name(value.qtype)
               if isinstance(value, QuantizedTensor) else "dense")
        counts[key] = counts.get(key, 0) + 1
    logger.info("gguf-torch: read %s — %s", path,
                ", ".join(f"{k} ({v})" for k, v in sorted(counts.items())))
    return out


def _field_str(reader: Any, key: str) -> Optional[str]:
    field_ = reader.get_field(key)
    if field_ is None:
        return None
    return str(field_.parts[field_.data[-1]], encoding="utf-8")


def _orig_shape(reader: Any, name: str) -> Any:
    import torch

    field_ = reader.get_field(f"comfy.gguf.orig_shape.{name}")
    if field_ is None:
        return None
    return torch.Size(tuple(int(field_.parts[i][0]) for i in field_.data))


__all__ = [
    "ADAPTER_ATTR",
    "BASE_ATTR",
    "LEAF_MARKER",
    "MATERIALIZED_ATTR",
    "SPEC_ATTR",
    "GgufRead",
    "LoraPatch",
    "QuantSpec",
    "QuantizedTensor",
    "attach_lora",
    "dequant_ahead",
    "detach_lora",
    "gguf_leaves",
    "install_quantized_weights",
    "is_gguf_leaf",
    "materialize",
    "peak_transient_bytes",
    "punned_class",
    "quantized_bytes",
    "quantized_tensors_from_views",
    "read_gguf",
    "structural_base",
]
