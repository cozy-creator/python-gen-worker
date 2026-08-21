"""fp8-E4M3 weight storage as module STRUCTURE — a class pun that replaces only `forward` (upcast at use site) — never diffusers' forward-boundary mutation hooks, which are compile-hostile (torch.export refuses module.to(dtype=) at leaf boundaries). Coverage mirrors diffusers' own layerwise-casting rule and REFUSES BY NAME any leaf upstream would cast that it cannot restructure. HAZARD (shared with w8a8/w4a4 storage lanes): never .to(dtype=...) a restructured module — the cast upcasts the fp8 storage and silently doubles residency; device moves are fine."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

_PUNNED: Dict[Tuple[Any, str], type] = {}

LEAF_MARKER = "_cozy_fp8_storage_leaf"
BASE_ATTR = "_cozy_pun_base"

_DEFAULT_SKIP_PATTERNS: Tuple[str, ...] = (
    "pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$",
)


def _default_skip_patterns() -> Tuple[str, ...]:
    try:
        from diffusers.hooks.layerwise_casting import (
            DEFAULT_SKIP_MODULES_PATTERN,
        )

        return tuple(str(p) for p in DEFAULT_SKIP_MODULES_PATTERN)
    except Exception:
        logger.debug("fp8-storage: upstream skip patterns unavailable; "
                     "using the mirrored set", exc_info=True)
        return _DEFAULT_SKIP_PATTERNS


def _peft_skip_patterns() -> Tuple[str, ...]:
    try:
        from peft.tuners.loha.layer import LoHaLayer
        from peft.tuners.lokr.layer import LoKrLayer
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        return ()
    out: List[str] = []
    for layer in (LoHaLayer, LoKrLayer, LoraLayer):
        out.extend(str(n) for n in layer.adapter_layer_names)
    return tuple(out)


def skip_patterns(model: Any) -> Tuple[str, ...]:
    """The leaf-path patterns excluded from fp8 storage — the mirror of ``ModelMixin.enable_layerwise_casting``'s pattern composition (defaults + the model's declared fp32/skip modules + peft adapter names)."""
    pats = set(_default_skip_patterns())
    for attr in ("_keep_in_fp32_modules", "_skip_layerwise_casting_patterns"):
        pats.update(str(p) for p in (getattr(model, attr, None) or ()))
    pats.update(_peft_skip_patterns())
    return tuple(sorted(pats))


def _supported_types() -> Tuple[type, ...]:
    import torch.nn as nn

    try:
        from diffusers.hooks._common import (
            _GO_LC_SUPPORTED_PYTORCH_LAYERS,
        )

        return tuple(_GO_LC_SUPPORTED_PYTORCH_LAYERS)
    except Exception:
        logger.debug("fp8-storage: upstream layer set unavailable; using the "
                     "mirrored set", exc_info=True)
        return (
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.Linear, nn.Embedding,
        )


def _forwards() -> Dict[str, Any]:
    import torch.nn.functional as F

    def linear_forward(self: Any, x: Any) -> Any:
        bias = self.bias
        return F.linear(
            x, self.weight.to(self.compute_dtype),
            None if bias is None else bias.to(self.compute_dtype))

    def conv_forward(self: Any, x: Any) -> Any:
        bias = self.bias
        return self._conv_forward(
            x, self.weight.to(self.compute_dtype),
            None if bias is None else bias.to(self.compute_dtype))

    def conv_transpose_forward(self: Any, x: Any,
                               output_size: Optional[Any] = None) -> Any:
        if self.padding_mode != "zeros":
            raise ValueError(
                "fp8-storage: only `zeros` padding is supported for "
                f"transposed convolution (got {self.padding_mode!r})")
        dims = len(self.kernel_size)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,
            dims, self.dilation)
        fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[dims - 1]
        bias = self.bias
        return fn(
            x, self.weight.to(self.compute_dtype),
            None if bias is None else bias.to(self.compute_dtype),
            self.stride, self.padding, output_padding, self.groups,
            self.dilation)

    def embedding_forward(self: Any, x: Any) -> Any:
        return F.embedding(
            x, self.weight.to(self.compute_dtype), self.padding_idx,
            self.max_norm, self.norm_type, self.scale_grad_by_freq,
            self.sparse)

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
    """The fp8-storage twin of ``base``: same class, same attributes, only ``forward`` replaced by the upcast-at-use-site form."""
    cached = _PUNNED.get((base, kind))
    if cached is not None:
        return cached
    forward = _forwards()[kind]
    built = type(
        f"Fp8Storage{base.__name__}", (base,),
        {
            "forward": forward,
            LEAF_MARKER: True,
            BASE_ATTR: base,
            "__doc__": (
                f"{base.__name__} holding fp8-E4M3 weights, upcast to "
                "``compute_dtype`` at the use site inside forward (pgw#727)."
            ),
        },
    )
    return _PUNNED.setdefault((base, kind), built)


def is_fp8_storage_leaf(module: Any) -> bool:
    """True for a restructured leaf."""
    return bool(getattr(type(module), LEAF_MARKER, False))


def structural_base(module: Any) -> type:
    """The module's structure-relevant class: a restructured leaf answers with the plain class it was restructured from, everything else with its own type."""
    return getattr(type(module), BASE_ATTR, type(module))


def _leaf_paths(model: Any, *, patterns: Tuple[str, ...],
                supported: Tuple[type, ...]) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []

    def walk(module: Any, prefix: str) -> None:
        if any(re.search(p, prefix) for p in patterns):
            return
        if isinstance(module, supported):
            out.append((prefix, module))
            return
        for name, child in module.named_children():
            walk(child, f"{prefix}.{name}" if prefix else name)

    walk(model, "")
    return out


def castable_leaves(model: Any) -> List[str]:
    """The coverage set: paths of every leaf that gets fp8 storage."""
    patterns = skip_patterns(model)
    supported = _supported_types()
    return [p for p, _m in _leaf_paths(model, patterns=patterns,
                                       supported=supported)]


def fp8_storage_leaves(model: Any) -> Dict[str, Any]:
    """``path -> leaf`` for every already-restructured leaf under ``model``."""
    return {n: m for n, m in model.named_modules() if is_fp8_storage_leaf(m)}


def restructure_fp8_storage(model: Any, *, storage_dtype: Any,
                            compute_dtype: Any) -> List[str]:
    """Restructure ``model``'s cast-eligible leaves into fp8 storage modules."""
    patterns = skip_patterns(model)
    supported = _supported_types()
    leaves = _leaf_paths(model, patterns=patterns, supported=supported)

    unsupported = [f"{p} ({type(m).__name__})" for p, m in leaves
                   if not is_fp8_storage_leaf(m) and not _kind(m)]
    if unsupported:
        raise ValueError("fp8-storage: no restructuring for "
                         + ", ".join(unsupported[:8]))
    hooked = [n for n, m in model.named_modules() if _has_cast_hook(m)]
    if hooked:
        raise ValueError(
            "fp8-storage: diffusers layerwise-cast hooks are already armed on "
            f"{len(hooked)} module(s) (first: {hooked[0]!r}) — the mutation "
            "lane and the structural lane must never compose")
    shadowed = [p for p, m in leaves if "forward" in m.__dict__]
    if shadowed:
        raise ValueError(
            "fp8-storage: leaves already carry an instance forward "
            f"(hooks or a LoRA wrap must be applied AFTER): {shadowed[:5]}")

    covered: List[str] = []
    for path, leaf in leaves:
        if not is_fp8_storage_leaf(leaf):
            leaf.__class__ = punned_class(type(leaf), _kind(leaf))
            for name in ("weight", "bias"):
                _to_storage_buffer(leaf, name, storage_dtype)
        leaf.compute_dtype = compute_dtype
        leaf.storage_dtype = storage_dtype
        covered.append(path)
    logger.info("fp8 storage restructured: %d leaves (compute %s, storage %s)",
                len(covered), compute_dtype, storage_dtype)
    if covered:
        _release_freed_blocks()
    return sorted(covered)


def _release_freed_blocks() -> None:
    try:
        import torch

        if cuda_ready() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("fp8-storage: empty_cache after restructure failed",
                     exc_info=True)


def _to_storage_buffer(leaf: Any, name: str, storage_dtype: Any) -> None:
    tensor = getattr(leaf, name, None)
    if tensor is None:
        return
    data = tensor.detach()
    if data.is_floating_point() and data.dtype != storage_dtype:
        data = data.to(storage_dtype)
    param = leaf._parameters.get(name)
    if param is not None and param.data.data_ptr() != data.data_ptr():
        # REBIND the outgoing Parameter onto the fp8 storage before dropping it: anything still holding the old Parameter (accelerate device hooks, low_cpu_mem_usage bookkeeping, an earlier list(model.parameters())) would keep the bf16 storage alive forever beside the fp8 copy — measured +50% VRAM. The one assignment restores the hook lane's property that every holder follows the cast.
        param.data = data
    leaf._parameters.pop(name, None)
    leaf._buffers.pop(name, None)
    leaf.register_buffer(name, data, persistent=True)


def _has_cast_hook(module: Any) -> bool:
    registry = getattr(module, "_diffusers_hook", None)
    if registry is None:
        return False
    get = getattr(registry, "get_hook", None)
    if not callable(get):
        return False
    try:
        return get("layerwise_casting") is not None
    except Exception:
        return False


__all__ = [
    "BASE_ATTR",
    "LEAF_MARKER",
    "castable_leaves",
    "fp8_storage_leaves",
    "is_fp8_storage_leaf",
    "punned_class",
    "restructure_fp8_storage",
    "skip_patterns",
    "structural_base",
]
