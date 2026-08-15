"""fp8-E4M3 weight storage as module STRUCTURE, not a forward-boundary
mutation.

The lane's semantics are "weights RESIDE in fp8, compute happens in the
compute dtype". diffusers expresses that as a MUTATION at the forward
boundary (``LayerwiseCastingHook``: ``module.to(dtype=compute)`` pre,
``module.to(dtype=storage)`` post). We deliberately do NOT use that lane: it is
COMPILE-HOSTILE, because ``module.to(dtype=...)`` at every leaf boundary defeats
the compiler — dynamo buys it nothing and ``torch.export`` refuses it outright
(``_apply(): Couldn't swap Linear.weight``). The identical semantics expressed
as STRUCTURE — hold the fp8 tensor, upcast at the use site inside ``forward`` —
is faster under dynamo, exportable, and free of the ``p.data``
mutation-aliasing hazard the offload rung shares.

**Coverage is upstream's, not ours.** Which leaves get fp8 storage is
decided by diffusers' rule (``_apply_layerwise_casting``: its supported layer
set, its default skip patterns, the model's own
``_skip_layerwise_casting_patterns`` / ``_keep_in_fp32_modules``, and peft's
adapter tensor names). This module mirrors that rule and REFUSES BY NAME any
leaf upstream would cast that it cannot restructure — a silently narrower
coverage set would be a silent VRAM regression AND a silent numerics change.
``tests/test_fp8_storage_pgw727.py`` asserts set-equality against the
installed diffusers, so upstream drift fails loud in CI rather than on a pod.

**Residency.** Two separate facts, and the first does not imply the second:

1. *Coverage parity*: the same leaves hold the same bytes at the same
   precision as the hook lane, and outputs are bitwise equal. Peak transient
   is one leaf's upcast either way — diffusers hooks per LEAF, not per block.
2. *No orphaned originals*: a class pun replaces the weight tensor OBJECT, so
   the bf16 original survives anywhere it is still held, and both copies
   resident is a ~50% VRAM regression. ``_to_storage_buffer`` rebinds the
   outgoing Parameter onto the fp8 storage, restoring the hook lane's property
   that every holder follows the cast. A module-only walk cannot see this
   failure — it reports "identical residency" while a pod says otherwise.

The restructure is a CLASS PUN: the leaf keeps its identity, parameters,
FQNs and kernel attributes, and only its ``forward`` changes (``nn.Linear``
-> ``Fp8StorageLinear(nn.Linear)``). ``isinstance`` stays true, so the
offload rung, LoRA branch targeting and dtype introspection keep working on
the same objects.

HAZARD (shared with the w8a8 / w4a4 storage lanes): never ``.to(dtype=...)``
a restructured module — a dtype cast upcasts the fp8 storage and silently
doubles residency. Device moves are fine.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

#: (plain class, kind) -> the punned class. One class object per pair: the
#: module TYPE enters the cell's composition digest, so it must be stable for
#: the life of the process.
_PUNNED: Dict[Tuple[Any, str], type] = {}

#: Class attribute marking a restructured leaf, and (on the class) the plain
#: class it was restructured from. Structural markers, not name checks: they
#: survive refactors and record no checkpoint-specific data.
LEAF_MARKER = "_cozy_fp8_storage_leaf"
BASE_ATTR = "_cozy_fp8_storage_base"

#: diffusers 0.39 ``hooks.layerwise_casting.DEFAULT_SKIP_MODULES_PATTERN``,
#: mirrored so a renamed upstream constant degrades to the same coverage
#: instead of silently widening it. Equivalence is asserted by the tape.
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
    """peft adapter tensor names, exactly as diffusers excludes them when the
    caller passed no patterns of its own (their footprint is negligible and
    casting them costs quality)."""
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
    """The leaf-path patterns excluded from fp8 storage — the mirror of
    ``ModelMixin.enable_layerwise_casting``'s pattern composition (defaults +
    the model's declared fp32/skip modules + peft adapter names)."""
    pats = set(_default_skip_patterns())
    for attr in ("_keep_in_fp32_modules", "_skip_layerwise_casting_patterns"):
        pats.update(str(p) for p in (getattr(model, attr, None) or ()))
    pats.update(_peft_skip_patterns())
    return tuple(sorted(pats))


def _supported_types() -> Tuple[type, ...]:
    """The layer set diffusers casts (its ``_GO_LC_SUPPORTED_PYTORCH_LAYERS``)
    — mirrored on import failure."""
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
    """One ``forward`` per layer KIND. Each is the stock torch forward with
    the weight (and bias) upcast at the use site — upstream's own
    weight-parameterized entry point where one exists (``_conv_forward``), so
    padding modes and output-padding math are never reimplemented."""
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
    """The restructuring kind for one leaf, ``""`` when it has none. Order
    matters: transposed convs are not ``nn.ConvNd`` subclasses, and embeddings
    carry no bias."""
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
    """The fp8-storage twin of ``base``: same class, same attributes, only
    ``forward`` replaced by the upcast-at-use-site form. Cached, so one class
    object per (base, kind) — the module type enters the cell's composition
    digest and must be stable within a process."""
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
    """The module's structure-relevant class: a restructured leaf answers with
    the plain class it was restructured from, everything else with its own
    type. Consumers that select modules by EXACT type (LoRA branch targeting)
    must ask this instead of ``type()``."""
    return getattr(type(module), BASE_ATTR, type(module))


def _leaf_paths(model: Any, *, patterns: Tuple[str, ...],
                supported: Tuple[type, ...]) -> List[Tuple[str, Any]]:
    """``(path, leaf)`` for every leaf diffusers' rule would cast: recursion
    pruned at any path prefix matching a skip pattern, stopping at the first
    supported layer — a transcription of ``_apply_layerwise_casting``."""
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
    """Restructure ``model``'s cast-eligible leaves into fp8 storage modules.

    Returns the covered leaf paths (sorted). Idempotent: already-restructured
    leaves are counted, not re-cast. Raises when the model cannot be
    restructured faithfully — a leaf kind this module does not implement, an
    instance ``forward`` already installed (a hook or a LoRA wrap would
    shadow the punned class), or diffusers cast hooks already armed. Refusing
    is the point: narrower coverage would be a silent numerics change.
    """
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
    """Return the freed bf16 blocks to the driver, not just to torch's caching
    allocator. This is a load-time fit-ladder correctness detail, not a
    micro-optimization: the restructure frees roughly half the denoiser's
    bytes, and the ladder reads DRIVER-level free VRAM
    (``memory.get_available_vram_gb`` -> ``torch.cuda.mem_get_info``), which
    still counts allocator-cached blocks as used. Without this the rung
    decision made moments later sees VRAM that is actually free as taken.
    Never initializes CUDA."""
    try:
        import torch

        if cuda_ready() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("fp8-storage: empty_cache after restructure failed",
                     exc_info=True)


def _to_storage_buffer(leaf: Any, name: str, storage_dtype: Any) -> None:
    """Move one tensor to fp8 storage as a persistent BUFFER, not a parameter.

    Buffer, for a load-bearing reason beyond taste (these are read-only at
    inference, like the w8a8 lane's weights): diffusers resolves
    ``ModelMixin.dtype`` from the first floating-point PARAMETER, with one
    special case — an armed layerwise-cast hook, whose ``compute_dtype`` it
    returns instead. Remove the hook and leave fp8 PARAMETERS behind and
    ``model.dtype`` starts reporting fp8, which breaks every denoiser that
    casts to ``self.dtype`` inside forward (``UNet2DModel.get_time_embed``
    does, measured: ``mat1 and mat2 must have the same dtype``). As buffers
    the fp8 weights are invisible to that walk, the remaining parameters are
    all at compute precision, and ``model.dtype`` answers correctly with no
    hook and no property override. state_dict keys are unchanged.
    """
    tensor = getattr(leaf, name, None)
    if tensor is None:
        return
    data = tensor.detach()
    if data.is_floating_point() and data.dtype != storage_dtype:
        data = data.to(storage_dtype)
    param = leaf._parameters.get(name)
    if param is not None and param.data.data_ptr() != data.data_ptr():
        # REBIND the outgoing Parameter onto the fp8 storage before dropping
        # it. Measured, and it is the whole ballgame for VRAM: replacing the
        # tensor OBJECT lets anything still holding the old Parameter —
        # accelerate's device hooks, `low_cpu_mem_usage` bookkeeping, any
        # earlier `list(model.parameters())` — keep the bf16 storage alive
        # forever, next to the fp8 copy. On an L4 that measured +50.3% (fp8
        # storage 7.35 GB vs plain bf16 4.89 GB: BOTH copies resident), and
        # +49.9% reproduced on CPU. The hook lane never had this failure mode
        # because `module.to(dtype=)` swaps storage INSIDE the same Parameter,
        # so every holder follows the cast. One assignment restores that
        # property: holders now share the single fp8 storage, and the bf16
        # storage drops at its last reference.
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
