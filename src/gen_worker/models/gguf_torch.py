"""GGML-quantized weights served by the ORDINARY torch path.

Today a GGUF checkpoint can only be served by a separate llama.cpp engine on a
torch-free image (``serving/gguf_fit.py``) — llama-shaped LLMs only, and it gets
none of the torch stack: no diffusers pipeline, no LoRA, no ladder, no VAE
tiling, no compiled graphs. This module makes GGML a weight STORAGE format
inside the same torch serving path as bf16, so a 4-bit flux/wan/sdxl runs on a
small card with everything else intact.

**The weights stay quantized — as far as the lease says they must.** Each leaf
holds its own block bytes on the device and, once per forward, decodes them
(:mod:`gen_worker.models.gguf_dequant`), matmuls at compute precision, and drops
the dense copy. Residency is the quantized size; peak transient is ONE leaf's
dense weight; the decode is memory-bound and costs ~10-20% of step time.

That is one end of a DIAL, not the only setting (Paul, 2026-08-19). A worker
handed surplus memory should use it: :func:`dequant_ahead` decodes as many
weights ONCE at load as the surplus pays for, largest first, because paying a
per-forward decode every step for the life of the endpoint is worse than paying
it once. ``surplus_bytes=0`` is the constrained tier, ``math.inf`` the surplus
tier, anything between graduates per layer. Nothing is ever RE-quantized — the
dial only moves a weight from "decoded every step" to "decoded once", and the
ladder still SELECTS artifacts rather than manufacturing them (``rung.py``).
LoRA semantics are identical on both sides of the dial.

Three deliberate divergences from the reference (city96's ComfyUI-GGUF, which
these kernels and the attach-never-merge LoRA are ported from):

1. **No lying tensor subclass.** The reference wraps every tensor in a
   ``torch.Tensor`` subclass whose ``.shape`` reports the DEQUANTIZED shape, so
   ComfyUI's shape-sniffing model detection keeps working on a quantized state
   dict. We build models from a config-only snapshot, never by sniffing shapes,
   so we pay none of that — and a lying ``.shape`` would corrupt exactly the
   thing this lane exists for: every VRAM/residency walk in the worker reads
   buffer shapes, and would over-report a 4-bit denoiser by ~4x. The qtype and
   the logical shape live on the LEAF (:class:`QuantSpec`), where they are also
   free of the subclass/``torch.compile`` interaction the reference has to
   version-gate. Nothing traces a subclass because there is no subclass.
2. **Blocks are uint8 buffers, not parameters.** ``nn.Module._apply`` only casts
   FLOATING-POINT tensors, so ``model.to(dtype=bf16)`` silently skips them —
   the fp8-storage lane's standing "never ``.to(dtype=...)`` a restructured
   module" hazard simply does not exist here. Device moves work normally.
3. **mmap release is a copy at install**, not the reference's
   ``.to(load_device).to(offload_device)`` bounce. A ``.gguf`` read through
   ``GGUFReader`` hands out numpy views of an mmap; leaving them installed means
   every step faults through the page cache and the tensors can never be pinned.
   One forced copy at install time is the same fix stated directly.

The restructure itself is the repo's CLASS PUN (see
:mod:`gen_worker.models.fp8_storage`): the leaf keeps its identity, FQNs and
attributes, and only ``forward`` changes, so ``isinstance`` stays true for the
offload rung, LoRA branch targeting and dtype introspection.

**LoRA is attached, never merged.** Merging into a 4-bit grid would requantize
and lose the adapter; attaching costs one rank-r product per forward and is
LOSSLESS, with instant unpatch. See :func:`attach_lora` for why this does NOT
trip the refuse-adapters-on-a-quantized-grid rule.

Where the bytes come from is not this module's business. :func:`read_gguf` is
the EDGE reader for a community ``.gguf`` file; the normalized path is per-tensor
block bytes out of the CAS, which arrive at :func:`install_quantized_weights` as
the same :class:`QuantizedTensor` values.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import gguf_dequant
from .fp8_storage import BASE_ATTR, structural_base

logger = logging.getLogger(__name__)

#: Class attribute marking a punned leaf. Structural marker, never a name check.
LEAF_MARKER = "_cozy_gguf_leaf"

#: ``BASE_ATTR`` and :func:`structural_base` are IMPORTED, not redefined. "the
#: plain class this leaf was punned from" is ONE concept with ONE spelling, so
#: every module walker that already resolves an fp8-storage leaf
#: (``w8a8_lora.branch_modules`` above all) resolves a GGML leaf by construction
#: rather than by a second lookup somebody has to remember to add. Two attribute
#: names left a GGML Linear invisible to LoRA branch targeting while every walk
#: still looked correct — pgw#1498's wiring lane, red arm on record.

#: Per-leaf attribute: ``{"weight": QuantSpec, ...}`` for the tensors that are
#: block bytes. A tensor absent from this map is dense and used as-is.
SPEC_ATTR = "gguf_specs"

#: Per-leaf attribute: ``{"weight": (LoraPatch, ...)}`` applied post-dequant.
ADAPTER_ATTR = "gguf_adapters"

#: Per-leaf attribute: the specs of tensors :func:`dequant_ahead` already
#: decoded. They are dense now, but the lane still owns them — an adapter may
#: still attach, and residency reporting must not lose track of where the bytes
#: came from.
MATERIALIZED_ATTR = "gguf_materialized"

#: (plain class, kind) -> punned class. One class object per pair: the module
#: TYPE enters the compiled graph's composition digest and must be stable for
#: the life of the process.
_PUNNED: Dict[Tuple[Any, str], type] = {}


@dataclass(frozen=True)
class QuantSpec:
    """What a leaf needs to decode one tensor: the GGML type and the shape the
    flat block stream expands to."""

    qtype: int
    shape: Any

    def __post_init__(self) -> None:
        if not gguf_dequant.is_supported(self.qtype):
            raise NotImplementedError(
                f"gguf-torch: {gguf_dequant.qtype_name(self.qtype)} has no "
                "vectorized decode")


@dataclass(frozen=True)
class QuantizedTensor:
    """Block bytes plus what they decode to — the unit that crosses into this
    module, whether it came from a ``.gguf`` file or from the CAS."""

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
    """One low-rank adapter branch, applied to the DEQUANTIZED weight.

    ``down`` is ``[rank, in...]`` and ``up`` is ``[out, rank...]``; conv branches
    carry the extra kernel dims, which flatten into the rank product. ``scale``
    is the caller's strength already folded with ``alpha / rank``.
    """

    down: Any
    up: Any
    scale: float = 1.0

    def delta(self, like: Any) -> Any:
        up = self.up.to(device=like.device, dtype=like.dtype)
        down = self.down.to(device=like.device, dtype=like.dtype)
        product = up.flatten(1) @ down.flatten(1)
        return product.reshape(like.shape) * self.scale


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _materialize(leaf: Any, name: str, dtype: Any) -> Any:
    """One tensor of ``leaf``, dense and at ``dtype``, for THIS forward.

    Three cases, one expression. Block bytes -> decode. Already dense, because
    the packer left it alone (norms, biases) or because :func:`dequant_ahead`
    decoded it at load -> a cast, which is free when the dtype already matches.
    Then every attached adapter, in BOTH cases: LoRA semantics must not depend
    on which side of the budget dial a leaf landed on.
    """
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
    """The dtype this op must run in.

    THE ACTIVATION DECIDES, not the load-time declaration. A weight that does
    not match its input is not a precision choice, it is a `RuntimeError` —
    caught by running the stack at bf16 with fp32 activations, which is exactly
    the mixed state a partially-cast pipeline is in. ``compute_dtype`` answers
    only where the input carries no float dtype of its own: an Embedding is
    indexed by int64, so its output precision cannot be read off its input.
    ``dequant_dtype`` remains the separate knob for the DECODE's own precision.
    """
    import torch

    if x is not None and x.is_floating_point():
        return x.dtype
    return getattr(leaf, "compute_dtype", None) or torch.get_default_dtype()


@functools.lru_cache(maxsize=1)
def _forwards() -> Dict[str, Any]:
    """One ``forward`` per layer KIND — the stock torch forward with the weight
    decoded at the use site. Upstream's own weight-parameterized entry point is
    used where one exists (``_conv_forward``), so padding modes and
    output-padding math are never reimplemented."""
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
    """The pun kind for one leaf, ``""`` when it has none. Order matters:
    transposed convs are not ``nn.ConvNd`` subclasses."""
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
    """The GGML-storage twin of ``base``: same class, same attributes, only
    ``forward`` replaced by the decode-at-use-site form. Cached, so one class
    object per (base, kind)."""
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


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install_quantized_weights(
    model: Any,
    tensors: Mapping[str, Any],
    *,
    compute_dtype: Any,
    device: Any = None,
    dequant_dtype: Any = None,
) -> List[str]:
    """Install ``tensors`` into ``model``, punning every leaf that gets blocks.

    ``tensors`` maps a state-dict key (``"...to_q.weight"``) to either a
    :class:`QuantizedTensor` or a plain dense tensor. Dense tensors land
    unchanged at ``compute_dtype``; quantized ones land as uint8 buffers on a
    punned leaf. Returns the punned leaf paths, sorted.

    Every tensor is COPIED onto ``device`` on the way in. That is not an
    optimization: a ``.gguf`` read hands out numpy views of an mmap, and an
    installed mmap view faults through the page cache on every step and can
    never be pinned. One copy here is the reference's mmap-release bounce,
    stated directly.
    """
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
    """``tensor`` on ``device``, guaranteed NOT to alias its source storage.

    ``copy=True`` is the whole point and is never redundant: a ``.gguf`` read
    hands out numpy views of an mmap, and ``.to(same_device)`` would keep the
    view. Installed mmap views fault through the page cache on every step and
    can never be pinned — the reference works around this after the fact by
    bouncing each module ``.to(load_device).to(offload_device)``; forcing the
    copy once, here, is the same fix said directly.
    """
    if device is None:
        return tensor.detach().clone()
    return tensor.detach().to(device=device, copy=True)


def _install_parameter(leaf: Any, name: str, tensor: Any) -> None:
    """Install a DENSE tensor, keeping it a Parameter.

    Load-bearing, and the opposite call from :func:`_install_buffer` on
    purpose. diffusers resolves ``ModelMixin.dtype`` by walking for the first
    floating-point PARAMETER, and a denoiser that casts to ``self.dtype``
    inside forward breaks when that walk finds nothing. Installing a whole
    checkpoint through here is exactly the case that would leave a model with
    no floating-point parameters at all — every dense tensor demoted to a
    buffer — so dense tensors stay parameters and only the block bytes become
    buffers. The blocks need no such care: they are uint8 and the walk skips
    them either way.
    """
    import torch

    param = leaf._parameters.get(name)
    # A `meta` placeholder holds no storage to point anywhere, and `set_data`
    # refuses it outright ("incompatible tensor type"). Config-only construction
    # (`gguf_diffusers.build_denoiser`) makes that the NORMAL case rather than an
    # edge: every parameter arrives as a meta placeholder and is replaced here.
    if param is not None and param.device.type != "meta":
        param.data = tensor
        param.requires_grad_(False)
        return
    leaf._parameters.pop(name, None)
    leaf._buffers.pop(name, None)
    leaf.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def _install_buffer(leaf: Any, name: str, tensor: Any) -> None:
    """Replace ``leaf.<name>`` with ``tensor`` as a persistent BUFFER.

    Buffer, not Parameter, because block bytes are storage rather than
    weights — nothing trains them, nothing should cast them, and they must not
    appear in a ``parameters()`` walk that expects weights. state_dict keys are
    unchanged.
    """
    param = leaf._parameters.get(name)
    if param is not None and param.device.type == "meta":
        # Nothing holds a meta placeholder worth following, and `set_data`
        # refuses it. Drop it; the buffer registration below is the whole move.
        param = None
    if param is not None and param.data.data_ptr() != tensor.data_ptr():
        # Rebind the outgoing Parameter onto the new storage before dropping it,
        # so anything still holding it (accelerate device hooks, an earlier
        # `list(model.parameters())`) follows instead of pinning the original.
        # `requires_grad` has to go first: torch refuses to point an autograd
        # leaf at integer storage, and block bytes are uint8. Nothing here
        # trains, so dropping it is the truth, not a workaround.
        param.requires_grad_(False)
        param.data = tensor
    leaf._parameters.pop(name, None)
    leaf._buffers.pop(name, None)
    leaf.register_buffer(name, tensor, persistent=True)


# ---------------------------------------------------------------------------
# the budget dial
# ---------------------------------------------------------------------------

def dequant_ahead(model: Any, *, surplus_bytes: float, dtype: Any) -> List[str]:
    """Decode as many weights ONCE at load as ``surplus_bytes`` pays for.

    Paul, 2026-08-19: quantized-resident and fully-dequantized are the two
    ENDPOINTS OF ONE DIAL, set from the residency lease at load time. A worker
    handed surplus memory should use it — a single dequant pass beats paying a
    per-forward decode on every step for the life of the endpoint — and a
    constrained worker pays per forward. Nothing is ever re-quantized: this only
    ever moves a weight from "decoded every step" to "decoded once".

    ``surplus_bytes`` is the lease's headroom over the quantized-resident plan,
    so ``0`` is the constrained tier, ``math.inf`` the surplus tier, and
    anything between graduates per layer. The price of one weight is the DELTA
    (dense minus its blocks), because its blocks are dropped.

    LARGEST FIRST, and the order is not arbitrary. Per-step decode saved is
    proportional to bytes either way, so that alone would not pick an order —
    but the transient headroom the fit plan must reserve is set by the LARGEST
    still-quantized weight, and largest-first is the only order that shrinks it.
    Spending continues past a weight too big for what is left, so a small
    remainder is not stranded.

    Returns the materialized ``"<path>.<tensor>"`` keys, largest first.
    """
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
    """Decode one weight once and drop its blocks. The leaf STAYS punned, so
    adapters keep applying exactly as they did on the quantized side."""
    specs = getattr(leaf, SPEC_ATTR, {})
    spec = specs.get(name)
    if spec is None:
        return
    dense = gguf_dequant.dequantize(getattr(leaf, name), spec.qtype, spec.shape,
                                    dtype=dtype)
    del specs[name]
    getattr(leaf, MATERIALIZED_ATTR)[name] = spec
    # A Parameter, not a buffer: past the dial this leaf IS an ordinary dense
    # layer and should look like one to every walk in the worker.
    _install_parameter(leaf, name, dense.contiguous())


def peak_transient_bytes(model: Any, *, dtype: Any) -> int:
    """The largest weight still decoded per forward — the headroom a fit plan
    must reserve on top of the resident bytes. Falls as the dial turns up."""
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
    """Resident bytes of block storage — the number the small-card ladder cares
    about. Honest precisely because ``.shape`` was not made to lie."""
    total = 0
    for leaf in gguf_leaves(model).values():
        for name in getattr(leaf, SPEC_ATTR, {}):
            tensor = getattr(leaf, name, None)
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

def attach_lora(leaf: Any, patches: Iterable[LoraPatch], *,
                name: str = "weight") -> None:
    """Attach adapter branches to one leaf, applied AFTER the decode.

    Never merged. Merging a LoRA into a 4-bit grid means dequantize, add,
    requantize — which rounds the delta away on exactly the small quants this
    lane exists to serve, and cannot be undone. Attaching costs one rank-r
    product per forward, is numerically identical to applying the adapter to the
    bf16 weight, and unpatching is :func:`detach_lora`.

    RECONCILED with the refuse-adapters-on-a-quantized-grid rule
    (``models/adapter_fidelity.py``): that rule refuses to WRITE an adapter INTO
    a quantized grid, because the grid's scales were fit to the base weights and
    a merged delta is silently rounded. Nothing here writes to the grid. The
    block bytes are read-only and byte-identical before and after; the delta is
    added to a dense fp copy that lives for one op. The refusal's premise —
    "the merge damages the quantization" — has no referent when there is no
    merge, so it does not apply and must not be extended here.
    """
    if not is_gguf_leaf(leaf):
        raise ValueError("gguf-torch: attach_lora expects a punned leaf")
    if name not in getattr(leaf, SPEC_ATTR, {}) and \
            name not in getattr(leaf, MATERIALIZED_ATTR, {}):
        raise ValueError(
            f"gguf-torch: {name!r} on this leaf never held block bytes — "
            "apply the adapter through the ordinary LoRA path")
    getattr(leaf, ADAPTER_ATTR)[name] = tuple(patches)


def detach_lora(model: Any) -> int:
    """Drop every attached adapter. Instant, because nothing was ever merged."""
    dropped = 0
    for leaf in gguf_leaves(model).values():
        adapters = getattr(leaf, ADAPTER_ATTR, None)
        if adapters:
            dropped += len(adapters)
            adapters.clear()
    return dropped


# ---------------------------------------------------------------------------
# the store: per-tensor block bytes, no container
# ---------------------------------------------------------------------------

def quantized_tensors_from_views(views: Mapping[str, Any], *,
                                 pin_memory: bool = False) -> Dict[str, Any]:
    """The SERVED path: :class:`tensorfs.tensors.TensorView` -> installable values.

    No ``.gguf`` file is involved and none is composed. tensorfs' ``gguf-v1``
    planner already cuts a container into one CAS region per tensor, and a
    ``TensorView`` already carries the two facts a quantized parameter needs
    that safetensors has no equivalent for: the GGML type (``dtype``, e.g.
    ``"Q4_K"``) and the block geometry (``block``, e.g. ``(256, 144)``). The
    block BYTES cross unchanged — dequantizing at ingest would forfeit the 4x
    that is the entire reason this lane exists.

    ``shape`` is translated: GGUF stores ``ne`` fastest-varying-first, torch is
    row-major, so the dims reverse. That reversal is the only transformation
    applied to anything here.
    """
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


# ---------------------------------------------------------------------------
# the .gguf edge
# ---------------------------------------------------------------------------

@dataclass
class GgufRead:
    """What one ``.gguf`` file yields: tensors keyed by their name in the file,
    plus the container's own metadata."""

    tensors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    architecture: str = ""


def read_gguf(path: Any, *, prefix: str = "") -> GgufRead:
    """Read a ``.gguf`` container into :class:`QuantizedTensor` values.

    This is an EDGE reader — community ingest and local development. The served
    path takes block bytes per-tensor out of the CAS and never sees a container.

    GGML stores a quantized tensor as flat bytes, so the logical shape is
    metadata: the reversed ``tensor.shape`` dims, unless the packer recorded a
    true original shape under ``comfy.gguf.orig_shape.<name>`` (5-D tensors get
    split by every packer, and conv weights lose their trailing 1s).
    """
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
