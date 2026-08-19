"""The SEAM that makes :mod:`gen_worker.models.gguf_torch` reachable — a
diffusers denoiser built from its CONFIG and filled with GGML block bytes.

pgw#1498's core module was correct and unreachable: nothing on the serving path
constructed it, because ``loading.load_gguf_pipeline`` handed the whole decode
to diffusers' ``GGUFQuantizationConfig``. That path works and is not ours: it
puns ``nn.Linear`` ONLY (a quantized conv or embedding lands as a byte-shaped
tensor in a dense parameter), it has no adapter story, no budget dial, and its
``GGUFParameter`` reports the DEQUANTIZED shape — so every residency walk in the
worker over-reports a 4-bit denoiser by the compression ratio, which is the one
number this lane exists to move.

**The bytes come from a SOURCE and nothing else about this module changes with
them.** Two exist:

* :class:`SingleFileGguf` — the community ``.gguf`` edge. A container names its
  tensors whatever its packer chose, so this source (and only this source)
  borrows diffusers' single-file key mapping to reach our key layout.
* :class:`NormalizedTensors` — the SERVED path (Paul's storage ruling,
  2026-08-19): per-tensor block bytes straight out of the CAS, already under our
  key layout, no container involved. It is one constructor away and is live the
  moment the ingest half lands (``gguf_torch.quantized_tensors_from_views``).

Both hand :func:`build_denoiser` the same ``{state-dict key: QuantizedTensor |
dense tensor}`` mapping, which is exactly what ``install_quantized_weights``
takes. The seam is the point: swapping the edge for the store is a change of
one constructor at the call site, not a rewrite of the loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol

from . import gguf_torch

logger = logging.getLogger(__name__)


class GgufTensorSource(Protocol):
    """Where a GGML-quantized denoiser's tensors come from.

    ``model`` is the config-built (weightless) denoiser: a source may consult
    its state-dict keys to decide whether a key translation is needed, and must
    return values keyed the way the MODEL names them.
    """

    def tensors(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class SingleFileGguf:
    """The community-ingest edge: one ``.gguf`` container on disk.

    OUR reader answers first (:func:`gen_worker.models.gguf_torch.read_gguf`).
    When the container already names its tensors the way the model does — the
    normalized case, and plenty of community packs — that is the whole job, and
    it is the better reader for it: it honours ``comfy.gguf.orig_shape.*``, so a
    conv or 5-D weight keeps its true shape instead of the ``[out, in*kh*kw]``
    that block geometry alone can derive.

    Only a container whose names are its PACKER's needs a translation, and that
    is the one thing we borrow: diffusers' ``checkpoint_mapping_fn``, applied to
    diffusers' ``GGUFParameter`` values (whose ``__torch_function__`` carries the
    GGML type through the splits and concatenations those converters do). We take
    the mapping and drop the quantizer — the values become
    :class:`~gen_worker.models.gguf_torch.QuantizedTensor` and are decoded by OUR
    kernels, on OUR leaves.
    """

    path: Path

    def tensors(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        wanted = set(model.state_dict())
        native = gguf_torch.read_gguf(self.path)
        if wanted <= set(native.tensors):
            return dict(native.tensors)
        logger.info(
            "gguf: %s names its tensors its packer's way (%d of %d model keys "
            "present); translating through the single-file key mapping",
            self.path.name, len(wanted & set(native.tensors)), len(wanted))
        return self._mapped(model, config)

    def _mapped(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        from diffusers.loaders import single_file_model as sfm
        from diffusers.loaders import single_file_utils as sfu

        read: Any = sfu.load_single_file_checkpoint
        mapping_class_of: Any = sfm._get_single_file_loadable_mapping_class
        loadable: Any = sfm.SINGLE_FILE_LOADABLE_CLASSES

        mapping_class = mapping_class_of(type(model))
        if mapping_class is None:
            raise ValueError(
                f"gguf: {type(model).__name__} has no single-file key mapping, "
                f"and {self.path.name} does not name its tensors the way the "
                "model does — nothing can say which tensor is which weight")
        checkpoint = loadable[mapping_class]["checkpoint_mapping_fn"](
            config=dict(config), checkpoint=read(str(self.path)))
        if not checkpoint:
            raise ValueError(
                f"gguf: the {mapping_class} key mapping found no weights in "
                f"{self.path.name}")
        return {key: _installable(value) for key, value in checkpoint.items()}


@dataclass(frozen=True)
class NormalizedTensors:
    """The SERVED path: ``tensorfs`` ``TensorView``s out of a ``LocalCAS``.

    Already under our key layout, already cut per tensor, block bytes verbatim —
    no container is read and none is composed. Nothing here needs the model,
    which is the whole difference from the edge source above.
    """

    views: Mapping[str, Any]
    pin_memory: bool = False

    def tensors(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        return gguf_torch.quantized_tensors_from_views(
            self.views, pin_memory=self.pin_memory)


def _installable(value: Any) -> Any:
    """One diffusers checkpoint value as ``install_quantized_weights`` takes it.

    A ``GGUFParameter`` carries the two facts a dense tensor cannot: the GGML
    type, and ``quant_shape`` — the shape the flat block stream expands to,
    recomputed by the subclass after every converter op. Everything else is
    already dense and passes through.
    """
    import torch

    quant_type = getattr(value, "quant_type", None)
    if quant_type is None:
        return value
    blocks = value.as_tensor().contiguous().view(torch.uint8).reshape(-1)
    spec = gguf_torch.QuantSpec(int(quant_type), torch.Size(tuple(value.quant_shape)))
    return gguf_torch.QuantizedTensor(blocks, spec)


def _conform_shapes(model: Any, tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Re-state every tensor's LOGICAL shape from the model, not the container.

    A GGML row is the flattened per-output row, so a quantized conv weight's
    block stream carries no memory of its kernel dims — the container records
    ``[out, in*kh*kw]`` and that is all a reader can derive from block geometry.
    (The reference packer works around this with a ``comfy.gguf.orig_shape.*``
    metadata key that no other producer is obliged to write, and that diffusers'
    reader ignores.) The model built from its own config knows the true shape
    for every key it owns; the container knows the bytes. Take each fact from
    the side that holds it.

    A count mismatch is not reshaped away — it means these bytes are not this
    weight, and it refuses by name.
    """
    reference = model.state_dict()
    out: Dict[str, Any] = {}
    for key, value in tensors.items():
        want = reference.get(key)
        if want is None:
            out[key] = value
            continue
        if isinstance(value, gguf_torch.QuantizedTensor):
            have = value.spec.shape
            if tuple(have) != tuple(want.shape):
                _refuse_element_mismatch(key, have, want.shape)
                value = gguf_torch.QuantizedTensor(
                    value.blocks,
                    gguf_torch.QuantSpec(value.spec.qtype, want.shape))
        elif tuple(value.shape) != tuple(want.shape):
            _refuse_element_mismatch(key, value.shape, want.shape)
            value = value.reshape(want.shape)
        out[key] = value
    return out


def _refuse_element_mismatch(key: str, have: Any, want: Any) -> None:
    def count(shape: Any) -> int:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total

    if count(have) != count(want):
        raise ValueError(
            f"gguf: {key!r} decodes to {count(have)} elements and the model "
            f"wants {count(want)} ({tuple(have)} vs {tuple(want)}) — these "
            "bytes are not this weight")


def build_denoiser(
    denoiser_cls: Any,
    config_dir: Any,
    source: GgufTensorSource,
    *,
    compute_dtype: Any,
    device: Any = None,
    dequant_dtype: Any = None,
) -> Any:
    """A denoiser built from its config alone, then filled with block bytes.

    CONFIG-ONLY construction is why this lane needs no lying tensor subclass:
    the reference wraps every tensor so a shape-sniffing model detector keeps
    working on a quantized state dict, and we never sniff — the config states
    the architecture and the block bytes are just storage.

    Parameters are allocated on ``meta`` and REPLACED by
    :func:`~gen_worker.models.gguf_torch.install_quantized_weights`; buffers are
    allocated for real, because a model's computed buffers (rotary tables,
    position grids) are never in a checkpoint and would otherwise be left on
    meta to die at the first forward. Any parameter the source did not cover is
    a LOUD refusal here — a meta weight is a pipeline that builds, loads,
    advertises and then fails the first request.
    """
    from accelerate import init_empty_weights

    config = denoiser_cls.load_config(str(config_dir))
    with init_empty_weights():
        model = denoiser_cls.from_config(config)

    tensors = _conform_shapes(model, source.tensors(model, config))
    expected = set(model.state_dict())
    unexpected = sorted(set(tensors) - expected)
    if unexpected:
        # Not fatal — a container routinely carries tensors a diffusers config
        # has no module for — but it is never silent.
        logger.info("gguf: %d tensor(s) name no module and are dropped "
                    "(first: %s)", len(unexpected), unexpected[0])
        tensors = {k: v for k, v in tensors.items() if k not in set(unexpected)}

    punned = gguf_torch.install_quantized_weights(
        model, tensors, compute_dtype=compute_dtype, device=device,
        dequant_dtype=dequant_dtype)
    _refuse_meta_parameters(model, denoiser_cls)
    model.eval()
    logger.info(
        "gguf: %s built from config with %d GGML leaves, %.2f GiB of block "
        "bytes resident", denoiser_cls.__name__, len(punned),
        gguf_torch.quantized_bytes(model) / float(1 << 30))
    return model


def _refuse_meta_parameters(model: Any, denoiser_cls: Any) -> None:
    left = sorted(
        name for name, p in model.named_parameters() if p.device.type == "meta")
    if not left:
        return
    raise ValueError(
        f"gguf: {denoiser_cls.__name__} has {len(left)} parameter(s) the "
        f"source never supplied and they are still on `meta` (first: "
        f"{left[0]!r}). A meta weight builds, loads and advertises, then dies "
        "on the first forward — refuse here instead.")


__all__ = [
    "GgufTensorSource",
    "NormalizedTensors",
    "SingleFileGguf",
    "build_denoiser",
]
