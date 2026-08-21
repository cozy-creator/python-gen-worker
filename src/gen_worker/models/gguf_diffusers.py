"""The SEAM that makes :mod:`gen_worker.models.gguf_torch` reachable — a diffusers denoiser built from its CONFIG and filled with GGML block bytes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol

from . import gguf_torch

logger = logging.getLogger(__name__)


class GgufTensorSource(Protocol):
    """Where a GGML-quantized denoiser's tensors come from."""

    def tensors(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class SingleFileGguf:
    """The community-ingest edge: one ``.gguf`` container on disk."""

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
    """The SERVED path: ``tensorfs`` ``TensorView``s out of a ``LocalCAS``."""

    views: Mapping[str, Any]
    pin_memory: bool = False

    def tensors(self, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
        return gguf_torch.quantized_tensors_from_views(
            self.views, pin_memory=self.pin_memory)


def _installable(value: Any) -> Any:
    import torch

    quant_type = getattr(value, "quant_type", None)
    if quant_type is None:
        return value
    blocks = value.as_tensor().contiguous().view(torch.uint8).reshape(-1)
    spec = gguf_torch.QuantSpec(int(quant_type), torch.Size(tuple(value.quant_shape)))
    return gguf_torch.QuantizedTensor(blocks, spec)


def _conform_shapes(model: Any, tensors: Dict[str, Any]) -> Dict[str, Any]:
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
    """A denoiser built from its config alone, then filled with block bytes."""
    from accelerate import init_empty_weights

    config = denoiser_cls.load_config(str(config_dir))
    with init_empty_weights():
        model = denoiser_cls.from_config(config)

    tensors = _conform_shapes(model, source.tensors(model, config))
    expected = set(model.state_dict())
    unexpected = sorted(set(tensors) - expected)
    if unexpected:
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
