"""LoadedComponent — opaque payload yielded by Source.iter_hf_components* methods.

Each LoadedComponent has been resolved to one of four states:

  - ``quantized``    — already loaded into memory via HF's
                       ``from_pretrained(quantization_config=...)`` (so the
                       in-place quant has happened and the result is on
                       cuda / cpu / disk per ``device_map='auto'`` +
                       ``offload_folder``). ``save_to`` writes the component
                       subdir under ``out_dir`` via ``save_pretrained``.
  - ``bf16_cpu``     — loaded into CPU memory in ``torch_dtype``, NOT
                       quantized. Yielded by ``iter_hf_components_streaming``
                       for tenants that want to drive quantization themselves
                       (e.g. ``torchao.quantization.quantize_(component.module,
                       config, device='cuda')``, which streams one Linear at
                       a time to GPU rather than staging the full bf16 there).
                       After the tenant mutates the module in-place,
                       ``save_to`` writes it the same way as ``quantized``.
  - ``passthrough``  — pure file copy; kind for tokenizer / scheduler / vae /
                       feature_extractor / safety_checker etc. Tenant doesn't
                       quantize these and the library doesn't load them; the
                       library just remembers the source dir and ``save_to``
                       copies it verbatim into ``out_dir / <name>``.
  - ``root``         — top-level snapshot files (``model_index.json``,
                       ``README.md``, ``LICENSE.md``). One synthetic
                       LoadedComponent is yielded last with ``name='_root'``;
                       its ``save_to`` copies those files into ``out_dir``
                       directly (not into a subdir).

Tenants don't construct LoadedComponent — the library does. Tenants observe
``.name`` and ``.kind`` for logging / spec metadata, optionally reach into
``.module`` for streaming-quant or other in-place transforms, then call
``component.save_to(out_dir)`` and move on.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


def _maybe_unwrap_torchao_subclasses(module: Any) -> None:
    """If `module` contains torchao tensor subclasses (AffineQuantizedTensor,
    etc.), unwrap them so the underlying raw tensors become safetensors-
    serializable. No-op if torchao isn't installed or the module has no such
    subclasses to unwrap.

    torchao 0.15+ ships native safetensors hooks on its new per-config tensor
    classes (Float8Tensor, Int4TilePackedTo4dTensor, IntxUnpackedToInt8Tensor)
    so this unwrap is in principle unneeded for those. Kept as a defensive
    fallback because (1) older AffineQuantizedTensor paths still appear in
    some torchao Config combinations and (2) the call is a cheap idempotent
    no-op when the model already serializes natively. Remove only after
    verifying every Config we ship round-trips through save_pretrained
    + from_pretrained without the unwrap.

    Called by `LoadedComponent.save_to` for kind in ('quantized', 'bf16_cpu').
    """
    try:
        from torchao.utils import unwrap_tensor_subclass
    except Exception:
        return
    try:
        unwrap_tensor_subclass(module)
    except Exception:
        # If the module doesn't contain torchao subclasses (or torchao raised
        # for some other reason), swallow — the subsequent save_pretrained
        # will surface a clear error if the module still has incompatible
        # tensor types.
        pass


ComponentKind = Literal["quantized", "bf16_cpu", "passthrough", "root"]


@dataclass
class LoadedComponent:
    name: str
    kind: ComponentKind
    # Populated when kind in ("quantized", "bf16_cpu"): the loaded HF module.
    #   - "quantized": already carries quantization state (Linear8bitLt /
    #     Linear4bit / torchao AffineQuantizedTensor / modelopt fake-quant
    #     subclasses) from HF's from_pretrained(quantization_config=...).
    #   - "bf16_cpu": plain bf16 module on CPU; tenant will mutate in-place
    #     (e.g. torchao.quantize_(module, config, device='cuda')) before save.
    # save_to calls .save_pretrained on it.
    _module: Any = None
    # Populated when kind == "passthrough": absolute path to the source-side
    # subdir for this component. save_to copies it into out_dir / <name>.
    _source_path: Path | None = None
    # Populated when kind == "root": list of absolute paths to top-level
    # snapshot files. save_to copies each into out_dir directly (not nested).
    _root_files: list[Path] = field(default_factory=list)
    # Optional metadata about how the component was loaded — useful for
    # tenants that want to capture provenance into ProducedFlavor attributes.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> Any:
        """The loaded HF module for kind in ('quantized', 'bf16_cpu').

        Streaming-quant tenants use this to invoke library-specific
        in-place transforms before ``save_to``. Raises if accessed on a
        kind that doesn't carry a module (``passthrough`` / ``root``).
        """
        if self.kind not in ("quantized", "bf16_cpu"):
            raise AttributeError(
                f"LoadedComponent({self.name}): .module is only valid for "
                f"kind in ('quantized', 'bf16_cpu'), got kind={self.kind!r}"
            )
        if self._module is None:
            raise RuntimeError(
                f"LoadedComponent({self.name}): kind={self.kind} but no "
                "loaded module attached"
            )
        return self._module

    def save_to(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.kind in ("quantized", "bf16_cpu"):
            if self._module is None:
                raise RuntimeError(
                    f"LoadedComponent({self.name}): kind={self.kind} but no "
                    "loaded module attached"
                )
            target = out_dir / self.name
            target.mkdir(parents=True, exist_ok=True)
            # safetensors is required (Cozy policy: safetensors, never
            # pickle). torchao's AffineQuantizedTensor and similar
            # tensor subclasses aren't directly serializable by safetensors —
            # they wrap raw int_data + scale + zero_point in one Python
            # object. Call `unwrap_tensor_subclass` to flatten the subclasses
            # back into raw state-dict entries that safetensors can write.
            # No-op for modules without torchao subclasses, and a no-op if
            # torchao isn't installed. bf16_cpu modules that were quantized
            # in-place by the tenant (e.g. torchao streaming-quant) carry the
            # same subclasses, so the unwrap applies to them too.
            _maybe_unwrap_torchao_subclasses(self._module)
            self._module.save_pretrained(str(target), safe_serialization=True)
            return

        if self.kind == "passthrough":
            if self._source_path is None or not self._source_path.is_dir():
                raise RuntimeError(
                    f"LoadedComponent({self.name}): kind=passthrough but no "
                    "source path attached"
                )
            target = out_dir / self.name
            # dirs_exist_ok=True so re-runs (or merging into a partially
            # written out_dir) don't raise.
            shutil.copytree(self._source_path, target, dirs_exist_ok=True)
            return

        if self.kind == "root":
            for f in self._root_files:
                if f.is_file():
                    shutil.copy2(f, out_dir / f.name)
            return

        raise RuntimeError(f"LoadedComponent({self.name}): unknown kind={self.kind!r}")


__all__ = ["LoadedComponent", "ComponentKind"]
