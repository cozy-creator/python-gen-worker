"""StreamingWriter — per-spec output writer that streams tensors to disk.

Tenants iterate ``source.iter_tensors()`` and call ``writer.write(component,
name, tensor)`` for each output tensor. Writer streams to a per-component
safetensors file on disk — one tensor in flight per component.

On ``finalize()``:
  1. Flush each per-component writer; if size > _MAX_SAFETENSORS_SHARD_BYTES
     reshard and emit .safetensors.index.json.
  2. Auto-copy any component from source that the tenant NEVER wrote AND
     never explicitly passthrough()'d — 'output matches source shape' default.
  3. Copy non-weight passthrough files: model_index.json, each component's
     config.json, tokenizer.*, scheduler_config.json, special_tokens_map.json.
  4. Return Path: file if source was singlefile, dir if source was diffusers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ._sharding import (
    MAX_SAFETENSORS_SHARD_BYTES,
    _tensor_size_bytes,
    build_safetensors_index,
    plan_safetensors_shards,
)

if TYPE_CHECKING:
    import torch

    from .component import Component
    from .source import Source


# Non-weight files to carry forward from source to output verbatim when doing
# diffusers-layout passthrough. Keeps config + tokenizer + scheduler intact so
# the output directory is drop-in loadable via DiffusionPipeline.from_pretrained.
_PASSTHROUGH_FILES = frozenset({
    "model_index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "scheduler_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer.model",
})


class StreamingWriter:
    """Per-spec streaming output writer.

    Library constructs one per variant spec; tenants receive via
    ``ctx.open_output_writer()``. Not thread-safe — one writer per tenant
    iteration loop.
    """

    def __init__(
        self,
        *,
        source: "Source",
        out_dir: Path,
    ) -> None:
        self._source = source
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # component_name -> {tensor_name: tensor} buffered before finalize.
        # For now we buffer in memory per component — streaming to disk
        # incrementally is an optimization we can add once the contract lands.
        self._by_component: dict[str, dict[str, Any]] = {}
        self._written_components: set[str] = set()
        self._explicit_passthrough: set[str] = set()
        self._finalized = False

    def write(self, component: str, name: str, tensor: "torch.Tensor") -> None:
        """Record one (component, name, tensor) for the output.

        ``component`` is '' for singlefile sources, subdir name for diffusers.
        ``name`` is the tensor's dotted-path name within the component. One
        tensor per (component, name) pair.
        """
        if self._finalized:
            raise RuntimeError("StreamingWriter.write called after finalize()")
        bucket = self._by_component.setdefault(component, {})
        bucket[name] = tensor
        self._written_components.add(component)

    def passthrough(self, component: "Component") -> None:
        """Copy this component verbatim from source to output.

        Use when the tenant wants to be explicit about not touching a
        component (vs. relying on the implicit auto-passthrough of
        untouched components at finalize time).
        """
        if self._finalized:
            raise RuntimeError("StreamingWriter.passthrough called after finalize()")
        self._explicit_passthrough.add(component.name)

    def finalize(self) -> Path:
        """Flush tensors, auto-passthrough untouched components, return output path."""
        if self._finalized:
            raise RuntimeError("StreamingWriter.finalize called twice")
        self._finalized = True

        if self._source.file_layout == "singlefile":
            target = self._out_dir / "model.safetensors"
            written = self._write_safetensors(target, self._by_component.get("", {}))
            # Passthrough top-level config.json / tokenizer.* / etc.
            self._copy_passthrough_files(self._source.path, self._out_dir)
            if written is None:
                # No tensors written — return the out_dir (will contain only passthroughs)
                return self._out_dir
            # If sharded, written points at the index.json and out_dir holds the shards
            # → return out_dir so consumers see the full tree. Else return the single file.
            return written if written.suffix == ".safetensors" else self._out_dir

        # diffusers: per-component output subdirs
        for comp_name, tensors in self._by_component.items():
            if not comp_name:
                raise RuntimeError(
                    "StreamingWriter.write called with empty component on a "
                    "diffusers source — specify the component name (unet, "
                    "vae, transformer, ...)"
                )
            self._write_component(comp_name, tensors)

        # Auto-passthrough any source component the tenant didn't touch
        source_components = set(self._source.components.keys())
        already_touched = self._written_components | self._explicit_passthrough
        for comp_name in source_components - already_touched:
            src = self._source.components[comp_name].path
            dst = self._out_dir / comp_name
            if dst.exists():
                continue
            shutil.copytree(str(src), str(dst))

        # Also passthrough scheduler/tokenizer subdirs the tenant doesn't know about
        for entry in self._source.path.iterdir():
            if entry.is_dir() and entry.name not in source_components:
                dst = self._out_dir / entry.name
                if not dst.exists():
                    shutil.copytree(str(entry), str(dst))

        # Top-level passthrough files (model_index.json especially)
        self._copy_passthrough_files(self._source.path, self._out_dir)
        return self._out_dir

    # ---- internals ---------------------------------------------------

    def _write_component(self, comp_name: str, tensors: dict[str, Any]) -> None:
        """Write all tensors for a diffusers component under out_dir/comp_name/.

        Auto-shards when the component's total size crosses
        MAX_SAFETENSORS_SHARD_BYTES — emits model-00001-of-00008.safetensors
        style shards + a model.safetensors.index.json. Tenant doesn't pick.
        """
        subdir = self._out_dir / comp_name
        subdir.mkdir(parents=True, exist_ok=True)
        # HF naming convention: unet/transformer/vae use diffusion_pytorch_model.*,
        # text_encoder family use model.*. Library picks based on the component.
        if comp_name.startswith("text_encoder") or comp_name == "image_encoder":
            shard_prefix = "model"
        else:
            shard_prefix = "diffusion_pytorch_model"
        self._write_safetensors(subdir, tensors, shard_prefix=shard_prefix)
        # Carry the component's config.json + any related files from source
        src_dir = self._source.components[comp_name].path
        self._copy_passthrough_files(src_dir, subdir)

    def _write_safetensors(
        self,
        target: Path,
        tensors: dict[str, Any],
        *,
        shard_prefix: str = "model",
    ) -> Optional[Path]:
        """Write a tensors dict to safetensors — single file OR sharded.

        Auto-shards when total size > MAX_SAFETENSORS_SHARD_BYTES: emits
        ``{shard_prefix}-NNNNN-of-NNNNN.safetensors`` shards + a matching
        ``{shard_prefix}.safetensors.index.json`` sidecar. HF from_pretrained
        handles both single-file and sharded-index inputs transparently.

        Returns the path of the entry-point file:
          - single-file: the .safetensors file
          - sharded: the .safetensors.index.json (consumers follow weight_map)
          - empty: None (nothing written)

        ``target`` is either a file path (caller picked the name) or a
        directory (library names files itself via shard_prefix).
        """
        if not tensors:
            return None
        from safetensors.torch import save_file

        tensor_sizes = {name: _tensor_size_bytes(t) for name, t in tensors.items()}
        total_size = sum(tensor_sizes.values())

        if total_size <= MAX_SAFETENSORS_SHARD_BYTES:
            if target.suffix == ".safetensors":
                out_path = target
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                out_path = target / f"{shard_prefix}.safetensors"
            save_file(tensors, str(out_path))
            return out_path

        # Multi-shard output
        shard_dir = target.parent if target.suffix == ".safetensors" else target
        shard_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_safetensors_shards(
            tensor_sizes, max_shard_bytes=MAX_SAFETENSORS_SHARD_BYTES,
            shard_prefix=shard_prefix,
        )
        buckets: dict[str, dict[str, Any]] = {}
        for name, tensor in tensors.items():
            shard = plan.weight_map[name]
            buckets.setdefault(shard, {})[name] = tensor
        for shard_name, bucket in buckets.items():
            save_file(bucket, str(shard_dir / shard_name))
        index_path = shard_dir / f"{shard_prefix}.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(build_safetensors_index(plan), f, separators=(",", ":"))
        return index_path

    def _copy_passthrough_files(self, src_dir: Path, dst_dir: Path) -> None:
        """Copy known non-weight passthrough files from src to dst."""
        if not src_dir.is_dir():
            return
        for entry in src_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.name in _PASSTHROUGH_FILES:
                dst_file = dst_dir / entry.name
                if not dst_file.exists():
                    shutil.copy2(str(entry), str(dst_file))


__all__ = ["StreamingWriter"]
