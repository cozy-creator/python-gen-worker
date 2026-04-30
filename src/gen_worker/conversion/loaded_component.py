"""LoadedComponent — opaque payload yielded by ``Source.iter_hf_components``.

Each LoadedComponent has been resolved to one of three states:

  - ``quantized``    — already loaded into memory via HF's
                       ``from_pretrained(quantization_config=...)`` (so the
                       in-place quant has happened and the result is on
                       cuda / cpu / disk per ``device_map='auto'`` +
                       ``offload_folder``). ``save_to`` writes the component
                       subdir under ``out_dir`` via ``save_pretrained``.
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
``.name`` and ``.kind`` for logging / spec metadata, then call
``component.save_to(out_dir)`` and move on.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ComponentKind = Literal["quantized", "passthrough", "root"]


@dataclass
class LoadedComponent:
    name: str
    kind: ComponentKind
    # Populated when kind == "quantized": the loaded HF module/pipeline that
    # already carries quantization state (Linear8bitLt / Linear4bit / torchao
    # AffineQuantizedTensor / modelopt fake-quant subclasses, depending on the
    # tenant's quant_config). save_to calls .save_pretrained on it.
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

    def save_to(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.kind == "quantized":
            if self._module is None:
                raise RuntimeError(
                    f"LoadedComponent({self.name}): kind=quantized but no "
                    "loaded module attached"
                )
            target = out_dir / self.name
            target.mkdir(parents=True, exist_ok=True)
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
