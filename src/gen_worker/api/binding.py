"""Model bindings for ``@endpoint(models={...})``.

A binding names one remote model source. The slot name is NEVER a
constructor argument — it comes from the ``models={}`` dict key (or, for a
single ``model=`` binding, the ``setup()``/handler parameter name).

    HF("black-forest-labs/FLUX.1-dev", dtype="bf16")
    Hub("owner/repo", tag="canary", flavor="nf4")
    Civitai("123456", version="789012")
    ModelScope("circlestone-labs/Anima", files=("split_files/*.safetensors",))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Union


def _clean(s: object) -> str:
    return str(s or "").strip()


# Weight STORAGE precisions a binding may request (th#546 two-format policy).
# "fp8" = fp8-E4M3 weight storage with per-layer upcast to the compute dtype
# (diffusers layerwise casting) — the universal VRAM-fit mechanism; works on
# cards without fp8 units. Applied by the loading layer; also auto-applied
# when the snapshot itself stores fp8 (an `#fp8` flavor artifact).
STORAGE_DTYPES: tuple[str, ...] = ("fp8",)


def _clean_storage_dtype(v: object) -> str:
    q = _clean(v).lower()
    if q and q not in STORAGE_DTYPES:
        raise ValueError(
            f"unknown storage_dtype {q!r}; expected one of {STORAGE_DTYPES}"
        )
    return q


@dataclass(frozen=True)
class Hub:
    """Tensorhub-backed binding: ``Hub("owner/repo", tag=, flavor=, storage_dtype=)``."""

    PROVIDER: ClassVar[str] = "tensorhub"

    ref: str
    tag: str = "prod"
    flavor: str = ""
    storage_dtype: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "tag", _clean(self.tag) or "prod")
        object.__setattr__(self, "flavor", _clean(self.flavor))
        object.__setattr__(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))
        if not self.ref:
            raise ValueError(f"{type(self).__name__} requires a non-empty ref")

    @property
    def provider(self) -> str:
        return type(self).PROVIDER


@dataclass(frozen=True)
class HF:
    """HuggingFace-backed binding: ``HF(id, revision=, dtype=, subfolder=, files=, storage_dtype=)``.

    ``files`` are ``snapshot_download`` ``allow_patterns`` globs — set them to
    fetch only specific files (ComfyUI / split-checkpoint repos with no
    ``model_index.json``). ``dtype`` selects the torch COMPUTE precision at
    load time (``"bf16"`` / ``"fp16"`` / ``"fp32"``). ``storage_dtype="fp8"``
    keeps denoiser weights in fp8-E4M3 storage with per-layer upcast to the
    compute dtype (VRAM fit on any card; see ``STORAGE_DTYPES``).
    """

    PROVIDER: ClassVar[str] = "hf"

    ref: str
    revision: str = ""
    dtype: str = ""
    subfolder: str = ""
    files: tuple[str, ...] = ()
    storage_dtype: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "revision", _clean(self.revision))
        object.__setattr__(self, "dtype", _clean(self.dtype))
        object.__setattr__(self, "subfolder", _clean(self.subfolder))
        object.__setattr__(
            self, "files", tuple(_clean(p) for p in self.files if _clean(p))
        )
        object.__setattr__(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))
        if "/" not in self.ref:
            raise ValueError(f"HF(id=) must be 'owner/repo', got {self.ref!r}")

    @property
    def provider(self) -> str:
        return type(self).PROVIDER


@dataclass(frozen=True)
class Civitai:
    """Civitai-backed binding: ``Civitai(model_id, version=)``.

    ``ref`` is the Civitai MODEL id; pin a specific model-version with
    ``version=``.
    """

    PROVIDER: ClassVar[str] = "civitai"

    ref: str
    version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "version", _clean(self.version))
        if not self.ref:
            raise ValueError("Civitai requires a non-empty model id")

    @property
    def provider(self) -> str:
        return type(self).PROVIDER


@dataclass(frozen=True)
class ModelScope:
    """ModelScope-backed binding: ``ModelScope(id, revision=, files=)``.

    File-oriented (no diffusers-layout requirement) — the clean source for
    ComfyUI / DiffSynth split checkpoints.
    """

    PROVIDER: ClassVar[str] = "modelscope"

    ref: str
    revision: str = ""
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "revision", _clean(self.revision))
        object.__setattr__(
            self, "files", tuple(_clean(p) for p in self.files if _clean(p))
        )
        if "/" not in self.ref:
            raise ValueError(f"ModelScope(id=) must be 'owner/repo', got {self.ref!r}")

    @property
    def provider(self) -> str:
        return type(self).PROVIDER


Binding = Union[Hub, HF, Civitai, ModelScope]
BINDING_TYPES: tuple[type, ...] = (Hub, HF, Civitai, ModelScope)


def wire_ref(binding: Binding) -> str:
    """Canonical ref string for the wire / cache key.

    Hub refs carry ``:tag`` (non-prod) and ``#flavor`` suffixes; HF refs
    carry ``@revision``. Load-time metadata (dtype/subfolder/files/
    storage_dtype) never enters the ref.
    """
    out = binding.ref
    if isinstance(binding, Hub):
        if binding.tag and binding.tag != "prod":
            out = f"{out}:{binding.tag}"
        if binding.flavor:
            out = f"{out}#{binding.flavor}"
    elif isinstance(binding, HF) and binding.revision:
        out = f"{out}@{binding.revision}"
    return out


__all__ = [
    "Binding", "BINDING_TYPES", "Civitai", "HF", "Hub", "ModelScope",
    "STORAGE_DTYPES", "wire_ref",
]
