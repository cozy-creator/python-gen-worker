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
from typing import Any, ClassVar, Union


def _clean(s: object) -> str:
    return str(s or "").strip()


# Weight STORAGE precisions a binding may request (th#546 two-format policy).
# "fp8" = fp8-E4M3 weight storage with per-layer upcast to the compute dtype
# (diffusers layerwise casting) — the universal VRAM-fit mechanism; works on
# cards without fp8 units. Applied by the loading layer; also auto-applied
# when the snapshot itself stores fp8 (an `#fp8` flavor artifact).
# "fp8+te" additionally casts the pipeline's text encoders via the
# transformers-aware path (linear weights fp8; embeddings/norms/tied weights
# stay at compute dtype — component fit-ladder rung 2, gw#460).
STORAGE_DTYPES: tuple[str, ...] = ("fp8", "fp8+te")


def _clean_storage_dtype(v: object) -> str:
    q = _clean(v).lower()
    if q and q not in STORAGE_DTYPES:
        raise ValueError(
            f"unknown storage_dtype {q!r}; expected one of {STORAGE_DTYPES}"
        )
    return q


@dataclass(frozen=True)
class Hub:
    """Tensorhub-backed binding: ``Hub("owner/repo", tag=, flavor=, storage_dtype=, allow_lora=)``.

    ``allow_lora=True`` opts the slot into per-request LoRA overlays
    (``_models.<slot>.loras``, ie#358) — the endpoint must also declare
    ``Compile(family=...)`` so the hub's architecture gate can police
    adapter targets.
    """

    PROVIDER: ClassVar[str] = "tensorhub"

    ref: str
    tag: str = "latest"
    flavor: str = ""
    storage_dtype: str = ""
    allow_lora: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "tag", _clean(self.tag) or "latest")
        object.__setattr__(self, "flavor", _clean(self.flavor))
        object.__setattr__(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))
        object.__setattr__(self, "allow_lora", bool(self.allow_lora))
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
    allow_lora: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref))
        object.__setattr__(self, "revision", _clean(self.revision))
        object.__setattr__(self, "dtype", _clean(self.dtype))
        object.__setattr__(self, "subfolder", _clean(self.subfolder))
        object.__setattr__(
            self, "files", tuple(_clean(p) for p in self.files if _clean(p))
        )
        object.__setattr__(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))
        object.__setattr__(self, "allow_lora", bool(self.allow_lora))
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
    """Normal-form ref string for the wire / cache key — delegates to the ONE
    grammar module (``gen_worker.models.refs``, gw#492).

    Hub refs carry ``:tag`` (elided when ``latest``, the grammar default) and
    ``#flavor`` suffixes; HF refs carry ``@revision``. Load-time metadata
    (dtype/subfolder/files/storage_dtype) never enters the ref.
    """
    from ..models.refs import HuggingFaceRef, fold_ref

    if isinstance(binding, Hub):
        # The default tag never overrides one embedded in ``ref``.
        tag = binding.tag if binding.tag != "latest" else ""
        return fold_ref(binding.ref, tag=tag, flavor=binding.flavor)
    if isinstance(binding, HF):
        return HuggingFaceRef(binding.ref, binding.revision or None).canonical()
    return binding.ref


def rebind_pick(
    binding: Binding,
    *,
    resolved_ref: str = "",
    flavor: "str | None" = None,
    cast: str = "",
) -> Binding:
    """THE fold of a precision pick into a binding (gw#494) — the hub
    HelloAck path (``resolved_ref`` + ``cast``) and the local-ladder path
    (``flavor`` + ``cast``) share this one implementation.

    ``resolved_ref`` is authoritative when given: its flavor (possibly none)
    replaces the binding's. ``flavor=None`` leaves the binding's flavor
    untouched. Raises ``ValueError`` when the pick cannot round-trip through
    ``wire_ref`` — a pick the rebound binding cannot re-mint would split the
    slot into two residency identities (the th#736 mechanic).
    """
    import dataclasses

    from ..models.refs import fold_ref, normalize_model_ref, parse_model_ref

    if resolved_ref:
        parsed = parse_model_ref(resolved_ref)
        if parsed.tensorhub is None:
            raise ValueError(f"resolution {resolved_ref!r} is not a tensorhub ref")
        flavor = parsed.tensorhub.flavor or ""
    rebound: Any = binding
    try:
        if flavor is not None and flavor != getattr(binding, "flavor", ""):
            rebound = dataclasses.replace(rebound, flavor=flavor)
        if cast:
            rebound = dataclasses.replace(rebound, storage_dtype=cast)
    except TypeError as exc:
        # e.g. folding a flavor into an HF/Civitai binding that has no
        # flavor axis — same rejection channel as a round-trip failure.
        raise ValueError(f"pick does not fit binding {binding!r}: {exc}") from exc
    if resolved_ref:
        expected = normalize_model_ref(resolved_ref)
    elif flavor:
        expected = fold_ref(wire_ref(binding), flavor=flavor)
    else:
        expected = None
    if expected is not None and wire_ref(rebound) != expected:
        raise ValueError(
            f"pick {expected!r} does not round-trip through the binding "
            f"(got {wire_ref(rebound)!r})"
        )
    return rebound


__all__ = [
    "Binding", "BINDING_TYPES", "Civitai", "HF", "Hub", "ModelScope",
    "STORAGE_DTYPES", "rebind_pick", "wire_ref",
]
