"""Model bindings for ``@endpoint(models={...})``.

A binding names one remote model source. The slot name is NEVER a
constructor argument — it comes from the ``models={}`` dict key (or, for a
single ``model=`` binding, the ``setup()``/handler parameter name).

    HF("black-forest-labs/FLUX.1-dev", dtype="bf16")
    Hub("owner/repo", tag="canary", flavor="nf4")
    Civitai("123456", version="789012")
    ModelScope("circlestone-labs/Anima", files=("split_files/*.safetensors",))

``ModelRef`` (pgw#511) is the ONE structured type: ``source`` is an explicit
field (never inferred from which factory built the value), ``path`` is the
bare repo/model id. ``Hub``/``HF``/``Civitai``/``ModelScope`` are thin
FACTORY FUNCTIONS over ``ModelRef`` — sugar, not a second type — that pin
``source`` and keep each registry's historical constructor signature and
validation.
"""

from __future__ import annotations

from typing import Any, Literal

import msgspec

ModelSource = Literal["tensorhub", "huggingface", "civitai", "modelscope"]


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


class ModelRef(msgspec.Struct, frozen=True):
    """ONE structured model reference (pgw#511): ``source`` is explicit, never
    inferred from shape or which factory built the value.

    Carries the union of every registry's per-source fields (tensorhub:
    ``tag``/``flavor``/``allow_lora``; huggingface: ``revision``/``dtype``/
    ``subfolder``/``files``; civitai: ``version``; modelscope: ``revision``/
    ``files``). ``storage_dtype`` is shared by tensorhub/huggingface. Build
    one via ``Hub``/``HF``/``Civitai``/``ModelScope`` rather than the raw
    constructor — they pin ``source`` and apply the per-registry validation
    below (mirrored in ``__post_init__`` so it holds for direct construction
    too, e.g. ``msgspec.structs.replace``).

    ``.provider`` and ``.ref`` are back-compat aliases for call sites that
    predate the explicit ``source``/``path`` fields: ``.provider`` returns
    the OLDER, narrower vocabulary (``"hf"`` not ``"huggingface"``; no
    ``"modelscope"``) that tensorhub's build-manifest ``bindings.<slot>.provider``
    column is DB-CHECK-constrained to — do not repoint manifest/download code
    at ``.source`` without also widening that constraint.
    """

    source: ModelSource
    path: str
    tag: str = ""
    flavor: str = ""
    revision: str = ""
    subfolder: str = ""
    dtype: str = ""
    storage_dtype: str = ""
    version: str = ""
    files: tuple[str, ...] = ()
    allow_lora: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _clean(self.path))
        object.__setattr__(self, "tag", _clean(self.tag))
        object.__setattr__(self, "flavor", _clean(self.flavor))
        object.__setattr__(self, "revision", _clean(self.revision))
        object.__setattr__(self, "subfolder", _clean(self.subfolder))
        object.__setattr__(self, "dtype", _clean(self.dtype))
        object.__setattr__(self, "version", _clean(self.version))
        object.__setattr__(
            self, "files", tuple(_clean(p) for p in self.files if _clean(p))
        )
        object.__setattr__(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))
        object.__setattr__(self, "allow_lora", bool(self.allow_lora))

        if self.source == "tensorhub":
            if not self.tag:
                object.__setattr__(self, "tag", "latest")
            if not self.path:
                raise ValueError("Hub requires a non-empty ref")
        elif self.source == "huggingface":
            if "/" not in self.path:
                raise ValueError(f"HF(id=) must be 'owner/repo', got {self.path!r}")
        elif self.source == "civitai":
            if not self.path:
                raise ValueError("Civitai requires a non-empty model id")
        elif self.source == "modelscope":
            if "/" not in self.path:
                raise ValueError(f"ModelScope(id=) must be 'owner/repo', got {self.path!r}")
        else:
            raise ValueError(f"unknown ModelRef source {self.source!r}")

    @property
    def provider(self) -> str:
        """Back-compat alias, OLD narrower vocabulary: ``"hf"`` (not
        ``"huggingface"``) for huggingface refs, ``source`` verbatim
        otherwise. This is what tensorhub's build-manifest
        ``bindings.<slot>.provider`` column expects — use ``.source`` for
        anything new (pgw#511 wire vocabulary)."""
        return "hf" if self.source == "huggingface" else self.source

    @property
    def ref(self) -> str:
        """Back-compat alias for ``.path``."""
        return self.path


def Hub(
    ref: str,
    *,
    tag: str = "latest",
    flavor: str = "",
    storage_dtype: str = "",
    allow_lora: bool = False,
) -> ModelRef:
    """Tensorhub-backed binding: ``Hub("owner/repo", tag=, flavor=, storage_dtype=, allow_lora=)``.

    ``allow_lora=True`` opts the slot into per-request LoRA overlays
    (``_models.<slot>.loras``, ie#358) — the endpoint must also declare
    ``Compile(family=...)`` so the hub's architecture gate can police
    adapter targets.
    """
    return ModelRef(
        source="tensorhub", path=ref, tag=tag, flavor=flavor,
        storage_dtype=storage_dtype, allow_lora=allow_lora,
    )


def HF(
    ref: str,
    *,
    revision: str = "",
    dtype: str = "",
    subfolder: str = "",
    files: tuple[str, ...] = (),
    storage_dtype: str = "",
    allow_lora: bool = False,
) -> ModelRef:
    """HuggingFace-backed binding: ``HF(id, revision=, dtype=, subfolder=, files=, storage_dtype=)``.

    ``files`` are ``snapshot_download`` ``allow_patterns`` globs — set them to
    fetch only specific files (ComfyUI / split-checkpoint repos with no
    ``model_index.json``). ``dtype`` selects the torch COMPUTE precision at
    load time (``"bf16"`` / ``"fp16"`` / ``"fp32"``). ``storage_dtype="fp8"``
    keeps denoiser weights in fp8-E4M3 storage with per-layer upcast to the
    compute dtype (VRAM fit on any card; see ``STORAGE_DTYPES``).
    """
    return ModelRef(
        source="huggingface", path=ref, revision=revision, dtype=dtype,
        subfolder=subfolder, files=files, storage_dtype=storage_dtype,
        allow_lora=allow_lora,
    )


def Civitai(ref: str, *, version: str = "") -> ModelRef:
    """Civitai-backed binding: ``Civitai(model_id, version=)``.

    ``ref`` is the Civitai MODEL id; pin a specific model-version with
    ``version=``.
    """
    return ModelRef(source="civitai", path=ref, version=version)


def ModelScope(
    ref: str, *, revision: str = "", files: tuple[str, ...] = (),
) -> ModelRef:
    """ModelScope-backed binding: ``ModelScope(id, revision=, files=)``.

    File-oriented (no diffusers-layout requirement) — the clean source for
    ComfyUI / DiffSynth split checkpoints.
    """
    return ModelRef(source="modelscope", path=ref, revision=revision, files=files)


Binding = ModelRef
BINDING_TYPES: tuple[type, ...] = (ModelRef,)


def wire_ref(binding: Binding) -> str:
    """Normal-form ref string for the wire / cache key — delegates to the ONE
    grammar module (``gen_worker.models.refs``, gw#492).

    Hub refs carry ``:tag`` (elided when ``latest``, the grammar default) and
    ``#flavor`` suffixes; HF refs carry ``@revision``. Load-time metadata
    (dtype/subfolder/files/storage_dtype) never enters the ref.
    """
    from ..models.refs import HuggingFaceRef, fold_ref

    if binding.source == "tensorhub":
        # The default tag never overrides one embedded in ``ref``.
        tag = binding.tag if binding.tag != "latest" else ""
        return fold_ref(binding.path, tag=tag, flavor=binding.flavor)
    if binding.source == "huggingface":
        return HuggingFaceRef(binding.path, binding.revision or None).canonical()
    return binding.path


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
    slot into two residency identities (the th#736 mechanic). This is also
    how a flavor/cast fold onto a source without that axis (e.g. an HF ref's
    flavor, which never enters its wire_ref) gets rejected — the rebound
    struct always accepts the field (msgspec has no per-source shape), so the
    round-trip mismatch is the enforcement point.
    """
    from msgspec import structs

    from ..models.refs import fold_ref, normalize_model_ref, parse_model_ref

    if resolved_ref:
        parsed = parse_model_ref(resolved_ref)
        if parsed.tensorhub is None:
            raise ValueError(f"resolution {resolved_ref!r} is not a tensorhub ref")
        flavor = parsed.tensorhub.flavor or ""
    rebound: Any = binding
    try:
        if flavor is not None and flavor != binding.flavor:
            rebound = structs.replace(rebound, flavor=flavor)
        if cast:
            rebound = structs.replace(rebound, storage_dtype=cast)
    except TypeError as exc:
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
    "Binding", "BINDING_TYPES", "Civitai", "HF", "Hub", "ModelRef",
    "ModelScope", "STORAGE_DTYPES", "rebind_pick", "wire_ref",
]
