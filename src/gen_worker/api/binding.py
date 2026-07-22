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
    inferred from shape or which factory built the value. Pure identity +
    fetch scope — no permission fields live here (pgw#523: overlay
    permission is a slot-policy concern, not an identity-struct flag).

    Carries the union of every registry's per-source fields (tensorhub:
    ``tag``/``flavor``; huggingface: ``revision``/``dtype``/``subfolder``/
    ``files``; civitai: ``version``; modelscope: ``revision``/``files``).
    ``storage_dtype`` is shared by tensorhub/huggingface. Build one via
    ``Hub``/``HF``/``Civitai``/``ModelScope`` rather than the raw
    constructor — they pin ``source`` and apply the per-registry validation
    below (mirrored in ``__post_init__`` so it holds for direct construction
    too, e.g. ``msgspec.structs.replace``).

    ``components`` (pgw#505, tensorhub/huggingface only): restricts the
    fetch to the named pipeline component subfolders — e.g. a full SDXL repo
    bound only for its VAE: ``Hub("owner/sdxl-repo", components=("vae",))``.
    Root config files (``model_index.json`` and other root ``*.json``) are
    always kept alongside the named subfolders. Empty (default) fetches the
    whole repo — today's behavior. Civitai/modelscope reject it: civitai
    artifacts aren't component-structured, and modelscope's ``files=`` glob
    already covers the split-checkpoint case.
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
    components: tuple[str, ...] = ()
    # pgw#617 hierarchical bindings (tensorhub only): sorted (component name,
    # canonical ref) substitutions on the base composition — stamped from
    # ModelBinding.components at dispatch, never declared by endpoint code.
    # Part of the binding's identity: a component-only rebind derives a new
    # instance/residency identity.
    component_overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # msgspec.structs.force_setattr, NOT object.__setattr__: the latter
        # raises "can't apply this __setattr__" on frozen msgspec Structs
        # under CPython 3.12 (every serve image) while passing on 3.13 (dev
        # venvs + CI) — J24M run16 shipped an image whose endpoint import
        # died at decoration and discovery found no functions.
        force = msgspec.structs.force_setattr
        force(self, "path", _clean(self.path))
        force(self, "tag", _clean(self.tag))
        force(self, "flavor", _clean(self.flavor))
        force(self, "revision", _clean(self.revision))
        force(self, "subfolder", _clean(self.subfolder))
        force(self, "dtype", _clean(self.dtype))
        force(self, "version", _clean(self.version))
        force(self, "files", tuple(_clean(p) for p in self.files if _clean(p)))
        force(self, "components", tuple(_clean(p) for p in self.components if _clean(p)))
        force(self, "component_overrides", tuple(sorted(
            (_clean(k), _clean(v))
            for k, v in self.component_overrides if _clean(k) and _clean(v)
        )))
        force(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))

        if self.component_overrides and self.source != "tensorhub":
            raise ValueError(
                "component_overrides= is tensorhub-only (pgw#617: refs are "
                "tensorhub-CAS, mirror-first)"
            )
        if self.source == "tensorhub":
            if not self.tag:
                msgspec.structs.force_setattr(self, "tag", "latest")
            if not self.path:
                raise ValueError("Hub requires a non-empty ref")
        elif self.source == "huggingface":
            if "/" not in self.path:
                raise ValueError(f"HF(id=) must be 'owner/repo', got {self.path!r}")
        elif self.source == "civitai":
            if not self.path:
                raise ValueError("Civitai requires a non-empty model id")
            if self.components:
                raise ValueError(
                    "Civitai bindings do not support components= "
                    "(civitai artifacts aren't component-structured)"
                )
        elif self.source == "modelscope":
            if "/" not in self.path:
                raise ValueError(f"ModelScope(id=) must be 'owner/repo', got {self.path!r}")
            if self.components:
                raise ValueError(
                    "ModelScope bindings do not support components= (use files= instead)"
                )
        else:
            raise ValueError(f"unknown ModelRef source {self.source!r}")


def Hub(
    ref: str,
    *,
    tag: str = "latest",
    flavor: str = "",
    storage_dtype: str = "",
    components: tuple[str, ...] = (),
) -> ModelRef:
    """Tensorhub-backed binding: ``Hub("owner/repo", tag=, flavor=, storage_dtype=, components=)``.

    ``components=`` (pgw#505) fetches only the named pipeline component
    subfolders (+ root config files) instead of the whole repo — e.g. a
    full SDXL checkpoint bound only for its VAE:
    ``Hub("owner/sdxl-repo", components=("vae",))``.
    """
    return ModelRef(
        source="tensorhub", path=ref, tag=tag, flavor=flavor,
        storage_dtype=storage_dtype, components=components,
    )


def HF(
    ref: str,
    *,
    revision: str = "",
    dtype: str = "",
    subfolder: str = "",
    files: tuple[str, ...] = (),
    components: tuple[str, ...] = (),
    storage_dtype: str = "",
) -> ModelRef:
    """HuggingFace-backed binding: ``HF(id, revision=, dtype=, subfolder=, files=, components=, storage_dtype=)``.

    ``files`` are ``snapshot_download`` ``allow_patterns`` globs — set them to
    fetch only specific files (ComfyUI / split-checkpoint repos with no
    ``model_index.json``). ``components=`` (pgw#505) is the diffusers-layout
    counterpart: name the pipeline component subfolders to fetch (e.g.
    ``components=("unet", "text_encoder")``); root config files
    (``model_index.json`` + other root ``*.json``) are always kept. When both
    are set, ``files=`` is matched within the ``components=``-narrowed
    listing. ``dtype`` selects the torch COMPUTE precision at load time
    (``"bf16"`` / ``"fp16"`` / ``"fp32"``). ``storage_dtype="fp8"`` keeps
    denoiser weights in fp8-E4M3 storage with per-layer upcast to the compute
    dtype (VRAM fit on any card; see ``STORAGE_DTYPES``).
    """
    return ModelRef(
        source="huggingface", path=ref, revision=revision, dtype=dtype,
        subfolder=subfolder, files=files, components=components,
        storage_dtype=storage_dtype,
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
