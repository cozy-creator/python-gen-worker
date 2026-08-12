"""Model bindings for ``@endpoint(models={...})``.

A binding names one remote model source. The slot name is NEVER a
constructor argument — it comes from the ``models={}`` dict key (or, for a
single ``model=`` binding, the ``setup()``/handler parameter name).

    HF("black-forest-labs/FLUX.1-dev", dtype="bf16")
    Hub("owner/repo", tag="canary")
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
from msgspec import structs

from ..models.refs import (
    DEFAULT_REF_TAG,
    HuggingFaceRef,
    fold_ref,
    normalize_model_ref,
    parse_model_ref,
    refuse_flavor_selector,
)

ModelSource = Literal["tensorhub", "huggingface", "civitai", "modelscope"]


def _clean(s: object) -> str:
    return str(s or "").strip()


# Weight STORAGE precisions a binding may request (th#546 two-format policy).
# "fp8" = fp8-E4M3 weight storage with per-layer upcast to the compute dtype
# (diffusers layerwise casting) — the universal VRAM-fit mechanism; works on
# cards without fp8 units. Applied by the loading layer; also auto-applied
# when the snapshot itself stores fp8 (a checkpoint whose tensor-layout
# contract is an fp8 one).
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
    ``tag``; huggingface: ``revision``/``dtype``/``subfolder``/
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
        # §1.32(d)/th#1803: THE FLAVOR SYSTEM IS DEAD. A `#` in ANY binding
        # path is a weight selector and refuses typed here, at the site the
        # author wrote — never as a hub 400 three layers away. Cell refs are
        # not bindings and never reach this constructor.
        refuse_flavor_selector(self.path, where=f"{self.source} binding")
        if self.source == "tensorhub":
            if not self.tag:
                msgspec.structs.force_setattr(self, "tag", DEFAULT_REF_TAG)
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

    @property
    def label(self) -> str:
        """Human-readable label for ``model_used`` metadata / logging —
        mirrors the wire form each source's registry keys on (pgw#654:
        endpoints delete their per-file ``_ref_label`` copies)."""
        label = self.path
        if self.source == "civitai" and self.version:
            label += f"@{self.version}"
        elif self.tag and self.tag != DEFAULT_REF_TAG:
            label += f":{self.tag}"
        return label


def Hub(
    ref: str,
    *,
    tag: str = DEFAULT_REF_TAG,
    storage_dtype: str = "",
    components: tuple[str, ...] = (),
) -> ModelRef:
    """Tensorhub-backed binding: ``Hub("owner/repo", tag=, storage_dtype=, components=)``.

    ``components=`` (pgw#505) fetches only the named pipeline component
    subfolders (+ root config files) instead of the whole repo — e.g. a
    full SDXL checkpoint bound only for its VAE:
    ``Hub("owner/sdxl-repo", components=("vae",))``.
    """
    return ModelRef(
        source="tensorhub", path=ref, tag=tag,
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

    Hub refs carry ``:tag`` (elided when ``prod``, the grammar default — an
    explicit ``tag="latest"`` is stamped verbatim, th#1276); HF refs carry
    ``@revision``. Load-time metadata (dtype/subfolder/files/storage_dtype)
    never enters the ref, and pgw#1148 deleted the ``#flavor`` suffix this
    used to fold — a binding has no second selector to mint.
    """

    if binding.source == "tensorhub":
        # The default tag never overrides one embedded in ``ref``.
        tag = binding.tag if binding.tag != DEFAULT_REF_TAG else ""
        return fold_ref(binding.path, tag=tag)
    if binding.source == "huggingface":
        return HuggingFaceRef(binding.path, binding.revision or None).canonical()
    return binding.path


def component_overrides(binding: Binding) -> tuple[tuple[str, str], ...]:
    """The ``(component, canonical ref)`` substitutions ``binding`` carries
    (pgw#617), normalized and sorted by :class:`ModelRef` itself.

    pgw#982: this and :func:`binding_wire_refs` live HERE, beside the field
    they read — they answer "what does this binding resolve to", which is not
    an executor internal. Every consumer (the executor, the rotation preload
    driver) names the SAME derivation, so placement semantics cannot fork.

    ``getattr`` so an externally-constructed stand-in is answered, not raised
    at.
    """
    return tuple(getattr(binding, "component_overrides", ()) or ())


def binding_wire_refs(binding: Binding) -> list[str]:
    """The base :func:`wire_ref` plus every component-override ref (pgw#617):
    the full set of refs materializing this binding pins and downloads."""
    return [wire_ref(binding), *(ref for _, ref in component_overrides(binding))]


def rebind_pick(
    binding: Binding,
    *,
    resolved_ref: str = "",
    cast: str = "",
) -> Binding:
    """THE fold of a hub pick into a binding (gw#494): the HelloAck path
    (``resolved_ref`` + ``cast``).

    ``resolved_ref`` is authoritative when given — since th#1803 the hub's
    ladder expresses its pick as a DIGEST (``owner/repo@sha256:…``), never a
    ``#flavor``. Raises ``ValueError`` when the pick cannot round-trip
    through ``wire_ref`` — a pick the rebound binding cannot re-mint would
    split the slot into two residency identities (the th#736 mechanic).

    pgw#1148 deleted the ``flavor=`` arm and with it the LOCAL-ladder caller
    that used it: there is no within-tag-group selector left to fold, so a
    pick is a digest or it is nothing.
    """

    rebound: Any = binding
    if resolved_ref:
        parsed = parse_model_ref(resolved_ref)
        if parsed.tensorhub is None:
            raise ValueError(f"resolution {resolved_ref!r} is not a tensorhub ref")
        refuse_flavor_selector(resolved_ref, where="hub resolution")
    try:
        if cast:
            rebound = structs.replace(rebound, storage_dtype=cast)
    except TypeError as exc:
        raise ValueError(f"pick does not fit binding {binding!r}: {exc}") from exc
    if resolved_ref:
        expected = normalize_model_ref(resolved_ref)
        if wire_ref(rebound) != expected:
            raise ValueError(
                f"pick {expected!r} does not round-trip through the binding "
                f"(got {wire_ref(rebound)!r})"
            )
    return rebound


__all__ = [
    "Binding", "BINDING_TYPES", "Civitai", "HF", "Hub", "ModelRef",
    "ModelScope", "STORAGE_DTYPES", "binding_wire_refs", "component_overrides",
    "rebind_pick", "wire_ref",
]
