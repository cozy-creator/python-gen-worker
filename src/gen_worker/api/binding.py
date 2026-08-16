"""Model bindings for ``@endpoint(models={...})``.

A binding names one remote model source. The slot name is NEVER a
constructor argument — it comes from the ``models={}`` dict key (or, for a
single ``model=`` binding, the ``setup()``/handler parameter name).

    HF("black-forest-labs/FLUX.1-dev", dtype="bf16")
    Hub("owner/repo", release="2026.08")
    Civitai("123456", version="789012")
    ModelScope("circlestone-labs/Anima", files=("split_files/*.safetensors",))

``ModelRef`` is the ONE structured type: ``source`` is an explicit
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
    HuggingFaceRef,
    WireRef,
    fold_ref,
    normalize_model_ref,
    parse_model_ref,
    refuse_ref_fragment,
)

ModelSource = Literal["tensorhub", "huggingface", "civitai", "modelscope"]


def _clean(s: object) -> str:
    return str(s or "").strip()


# Weight STORAGE precisions a binding may request.
# "fp8" = fp8-E4M3 weight storage with per-layer upcast to the compute dtype
# (diffusers layerwise casting) — the universal VRAM-fit mechanism; works on
# cards without fp8 units. Applied by the loading layer; also auto-applied
# when the snapshot itself stores fp8 (a checkpoint whose tensor-layout
# contract is an fp8 one).
# "fp8+te" additionally casts the pipeline's text encoders via the
# transformers-aware path (linear weights fp8; embeddings/norms/tied weights
# stay at compute dtype — component fit-ladder rung 2).
STORAGE_DTYPES: tuple[str, ...] = ("fp8", "fp8+te")


def _clean_storage_dtype(v: object) -> str:
    q = _clean(v).lower()
    if q and q not in STORAGE_DTYPES:
        raise ValueError(
            f"unknown storage_dtype {q!r}; expected one of {STORAGE_DTYPES}"
        )
    return q


class ModelRef(msgspec.Struct, frozen=True):
    """ONE structured model reference: ``source`` is explicit, never
    inferred from shape or which factory built the value. Pure identity +
    fetch scope — no permission fields live here: overlay permission is a
    slot-policy concern, not an identity-struct flag.

    Carries the union of every registry's per-source fields (tensorhub:
    ``release``; huggingface: ``revision``/``dtype``/``subfolder``/
    ``files``; civitai: ``version``; modelscope: ``revision``/``files``).
    ``storage_dtype`` is shared by tensorhub/huggingface. Build one via
    ``Hub``/``HF``/``Civitai``/``ModelScope`` rather than the raw
    constructor — they pin ``source`` and apply the per-registry validation
    below (mirrored in ``__post_init__`` so it holds for direct construction
    too, e.g. ``msgspec.structs.replace``).

    ``components`` (tensorhub/huggingface only): restricts the
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
    #: th#1987: the tensorhub RELEASE this binding pins. There is no default —
    #: a binding that names none addresses the repo and no artifact in it, and
    #: the hub answers `release_not_found`.
    release: str = ""
    revision: str = ""
    subfolder: str = ""
    dtype: str = ""
    storage_dtype: str = ""
    version: str = ""
    files: tuple[str, ...] = ()
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # msgspec.structs.force_setattr, NOT object.__setattr__: the latter
        # raises "can't apply this __setattr__" on frozen msgspec Structs under
        # CPython 3.12 (every serve image) while passing on 3.13 (dev venvs +
        # CI), so the endpoint import dies at decoration in the image only.
        force = msgspec.structs.force_setattr
        force(self, "path", _clean(self.path))
        force(self, "release", _clean(self.release))
        force(self, "revision", _clean(self.revision))
        force(self, "subfolder", _clean(self.subfolder))
        force(self, "dtype", _clean(self.dtype))
        force(self, "version", _clean(self.version))
        force(self, "files", tuple(_clean(p) for p in self.files if _clean(p)))
        force(self, "components", tuple(_clean(p) for p in self.components if _clean(p)))
        force(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))

        # THE FLAVOR SYSTEM IS DEAD. A `#` in ANY binding path is a weight
        # selector and refuses typed here, at the site the author wrote — never
        # as a hub 400 three layers away. Cell refs are not bindings and never
        # reach this constructor.
        refuse_ref_fragment(self.path, where=f"{self.source} binding")
        # th#1987: `release` is the TENSORHUB axis. On any other source it
        # named nothing, was fetched by nothing, and reached only `label` —
        # where it silently vanished behind a civitai `version=`. One home per
        # value: refuse it where it cannot mean anything.
        if self.source != "tensorhub" and self.release:
            raise ValueError(
                f"{self.source} bindings have no release axis (release="
                f"{self.release!r}); it is a tensorhub release identifier. Use "
                "revision= for a huggingface/modelscope commit or version= for "
                "a civitai model-version."
            )
        if self.source == "tensorhub":
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
        mirrors the wire form each source's registry keys on.

        INJECTIVE (ie#727 residue): two bindings that pin different artifacts
        never share a label. The old fold appended the release only when the
        label carried no ``@`` at all, which silently dropped it twice — off a
        civitai ref that also pinned a ``version=``, and off a hub ref whose
        ``path`` already carried a release the side-channel ``release=``
        OVERRIDES. The second case made ``label`` and :func:`wire_ref` name
        different artifacts, so ``model_used`` reported the ref that was not
        fetched. The hub branch is now the same fold the wire uses.
        """
        if self.source == "tensorhub":
            return str(fold_ref(self.path, release=self.release))
        if self.source == "civitai":
            return f"{self.path}@{self.version}" if self.version else self.path
        return f"{self.path}@{self.revision}" if self.revision else self.path


def Hub(
    ref: str,
    *,
    release: str = "",
    storage_dtype: str = "",
    components: tuple[str, ...] = (),
) -> ModelRef:
    """Tensorhub-backed binding: ``Hub("owner/repo@<release>", storage_dtype=, components=)``.

    The release may ride the ref (``Hub("owner/repo@2026.08")``) or be given
    beside it (``release="2026.08"``); the side channel wins. A binding that
    names neither pins no artifact — th#1987 deleted the default.

    ``components=`` fetches only the named pipeline component
    subfolders (+ root config files) instead of the whole repo — e.g. a
    full SDXL checkpoint bound only for its VAE:
    ``Hub("owner/sdxl-repo", components=("vae",))``.
    """
    return ModelRef(
        source="tensorhub", path=ref, release=release,
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
    ``model_index.json``). ``components=`` is the diffusers-layout
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


def wire_ref(binding: Binding) -> WireRef:
    """Normal-form ref string for the wire / cache key — delegates to the ONE
    grammar module (``gen_worker.models.refs``).

    Hub refs carry ``@<release>`` (nothing is elided — th#1987 deleted the
    default); HF refs carry ``@revision``. Load-time metadata (dtype/subfolder/files/storage_dtype)
    never enters the ref, and there is no ``#flavor`` suffix — a binding has no
    second selector to mint.
    """

    if binding.source == "tensorhub":
        return fold_ref(binding.path, release=binding.release)
    if binding.source == "huggingface":
        return HuggingFaceRef(binding.path, binding.revision or None).canonical()
    # civitai/modelscope: the path is ASSERTED to be normal form rather than
    # derived through the grammar, which the other two branches do. The
    # `WireRef(...)` makes that the one visible assertion in this function
    # instead of a silent widening of its whole return type.
    return WireRef(binding.path)


def rebind_pick(
    binding: Binding,
    *,
    resolved_ref: str = "",
    cast: str = "",
) -> Binding:
    """THE fold of a hub pick into a binding: the HelloAck path
    (``resolved_ref`` + ``cast``).

    ``resolved_ref`` is authoritative when given — the hub's ladder expresses
    its pick as a DIGEST (``owner/repo@sha256:…``), never a ``#flavor``. Raises
    ``ValueError`` when the pick cannot round-trip through ``wire_ref`` — a
    pick the rebound binding cannot re-mint would split the slot into two
    residency identities.

    There is no within-release selector left to fold, so a pick is a digest
    or it is nothing.
    """

    rebound: Any = binding
    if resolved_ref:
        parsed = parse_model_ref(resolved_ref)
        if parsed.tensorhub is None:
            raise ValueError(f"resolution {resolved_ref!r} is not a tensorhub ref")
        refuse_ref_fragment(resolved_ref, where="hub resolution")
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
    "ModelScope", "STORAGE_DTYPES", "rebind_pick", "wire_ref",
]
