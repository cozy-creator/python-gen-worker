"""Binding model for `@inference(models=...)` — `Repo` + `Dispatch`.

Replaces the old `Annotated[T, ModelRef(...)]` injection pattern. See
`progress.json` issue #9 (decorator-table-model-bindings) for the spec.

A binding is one of two **picks**:

- **Fixed pick** (`Repo`): function pins one specific `(repo, flavor?, tag?)`.
  Tag defaults to ``"prod"``. Flavor is optional. Resolves to a concrete
  checkpoint at deploy time.
- **Dispatch pick** (`Dispatch`): function pins a *set* of `(repo, flavor?,
  tag?)` picks keyed by a single discriminator field in the payload (a
  ``Literal[...]``-typed field). At invoke time the discriminator selects
  which pick is used.

Each pick supports an optional ``.allow_override(*classes)`` chainable
method. When called with one or more pipeline class arguments, the invoker
may substitute the default pick with an arbitrary ref of their choice —
subject to the constraint that the supplied ref's ``pipeline_class`` must
be in the explicit allowlist.

All modifier methods return new immutable instances; chain order is
commutative. Bare zero-arg ``.allow_override()`` raises ``ValueError`` at
decoration time — the framework does NOT auto-derive the constraint from
the function's param annotation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Mapping, Optional, Union


def _qualname(t: Any) -> str:
    """Return a class's fully-qualified name (`module.qualname`)."""
    mod = getattr(t, "__module__", "") or ""
    qn = getattr(t, "__qualname__", None) or getattr(t, "__name__", "") or ""
    if mod and qn:
        return f"{mod}.{qn}"
    return repr(t)


def _normalize_classes(classes: tuple[Any, ...]) -> tuple[str, ...]:
    """Normalize ``*classes`` (class objects or string FQNs) to a unique
    tuple of FQN strings, preserving order.

    A zero-arg call raises ``ValueError``. The framework requires the tenant
    to explicitly enumerate which pipeline classes the invoker may substitute.
    """
    if not classes:
        raise ValueError(
            "allow_override() requires at least one pipeline class argument. "
            "Pass class objects (preferred) or string FQNs, e.g. "
            "allow_override(MyPipelineClass) or allow_override('pkg.mod.MyPipelineClass')."
        )
    out: list[str] = []
    seen: set[str] = set()
    for c in classes:
        if isinstance(c, str):
            name = c.strip()
            if not name:
                raise ValueError("allow_override() string FQN must be non-empty")
        elif isinstance(c, type):
            name = _qualname(c)
        else:
            raise TypeError(
                f"allow_override() arguments must be class objects or string FQNs; "
                f"got {type(c).__name__}: {c!r}"
            )
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


_SLOT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_slot_name(name: str) -> str:
    out = str(name or "").strip()
    if not out:
        return ""
    if not _SLOT_NAME_RE.match(out):
        raise ValueError(
            "model slot name must be a Python/env-style identifier "
            "(letters, numbers, underscores; cannot start with a number)"
        )
    return out


@dataclass(frozen=True, init=False)
class Repo:
    """Repository handle — also usable as a binding with defaults.

    A bare ``Repo("owner/repo")`` is a **tensorhub** reference. For
    huggingface or civitai use the explicit subclasses :class:`HFRepo` /
    :class:`CivitaiRepo` — the provider is encoded in the class chosen, not
    in a string prefix. See issue #10 (typed-provider-repo-classes).

    Module-level convention::

        flux = Repo("black-forest-labs/flux.2-klein-4b-base")  # tensorhub
        qwen = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")             # huggingface
        sdxl = CivitaiRepo("123456")                            # civitai

    A bare ``Repo("owner/repo")`` already serves as a valid fixed binding
    (tag defaults to ``"prod"``, no flavor, no override). Refine with the
    chainable modifiers::

        flux.flavor("nf4")
        flux.tag("canary").flavor("bf16")
        flux.flavor("nf4").allow_override(MyPipelineClass)

    All modifier methods return new immutable instances; chain order is
    commutative.
    """

    # Class-level provider constant — ClassVar so dataclass does NOT
    # treat it as a field. Subclasses override with their own value.
    # "tensorhub" = cozy hub (default), "hf" = huggingface, "civitai" = civitai.
    PROVIDER: ClassVar[str] = "tensorhub"

    ref: str
    _slot_name: str = ""
    _flavor: str = ""
    _tag: str = "prod"
    _allow_override: bool = False
    _pipeline_classes: tuple[str, ...] = ()
    _allow_lora: bool = False

    def __init__(
        self,
        ref_or_name: str | None = None,
        default_ref: str | None = None,
        *,
        name: str = "",
        ref: str = "",
        _slot_name: str = "",
        _flavor: str = "",
        _tag: str = "prod",
        _allow_override: bool = False,
        _pipeline_classes: tuple[str, ...] = (),
        _allow_lora: bool = False,
    ) -> None:
        slot_name = _normalize_slot_name(name or _slot_name)
        if default_ref is not None:
            if not slot_name:
                slot_name = _normalize_slot_name(str(ref_or_name or ""))
            ref_value = default_ref
        else:
            ref_value = ref or str(ref_or_name or "")
            if name:
                slot_name = _normalize_slot_name(name)

        object.__setattr__(self, "ref", ref_value)
        object.__setattr__(self, "_slot_name", slot_name)
        object.__setattr__(self, "_flavor", str(_flavor or "").strip())
        object.__setattr__(self, "_tag", str(_tag or "prod").strip() or "prod")
        object.__setattr__(self, "_allow_override", bool(_allow_override))
        object.__setattr__(self, "_pipeline_classes", tuple(_pipeline_classes or ()))
        object.__setattr__(self, "_allow_lora", bool(_allow_lora))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError(f"{type(self).__name__}(ref=) must be a non-empty string, got {self.ref!r}")
        object.__setattr__(self, "ref", self.ref.strip())
        object.__setattr__(self, "_slot_name", _normalize_slot_name(self._slot_name))

    @property
    def provider(self) -> str:
        """Resolver provider: ``"tensorhub"`` (default), ``"hf"``, or ``"civitai"``."""
        return type(self).PROVIDER

    @property
    def slot_name(self) -> str:
        """Stable mutable-config key for this model binding, if explicitly declared."""
        return self._slot_name

    def flavor(self, name: str) -> "Repo":
        """Return a new Repo bound to the given flavor."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("Repo.flavor() requires a non-empty flavor name")
        return replace(self, _flavor=name)

    def tag(self, name: str) -> "Repo":
        """Return a new Repo bound to the given tag.

        Tag defaults to ``"prod"``; override only for non-prod rollouts.
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("Repo.tag() requires a non-empty tag name")
        return replace(self, _tag=name)

    def allow_override(self, *classes: Any) -> "Repo":
        """Return a new Repo whose default may be overridden at invoke time.

        ``*classes`` is the explicit allowlist of acceptable pipeline classes
        for caller-supplied overrides. Accepts class objects (preferred —
        autocomplete + import-time check) or string FQNs (escape hatch).
        A bare zero-arg call raises ``ValueError``.
        """
        fqns = _normalize_classes(classes)
        return replace(self, _allow_override=True, _pipeline_classes=fqns)

    def allow_lora(self) -> "Repo":
        """Return a new Repo whose injected runtime may receive LoRA overlays."""
        return replace(self, _allow_lora=True)


@dataclass(frozen=True, init=False)
class HFRepo(Repo):
    """HuggingFace-backed binding.

    The ref is the canonical HuggingFace repo id (``"owner/repo"``). Pin to
    a specific git revision with :meth:`revision`. Select load-time torch
    precision with :meth:`dtype`::

        qwen = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")
        qwen_pinned = qwen.revision("a1b2c3d")            # branch / tag / sha
        qwen_bf16 = qwen.dtype("bf16")                    # torch_dtype at load

    HuggingFace does NOT have a "flavor" ref selector — flavor is a
    tensorhub-side concept (the ``#flavor`` suffix in
    ``Repo("owner/repo:tag#flavor")``). Calling :meth:`flavor` on an HFRepo
    emits a :class:`DeprecationWarning` and maps the value to
    :meth:`dtype` for one release; the shim drops in 0.8.x.
    """

    PROVIDER: ClassVar[str] = "hf"
    _revision: str = ""
    # torch dtype for `from_pretrained(torch_dtype=...)`. Empty string means
    # "let the loader pick a default" (today: `torch.bfloat16`). The wire
    # format carries this alongside `allow_override` / `pipeline_classes`
    # on the binding row; it is NOT encoded into the ref string.
    _dtype: str = ""
    # Subfolder inside the HF repo to load a single component from (issue
    # #337 SharedBase: a text_encoder / vae lives under a named subfolder of
    # a full SDXL repo). Empty = load the repo root.
    _subfolder: str = ""
    # File-selection globs (snapshot_download allow_patterns) to fetch ONLY
    # specific files from the repo. Set via .files(...). When present, the
    # downloader does a direct snapshot_download(allow_patterns=...) and skips
    # the diffusers-layout inference — so ComfyUI / split-checkpoint repos
    # (no model_index.json) resolve by explicitly listing their component files.
    _allow_patterns: tuple[str, ...] = ()

    def __init__(
        self,
        ref_or_name: str | None = None,
        default_ref: str | None = None,
        *,
        name: str = "",
        ref: str = "",
        subfolder: str = "",
        _slot_name: str = "",
        _flavor: str = "",
        _tag: str = "prod",
        _allow_override: bool = False,
        _pipeline_classes: tuple[str, ...] = (),
        _allow_lora: bool = False,
        _revision: str = "",
        _dtype: str = "",
        _subfolder: str = "",
        _allow_patterns: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            ref_or_name,
            default_ref,
            name=name,
            ref=ref,
            _slot_name=_slot_name,
            _flavor=_flavor,
            _tag=_tag,
            _allow_override=_allow_override,
            _pipeline_classes=_pipeline_classes,
            _allow_lora=_allow_lora,
        )
        object.__setattr__(self, "_revision", str(_revision or "").strip())
        object.__setattr__(self, "_dtype", str(_dtype or "").strip())
        object.__setattr__(self, "_subfolder", str(subfolder or _subfolder or "").strip())
        object.__setattr__(self, "_allow_patterns", tuple(_allow_patterns or ()))
        self.__post_init__()

    def files(self, *patterns: str) -> "HFRepo":
        """Return a new HFRepo that downloads only the matching files.

        Patterns are ``huggingface_hub.snapshot_download`` ``allow_patterns``
        globs (e.g. ``"split_files/vae/*.safetensors"``). When set, the worker
        fetches exactly these files and SKIPS the diffusers-layout inference —
        this is how a ComfyUI / split-checkpoint repo (no ``model_index.json``,
        weights nested under subdirs) loads from HF, and it avoids pulling
        unused shards / sibling models.
        """
        pats = tuple(str(p).strip() for p in patterns if str(p).strip())
        if not pats:
            raise ValueError("HFRepo.files() requires at least one pattern")
        return replace(self, _allow_patterns=pats)

    def subfolder(self, name: str) -> "HFRepo":
        """Return a new HFRepo that loads the named subfolder of the repo.

        Used for :class:`SharedBase` components that live under a folder of a
        larger repo (e.g. ``text_encoder`` inside an SDXL repo).
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("HFRepo.subfolder() requires a non-empty subfolder name")
        return replace(self, _subfolder=name)

    def __post_init__(self) -> None:
        super().__post_init__()
        if "/" not in self.ref:
            raise ValueError(
                f"HFRepo(ref=) must be 'owner/repo', got {self.ref!r}"
            )

    def revision(self, rev: str) -> "HFRepo":
        """Return a new HFRepo pinned to a specific git revision."""
        rev = str(rev or "").strip()
        if not rev:
            raise ValueError("HFRepo.revision() requires a non-empty revision")
        return replace(self, _revision=rev)

    def dtype(self, name: str) -> "HFRepo":
        """Return a new HFRepo whose weights load at the given torch dtype.

        Accepts diffusers / transformers-friendly names: ``"bf16"`` /
        ``"bfloat16"``, ``"fp16"`` / ``"float16"``, ``"fp32"`` /
        ``"float32"``. The value flows through
        :class:`~gen_worker.pipeline.loader.PipelineConfig.dtype` into
        ``from_pretrained(torch_dtype=...)``.
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("HFRepo.dtype() requires a non-empty dtype name")
        return replace(self, _dtype=name)

    def flavor(self, name: str) -> "HFRepo":
        """**Deprecated.** Maps to :meth:`dtype` for one release.

        ``flavor`` is a tensorhub-side concept (the ``#flavor`` suffix in
        ``Repo("owner/repo:tag#flavor")``). HuggingFace has no analogous
        ref selector — what you actually want on an HFRepo is the
        load-time torch precision, which is :meth:`dtype`.

        Drops in 0.8.x.
        """
        warnings.warn(
            "HFRepo.flavor() is deprecated; use .dtype(...) instead — flavor "
            "is tensorhub-only. Drops in 0.8.x.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dtype(name)


@dataclass(frozen=True, init=False)
class CivitaiRepo(Repo):
    """Civitai-backed binding.

    The ref is the Civitai model id (as a string). Pin to a specific
    version with :meth:`version`::

        lora = CivitaiRepo("123456")
        lora_v2 = lora.version("789012")
    """

    PROVIDER: ClassVar[str] = "civitai"
    _version_id: str = ""

    def __init__(
        self,
        ref_or_name: str | None = None,
        default_ref: str | None = None,
        *,
        name: str = "",
        ref: str = "",
        _slot_name: str = "",
        _flavor: str = "",
        _tag: str = "prod",
        _allow_override: bool = False,
        _pipeline_classes: tuple[str, ...] = (),
        _allow_lora: bool = False,
        _version_id: str = "",
    ) -> None:
        super().__init__(
            ref_or_name,
            default_ref,
            name=name,
            ref=ref,
            _slot_name=_slot_name,
            _flavor=_flavor,
            _tag=_tag,
            _allow_override=_allow_override,
            _pipeline_classes=_pipeline_classes,
            _allow_lora=_allow_lora,
        )
        object.__setattr__(self, "_version_id", str(_version_id or "").strip())

    def version(self, version_id: str) -> "CivitaiRepo":
        """Return a new CivitaiRepo pinned to a specific version id."""
        version_id = str(version_id or "").strip()
        if not version_id:
            raise ValueError("CivitaiRepo.version() requires a non-empty version id")
        return replace(self, _version_id=version_id)


@dataclass(frozen=True, init=False)
class ModelScopeRepo(Repo):
    """ModelScope-backed binding.

    The ref is the ModelScope repo id (``"owner/repo"``). ModelScope is
    DiffSynth-Studio's native model source and is **file-oriented** (no
    diffusers-layout requirement), so this is the clean way to fetch
    ComfyUI / DiffSynth-style split checkpoints (e.g. ``circlestone-labs/Anima``)
    that the HuggingFace resolver rejects.

        anima = ModelScopeRepo("circlestone-labs/Anima")
        anima = anima.revision("master")
        # fetch only specific files (skip unused shards / sibling models):
        anima = anima.files("split_files/diffusion_models/anima-base-v1.0.safetensors")

    ``.files(*patterns)`` carries ``allow_patterns`` as binding metadata (like
    :meth:`HFRepo.dtype` / :meth:`HFRepo.subfolder`) — it is NOT encoded into the
    ref string; the downloader passes it to
    ``modelscope.snapshot_download(allow_patterns=...)``.
    """

    PROVIDER: ClassVar[str] = "modelscope"
    _revision: str = ""
    _allow_patterns: tuple[str, ...] = ()

    def __init__(
        self,
        ref_or_name: str | None = None,
        default_ref: str | None = None,
        *,
        name: str = "",
        ref: str = "",
        _slot_name: str = "",
        _flavor: str = "",
        _tag: str = "prod",
        _allow_override: bool = False,
        _pipeline_classes: tuple[str, ...] = (),
        _allow_lora: bool = False,
        _revision: str = "",
        _allow_patterns: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            ref_or_name,
            default_ref,
            name=name,
            ref=ref,
            _slot_name=_slot_name,
            _flavor=_flavor,
            _tag=_tag,
            _allow_override=_allow_override,
            _pipeline_classes=_pipeline_classes,
            _allow_lora=_allow_lora,
        )
        object.__setattr__(self, "_revision", str(_revision or "").strip())
        object.__setattr__(self, "_allow_patterns", tuple(_allow_patterns or ()))
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if "/" not in self.ref:
            raise ValueError(
                f"ModelScopeRepo(ref=) must be 'owner/repo', got {self.ref!r}"
            )

    def revision(self, rev: str) -> "ModelScopeRepo":
        """Return a new ModelScopeRepo pinned to a specific revision."""
        rev = str(rev or "").strip()
        if not rev:
            raise ValueError("ModelScopeRepo.revision() requires a non-empty revision")
        return replace(self, _revision=rev)

    def files(self, *patterns: str) -> "ModelScopeRepo":
        """Return a new ModelScopeRepo that downloads only the matching files.

        Patterns are ``modelscope.snapshot_download`` ``allow_patterns`` globs
        (e.g. ``"split_files/vae/*.safetensors"``). Use this to fetch only the
        needed components and skip unused shards / sibling models.
        """
        pats = tuple(str(p).strip() for p in patterns if str(p).strip())
        if not pats:
            raise ValueError("ModelScopeRepo.files() requires at least one pattern")
        return replace(self, _allow_patterns=pats)


@dataclass(frozen=True)
class Dispatch:
    """Payload-driven dispatch pick.

    Construct via the free :func:`dispatch` function::

        dispatch(field="variant", table={
            "nf4": flux.flavor("nf4"),
            "int8": flux.flavor("int8"),
        })

    The discriminator ``field`` must be a ``Literal[...]``-typed member of
    the function's payload struct; every ``table`` key must be one of the
    Literal's members. Validated at decoration time by
    :func:`inference_function`.
    """

    field: str
    table: Mapping[str, "Repo"]
    _allow_override: bool = False
    _pipeline_classes: tuple[str, ...] = ()
    _allow_lora: bool = False

    def __post_init__(self) -> None:
        f = str(self.field or "").strip()
        if not f:
            raise ValueError("dispatch() requires a non-empty discriminator field name")
        object.__setattr__(self, "field", f)

        if not self.table:
            raise ValueError("dispatch() requires a non-empty table")
        # Freeze the table to a plain dict (msgspec.to_builtins safety).
        frozen_table: dict[str, Repo] = {}
        for k, v in self.table.items():
            key = str(k or "").strip()
            if not key:
                raise ValueError(f"dispatch() table has empty key: {k!r}")
            if not isinstance(v, Repo):
                raise TypeError(
                    f"dispatch() table values must be Repo instances; "
                    f"got {type(v).__name__} for key {key!r}"
                )
            frozen_table[key] = v
        object.__setattr__(self, "table", frozen_table)

    def allow_override(self, *classes: Any) -> "Dispatch":
        """Return a new Dispatch whose default may be overridden at invoke time.

        Same semantics as :meth:`Repo.allow_override`.
        """
        fqns = _normalize_classes(classes)
        return replace(self, _allow_override=True, _pipeline_classes=fqns)

    def allow_lora(self) -> "Dispatch":
        """Return a new Dispatch whose injected runtime may receive LoRA overlays."""
        return replace(self, _allow_lora=True)


def dispatch(field: str, table: Mapping[str, Repo]) -> Dispatch:
    """Construct a :class:`Dispatch` binding.

    Args:
        field: The discriminator payload field name. Must reference a
            ``Literal[...]``-typed field on the function's payload struct.
        table: Mapping from Literal member values → :class:`Repo` picks.
            Every Literal value must appear as a key. Validated at
            decoration time. Each value may be a plain :class:`Repo`/
            :class:`HFRepo` (full pipeline), a :class:`Variant` produced by
            :meth:`SharedBase.variant` (shared base + swappable slot), or a
            ``.allow_lora()`` Repo (LoRA overlay). See issue #337.
    """
    return Dispatch(field=field, table=dict(table))


# ============================================================================
# SharedBase + Variant — model-selectable endpoints (issue #337).
#
# An endpoint may expose a *set* of selectable fine-tunes that share a frozen
# stack of components (SDXL: two CLIP text encoders + the VAE) and differ only
# in one swappable slot (the UNet/transformer). Loading N full pipelines blows
# past VRAM; the shared-base contract loads the frozen stack ONCE, pins it in
# VRAM, and builds each variant pipeline pointing at the SAME component objects
# (diffusers pipelines are containers of references to nn.Modules). Memory then
# is (shared stack, once) + {LRU set of variant slots} instead of N×full.
#
# `SharedBase(pipeline_cls, **components)` declares the frozen stack. Its
# `.variant(**variant_slots)` returns a `Variant` — a `Repo` subclass so it
# drops straight into `dispatch(field=, table={...})` tables and the existing
# wire serializer / partial-readiness keying. The Variant's `.ref` is the
# primary variant slot's ref (the swap unit), so cache identity and download
# progress key off the thing that actually changes per model.
# ============================================================================


@dataclass(frozen=True, init=False)
class Variant(Repo):
    """One selectable model = a :class:`SharedBase` + its own variant slot(s).

    Produced by :meth:`SharedBase.variant`. A ``Variant`` IS a :class:`Repo`
    (its ``.ref`` is the primary variant slot's ref — the UNet/transformer that
    swaps per request), so it composes with :func:`dispatch` and the binding
    wire format unchanged. It additionally carries:

    - ``pipeline_class_fqn`` — the diffusers pipeline class to assemble.
    - ``shared_components`` — ``{component_name: Repo}`` loaded once + pinned.
    - ``variant_slots`` — ``{slot_name: Repo}`` swapped per request (the
      first entry is the primary slot and drives ``.ref``).
    """

    PROVIDER: ClassVar[str] = "hf"

    pipeline_class_fqn: str = ""
    shared_components: Mapping[str, "Repo"] = ()  # type: ignore[assignment]
    variant_slots: Mapping[str, "Repo"] = ()  # type: ignore[assignment]

    def __init__(
        self,
        *,
        pipeline_class_fqn: str,
        shared_components: Mapping[str, "Repo"],
        variant_slots: Mapping[str, "Repo"],
        _slot_name: str = "",
        _allow_override: bool = False,
        _pipeline_classes: tuple[str, ...] = (),
        _allow_lora: bool = False,
    ) -> None:
        if not variant_slots:
            raise ValueError(
                "SharedBase.variant() requires at least one variant slot "
                "(e.g. .variant(unet=HFRepo(...)))"
            )
        primary = next(iter(variant_slots.values()))
        # The Variant's identity (ref/provider) follows its primary swap slot
        # so cache keys + download progress track the per-model weights.
        super().__init__(
            ref=primary.ref,
            _slot_name=_slot_name,
            _flavor=getattr(primary, "_flavor", "") or "",
            _tag=getattr(primary, "_tag", "prod") or "prod",
            _allow_override=_allow_override,
            _pipeline_classes=_pipeline_classes,
            _allow_lora=_allow_lora,
        )
        object.__setattr__(self, "pipeline_class_fqn", str(pipeline_class_fqn or ""))
        object.__setattr__(self, "shared_components", dict(shared_components or {}))
        object.__setattr__(self, "variant_slots", dict(variant_slots or {}))

    @property
    def provider(self) -> str:
        # A Variant's effective provider is its primary slot's provider.
        primary = next(iter(self.variant_slots.values()), None)
        return getattr(primary, "provider", type(self).PROVIDER) if primary else type(self).PROVIDER

    @property
    def primary_slot_name(self) -> str:
        """Name of the swappable slot that drives this variant's identity."""
        return next(iter(self.variant_slots.keys()), "")

    def _respawn(
        self,
        *,
        _allow_override: Optional[bool] = None,
        _pipeline_classes: Optional[tuple[str, ...]] = None,
        _allow_lora: Optional[bool] = None,
    ) -> "Variant":
        """Return a copy with modifier flags changed, preserving variant fields.

        ``dataclasses.replace`` (used by the base ``Repo`` modifiers) can't
        reconstruct a ``Variant`` because its ``__init__`` signature differs, so
        the chainable modifiers route through here instead.
        """
        return Variant(
            pipeline_class_fqn=self.pipeline_class_fqn,
            shared_components=dict(self.shared_components),
            variant_slots=dict(self.variant_slots),
            _slot_name=self._slot_name,
            _allow_override=self._allow_override if _allow_override is None else _allow_override,
            _pipeline_classes=self._pipeline_classes if _pipeline_classes is None else _pipeline_classes,
            _allow_lora=self._allow_lora if _allow_lora is None else _allow_lora,
        )

    def allow_override(self, *classes: Any) -> "Variant":
        return self._respawn(_allow_override=True, _pipeline_classes=_normalize_classes(classes))

    def allow_lora(self) -> "Variant":
        return self._respawn(_allow_lora=True)

    def flavor(self, name: str) -> "Variant":  # noqa: D401
        raise TypeError(
            "flavor()/tag() are set on the per-slot Repos passed to "
            "SharedBase.variant(...), not on the Variant itself."
        )

    def tag(self, name: str) -> "Variant":  # noqa: D401
        return self.flavor(name)


class SharedBase:
    """A frozen, load-once component stack shared by reference across variants.

    ::

        sdxl_base = SharedBase(
            StableDiffusionXLPipeline,
            text_encoder   = HFRepo("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder"),
            text_encoder_2 = HFRepo("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2"),
            vae            = HFRepo("madebyollin/sdxl-vae-fp16-fix"),
        )
        MODELS = {
            "illustrious": sdxl_base.variant(unet=HFRepo("OnomaAIResearch/Illustrious-XL-v1.0")),
            "animagine":   sdxl_base.variant(unet=HFRepo("cagliostrolab/animagine-xl-3.1")),
        }

    The components passed here are loaded ONCE, pinned in VRAM, and shared by
    reference into every variant pipeline. An empty SharedBase (no components)
    is valid — it means each variant is a standalone full pipeline (the
    escape hatch for fine-tunes whose CLIP/VAE differ from the base).
    """

    def __init__(self, pipeline_cls: Any, **components: "Repo") -> None:
        if pipeline_cls is None:
            raise ValueError("SharedBase requires a pipeline class as the first argument")
        for name, comp in components.items():
            if not isinstance(comp, Repo):
                raise TypeError(
                    f"SharedBase component {name!r} must be a Repo/HFRepo, "
                    f"got {type(comp).__name__}"
                )
        self.pipeline_cls = pipeline_cls
        self.pipeline_class_fqn = _qualname(pipeline_cls)
        self.components: dict[str, Repo] = dict(components)

    def variant(self, **variant_slots: "Repo") -> Variant:
        """Build a selectable :class:`Variant` = this shared base + slots.

        ``variant_slots`` are the per-model swap units (typically
        ``unet=HFRepo(...)`` or ``transformer=HFRepo(...)``). The first slot
        is the primary one and drives the variant's cache identity.
        """
        if not variant_slots:
            raise ValueError(
                "SharedBase.variant() requires at least one slot, e.g. "
                ".variant(unet=HFRepo('owner/repo'))"
            )
        for name, comp in variant_slots.items():
            if not isinstance(comp, Repo):
                raise TypeError(
                    f"variant slot {name!r} must be a Repo/HFRepo, "
                    f"got {type(comp).__name__}"
                )
        return Variant(
            pipeline_class_fqn=self.pipeline_class_fqn,
            shared_components=dict(self.components),
            variant_slots=dict(variant_slots),
        )


Binding = Union[Repo, Dispatch]


__all__ = [
    "Binding",
    "CivitaiRepo",
    "Dispatch",
    "HFRepo",
    "Repo",
    "SharedBase",
    "Variant",
    "dispatch",
]
