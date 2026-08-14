"""The repackage declaration schema — typed maps the SDK engine executes.

``convert/`` used to be the one package with no
registration hook, so every family's layout knowledge lived as a module-level
private constant inside the SDK — ~430 lines of hand-ported per-family tensor
surgery, four overlapping detection ladders, and two copies of the SDXL
signature that disagreed with each other.

This module defines the *vocabulary*; :mod:`gen_worker.convert.repack_registry`
holds the registrations; :mod:`gen_worker.convert.repack_engine` executes them.
No family appears in any of the three.

The unit of declaration is :class:`RepackageFamily`. Onboarding a family means
writing one of these in the endpoint repo and registering it — zero SDK changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

RenameKind = Literal["prefix", "substring", "alternation"]


class DeclarationError(ValueError):
    """A declaration is malformed. Raised at registration time, never at run time."""


@dataclass(frozen=True)
class RenameRule:
    """One ordered key-rewrite pass over a component's state-dict keys.

    ``prefix``      — for every pair, if the key starts with ``src`` the first
                      occurrence of ``src`` is replaced by ``dst``. Pairs are
                      applied in order and a key may match more than one.
    ``substring``   — every occurrence of ``src`` is replaced by ``dst``, pairs
                      applied in order.
    ``alternation`` — all pairs compiled into ONE regex alternation and applied
                      in a single pass, so an earlier replacement's output can
                      never be rewritten by a later pair. Longest source first,
                      which is what the CLIP keyspaces require.
    """

    kind: RenameKind
    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.kind not in ("prefix", "substring", "alternation"):
            raise DeclarationError(f"unknown rename rule kind: {self.kind!r}")
        if not self.pairs:
            raise DeclarationError(f"rename rule kind={self.kind} declares no pairs")
        for pair in self.pairs:
            if len(pair) != 2 or not str(pair[0] or "").strip():
                raise DeclarationError(f"malformed rename pair: {pair!r}")

    def apply(self, key: str) -> str:
        if self.kind == "prefix":
            out = key
            for src, dst in self.pairs:
                if out.startswith(src):
                    out = out.replace(src, dst, 1)
            return out
        if self.kind == "substring":
            out = key
            for src, dst in self.pairs:
                out = out.replace(src, dst)
            return out
        return self._alternation().sub(self._alternation_repl, key)

    def _alternation(self) -> re.Pattern[str]:
        cached = _ALTERNATION_CACHE.get(self.pairs)
        if cached is None:
            sources = sorted({src for src, _ in self.pairs}, key=len, reverse=True)
            cached = re.compile("|".join(re.escape(s) for s in sources))
            _ALTERNATION_CACHE[self.pairs] = cached
        return cached

    def _alternation_repl(self, match: re.Match[str]) -> str:
        for src, dst in self.pairs:
            if src == match.group(0):
                return dst
        return match.group(0)


_ALTERNATION_CACHE: dict[tuple[tuple[str, str], ...], re.Pattern[str]] = {}


@dataclass(frozen=True)
class RepackVariant:
    """One way a component may need converting. First match wins.

    ``probe_key`` is a raw state-dict key whose presence selects this variant
    (SD 2.0's ``text_model.embeddings.position_ids`` is the canonical case).
    A variant with ``probe_key=None`` matches unconditionally and must be last.
    """

    out_prefix: str = ""
    rules: tuple[RenameRule, ...] = ()
    probe_key: str | None = None
    transpose_keys: tuple[str, ...] = ()

    def matches(self, state_dict_keys: frozenset[str]) -> bool:
        return self.probe_key is None or self.probe_key in state_dict_keys


@dataclass(frozen=True)
class ComponentRepack:
    """How ONE diffusers component folder becomes part of a single-file export."""

    name: str
    variants: tuple[RepackVariant, ...]
    safetensors_bases: tuple[str, ...] = ("diffusion_pytorch_model",)
    bin_base: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise DeclarationError("ComponentRepack.name is empty")
        if not self.variants:
            raise DeclarationError(f"component {self.name!r} declares no variants")
        if self.variants[-1].probe_key is not None:
            raise DeclarationError(
                f"component {self.name!r}: the last variant must be unconditional "
                "(probe_key=None) — otherwise a tree matching none of them is "
                "silently dropped from the export"
            )
        if not self.safetensors_bases:
            raise DeclarationError(f"component {self.name!r} declares no weight bases")

    def variant_for(self, state_dict_keys: frozenset[str]) -> RepackVariant:
        for variant in self.variants:
            if variant.matches(state_dict_keys):
                return variant
        raise DeclarationError(f"component {self.name!r}: no variant matched")


@dataclass(frozen=True)
class LayoutSignature:
    """The component set a tree MUST look like for this family's converter to run.

    This is the permanent guard: two copies of a family's signature can
    disagree and route a tree to the wrong converter. With the signature
    declared beside the converter there is only one copy, and the engine
    refuses by name when the
    tree on disk does not match — whether the family was auto-detected or passed
    in by a caller.
    """

    requires_all: tuple[str, ...] = ()
    requires_any: tuple[tuple[str, ...], ...] = ()
    forbids: tuple[str, ...] = ()

    def violations(self, present: frozenset[str]) -> list[str]:
        out: list[str] = []
        for name in self.requires_all:
            if name not in present:
                out.append(f"missing required component {name!r}")
        for group in self.requires_any:
            if not (set(group) & present):
                out.append(f"missing all of {list(group)!r} (at least one is required)")
        for name in self.forbids:
            if name in present:
                out.append(f"carries {name!r}, which this family must NOT have")
        return out


@dataclass(frozen=True)
class SinglefileTarget:
    """A diffusers pipeline class (+ optional config donor repos) to try, in order.

    ``config_repos`` are HuggingFace repo ids used only as *config* donors when a
    single-file checkpoint omits a component (DiT-only fine-tunes ship no text
    encoder). They are foreign-catalog coordinates, which is why they are
    declared rather than hardcoded in the SDK.
    """

    pipeline_class: str
    config_repos: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.pipeline_class or "").strip():
            raise DeclarationError("SinglefileTarget.pipeline_class is empty")


@dataclass(frozen=True)
class DtypePolicy:
    """How a requested output dtype maps onto the load/save dtype.

    Generic, not family knowledge: ``preserve_fp16`` exists because loading an
    fp16 source in bf16 rounds a 10-bit mantissa to 7 bits irrecoverably.
    """

    load_default: str = "bf16"
    preserve: tuple[str, ...] = ("fp16", "fp32")
    allowed_output: tuple[str, ...] = ("fp16", "bf16", "fp32")


@dataclass(frozen=True)
class RepackageFamily:
    """One family's complete repackage declaration.

    ``aliases`` / ``alias_prefixes`` replace the SDK's ``_normalize_family``
    ladder: a fine-tune lineage (``sdxl-illustrious``, ``flux1-dev``) names the
    family it repackages as, declared by whoever owns the family.
    """

    family: str
    aliases: tuple[str, ...] = ()
    alias_prefixes: tuple[str, ...] = ()
    signature: LayoutSignature = field(default_factory=LayoutSignature)
    to_singlefile: tuple[ComponentRepack, ...] = ()
    to_diffusers: tuple[SinglefileTarget, ...] = ()
    dtypes: DtypePolicy = field(default_factory=DtypePolicy)

    def __post_init__(self) -> None:
        if not str(self.family or "").strip():
            raise DeclarationError("RepackageFamily.family is empty")
        if self.family == "unknown":
            raise DeclarationError("'unknown' is the no-match sentinel, not a family")
        seen: set[str] = set()
        for comp in self.to_singlefile:
            if comp.name in seen:
                raise DeclarationError(
                    f"family {self.family!r} declares component {comp.name!r} twice"
                )
            seen.add(comp.name)
        if self.to_singlefile and not (
            self.signature.requires_all or self.signature.requires_any or self.signature.forbids
        ):
            raise DeclarationError(
                f"family {self.family!r} declares a diffusers->singlefile converter with an "
                "EMPTY layout signature — a converter with no signature is exactly the "
                "pgw#740 wrong-converter defect, and is refused at declaration time"
            )

    @property
    def supports_diffusers_to_singlefile(self) -> bool:
        return bool(self.to_singlefile)

    @property
    def supports_singlefile_to_diffusers(self) -> bool:
        return bool(self.to_diffusers)


__all__ = [
    "ComponentRepack",
    "DeclarationError",
    "DtypePolicy",
    "LayoutSignature",
    "RenameRule",
    "RepackVariant",
    "RepackageFamily",
    "SinglefileTarget",
]
