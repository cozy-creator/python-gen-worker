"""The ONE component-name vocabulary.

pgw#740 (B5/B6): the tuple ``("transformer", "unet", ...)`` was copy-pasted into
20+ modules across ``models/`` and ``convert/``. The copies disagreed — Wan MoE's
``transformer_2`` and LTX's ``connectors`` were absent from several of them, which
is how components ended up silently un-offloaded and un-quantized.

The vocabulary is now declared, not repeated. The SDK ships the *generic*
diffusers vocabulary (the names diffusers itself defines); an endpoint whose
composition carries additional components declares them with
:func:`declare_components` and every consumer sees them at once.

Nothing here encodes a family: ``transformer_2`` is a diffusers component name,
not "Wan". A family-specific component (``connectors``) arrives by declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock


@dataclass(frozen=True)
class ComponentVocabulary:
    """Component names grouped by the role a consumer selects on.

    Order is meaningful: consumers that pick "the denoiser" take the first
    present name, so the most specific/likely name leads.
    """

    denoisers: tuple[str, ...]
    text_encoders: tuple[str, ...]
    vaes: tuple[str, ...]
    extras: tuple[str, ...]
    #: Weight-bearing components that are not one of the three primary roles
    #: (``controlnet``, ``prior``, ``decoder``, ``image_encoder``). They carry
    #: tensors — so they are walked, sized and quantized — but no consumer ever
    #: picks "the denoiser" or "the VAE" and means one of these.
    auxiliaries: tuple[str, ...] = ()

    @property
    def all(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for group in (self.denoisers, self.text_encoders, self.vaes,
                      self.auxiliaries, self.extras):
            for name in group:
                seen.setdefault(name, None)
        return tuple(seen)

    def is_denoiser(self, name: str) -> bool:
        return _head(name) in self.denoisers

    def is_text_encoder(self, name: str) -> bool:
        return _head(name) in self.text_encoders

    def is_vae(self, name: str) -> bool:
        return _head(name) in self.vaes

    def role_of(self, name: str) -> str:
        head = _head(name)
        if head in self.denoisers:
            return "denoiser"
        if head in self.text_encoders:
            return "text_encoder"
        if head in self.vaes:
            return "vae"
        if head in self.auxiliaries:
            return "auxiliary"
        if head in self.extras:
            return "extra"
        return "unknown"


def _head(name: str) -> str:
    """``vae.decode`` / ``transformer/config.json`` -> ``vae`` / ``transformer``."""
    raw = str(name or "").strip().strip("/")
    for sep in (".", "/"):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
    return raw


# The generic diffusers vocabulary. ``transformer_2`` is diffusers' name for a
# second denoiser stack (MoE experts / high-noise+low-noise pairs); ``dit`` is
# the name single-file exports use. Neither is family knowledge.
_GENERIC = ComponentVocabulary(
    denoisers=("transformer", "unet", "transformer_2", "dit"),
    text_encoders=("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"),
    vaes=("vae", "vae_encoder", "vae_decoder"),
    extras=("scheduler", "tokenizer", "tokenizer_2", "tokenizer_3",
            "feature_extractor", "safety_checker"),
    auxiliaries=("image_encoder", "controlnet", "prior", "decoder"),
)

_lock = RLock()
_current = _GENERIC


def component_vocabulary() -> ComponentVocabulary:
    """The vocabulary every consumer reads. Never mutate the result."""
    with _lock:
        return _current


def declare_components(
    *,
    denoisers: tuple[str, ...] = (),
    text_encoders: tuple[str, ...] = (),
    vaes: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    auxiliaries: tuple[str, ...] = (),
) -> ComponentVocabulary:
    """Extend the vocabulary from an endpoint declaration. Additive and idempotent.

    Called at endpoint-module import time, before any component enumeration.
    Names already present keep their position, so a declaration cannot reorder
    the generic vocabulary out from under another endpoint in the same process.
    """
    global _current
    with _lock:
        _current = ComponentVocabulary(
            denoisers=_merge(_current.denoisers, denoisers),
            text_encoders=_merge(_current.text_encoders, text_encoders),
            vaes=_merge(_current.vaes, vaes),
            extras=_merge(_current.extras, extras),
            auxiliaries=_merge(_current.auxiliaries, auxiliaries),
        )
        return _current


def reset_component_vocabulary() -> None:
    """Restore the generic vocabulary. Tests only."""
    global _current
    with _lock:
        _current = replace(_GENERIC)


def _merge(existing: tuple[str, ...], added: tuple[str, ...]) -> tuple[str, ...]:
    out = list(existing)
    for raw in added:
        name = str(raw or "").strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def denoiser_components() -> tuple[str, ...]:
    return component_vocabulary().denoisers


def text_encoder_components() -> tuple[str, ...]:
    return component_vocabulary().text_encoders


def weight_components() -> tuple[str, ...]:
    """Components that carry weights worth quantizing/offloading/walking.

    Everything except the config-only extras (scheduler, tokenizers,
    feature_extractor, safety_checker).
    """
    vocab = component_vocabulary()
    return _concat(vocab.denoisers, vocab.text_encoders, vocab.vaes, vocab.auxiliaries)


def quant_candidate_components() -> tuple[str, ...]:
    """Weight-bearing components a quantization pass may target.

    :func:`weight_components` minus the VAEs. A quantized VAE is the first
    thing to show as visible artifacting, so QUANTIZATION-POLICY.md keeps it
    at compute precision and no default pass proposes it. Callers that want a
    VAE quantized must name it explicitly.
    """
    vocab = component_vocabulary()
    return _concat(vocab.denoisers, vocab.text_encoders, vocab.auxiliaries)


def pipeline_component_dirs() -> tuple[str, ...]:
    """Every known component subdirectory name in a diffusers snapshot."""
    return component_vocabulary().all


def _concat(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for group in groups:
        for name in group:
            seen.setdefault(name, None)
    return tuple(seen)


__all__ = [
    "ComponentVocabulary",
    "component_vocabulary",
    "declare_components",
    "denoiser_components",
    "pipeline_component_dirs",
    "quant_candidate_components",
    "reset_component_vocabulary",
    "text_encoder_components",
    "weight_components",
]
