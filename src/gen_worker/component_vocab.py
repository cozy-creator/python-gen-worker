"""The ONE component-name vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock


@dataclass(frozen=True)
class ComponentVocabulary:
    """Component names grouped by the role a consumer selects on."""

    denoisers: tuple[str, ...]
    text_encoders: tuple[str, ...]
    vaes: tuple[str, ...]
    extras: tuple[str, ...]
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
    raw = str(name or "").strip().strip("/")
    for sep in (".", "/"):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
    return raw


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
    """The vocabulary every consumer reads."""
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
    """Extend the vocabulary from an endpoint declaration."""
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
    """Restore the generic vocabulary."""
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
    """Components that carry weights worth quantizing/offloading/walking."""
    vocab = component_vocabulary()
    return _concat(vocab.denoisers, vocab.text_encoders, vocab.vaes, vocab.auxiliaries)


def quant_candidate_components() -> tuple[str, ...]:
    """Weight-bearing components a quantization pass may target: weight_components minus the VAEs — a quantized VAE is the first thing to show visible artifacting, so no default pass proposes one; callers must name a VAE explicitly."""
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
